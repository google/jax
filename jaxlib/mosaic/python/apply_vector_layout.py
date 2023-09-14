# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unroll and pad operations to match native architectural sizes.

The rewrites are driven by the `in_layout` and `out_layout` op attributes,
which should have been populated by an earlier inference pass.
"""

# mypy: ignore-errors
import abc
import collections
from collections.abc import Sequence
import dataclasses
import enum
import functools
import math
import re
from typing import Any, Callable, Literal, Optional, Union, overload

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import math as math_dialect
from jaxlib.mlir.dialects import scf
from jaxlib.mlir.dialects import vector
import numpy as np

from . import infer_memref_layout
from . import tpu


ValueLike = Union[ir.Value, ir.Operation, ir.OpView]

TargetTuple = collections.namedtuple("TargetTuple", ["sublanes", "lanes"])
TARGET_SHAPE = TargetTuple(8, 128)


@enum.unique
class Direction(enum.Enum):
  SUBLANES = "sublanes"
  LANES = "lanes"
  SUBELEMENTS = "subelements"

  def __repr__(self):
    return self.name.lower()
SUBLANES = Direction.SUBLANES
LANES = Direction.LANES
SUBELEMENTS = Direction.SUBELEMENTS


class Replicated(enum.Enum):
  REPLICATED = "*"

  def __repr__(self):
    return "*"
  __str__ = __repr__

  def __bool__(self):
    return False  # Useful because we can then say `offset or 0`
REPLICATED = Replicated.REPLICATED


Offset = Union[int, Literal[REPLICATED]]


class ImplicitDim(enum.IntEnum):
  MINOR = -1
  SECOND_MINOR = -2

  def __repr__(self) -> str:
    return str(int(self))


@dataclasses.dataclass(frozen=True)
class VectorLayout:
  """Describes a mapping of an arbitrarily sized values into vregs.

  First, let us consider the simplest case, when implicit_dim is None, bitwidth
  is 32, and tiling matches the vreg shape. Then, the two last dimensions of a
  vector are tiled over sublanes and lanes respectively. If a value is too large
  to fit within a single vreg, then it continues in another vector register.
  For example purposes, we assume that vregs have 4 sublanes and 5 lanes from
  now on. A matrix with elements:

    a b c d e
    f g h i j
    k l m n o
    p q r s t

  laid out with offsets (1, 2) will use four vregs as follows:

    vreg 1      vreg 2
  . . . . .    . . . . .
  . . a b c    d e . . .
  . . f g h    i j . . .
  . . . . .    . . . . .

    vreg 3      vreg 4
  . . k l m    n o . . .
  . . p q r    s t . . .
  . . . . .    . . . . .
  . . . . .    . . . . .

  The dot character indicates padding. Nothing should be assumed about the
  value of those entries.

  If a value with this layout has rank >2, the leading dimensions will be
  unrolled over vregs. That is, the total number of vregs used to represent
  a value is equal to the product of all leading dimension sizes, and the number
  of vregs necessary to lay out the last two dimensions (as in the example).

  ---

  The implicit_dim attribute makes it possible to tile only the last dimension
  of a value, by implicitly inserting a singleton dimension that is tiled over
  sublanes (when implicit_dim is MINOR) or lanes (when implicit_dim is
  SECOND_MINOR).

  When the value has only one dimension, implicit_dim must be specified.

  ---

  The tiling attribute makes it possible to subdivide a single vector register
  into multiple subtiles that traverse the last dimension of a value. For
  example, consider vregs of shape (4, 5) an array:

    a b c d e f g h i j
    k l m n o p q r s t

  If we used a tiling of (4, 5), we would need two vregs to store this value,
  with the lower half of every register containing padding. But, if we use a
  tiling of (2, 5), both tiles fit into a single vreg:

    vreg 0
  a b c d e | tile 0
  k l m n o |
  f g h i j    | tile 1
  p q r s t    |

  Tiling is especially useful for compact storage of 1D values. Without it,
  we could use at most one sublane of every vector register. But, with a tiling
  of (1, 128) and implicit_dim being SECOND_MINOR, we can use all entries in a
  register to store long vectors.

  ---

  Finally, when the element bitwidth becomes smaller than 32, we use a two
  level tiling scheme, where elements of consecutive rows are packed into
  subelements. In TPU documentation this is often called a compressed layout.
  Note that this puts restrictions on the tile sizes, as they cannot have fewer
  rows than the packing factor (32 / bitwidth).

  Attributes:
    bitwidth: The bitwidth of the stored values.
    offsets: The coordinates of the first valid element. If an offset is
      REPLICATED, then any offset is valid as the value does not vary across
      sublanes or lanes respectively.
    tiling: The tiling used to lay out values (see the XLA docs). For values of
      bitwidth < 32, an implicit (32 // bitwidth, 1) tiling is appended to the
      one specified as an attribute.
    implicit_dim: If specified, the value has an implicit dim inserted in either
      minormost or second minormost position.
  """
  bitwidth: int
  offsets: tuple[Offset, Offset]  # Replication applies only within a tile.
  tiling: tuple[int, int]
  implicit_dim: Optional[ImplicitDim]

  def __post_init__(self):
    # TODO(b/275751535): Allow more bitwidths.
    assert self.bitwidth.bit_count() == 1 and self.bitwidth <= 32
    # Tiling should neatly divide the target shape, so that every vector
    # register ends up having the same structure.
    # Also, every tile should occupy a fixed number of sublanes.
    assert ((self.tiling[0] * self.tiling[1]) % (self.packing * TARGET_SHAPE[1])
            == 0)
    # Offsets should not exceed the tile size. The data always starts within the
    # first tile of a vreg.
    assert all(0 <= (o or 0) < t for o, t in zip(self.offsets, self.tiling))

  @property
  def packing(self) -> int:
    """Returns the number of values stored in a vreg entry."""
    return 32 // self.bitwidth

  @property
  def layout_rank(self) -> int:
    """The number of minormost dimensions tiled by this layout."""
    return 1 + (self.implicit_dim is None)

  @property
  def has_natural_topology(self) -> bool:
    """True, if every vector register has a layout without jumps.

    By without jumps we mean that traversing vregs over (sub)lanes always leads
    to a contiguous traversal of the (second) minormost dimension of data. This
    is only true for 32-bit types, since narrower types use two level tiling.
    """
    return (
        self.bitwidth == 32
        and self.tiling == TARGET_SHAPE
        and self.implicit_dim is None
    )

  @property
  def has_native_tiling(self) -> bool:
    """True, if every vector register has a natural "packed" topology.

    This is equivalent to has_natural_topology for 32-bit types, but generalizes
    it to narrower values with packed layouts too.
    """
    return self.tiling == native_tiling(self.bitwidth)

  @property
  def tiles_per_vreg(self) -> int:
    """How many tiles fit in each vector register."""
    tile_elems = self.tiling[0] * self.tiling[1]
    vreg_capacity = self.packing * TARGET_SHAPE[0] * TARGET_SHAPE[1]
    tiles_per_vreg, rem = divmod(vreg_capacity, tile_elems)
    assert rem == 0
    return tiles_per_vreg

  @property
  def sublanes_per_tile(self) -> int:
    """The number of sublanes necessary to store each tile."""
    sublanes_per_tile, rem = divmod(TARGET_SHAPE.sublanes, self.tiles_per_vreg)
    assert rem == 0
    return sublanes_per_tile

  @property
  def vreg_slice(self) -> TargetTuple:
    """Returns the size of a window contained in a single vreg.

    We never reuse the same vector register to store data of multiple rows,
    so only the minormost dimension can increase.
    """
    return TargetTuple(self.tiling[0], self.tiles_per_vreg * self.tiling[1])

  def implicit_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
    if self.implicit_dim is None:
      return shape
    elif self.implicit_dim == ImplicitDim.MINOR:
      return (*shape, 1)
    elif self.implicit_dim == ImplicitDim.SECOND_MINOR:
      return (*shape[:-1], 1, shape[-1])
    else:
      raise AssertionError(f"Invalid implicit dim: {self.implicit_dim}")

  def tile_array_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Returns the shape of an ndarray of vregs needed to represent a value.

    All but the last two dimensions are unrolled over vregs. In the last two
    dims we need as many vregs as indicated by dividing the point at which
    the value ends (given by the start offset plus the dim size) divided by
    the respective vreg capacity in that dim (and a ceiling if non-integral).
    If a value is replicated, then any offset is valid and we pick 0 to
    minimize the number of vregs.

    Args:
      shape: The shape of a value this layout applies to.
    """
    implicit_shape = self.implicit_shape(shape)
    vreg_slice = self.vreg_slice
    offsets = self.offsets
    tiles_shape = (
        *implicit_shape[:-2],
        *(cdiv((o or 0) + f, t)
          for o, f, t in zip(offsets, implicit_shape[-2:], vreg_slice))
    )
    # Remove the implicit dimension --- it's always of size 1.
    if self.implicit_dim is None:
      return tiles_shape
    elif self.implicit_dim == ImplicitDim.MINOR:
      return tiles_shape[:-1]
    elif self.implicit_dim == ImplicitDim.SECOND_MINOR:
      return (*tiles_shape[:-2], tiles_shape[-1])
    else:
      raise AssertionError(f"Invalid implicit dim: {self.implicit_dim}")

  def generalizes(self, other: "VectorLayout",
                  shape: Optional[tuple[int, ...]] = None) -> bool:
    """Returns True if the other layout is a special case of this one.

    In here, other is considered "a special case" when the set of vector
    register entries that represent a value in that layout is also the set of
    entries in which self stores the value. This is of course true for layouts
    that are equivalent, but it does not need to hold both ways. For example,
    a layout that implies the value does not change along an axis of the vector
    register is more general than the layout that picks a fixed starting point
    for the value and does not encode that assumption.

    The generalization relation is a non-strict partial order. You can think of
    it as a partial <= on vector layouts, but we don't overload Python operators
    since there's no clear way to decide where the bottom and top should be.

    Args:
      other: The layout compared against self.
      shape: An optional shape of the vector to which both layouts apply.
        The generalization relation is larger than usual for some shapes.  That
        is, if self.generalizes(other) then also self.generalizes(other, shape)
        for any shape, but that implication does not hold the other way around
        for some shapes.
    """
    if self.bitwidth != other.bitwidth:
      return False
    for s, o in zip(self.offsets, other.offsets):
      if s != o and s is not REPLICATED:
        return False
    if self.implicit_dim != other.implicit_dim:
      # Don't fail yet!
      # If the second-minor dimension is of size 1, then it does not matter
      # whether we have a second minor implicit dim or not.
      second_minor = ImplicitDim.SECOND_MINOR
      if not (
          shape is not None
          and self.implicit_shape(shape)[-2] == 1
          and {self.implicit_dim, other.implicit_dim} == {second_minor, None}
      ):
        return False
    if self.tiling != other.tiling:
      # Don't fail yet!
      # If there is only one tile in both tilings, then they are equivalent.
      if shape is None:
        return False
      ishape = self.implicit_shape(shape)
      if not (
          shape is not None
          and self.tiling[-1] == other.tiling[-1] == TARGET_SHAPE.lanes
          and (self.offsets[-1] or 0) + ishape[-1] <= TARGET_SHAPE.lanes
          and (self.offsets[-2] or 0) + ishape[-2] <= self.tiling[-2]
          and (self.offsets[-2] or 0) + ishape[-2] <= other.tiling[-2]
      ):
        return False
    return True

  def equivalent_to(self, other: "VectorLayout",
                    shape: Optional[tuple[int, ...]] = None) -> bool:
    """Returns True if the two layouts are equivalent.

    That is, when all potential vector entries where the value can be stored
    (there might be multiple choices for some layouts!) are equal in both
    self and other.

    Args:
      other: The layout compared against self.
      shape: An optional shape of the vector to which both layouts apply. More
        layouts are considered equivalent when the shape is specified. Also see
        the docstring of the generalizes method.
    """
    return self.generalizes(other, shape) and other.generalizes(self, shape)

  def tile_data_bounds(
      self,
      full_shape: tuple[int, ...],
      ixs: tuple[int, ...],
      allow_replicated: Union[bool, TargetTuple] = False,
  ) -> "VRegDataBounds":
    """Returns the bounds of the given tile that hold useful data.

    Arguments:
      full_shape: The shape of the full vector this layout applies to.
      ixs: The indices into an array of tiles representing the full vector
        (see tile_array_shape for bounds) selecting the tile for which the
        bounds are queried.
      allow_replicated: If False, no offset is allowed to be REPLICATED. If
        True, offsets are allowed to be REPLICATED, but the bounds will span
        the full dimension of the tile (i.e. potentially multiple repeats of
        the actual data).

    Returns:
      A TargetTuple of slices, indicating the span of useful data within the
      tile selected by idx.
    """
    # TODO(apaszke): allow_replicated could have been generalized to specify
    # what action should be taken when a REPLICATED offset is encountered.
    # Right now it either disallows replication, or selects the whole dimension.
    if self.implicit_dim == ImplicitDim.MINOR:
      s = ixs[-1],
      l = 0
    elif self.implicit_dim == ImplicitDim.SECOND_MINOR:
      s = 0
      l = ixs[-1]
    else:
      *_, s, l = ixs
    ns, nl = self.implicit_shape(self.tile_array_shape(full_shape))[-2:]
    implicit_shape = self.implicit_shape(full_shape)
    so, lo = self.offsets
    if isinstance(allow_replicated, bool):
      allow_replicated = TargetTuple(allow_replicated, allow_replicated)
    if not self.has_natural_topology:
      if any(o is REPLICATED for o in self.offsets):
        raise NotImplementedError
      if not all(0 <= o < t for o, t in zip(self.offsets, self.tiling)):
        raise NotImplementedError
      if (
          self.tiling[0] == 1
          and self.tiling[-1] % TARGET_SHAPE.lanes == 0
          and self.implicit_dim == ImplicitDim.SECOND_MINOR
      ):
        assert s == 0
        start_offset = 0
        end_offset = values_per_vreg = (
            TARGET_SHAPE.sublanes * TARGET_SHAPE.lanes * self.packing
        )
        if l == 0:
          start_offset = self.offsets[-1]
        if l == nl - 1:
          if rem := (self.offsets[-1] + implicit_shape[-1]) % values_per_vreg:
            end_offset = rem
        return SingleRowVRegBounds(self, start_offset, end_offset)
      if self.tiling[-1] != TARGET_SHAPE.lanes:
        raise NotImplementedError
      start_sublanes = so if s == 0 else 0
      start_lanes = lo if l == 0 else 0
      end_sublanes = self.tiling[-2]
      if s == ns - 1:
        if rem_sub := (so + implicit_shape[-2]) % self.tiling[-2]:
          end_sublanes = rem_sub
      end_lanes = self.tiling[1]
      num_tiles = self.tiles_per_vreg
      if l == nl - 1:
        minormost_tiles = cdiv(lo + implicit_shape[-1], self.tiling[-1])
        if rem_tiles := minormost_tiles % self.tiles_per_vreg:
          num_tiles = rem_tiles
        if rem_lanes := (lo + implicit_shape[-1]) % self.tiling[-1]:
          end_lanes = rem_lanes
      return TiledRectangularVRegBounds(
          self,
          num_tiles,
          TargetTuple(start_sublanes, start_lanes),
          TargetTuple(end_sublanes, end_lanes),
      )
    # TODO(apaszke): Remove this path in favor of TiledVRegBounds.
    shift = TargetTuple(0 if so is REPLICATED else so,
                        0 if lo is REPLICATED else lo)
    sb, lb = 0, 0
    se, le = TARGET_SHAPE
    # First, deal with sublanes.
    if s == 0:
      sb = shift.sublanes
    if so is REPLICATED:
      if not allow_replicated.sublanes:
        raise AssertionError("unexpected replicated offset")
      # Otherwise, do nothing. We take the full slice.
    elif s == ns - 1:
      if srem := (shift.sublanes + implicit_shape[-2]) % TARGET_SHAPE.sublanes:
        se = srem
    # Now, we deal with lanes.
    if l == 0:
      lb = shift.lanes
    if lo is REPLICATED:
      if not allow_replicated.lanes:
        raise AssertionError("unexpected replicated offset")
    elif l == nl - 1:
      if lrem := (shift.lanes + implicit_shape[-1]) % TARGET_SHAPE.lanes:
        le = lrem
    assert se > sb
    assert le > lb
    return RectangularVRegBounds(TargetTuple(slice(sb, se), slice(lb, le)))


@dataclasses.dataclass(frozen=True)
class VRegDataBounds(abc.ABC):
  """A representation of the vector register entries that contain actual data.

  Note that we assume that the vreg contains at least one element (or else
  there would be no reason to use it).
  """

  @abc.abstractmethod
  def mask_varies_along(self, direction: Direction) -> bool:
    """Determines whether all indices along a direction contain useful data."""

  @property
  def complete(self) -> bool:
    """If True, all bits in the vector register are used to represent the value."""
    # If the mask is constant and the vreg is non-empty (by assumption), then it
    # has to be complete.
    for d in Direction:
      if self.mask_varies_along(d):
        return False
    return True

  @abc.abstractmethod
  def get_vector_mask(self, generation: int) -> ir.Value:
    """Constructs a vector mask value that is true iff the entry contains useful data.

    The returned value can be an int32 bitmask too, when the target does not
    have sufficiently expressive vector masks.

    Args:
      generation: The target TPU generation.
    """

  @abc.abstractmethod
  def get_sublane_mask(self) -> ir.Attribute:
    """Constructs a DenseBoolArrayAttr containing a sublane mask for the vreg.

    The sublane mask should never have True for sublanes that do not contain
    useful data, but having an unmasked sublane doesn't imply that all bits
    in that sublane are used to represent data (relevant for packed layouts).
    """


@dataclasses.dataclass(frozen=True)
class RectangularVRegBounds(VRegDataBounds):
  """Represents a rectangular region of data within a vector register.

  This class is very limited in its power and should only be used for 32-bit
  values with native tiling.

  Attributes:
    bounds: A TargetTuple of slices encoding the bounds of the rectangular
      data region.
  """
  bounds: TargetTuple

  def mask_varies_along(self, direction: Direction) -> bool:
    """See base class."""
    if direction == SUBELEMENTS:  # Only 32-bit types supported.
      return False
    dir_bounds = getattr(self.bounds, direction.value)
    return dir_bounds != slice(0, getattr(TARGET_SHAPE, direction.value))

  def get_vector_mask(self, generation: int) -> ir.Value:
    """See base class."""
    low = [self.bounds.sublanes.start, self.bounds.lanes.start]
    high = [self.bounds.sublanes.stop, self.bounds.lanes.stop]
    return tpu.CreateMaskOp(
        ir.VectorType.get(TARGET_SHAPE, ir.IntegerType.get_signless(1)),
        low=list(map(ix_cst, low)), high=list(map(ix_cst, high))).result

  def get_sublane_mask(self) -> ir.Attribute:
    """See base class."""
    sublane_mask = np.full((TARGET_SHAPE.sublanes,), False, dtype=bool)
    sublane_mask[self.bounds.sublanes] = True
    return ir.DenseBoolArrayAttr.get(sublane_mask.tolist())


@dataclasses.dataclass(frozen=True)
class SingleRowVRegBounds(VRegDataBounds):
  """Represents a subset of a (packed) 1D vector register.

  All indices below are scaled up by the packing. That is, the maximal stop
  offset for a register containing 16-bit values is twice as large as for
  a register containing 32-bit values.

  Standard 1D packing is used. The values start laid out in the low half of the
  first sublane, then wrap around to the higher half of the first sublane, etc.

  Attributes:
    layout: The layout used to generate the bounds.
    start_offset: Index of the element from which the mask begins (inclusive).
    stop_offset: Index of the element at which the mask ends (exclusive).
  """
  layout: VectorLayout
  start_offset: int
  stop_offset: int

  def __post_init__(self):
    assert 0 <= self.start_offset < self.stop_offset <= self.entries_per_vreg

  @property
  def entries_per_vreg(self) -> int:
    """Total number of entries contained in a vreg."""
    return TARGET_SHAPE.lanes * TARGET_SHAPE.sublanes * self.layout.packing

  def mask_varies_along(self, direction: Direction) -> bool:
    """See base class."""
    if self.start_offset == 0 and self.stop_offset == self.entries_per_vreg:
      return False
    if direction == SUBELEMENTS:
      return (
          self.start_offset % self.layout.packing != 0
          or self.stop_offset % self.layout.packing != 0
      )
    elif direction == SUBLANES:
      return (
          self.start_offset >= TARGET_SHAPE.lanes
          or self.stop_offset < self.entries_per_vreg - TARGET_SHAPE.lanes
      )
    elif direction == LANES:
      return True
    else:
      raise AssertionError("Unhandled direction?")

  def get_vector_mask(self, generation: int) -> ir.Value:
    """See base class."""
    if self.mask_varies_along(SUBELEMENTS):
      raise NotImplementedError
    i32_vreg = ir.VectorType.get(TARGET_SHAPE, i32())
    def constant(v):
      return arith.ConstantOp(
          i32_vreg,
          ir.DenseElementsAttr.get_splat(
              i32_vreg, ir.IntegerAttr.get(i32(), v)
          ),
      )
    if self.layout.bitwidth == 32:
      start = constant(self.start_offset)
      end = constant(self.stop_offset)
    else:
      if (
          self.start_offset % (TARGET_SHAPE.lanes * self.layout.packing) != 0
          or self.stop_offset % (TARGET_SHAPE.lanes * self.layout.packing) != 0
      ):
        raise NotImplementedError
      start = constant(self.start_offset // self.layout.packing)
      end = constant(self.stop_offset // self.layout.packing)
    iota = tpu.IotaOp(i32_vreg).result
    pred_ge = ir.IntegerAttr.get(i64(), 5)
    pred_lt = ir.IntegerAttr.get(i64(), 2)
    return arith.AndIOp(
        arith.CmpIOp(pred_ge, iota, start), arith.CmpIOp(pred_lt, iota, end)
    ).result

  def get_sublane_mask(self) -> ir.Attribute:
    """See base class."""
    start_sublane = (
        self.start_offset // self.layout.packing // TARGET_SHAPE.lanes
    )
    end_sublane = cdiv(
        cdiv(self.stop_offset, self.layout.packing), TARGET_SHAPE.lanes
    )
    sublane_mask = np.full((TARGET_SHAPE.sublanes,), False, dtype=bool)
    sublane_mask[start_sublane:end_sublane] = True
    return ir.DenseBoolArrayAttr.get(sublane_mask.tolist())


@dataclasses.dataclass(frozen=True)
class TiledRectangularVRegBounds(VRegDataBounds):
  """Represents the data bounds within a vector register with tiled and potentially packed data.

  Note that the (packed) sublane offset from start_offset and (packed) sublane
  bound from end_offsets apply to all tiles within a vreg. On the other hand,
  the lane offset from start_offset only applies to the first tile, while
  lane bound from end_offset only applies to the last used tile.

  Attributes:
    layout: The layout of the value, mainly used for its bitwidth and tiling.
      Note that the layout offsets SHOULD NOT be used.
    num_tiles: The number of tiles at the beginning of the vreg that contain
      actual data.
    start_offsets: The lane and (packed) sublane offset within the first tile.
    end_offsets: The lane and (packed) sublane offset within the last used tile.
  """
  layout: VectorLayout
  num_tiles: int
  # TODO(apaszke): Don't use TargetTuple! offsets are in rows, not sublanes!
  start_offsets: TargetTuple
  end_offsets: TargetTuple

  def __post_init__(self):
    assert self.layout.tiling[-1] == TARGET_SHAPE.lanes
    assert 0 < self.num_tiles <= self.layout.tiles_per_vreg
    assert all(0 <= o < t
               for o, t in zip(self.start_offsets, self.layout.tiling))
    assert all(0 <= o <= t
               for o, t in zip(self.end_offsets, self.layout.tiling))

  @property
  def uses_all_tiles(self) -> bool:
    """True, if every tile in the vreg contains some data."""
    return self.num_tiles == self.layout.tiles_per_vreg

  def mask_varies_along(self, direction: Direction) -> bool:
    """See base class."""
    if direction == SUBLANES:
      return (
          not self.uses_all_tiles
          or self.start_offsets.sublanes != 0
          or self.end_offsets.sublanes != self.layout.tiling[-2]
      )
    elif direction == LANES:
      return (
          self.start_offsets.lanes != 0
          or self.end_offsets.lanes != self.layout.tiling[-1]
      )
    elif direction == SUBELEMENTS:
      return (
          self.start_offsets.sublanes % self.layout.packing != 0
          or self.end_offsets.sublanes % self.layout.packing != 0
      )
    else:
      raise NotImplementedError

  def get_vector_mask(self, generation: int) -> ir.Value:
    """See base class."""
    i1 = ir.IntegerType.get_signless(1)
    # I'm pretty sure this works for all bitwidths, but it's untested.
    if self.mask_varies_along(SUBELEMENTS):
      if self.layout.packing > 2:
        raise NotImplementedError  # TODO(b/300082350): Generalize this
      # For older TPUs, we virtualize masking, but only for simple cases.
      if generation < 4:
        if self.num_tiles > 1:
          raise NotImplementedError
        mask_vreg_ty = ir.VectorType.get(TARGET_SHAPE, i1)
      else:
        mask_vreg_ty = ir.VectorType.get(
            (*TARGET_SHAPE, self.layout.packing), i1
        )
    else:
      mask_vreg_ty = ir.VectorType.get(TARGET_SHAPE, i1)
    if self.complete:
      return arith.ConstantOp(
          mask_vreg_ty,
          ir.DenseElementsAttr.get_splat(
              mask_vreg_ty, ir.IntegerAttr.get(i1, True)
          ))
    mask = None
    assert self.num_tiles
    packing = self.layout.packing
    start_sub = self.start_offsets.sublanes // packing
    end_sub = cdiv(self.end_offsets.sublanes, packing)
    assert 0 <= start_sub < end_sub <= TARGET_SHAPE.sublanes
    for tile in range(self.num_tiles):
      sublane_offset = self.layout.sublanes_per_tile * tile
      row_offset = sublane_offset * self.layout.packing
      start_lane = 0
      if tile == 0:
        start_lane = self.start_offsets.lanes
      end_lane = TARGET_SHAPE.lanes
      if tile == self.num_tiles - 1:
        end_lane = self.end_offsets.lanes
      assert 0 <= start_lane < end_lane <= TARGET_SHAPE.lanes
      # TODO(apaszke): For loads/stores whole sublanes are covered by the
      # sublane mask, so we can focus only on lanes and partial sublanes.
      tile_mask = tpu.CreateMaskOp(
          mask_vreg_ty,
          map(ix_cst, [sublane_offset + start_sub, start_lane]),
          map(ix_cst, [sublane_offset + end_sub, end_lane]),
      )
      if self.mask_varies_along(SUBELEMENTS):
        start_row = self.start_offsets.sublanes + row_offset
        end_row = self.end_offsets.sublanes + row_offset
        if generation >= 4:
          # Only use non-trivial start/end if they don't fall on sublane
          # boundary. Otherwise CreateMaskOp already does the right thing. This
          # lets us use cheaper instruction sequences on TPUv4.
          if self.start_offsets.sublanes % self.layout.packing == 0:
            start_row = 0
          if self.end_offsets.sublanes % self.layout.packing == 0:
            end_row = TARGET_SHAPE.sublanes * self.layout.packing
          submask = tpu.CreateSubelementMaskOp(
              mask_vreg_ty, start_row, end_row, self.layout.packing)
          tile_mask = arith.AndIOp(tile_mask, submask)
        else:  # generation < 4
          if self.num_tiles > 1:
            raise NotImplementedError(
                "TPU generations before 4 cannot handle all bf16 masking"
            )
          def mask_cst(v):
            int_mask_ty = ir.VectorType.get(TARGET_SHAPE, i32())
            return arith.ConstantOp(
                int_mask_ty,
                ir.DenseElementsAttr.get_splat(
                    int_mask_ty, ir.IntegerAttr.get(i32(), v)
                ),
            )
          tile_bitmask = arith.SelectOp(
              tile_mask, mask_cst(0xFFFFFFFF), mask_cst(0)
          )
          if start_row % 2 != 0:
            element_bitpattern = 0xffff << 16
            row_mask = tpu.CreateMaskOp(
                mask_vreg_ty,
                map(ix_cst, [start_row // 2, 0]),
                map(ix_cst, [start_row // 2 + 1, TARGET_SHAPE[1]]),
            )
            row_bitmask = arith.SelectOp(
                row_mask, mask_cst(element_bitpattern), mask_cst(0xffffffff)
            )
            tile_bitmask = arith.AndIOp(tile_bitmask, row_bitmask)
          if end_row % 2 != 0:
            element_bitpattern = 0xffff
            row_mask = tpu.CreateMaskOp(
                mask_vreg_ty,
                map(ix_cst, [end_row // 2, 0]),
                map(ix_cst, [end_row // 2 + 1, TARGET_SHAPE[1]]),
            )
            row_bitmask = arith.SelectOp(
                row_mask, mask_cst(element_bitpattern), mask_cst(0xffffffff)
            )
            tile_bitmask = arith.AndIOp(tile_bitmask, row_bitmask)
          return tile_bitmask.result
      mask = tile_mask if mask is None else arith.OrIOp(mask, tile_mask)
    assert mask is not None
    return mask.result

  def get_sublane_mask(self) -> ir.Attribute:
    """See base class."""
    mask = [False] * TARGET_SHAPE.sublanes
    start = self.start_offsets.sublanes // self.layout.packing
    end = cdiv(self.end_offsets.sublanes, self.layout.packing)
    sublane_bound = self.num_tiles * self.layout.sublanes_per_tile
    for sub in range(0, sublane_bound, self.layout.sublanes_per_tile):
      assert not any(mask[sub + start : sub + end])
      for i in range(sub + start, sub + end):
        mask[i] = True
    return ir.DenseBoolArrayAttr.get(mask)


Layout = Optional[VectorLayout]

PATTERN = re.compile(
    r'#tpu.vpad<"([0-9]+),{([*0-9]+),([*0-9]+)},\(([0-9]+),([0-9]+)\)(,-1|,-2)?">'
)


def parse_offset(s: str) -> Offset:
  if s == "*": return REPLICATED
  return int(s)


def parse_layout(attr: ir.Attribute, value_type: ir.Type) -> Layout:
  """Parse an MLIR attribute into a Layout."""
  if ir.StringAttr.isinstance(attr):
    encoding = ir.StringAttr(attr).value
  else:
    encoding = str(attr)
  if encoding == r'#tpu.vpad<"none">':
    return None
  vector_type = ir.VectorType(value_type)
  if match := PATTERN.match(encoding):
    assert int(match[1]) == type_bitwidth(vector_type.element_type)
    if match[6] is None:
      implicit_dim = None
    elif match[6] == ",-1":
      implicit_dim = ImplicitDim.MINOR
    elif match[6] == ",-2":
      implicit_dim = ImplicitDim.SECOND_MINOR
    return VectorLayout(
        int(match[1]),
        (parse_offset(match[2]), parse_offset(match[3])),
        (int(match[4]), int(match[5])),
        implicit_dim=implicit_dim,
    )
  raise NotImplementedError(f"Unrecognized layout: {attr}")


def print_layout(layout: Layout) -> str:
  if layout is None:
    return '#tpu.vpad<"none">'
  o = [str(x) for x in layout.offsets]
  if layout.implicit_dim is not None:
    implicit_dim = "," + str(layout.implicit_dim.value)
  else:
    implicit_dim = ""
  return f'#tpu.vpad<"{layout.bitwidth},{{{o[0]},{o[1]}}},({layout.tiling[0]},{layout.tiling[1]}){implicit_dim}">'


def select_tiles_from_rotated_row_vregs(
    rotated_row_vregs: np.ndarray,
    start_src_col: int,
    end_src_col: int,
    first_dst_tile_sublane_offset: int,
    dst_layout: VectorLayout,
    hw_generation: int,
) -> ValueLike:
  """Assembles a destination tile using partial data from rotated vregs using a divide-and-conquer strategy.

  Arguments:
    rotated_row_vregs: A row of rotated vregs, from which destination tile(s)
      is/are to be selected to assemble a new vreg.
    src_layout: The source layout.
    start_src_col: The first rotated vreg in the row of rotated vregs to
      process.
    end_src_col: The last rotated vreg in the row of rotated vreg to process.
    first_dst_tile_sublane_offset: Sublane offset where the first dst tile to be
      selected starts.
    dst_layout: Destination layout, based on which retiling is being performed.
    hw_generation: The generation of a target hardware.

  Returns:
    A new vreg assembled from dst tiles stored in given rotated vregs.
  """

  if start_src_col > end_src_col:
    raise ValueError("Invalid values for start and end column.")
  if start_src_col == end_src_col:
    return rotated_row_vregs[start_src_col]

  mid_src_col = start_src_col + (end_src_col - start_src_col) // 2

  left_partial_vreg = select_tiles_from_rotated_row_vregs(
      rotated_row_vregs,
      start_src_col,
      mid_src_col,
      first_dst_tile_sublane_offset,
      dst_layout,
      hw_generation,
  )

  left_tiles_count = mid_src_col - start_src_col + 1
  right_first_dst_tile_sublane_offset = (
      first_dst_tile_sublane_offset
      + left_tiles_count * dst_layout.sublanes_per_tile
  ) % TARGET_SHAPE.sublanes

  right_partial_vreg = select_tiles_from_rotated_row_vregs(
      rotated_row_vregs,
      mid_src_col + 1,
      end_src_col,
      right_first_dst_tile_sublane_offset,
      dst_layout,
      hw_generation,
  )

  i1 = ir.IntegerType.get_signless(1)
  mask_vreg_ty = (
      ir.VectorType.get((*TARGET_SHAPE, 2), i1)
      if dst_layout.packing == 2
      else ir.VectorType.get(TARGET_SHAPE, i1)
  )

  if first_dst_tile_sublane_offset < right_first_dst_tile_sublane_offset:
    # The useful data sublanes in left vregs do not wrap around in vreg.
    # For e.g. consider (2,128) destination tiling and we are trying to merge
    # two vregs as follows:
    #
    #  vreg 0:        vreg 1:
    #  x x x x x     dst_tile_2
    #  x x x x x     dst_tile_3
    #  dst_tile_4    x x x x x
    #  dst_tile_5    x x x x x
    #  dst_tile_6    x x x x x
    #  dst_tile_7    x x x x x
    #  x x x x x     dst_tile_0
    #  x x x x x     dst_tile_1
    #
    # In the above case, the data we want to select from vreg 1 wraps around,
    # whereas vreg 0 useful data is contiguous. It is easier to create '1' mask
    # for vreg 0.
    sublanes_mask = tpu.CreateMaskOp(
        mask_vreg_ty,
        map(ix_cst, [first_dst_tile_sublane_offset, 0]),
        map(ix_cst, [right_first_dst_tile_sublane_offset, TARGET_SHAPE.lanes]),
    )
    return arith.SelectOp(sublanes_mask, left_partial_vreg, right_partial_vreg)

  sublanes_mask = tpu.CreateMaskOp(
      mask_vreg_ty,
      map(ix_cst, [right_first_dst_tile_sublane_offset, 0]),
      map(ix_cst, [first_dst_tile_sublane_offset, TARGET_SHAPE.lanes]),
  )
  return arith.SelectOp(sublanes_mask, right_partial_vreg, left_partial_vreg)


def is_positive_pow_2(x: int) -> bool:
  """Returns true iff the integer argument is positive and a power of 2.

  Arguments:
    x: an integer argument.
  """
  return (x > 0) and (x & (x - 1)) == 0


def retile_to_reduced_sublanes(
    value_shape: tuple[int, ...],
    src_layout: VectorLayout,
    src_vreg_array: np.ndarray,
    dst_layout: VectorLayout,
    hw_generation: int,
) -> np.ndarray:
  """Retiles across vregs to match the destination layout when the sublane tiling dimension is reduced.

  Arguments:
    value_shape: The shape of the value which needs to be retiled in vregs.
    src_layout: The source layout.
    src_vreg_array: An array of vregs storing source tiles.
    dst_layout: The destination layout, with reduced sublane dimension, based on
      which the retiling will be performed.
    hw_generation: The generation of a target hardware.

  Returns:
    A new array of vregs that store tiles based on the destination layout.
  """
  dst_tiling_sublane = dst_layout.tiling[-2]
  assert dst_tiling_sublane > 0 and dst_tiling_sublane < src_layout.tiling[-2] and is_positive_pow_2(dst_tiling_sublane)
  assert src_layout.tiling[-1] == dst_layout.tiling[-1]

  dst_vreg_array = np.empty(
      dst_layout.tile_array_shape(value_shape), dtype=object
  )

  # We need to rotate each src tile in each src vreg once so that that they can
  # be merged to form new vregs. If a src vreg contains more than one src tile,
  # it will be rotated once per src tile. Consider (8,512) tensor stored with
  # layout (8,128) in a vreg array of shape (1, 4). Each src vreg
  # contains one src tile in this case. Given, the destination layout is
  # (2,128), each src tile is divided into 4 destination tiles as shown below:
  #
  #  src_vreg_0_0:     src_vreg_0_1:    src_vreg_0_2:   src_vreg_0_3:
  # dst_tile_0_0_0    dst_tile_0_0_1   dst_tile_0_0_2  dst_tile_0_0_3
  # dst_tile_1_0_0    dst_tile_1_0_1   dst_tile_1_0_2  dst_tile_1_0_3
  # dst_tile_2_0_0    dst_tile_2_0_1   dst_tile_2_0_2  dst_tile_2_0_3
  # dst_tile_3_0_0    dst_tile_3_0_1   dst_tile_3_0_2  dst_tile_3_0_3

  # In this example, each src tile in the src vreg is rotated by
  # col *  sublanes_per_tile to produce the following rotated src vregs:
  #
  # rot_src_vreg_0_0: rot_src_vreg_0_1: rot_src_vreg_0_2: rot_src_vreg_0_3:
  #    dst_tile_0_0_0    dst_tile_3_0_1    dst_tile_2_0_2    dst_tile_1_0_3
  #    dst_tile_1_0_0    dst_tile_0_0_1    dst_tile_3_0_2    dst_tile_2_0_3
  #    dst_tile_2_0_0    dst_tile_1_0_1    dst_tile_0_0_2    dst_tile_3_0_3
  #    dst_tile_3_0_0    dst_tile_2_0_1    dst_tile_1_0_2    dst_tile_0_0_3

  # If there were 2 src tiles in the src vreg, we would have rotated each src
  # vreg twice, producing 2 rotated src vreg per src vreg. The rotation amount
  # is calculated from the src and the dest tiling.

  rotated_src_vregs_array = np.empty(
      (
          *(src_vreg_array.shape[:-1]),
          # Each vreg may store more than one src tile. We may have to rotate a
          # vreg, once for every src tile in the vreg.
          src_vreg_array.shape[-1] * src_layout.tiles_per_vreg,
      ),
      dtype=object,
  )

  for *other_dims, row, idx in np.ndindex(rotated_src_vregs_array.shape):
    tile_idx = idx % dst_layout.tiles_per_vreg
    dst_sublane = tile_idx * dst_layout.sublanes_per_tile
    src_col, src_tile_offset = divmod(idx, src_layout.tiles_per_vreg)
    src_vreg = src_vreg_array[(*other_dims, row, src_col)]
    src_sublane = src_tile_offset * src_layout.sublanes_per_tile
    rotate_amt = dst_sublane - src_sublane
    if rotate_amt == 0:
      rotated_src_vregs_array[(*other_dims, row, idx)] = src_vreg
      continue
    if rotate_amt < 0:
      rotate_amt = TARGET_SHAPE.sublanes + rotate_amt
    rotated_src_vregs_array[(*other_dims, row, idx)] = tpu.RotateOp(
        src_vreg, amount=rotate_amt, dimension=0
    )
  # Assemble output vregs using tiles from rotated vregs using select.
  # Given, above example, destination vregs are then assembled as follows:
  #  dst_vreg_0_0:
  # dst_tile_0_0_0
  # dst_tile_0_0_1
  # dst_tile_0_0_2
  # dst_tile_0_0_3

  #  dst_vreg_1_0: (Notice dst tiles are not in correct offset!)
  # dst_tile_1_0_3
  # dst_tile_1_0_0
  # dst_tile_1_0_1
  # dst_tile_1_0_2

  #  dst_vreg_2_0: (Notice dst tiles are not in correct offset!)
  # dst_tile_2_0_2
  # dst_tile_2_0_3
  # dst_tile_2_0_0
  # dst_tile_2_0_1

  #  dst_vreg_3_0: (Notice dst tiles are not in correct offset!)
  # dst_tile_3_0_1
  # dst_tile_3_0_2
  # dst_tile_3_0_3
  # dst_tile_3_0_0

  # Each destination vreg is assembled from destination tiles in multiple
  # rotated src vregs. In the above example, if we wanted each destination tile
  # to be in correct sublane offset in a rotated vreg, say rot_src_vreg_0_1,
  # before assembling the destination tiles, we would have had to rotate
  # src_vreg_0_1 four times, creating 4 rotated vregs (instead of 1) for each
  # src vreg. In the above example, we instead rotated a src vreg src_vreg_0_1
  # only once to obtain rot_src_vreg_0_1 where the dst_tile_0_0_1 is in correct
  # final sublane offset, i.e. 2. But notice the sublane offset of
  # dst_tile_1_0_1 in the same rotated vreg. Its correct final destination
  # sublane offset is 2, but in rot_src_vreg_0_1, its offset is 4. Its sublane
  # offset is off by 2. We need to correct these sublane offsets in the final
  # assembled dst vregs. A single rotation of each assembled dst vreg is needed
  # to correct such sublane offsets. This strategy reduces the number of sublane
  # rotations required. See comments below.
  tile_sublane_change_factor = src_layout.tiling[-2] // dst_layout.tiling[-2]
  for *other_dims, row, col in np.ndindex(dst_vreg_array.shape):
    rotated_vreg_row, first_dst_tile_offset = divmod(
        row, tile_sublane_change_factor
    )
    first_dst_tile_sublane_offset = (
        first_dst_tile_offset * dst_layout.sublanes_per_tile
    )
    src_vreg_array_col_start = col * dst_layout.tiles_per_vreg
    src_vreg_array_col_end = (
        min(
            ((col + 1) * dst_layout.tiles_per_vreg),
            rotated_src_vregs_array.shape[-1],
        )
        - 1
    )
    dst_tile = select_tiles_from_rotated_row_vregs(
        rotated_row_vregs=rotated_src_vregs_array[
            (*other_dims, rotated_vreg_row, slice(None))
        ],
        start_src_col=src_vreg_array_col_start,
        end_src_col=src_vreg_array_col_end,
        first_dst_tile_sublane_offset=first_dst_tile_sublane_offset,
        dst_layout=dst_layout,
        hw_generation=hw_generation,
    )
    if first_dst_tile_sublane_offset == 0:
      # No need to rotate. First dst tile is already at offset 0, which means
      # rest of the dst tiles are also at correct sublane offset.
      dst_vreg_array[(*other_dims, row, col)] = dst_tile
    else:
      # Fix the destination tile sublane offset by rotating assembled dest vreg
      # once (See comments above). The dst vregs are fixed as follows:
      # No rotation needed.
      # dst_tile_0_0_0
      # dst_tile_0_0_1
      # dst_tile_0_0_2
      # dst_tile_0_0_3

      # Rotated by -1 * (sublanes_per_tile=2) * (row=1):
      # dst_tile_1_0_0
      # dst_tile_1_0_1
      # dst_tile_1_0_2
      # dst_tile_1_0_3

      # Rotated by -1 * (sublanes_per_tile=2) * (row=2):
      # dst_tile_2_0_0
      # dst_tile_2_0_1
      # dst_tile_2_0_2
      # dst_tile_2_0_3

      # Rotated by -1 * (sublanes_per_tile=2) * (row=3):
      # dst_tile_3_0_0
      # dst_tile_3_0_1
      # dst_tile_3_0_2
      # dst_tile_3_0_3
      dst_vreg_array[(*other_dims, row, col)] = tpu.RotateOp(
          dst_tile,
          amount=TARGET_SHAPE.sublanes - first_dst_tile_sublane_offset,
          dimension=0,
      )
  return dst_vreg_array


def is_supported_reduced_sublanes_retile(
    src_layout: VectorLayout, dst_layout: VectorLayout
) -> bool:
  """Returns true iff the layout changes involve reduced sublanes per tile.

  Arguments:
    src_layout: The existing layout.
    dst_layout: The new layout based on which the retiling is to be carried out.
  """

  return (
      src_layout.implicit_dim is None
      and dst_layout.implicit_dim is None
      and all(
          (os or 0) == (ot or 0)
          for os, ot in zip(src_layout.offsets, dst_layout.offsets)
      )
      # TODO (kumudbhandari): We have not tested any tile size where
      # tile[-1] != TARGET_SHAPE.lanes. It should work but needs to be tested.
      and src_layout.tiling[-1] == dst_layout.tiling[-1] == TARGET_SHAPE.lanes
      and dst_layout.tiling[-2] < src_layout.tiling[-2]
      and src_layout.bitwidth == dst_layout.bitwidth
      and is_positive_pow_2(src_layout.tiling[-2])
      and is_positive_pow_2(dst_layout.tiling[-2])
  )


# TODO(apaszke): Test this function properly
def relayout(
    v: ir.Value, src: VectorLayout, dst: VectorLayout, hw_generation: int
) -> ValueLike:
  """Changes the layout of a vector value.

  Arguments:
    v: The value to relayout.
    src: The current layout of v.
    dst: The target layout of v.
    hw_generation: The generation of a target hardware.

  Returns:
    A new MLIR vector value, laid out as requested by dst.
  """
  if (bitwidth := src.bitwidth) != dst.bitwidth:
    raise ValueError("Can't change bitwidth during a relayout")
  packing = src.packing
  vty = ir.VectorType(v.type)
  src_tiles = disassemble(src, v)
  dst_tiles_shape = dst.tile_array_shape(vty.shape)
  if src.generalizes(dst, vty.shape):
    return assemble(vty, dst, src_tiles)
  if src.offsets == (REPLICATED, REPLICATED) and src.tiles_per_vreg == 1:
    # A fully replicated value is always easy to relayout.
    src_tiles_list = list(src_tiles.flat)
    # TODO(apaszke): It would be nice to be able to assert this here, but
    # given replicated values our rules can introduce equivalent expressions.
    # assert all(t is src_tiles_list[0] for t in src_tiles_list)
    dst_tiles = np.full(dst.tile_array_shape(vty.shape),
                        src_tiles_list[0], dtype=object)
    return assemble(vty, dst, dst_tiles)

  # Try to reconcile differences in implicit dim.
  if src.implicit_dim != dst.implicit_dim:
    if dst.implicit_dim is None and vty.shape[src.implicit_dim] == 1:
      src = VectorLayout(src.bitwidth, src.offsets, src.tiling, None)

  # Handle retiling from (1, 128) to (8, 128) for 32-bit data.
  if (
      src.implicit_dim is None
      and dst.implicit_dim is None
      and src.bitwidth == 32
      and src.offsets == (0, 0)
      and (dst.offsets == (REPLICATED, 0) or dst.offsets == (0, 0))
      and src.tiling == (1, 128)
      and dst.tiling == (8, 128)
      and src_tiles.shape[-2] == 1
  ):
    src_tiles_retiled = np.empty(
        dst.tile_array_shape(vty.shape), dtype=object
    )
    for *batch_idx, dst_col in np.ndindex(
        src_tiles_retiled.shape[:-2] + src_tiles_retiled.shape[-1:]
    ):
      src_col = dst_col // 8
      slane_idx = dst_col % 8
      gather_indices = ir.DenseI32ArrayAttr.get([slane_idx] * 8)
      src_tile = src_tiles[(*batch_idx, 0, src_col)]
      src_tiles_retiled[(*batch_idx, slice(None), dst_col)] = tpu.GatherOp(
          src_tile.type, src_tile, gather_indices, 0
      )
    src = dst
    src_tiles = src_tiles_retiled

  # TODO(apaszke): Generalize retiling to general 16-bit types (might need to
  # use a different unpacking op).
  # (8,128) -> (16,128) tiling change for packed 16-bit types.
  if (
      src.implicit_dim is None
      and dst.implicit_dim is None
      and ir.BF16Type.isinstance(vty.element_type)
      and src.offsets == dst.offsets
      and src.tiling == (8, 128)
      and dst.tiling == (16, 128)
  ):
    new_src = VectorLayout(src.bitwidth, src.offsets, dst.tiling, None)
    src_tiles_retiled = np.empty(
        new_src.tile_array_shape(vty.shape), dtype=object)
    for (*batch_idx, dst_row, dst_col) in np.ndindex(src_tiles_retiled.shape):
      src_row1 = src_tiles[(*batch_idx, dst_row * 2, dst_col // 2)]
      src_row2_row = min(dst_row * 2 + 1, src_tiles.shape[-2] - 1)
      src_row2 = src_tiles[(*batch_idx, src_row2_row, dst_col // 2)]

      vreg_part = dst_col % 2
      vreg_f32 = ir.VectorType.get(TARGET_SHAPE, ir.F32Type.get())
      half_row1 = tpu.UnpackSubelementsOp(vreg_f32, src_row1, vreg_part)
      half_row2 = tpu.UnpackSubelementsOp(vreg_f32, src_row2, vreg_part)
      src_tiles_retiled[(*batch_idx, dst_row, dst_col)] = tpu.PackSubelementsOp(
          src_row1.type, [half_row1, half_row2]
      )
    src = new_src
    src_tiles = src_tiles_retiled

  if is_supported_reduced_sublanes_retile(src, dst):
    src_tiles = retile_to_reduced_sublanes(
        value_shape=vty.shape,
        src_layout=src,
        src_vreg_array=src_tiles,
        dst_layout=dst,
        hw_generation=hw_generation,
    )
    src = dst

  # Fix up the offsets, assuming everything else matches between src and dst.
  if src.tiling == dst.tiling and src.implicit_dim == dst.implicit_dim:
    # TODO(apaszke): Changing an offset might add or remove one vreg.
    if dst_tiles_shape != src_tiles.shape:
      raise NotImplementedError("Offsets changing the vreg array shape")
    dst_tiles = src_tiles.copy()

    # Shifting rows.
    if src.offsets[0] is REPLICATED:
      row_diff = 0  # No data movement needed.
    elif dst.offsets[0] is REPLICATED:
      raise NotImplementedError("Sublane broadcast not implemented")
    else:
      row_diff = dst.offsets[0] - src.offsets[0]
    if row_diff != 0:
      # This is an easy case, because we never need to combine multiple vregs.
      if src.implicit_shape(vty.shape)[-2] != 1:
        raise NotImplementedError("Row shifts for multi-row values")
      src_sublane = src.offsets[0] // packing
      dst_sublane = dst.offsets[0] // packing
      if dst_sublane != src_sublane:
        sublane_diff = dst_sublane - src_sublane
        if sublane_diff < 0:
          sublane_diff += TARGET_SHAPE.sublanes
        sublane_diff_attr = ir.IntegerAttr.get(
            ir.IntegerType.get_signed(32), sublane_diff
        )
        for idx, tile in np.ndenumerate(src_tiles):
          dst_tiles[idx] = tpu.RotateOp(
              tile, amount=sublane_diff_attr, dimension=0
          ).result
      src_subelem = src.offsets[0] % packing
      dst_subelem = dst.offsets[0] % packing
      if dst_subelem != src_subelem:
        subelem_diff = dst_subelem - src_subelem
        shift_bits = bitwidth * abs(subelem_diff)
        bits_vreg_ty = ir.VectorType.get(TARGET_SHAPE, i32())
        shift_vreg = arith.ConstantOp(
            bits_vreg_ty,
            ir.DenseElementsAttr.get_splat(
                bits_vreg_ty, ir.IntegerAttr.get(i32(), shift_bits)
            ),
        )
        for idx, tile in np.ndenumerate(dst_tiles):
          bit_tile = tpu.BitcastOp(bits_vreg_ty, tile)
          if subelem_diff > 0:
            shift_tile = arith.ShLIOp(bit_tile, shift_vreg)
          elif subelem_diff < 0:
            shift_tile = arith.ShRUIOp(bit_tile, shift_vreg)
          else:
            raise AssertionError("unexpected equal subelements")
          dst_tiles[idx] = tpu.BitcastOp(tile.type, shift_tile).result

    # Shifting columns.
    if src.offsets[1] is REPLICATED:
      col_diff = 0
    elif dst.offsets[1] is REPLICATED:
      raise NotImplementedError("Lane broadcast not implemented")
    else:
      col_diff = dst.offsets[1] - src.offsets[1]
    if col_diff != 0:
      if row_diff != 0:
        raise NotImplementedError("Both columns and rows are shifted")
      if col_diff < 0:
        raise NotImplementedError("Shifts to the left")
      if bitwidth != 32:
        raise NotImplementedError("Only 32-bit column shifts supported")
      sublane_diff = col_diff
      sublane_diff_attr = ir.IntegerAttr.get(
          ir.IntegerType.get_signed(32), sublane_diff
      )
      if src_tiles.shape[-1] > 1:
        mask = tpu.CreateMaskOp(
            ir.VectorType.get(TARGET_SHAPE, ir.IntegerType.get_signless(1)),
            low=list(map(ix_cst, [0, 0])),
            high=list(map(ix_cst, [TARGET_SHAPE[0], col_diff])))
      for idx, tile in np.ndenumerate(src_tiles):
        rot_tile = tpu.RotateOp(tile, amount=sublane_diff_attr, dimension=1)
        if idx[-1] != 0:
          prev_rot_tile = dst_tiles[(*idx[:-1], idx[-1] - 1)]
          rot_tile = arith.SelectOp(mask, prev_rot_tile, rot_tile)
        dst_tiles[idx] = rot_tile

    return assemble(vty, dst, dst_tiles)
  # TODO(apaszke): Implement general relayout
  raise NotImplementedError(
      f"unsupported layout change for {vty}: {src} -> {dst}")


@dataclasses.dataclass
class RewriteContext:
  """A context provided to all rewrite rules.

  This class is meant to provide stateful helpers for common functionality
  to rewrite rules. For example, MLIR op deletions need to be delayed until
  all rewrites are complete, to work around issues in Python bindings to MLIR.
  """
  func: func.FuncOp
  hardware_generation: int

  def erase(self, op: Union[ir.Operation, ir.OpView]):
    if isinstance(op, ir.OpView):
      op = op.operation
    op.erase()

  def replace(self, old: Union[ir.Operation, ir.OpView], new: ValueLike):
    self.replace_all_uses_with(old, new)
    self.erase(old)

  def replace_all_uses_with(
      self, old: Union[ir.Operation, ir.OpView], new: ValueLike
  ):
    if isinstance(new, (ir.Operation, ir.OpView)):
      new = new.results
    else:
      new = [new]
    new_types = [v.type for v in new]
    assert new_types == old.results.types, f"{old.results.types} -> {new_types}"
    tpu.private_replace_all_uses_with(old, new)

  def set_operand(self, op: ir.Operation, idx: int, new: ir.Value):
    tpu.private_set_operand(op, idx, new)

  def set_operands(self, op: ir.Operation, new: list[ir.Value]):
    tpu.private_set_operands(op, new)

  def move_all_regions(self, src: ir.Operation, dst: ir.Operation):
    tpu.private_move_all_regions(src, dst)

  def append_constant(self, value: ir.DenseElementsAttr) -> ir.Value:
    """Hoist a vector constant as an additional argument of the function."""
    (entry_block,) = self.func.body
    value_ty = ir.VectorType(value.type)
    if ir.IntegerType.isinstance(value_ty.element_type):
      if ir.IntegerType(value_ty.element_type).width != 32:
        raise NotImplementedError("Only 32-bit constants supported")
    elif not ir.F32Type.isinstance(value_ty.element_type):
      raise NotImplementedError("Only 32-bit constants supported")
    arg_type = infer_memref_layout.infer_memref(
        ir.MemRefType.get(value_ty.shape, value_ty.element_type),
        self.hardware_generation)
    argument = tpu.private_insert_argument(
        len(entry_block.arguments) - 1, entry_block, arg_type
    )
    func_ty = self.func.type
    # Adjust the function type.
    new_arg_tys = [*func_ty.inputs[:-1], arg_type, func_ty.inputs[-1]]
    self.func.attributes["function_type"] = ir.TypeAttr.get(
        ir.FunctionType.get(new_arg_tys, func_ty.results)
    )
    # Adjust the constants attribute.
    if "vector_constants" in self.func.attributes:
      prev_cst = ir.ArrayAttr(self.func.attributes["vector_constants"])
      self.func.attributes["vector_constants"] = prev_cst + [value]
    else:
      self.func.attributes["vector_constants"] = ir.ArrayAttr.get([value])
    # Adjust window params for the extra operand.
    if "window_params" in self.func.attributes:
      iteration_rank = len(
          ir.DenseI64ArrayAttr(self.func.attributes["iteration_bounds"]))
      iter_idx = ",".join(f"x{i}" for i in range(iteration_rank))
      arg_idx = ",".join([str(0)] * arg_type.rank)
      new_param = ir.DictAttr.get({
          "transform_indices":
              ir.Attribute.parse(f"affine_map<({iter_idx}) -> ({arg_idx})>")
      })
      window_params = list(ir.ArrayAttr(self.func.attributes["window_params"]))
      window_params.insert(-1, new_param)
      self.func.attributes["window_params"] = ir.ArrayAttr.get(window_params)
    return argument


def apply_layout_func(ctx: RewriteContext, f: func.FuncOp):
  """Rewrites the function according to layout annotations of its operations.

  Args:
    ctx: The context used for rewriting.
    f: An MLIR function to be rewritten.
  """
  (entry_block,) = f.body
  apply_layout_block(ctx, entry_block)


def apply_layout_block(ctx: RewriteContext, block: ir.Block):
  # We'll be modifying the block, so make a list of operations beforehand.
  for op in list(block):
    apply_layout_op(ctx, op)


# TODO(apaszke): Implement a debug mode that inserts additional assertions.
# For example, we should verify that ops that were supposed to generate
# replicated outputs satisfy that requirement.
def apply_layout_op(ctx: RewriteContext, op: ir.OpView):
  """Rewrites the operation according to its layout annotations.

  Args:
    ctx: The context used for rewriting.
    op: An MLIR operation to be rewritten.
  """
  # When an operation does not have any operands, the layout_in tuple is empty.
  # If one of the operands is a scalar value, the corresponding entry in the
  # layout_in tuple will be None. The same applies to the results of the
  # operation and the layout_out tuple.
  if "out_layout" in op.attributes:
    arr_attr = ir.ArrayAttr(op.attributes["out_layout"])
    layout_out = tuple(
        parse_layout(a, o.type) for a, o in zip(arr_attr, op.results)
    )
  else:
    layout_out = ()
    assert not op.results, str(op)
  if "in_layout" in op.attributes:
    arr_attr = ir.ArrayAttr(op.attributes["in_layout"])
    layout_in = tuple(
        parse_layout(a, o.type) for a, o in zip(arr_attr, op.operands)
    )
    # Ensure that the out_layout at definition site matches the in_layout
    # declared here.
    # TODO(apaszke): Figure out a story for handling conflicts!
    assert len(layout_in) == len(op.operands)
    for idx, (v, li) in enumerate(zip(op.operands, layout_in)):
      vty = try_cast(v.type, ir.VectorType)
      if not vty:
        assert li is None
        continue
      def_op = v.owner
      assert isinstance(def_op, (ir.OpView, ir.Operation)), def_op
      res_idx = ir.OpResult(v).result_number
      arr_attr = ir.ArrayAttr(def_op.attributes["out_layout"])
      lo = parse_layout(arr_attr[res_idx], vty)
      if lo is None:
        raise ValueError("vector result should have a defined layout")
      if lo.generalizes(li):
        continue
      with ir.InsertionPoint(op), op.location:
        new_v = relayout(
            v, src=lo, dst=li, hw_generation=ctx.hardware_generation
        ).result
        ctx.set_operand(op, idx, new_v)
  else:
    layout_in = ()
    assert not op.operands, op
  no_vector_args = all(l is None for l in layout_out) and all(
      l is None for l in layout_in
  )
  if no_vector_args and not op.regions:
    # We don't need to do anything for scalar operations.
    if op.operands:
      del op.attributes["in_layout"]
    if op.results:
      del op.attributes["out_layout"]
  elif rule := _rules.get(op.OPERATION_NAME, None):
    if not layout_in:
      layout_in = None
    elif len(layout_in) == 1:
      layout_in = layout_in[0]
    if not layout_out:
      layout_out = None
    elif len(layout_out) == 1:
      layout_out = layout_out[0]
    with ir.InsertionPoint(op), op.location:
      rule(ctx, op, layout_in, layout_out)
  else:
    raise NotImplementedError(f"Unsupported operation: {op.OPERATION_NAME}")


def apply(module: ir.Module, hardware_generation: int):
  with module.context:
    for f in module.body:
      # TODO(apaszke): Don't allow constants in index transforms for now.
      ctx = RewriteContext(f, hardware_generation)
      if not isinstance(f, func.FuncOp):
        raise ValueError(f"Unexpected op in module body: {f.OPERATION_NAME}")
      apply_layout_func(ctx, f)


################################################################################
# Rewrite rules
################################################################################

_rules: dict[str, Callable[[RewriteContext, Any, Any, Any], None]] = {}


def _register_rule(name):
  def do_register(fn):
    _rules[name] = fn
    return fn
  return do_register


@_register_rule("arith.constant")
def _arith_constant_rule(ctx: RewriteContext, op: arith.ConstantOp,  # pylint: disable=missing-function-docstring
                         layout_in: Layout, layout_out: Layout):
  assert layout_in is None
  ty = op.result.type
  if vty := try_cast(ty, ir.VectorType):
    assert layout_out is not None
    value = ir.DenseElementsAttr(op.value)
    target_vty = native_vreg_ty(vty.element_type)
    if value.is_splat:
      if layout_out.offsets != (REPLICATED, REPLICATED):
        raise NotImplementedError("Non-replicated splat constants")
      new_value = ir.DenseElementsAttr.get_splat(
          target_vty, value.get_splat_value())
      tile = arith.ConstantOp(target_vty, new_value)
      tiles = np.full(layout_out.tile_array_shape(vty.shape), tile)
      return ctx.replace(op, assemble(ty, layout_out, tiles))
    else:
      if type_bitwidth(vty.element_type) != 32:
        raise NotImplementedError(
            "Only 32-bit non-splat constants are supported"
        )
      ref = ctx.append_constant(value)
      load_op = vector.LoadOp(vty, ref, [ix_cst(0)] * vty.rank)
      ctx.replace_all_uses_with(op, load_op.result)
      return _vector_load_rule(
          ctx,
          load_op,
          [None] * (vty.rank + 1),
          VectorLayout(32, (0, 0), TARGET_SHAPE, None),
      )
  raise NotImplementedError(f"Unsupported arith.const type: {ty}")


def _elementwise_op_rule(factory,  # pylint: disable=missing-function-docstring
                         ctx: RewriteContext, op: ir.Operation,
                         layout_in: Union[Layout, tuple[Layout, ...]],
                         layout_out: Layout):
  if not isinstance(layout_in, tuple):
    layout_in = (layout_in,)
  ty = ir.VectorType(op.result.type)
  check(all(isinstance(l, type(layout_out)) for l in layout_in),
        "inconsistent layout kinds in elementwise operation")
  check(all(l.generalizes(layout_out, ty.shape) for l in layout_in),
        "incompatible layouts in elementwise operation")
  in_tile_arrays = [disassemble(l, o) for l, o in zip(layout_in, op.operands)]
  # Note that we have to broadcast to handle replicate dimensions.
  bit = np.broadcast(*in_tile_arrays)
  out_tile_array = np.ndarray(bit.shape, dtype=object)
  for idx, tiles in zip(np.ndindex(bit.shape), bit):
    out_tile_array[idx] = factory(*tiles)
  return ctx.replace(op, assemble(ty, layout_out, out_tile_array))

_register_rule("arith.addf")(
    functools.partial(_elementwise_op_rule, arith.AddFOp))
_register_rule("arith.addi")(
    functools.partial(_elementwise_op_rule, arith.AddIOp))
_register_rule("arith.subf")(
    functools.partial(_elementwise_op_rule, arith.SubFOp))
_register_rule("arith.subi")(
    functools.partial(_elementwise_op_rule, arith.SubIOp))
_register_rule("arith.mulf")(
    functools.partial(_elementwise_op_rule, arith.MulFOp))
_register_rule("arith.muli")(  # go/NOTYPO
    functools.partial(_elementwise_op_rule, arith.MulIOp))
_register_rule("arith.divf")(
    functools.partial(_elementwise_op_rule, arith.DivFOp))
_register_rule("arith.divsi")(
    functools.partial(_elementwise_op_rule, arith.DivSIOp))
_register_rule("arith.remsi")(
    functools.partial(_elementwise_op_rule, arith.RemSIOp))
_register_rule("arith.maximumf")(
    functools.partial(_elementwise_op_rule, arith.MaximumFOp)
)
_register_rule("arith.minimumf")(
    functools.partial(_elementwise_op_rule, arith.MinimumFOp)
)
_register_rule("arith.select")(
    functools.partial(_elementwise_op_rule, arith.SelectOp))
_register_rule("arith.index_cast")(
    functools.partial(_elementwise_op_rule, arith.IndexCastOp))
_register_rule("arith.andi")(
    functools.partial(_elementwise_op_rule, arith.AndIOp))
_register_rule("arith.ori")(
    functools.partial(_elementwise_op_rule, arith.OrIOp))
_register_rule("arith.negf")(
    functools.partial(_elementwise_op_rule, arith.NegFOp))
_register_rule("arith.xori")(
    functools.partial(_elementwise_op_rule, arith.XOrIOp))
_register_rule("arith.shli")(
    functools.partial(_elementwise_op_rule, arith.ShLIOp))
_register_rule("arith.shrui")(
    functools.partial(_elementwise_op_rule, arith.ShRUIOp))
_register_rule("math.absi")(
    functools.partial(_elementwise_op_rule, math_dialect.AbsIOp)
)
_register_rule("math.exp")(
    functools.partial(_elementwise_op_rule, math_dialect.ExpOp))
_register_rule("math.exp2")(
    functools.partial(_elementwise_op_rule, math_dialect.Exp2Op))
_register_rule("math.cos")(
    functools.partial(_elementwise_op_rule, math_dialect.CosOp))
_register_rule("math.sin")(
    functools.partial(_elementwise_op_rule, math_dialect.SinOp))
_register_rule("math.powf")(
    functools.partial(_elementwise_op_rule, math_dialect.PowFOp))
_register_rule("math.fpowi")(
    functools.partial(_elementwise_op_rule, math_dialect.FPowIOp)
)
_register_rule("math.rsqrt")(
    functools.partial(_elementwise_op_rule, math_dialect.RsqrtOp))
_register_rule("math.tanh")(
    functools.partial(_elementwise_op_rule, math_dialect.TanhOp))
_register_rule("math.log")(
    functools.partial(_elementwise_op_rule, math_dialect.LogOp))


def _ext_op_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext, op, layout_in: VectorLayout, layout_out: VectorLayout
):
  result_ty = ir.VectorType(op.result.type)
  if layout_out.bitwidth != 32:
    raise NotImplementedError("Only extensions to 32-bit supported")
  input_vregs = disassemble(layout_in, op.in_)
  output_vregs = np.empty(
      layout_out.tile_array_shape(result_ty.shape), dtype=object
  )
  res_vreg_ty = native_vreg_ty(result_ty.element_type)
  if layout_in.implicit_dim != layout_out.implicit_dim:
    raise NotImplementedError("Change of layout during the cast")
  if layout_in.implicit_dim is not None:
    if layout_in.implicit_dim != ImplicitDim.SECOND_MINOR:
      raise NotImplementedError("Only casts of lane-oriented values supported")
    if input_vregs.shape != (1,) or output_vregs.shape != (1,):
      raise NotImplementedError
    if layout_in.offsets[0] >= TARGET_SHAPE.sublanes:
      raise NotImplementedError
    output_vregs[:] = tpu.UnpackSubelementsOp(
        res_vreg_ty, input_vregs.item(), 0
    )
  elif layout_in.implicit_dim is None:
    if layout_in.tiling != TARGET_SHAPE or layout_out.tiling != TARGET_SHAPE:
      raise NotImplementedError(f"Only {TARGET_SHAPE} tiling supported")
    packing = layout_in.packing
    for idx in np.ndindex(output_vregs.shape):
      input_vreg_idx = (*idx[:-1], idx[-1] // packing)
      vreg_part = idx[-1] % packing
      output_vregs[idx] = tpu.UnpackSubelementsOp(
          res_vreg_ty, input_vregs[input_vreg_idx], vreg_part
      )
  else:
    raise NotImplementedError("Unrecognized layout")
  return ctx.replace(op, assemble(result_ty, layout_out, output_vregs))


@_register_rule("arith.extf")
def _arith_extf_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext,
    op: arith.ExtFOp,
    layout_in: VectorLayout,
    layout_out: VectorLayout,
):
  if layout_in.bitwidth != 16 or layout_out.bitwidth != 32:
    raise NotImplementedError("Only 16-bit to 32-bit conversion supported")
  return _ext_op_rule(ctx, op, layout_in, layout_out)


@_register_rule("arith.extsi")
def _arith_extsi_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext,
    op: arith.ExtSIOp,
    layout_in: VectorLayout,
    layout_out: VectorLayout,
):
  return _ext_op_rule(ctx, op, layout_in, layout_out)


def _trunc_op_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext, op, layout_in: VectorLayout, layout_out: VectorLayout
):
  result_ty = ir.VectorType(op.result.type)
  input_vregs = disassemble(layout_in, op.in_)
  output_vregs = np.empty(
      layout_out.tile_array_shape(result_ty.shape), dtype=object
  )
  if layout_in.bitwidth != 32:
    raise NotImplementedError("Only 32-bit truncation supported")
  res_vreg_ty = native_vreg_ty(result_ty.element_type)
  if layout_in.implicit_dim is None and layout_out.implicit_dim is None:
    if layout_in.tiling != TARGET_SHAPE:
      raise NotImplementedError("Only (8,128) tiling supported")
    if layout_out.tiling == TARGET_SHAPE:
      packing = layout_out.packing
      for idx in np.ndindex(output_vregs.shape):
        parts = []
        col_base = idx[-1] * packing
        for i in range(packing):
          # Pack any data lying around if OOB
          col = min(col_base + i, input_vregs.shape[-1] - 1)
          parts.append(input_vregs[(*idx[:-1], col)])
        output_vregs[idx] = tpu.PackSubelementsOp(res_vreg_ty, parts)
    elif layout_out.bitwidth == 16 and layout_out.tiling == (16, 128):
      for idx in np.ndindex(output_vregs.shape):
        first = input_vregs[(*idx[:-2], idx[-2] * 2, idx[-1])]
        if idx[-2] * 2 + 1 == input_vregs.shape[-2]:
          second = first  # OOB, so we can pack any data lying around.
        else:
          second = input_vregs[(*idx[:-2], idx[-2] * 2 + 1, idx[-1])]
        output_vregs[idx] = tpu.PackSubelementsOp(res_vreg_ty, [first, second])
  return ctx.replace(op, assemble(result_ty, layout_out, output_vregs))


@_register_rule("arith.truncf")
def _arith_truncf_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext,
    op: arith.TruncFOp,
    layout_in: VectorLayout,
    layout_out: VectorLayout,
):
  if layout_in.bitwidth != 32 or layout_out.bitwidth != 16:
    raise NotImplementedError("Only 32-bit to 16-bit conversion supported")
  return _trunc_op_rule(ctx, op, layout_in, layout_out)


@_register_rule("arith.trunci")
def _arith_trunci_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext,
    op: arith.TruncIOp,
    layout_in: VectorLayout,
    layout_out: VectorLayout,
):
  return _trunc_op_rule(ctx, op, layout_in, layout_out)


@_register_rule("arith.cmpi")
def _arith_cmpi_rule(ctx: RewriteContext, op: arith.CmpIOp,
                     layout_in: VectorLayout, layout_out: VectorLayout):
  def factory(lhs, rhs):
    return arith.CmpIOp(op.attributes["predicate"], lhs, rhs)
  return _elementwise_op_rule(factory, ctx, op, layout_in, layout_out)


@_register_rule("arith.cmpf")
def _arith_cmpf_rule(ctx: RewriteContext, op: arith.CmpFOp,
                     layout_in: VectorLayout, layout_out: VectorLayout):
  def factory(lhs, rhs):
    return arith.CmpFOp(op.attributes["predicate"], lhs, rhs)
  return _elementwise_op_rule(factory, ctx, op, layout_in, layout_out)


@_register_rule("arith.extui")
def _arith_extui_rule(ctx: RewriteContext, op: arith.ExtUIOp,
                      layout_in: VectorLayout, layout_out: VectorLayout):
  dtype = ir.VectorType(op.result.type).element_type
  def factory(x):
    x_ty = ir.VectorType(x.type)
    out_ty = ir.VectorType.get(x_ty.shape, dtype)
    return arith.ExtUIOp(out_ty, x)
  return _elementwise_op_rule(factory, ctx, op, layout_in, layout_out)


@_register_rule("arith.sitofp")
def _arith_siofp_rule(ctx: RewriteContext, op: arith.SIToFPOp,
                      layout_in: VectorLayout, layout_out: VectorLayout):
  element_type = ir.VectorType(op.result.type).element_type
  def factory(tile):
    tile_shape = ir.VectorType(tile.type).shape
    return arith.SIToFPOp(ir.VectorType.get(tile_shape, element_type), tile)
  return _elementwise_op_rule(factory, ctx, op, layout_in, layout_out)


@_register_rule("arith.fptosi")
def _arith_fptosi_rule(ctx: RewriteContext, op: arith.FPToSIOp,
                       layout_in: VectorLayout, layout_out: VectorLayout):
  element_type = ir.VectorType(op.result.type).element_type
  def factory(tile):
    tile_shape = ir.VectorType(tile.type).shape
    return arith.FPToSIOp(ir.VectorType.get(tile_shape, element_type), tile)
  return _elementwise_op_rule(factory, ctx, op, layout_in, layout_out)


@_register_rule("func.return")
def _func_return_rule(ctx: RewriteContext, op: func.ReturnOp,
                      layout_in: Layout, layout_out: Layout):
  del ctx, op  # Unused.
  assert layout_out is None
  if not isinstance(layout_in, tuple):
    layout_in = (layout_in,)
  if any(l is not None for l in layout_in):
    raise ValueError("vector-typed return values are not supported")


@_register_rule("scf.if")
def _scf_if_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext,
    op: scf.IfOp,
    layout_in: None,  # pylint: disable=unused-argument
    layout_out: Union[Layout, tuple[Layout, ...]],
):
  if len(op.results) == 1:
    layout_out = (layout_out,)
  true_region, false_region = op.regions
  (true_block,) = true_region.blocks
  true_yield = true_block.operations[len(true_block.operations) - 1]
  assert isinstance(true_yield, scf.YieldOp)
  if (
      layout_out
      and true_yield.attributes["in_layout"] != op.attributes["out_layout"]
  ):
    raise NotImplementedError(
        "different layouts in then yield's operands and if's results"
    )
  apply_layout_block(ctx, true_block)
  if not false_region.blocks:
    assert (
        not op.results
    ), "expected no results if op does not have an else block"
    assert layout_out is None
    return
  (false_block,) = false_region.blocks
  false_yield = false_block.operations[len(false_block.operations) - 1]
  assert isinstance(false_yield, scf.YieldOp)
  if (
      layout_out
      and false_yield.attributes["in_layout"] != op.attributes["out_layout"]
  ):
    raise NotImplementedError(
        "different layouts in else yield's operands and if's results"
    )
  apply_layout_block(ctx, false_block)

  # Apply layout to results after applying layout in the true and false regions.
  if not op.results:
    assert layout_out is None
    return
  assert len(op.results) == len(layout_out)
  # If scf.if has results, it should have both non-empty true and false regions.
  assert true_region.blocks and false_region.blocks

  # Move true and false regions to the new if op whose result has same type
  # and layout as yield operand's.
  new_op = scf.IfOp(
      op.condition,
      [operand.type for operand in true_yield.operands],
      hasElse=True,
  )
  ctx.move_all_regions(op, new_op)

  index = 0  # Tracking the index of unrolled result in new_op.results.
  rolled_results = []
  for result, layout in zip(op.results, layout_out):
    if not ir.VectorType.isinstance(result.type):
      assert layout is None
      rolled_results.append(new_op.results[index])
      index += 1
    else:
      # When the result has a vector type, assemble the result.
      assert isinstance(layout, VectorLayout)
      tiles_shape = layout.tile_array_shape(result.type.shape)
      num_vectors = np.prod(tiles_shape)
      tiles = np.array(new_op.results[index : index + num_vectors]).reshape(
          tiles_shape
      )
      index += num_vectors
      rolled_op = assemble(result.type, layout, tiles)
      rolled_results.append(rolled_op.result)

  tpu.private_replace_all_uses_with(op, rolled_results)
  ctx.erase(op)


@_register_rule("scf.yield")
def _scf_yield_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext,
    op: scf.YieldOp,
    layout_in: Union[Layout, tuple[Layout, ...]],
    layout_out: None,  # pylint: disable=unused-argument
):
  if not op.operands:
    assert layout_in is None
    return
  if len(op.operands) == 1:
    layout_in = (layout_in,)
  assert len(op.operands) == len(layout_in)
  unrolled = []
  for operand, layout in zip(op.operands, layout_in):
    if not ir.VectorType.isinstance(operand.type):  # scalar
      assert layout is None
      unrolled.append(operand)
    else:
      # When the operand has a vector type, disassemble the operand.
      assert layout is not None
      tiles = disassemble(layout, operand)
      unrolled.extend(list(tiles.flat))

  # Replace old oprands with unrolled operands.
  return ctx.set_operands(op.operation, unrolled)


@_register_rule("tpu.concatenate")
def _tpu_concatenate_rule(
    ctx: RewriteContext,
    op: tpu.ConcatenateOp,
    layout_in: Sequence[VectorLayout],
    layout_out: VectorLayout
):
  if any(l != layout_out for l in layout_in):
    raise NotImplementedError("Inconsistent layouts")
  layout = layout_out
  if not layout.has_natural_topology:
    raise NotImplementedError
  res_ty = ir.VectorType(op.result.type)
  dimension = ir.IntegerAttr(op.dimension).value
  if dimension - res_ty.rank >= -2:
    raise NotImplementedError("Concatenation along the last two dimensions")
  tiles = [disassemble(layout, x) for x in op.operands]
  res_tiles = np.concatenate(tiles, axis=dimension)
  ctx.replace(op, assemble(res_ty, layout, res_tiles))


@_register_rule("tpu.load")
def _tpu_load_rule(
    ctx: RewriteContext,
    op: tpu.LoadOp,
    layout_in: Sequence[Layout],
    layout_out: VectorLayout,
):
  assert all(li is None for li in layout_in)
  ty = ir.VectorType(op.result.type)

  # We expect the result is already a native-sized vreg.
  if layout_out.bitwidth != 32:
    raise NotImplementedError("Only 32-bit loads supported")
  assert layout_out == VectorLayout(32, (0, 0), TARGET_SHAPE, None)

  indices = [get_int_const(v, "tpu.load index") for v in op.indices]
  if indices[1] % TARGET_SHAPE.lanes:
    raise NotImplementedError(
        f"Lane index is not a multiple of {TARGET_SHAPE.lanes}"
    )

  tile = tpu.LoadOp(
      ty,
      op.base,
      op.indices,
      op.sublane_mask,
      sublane_stride=op.sublane_stride,
  )
  return ctx.replace(op, assemble(ty, layout_out, np.asarray([[tile]])))


@_register_rule("tpu.store")
def _tpu_store_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext,
    op: tpu.StoreOp,
    layout_in: Sequence[Layout],
    layout_out: None,  # pylint: disable=unused-argument
):
  to_store_layout, *other_layouts = layout_in
  assert all(li is None for li in other_layouts)

  # We expect the value to store is already a native-sized vreg.
  if to_store_layout.bitwidth != 32:
    raise NotImplementedError("Only 32-bit stores supported")
  assert to_store_layout == VectorLayout(32, (0, 0), TARGET_SHAPE, None)

  indices = [get_int_const(v, "tpu.store index") for v in op.indices]
  if indices[1] % TARGET_SHAPE.lanes:
    raise NotImplementedError(
        f"Lane index is not a multiple of {TARGET_SHAPE.lanes}"
    )

  tiles = disassemble(to_store_layout, op.valueToStore)
  assert tiles.shape == (1, 1)
  tpu.StoreOp(
      tiles[0][0],
      op.base,
      op.indices,
      op.sublane_mask,
      mask=op.mask,
      sublane_stride=op.sublane_stride,
  )
  return ctx.erase(op)


@_register_rule("tpu.trace")
def _tpu_trace_rule(ctx: RewriteContext, op: tpu.TraceOp,  # pylint: disable=missing-function-docstring
                    layout_in: Layout, layout_out: Layout):
  if op.operands or op.results:
    raise NotImplementedError("tpu.traced_block with inputs or outputs")
  assert layout_out is None
  assert layout_in is None
  # We don't modify the op, but we do rewrite the branch bodies.
  (region,) = op.regions
  (block,) = region.blocks
  apply_layout_block(ctx, block)


@_register_rule("tpu.region")
def _tpu_region_rule(ctx: RewriteContext, op: tpu.TraceOp,  # pylint: disable=missing-function-docstring
                     layout_in: Layout, layout_out: Layout):
  if op.operands or op.results:
    raise NotImplementedError("tpu.region_block with inputs or outputs")
  assert layout_out is None
  assert layout_in is None
  # We don't modify the op, but we do rewrite the branch bodies.
  (region,) = op.regions
  (block,) = region.blocks
  apply_layout_block(ctx, block)


@_register_rule("tpu.iota")
def _tpu_iota_rule(ctx: RewriteContext, op: tpu.IotaOp,  # pylint: disable=missing-function-docstring
                   layout_in: None, layout_out: VectorLayout):
  assert layout_in is None
  ty = ir.VectorType(op.result.type)
  vreg = native_vreg_ty(ty.element_type)
  assert ir.IntegerType.isinstance(ty.element_type)
  if layout_out.implicit_dim is not None:
    raise NotImplementedError("Only 2D layouts supported")
  tile_array_shape = layout_out.tile_array_shape(ty.shape)
  dimension = op.dimension
  if dimension is None:
    raise NotImplementedError
  elif dimension.value == ty.rank - 1:
    if layout_out.offsets[1] != 0:
      raise NotImplementedError("Unsupported offset")
    num_tiles = tile_array_shape[-1]
    tiles = np.empty(shape=(num_tiles,), dtype=object)
    iota = tpu.IotaOp(vreg, dimension=1)
    for i in range(num_tiles):
      offset = arith.ConstantOp(
          vreg,
          ir.DenseElementsAttr.get_splat(
              vreg, ir.IntegerAttr.get(ty.element_type, i * vreg.shape[-1])
          ))
      tiles[i] = arith.AddIOp(iota, offset)
    ctx.replace(
        op, assemble(ty, layout_out, np.broadcast_to(tiles, tile_array_shape)))
  elif dimension.value == ty.rank - 2:
    if layout_out.offsets[0] != 0:
      raise NotImplementedError("Unsupported offset")
    num_tiles = tile_array_shape[-2]
    tiles = np.empty(shape=(num_tiles,), dtype=object)
    iota = tpu.IotaOp(vreg, dimension=0)
    for i in range(num_tiles):
      offset = arith.ConstantOp(
          vreg,
          ir.DenseElementsAttr.get_splat(
              vreg, ir.IntegerAttr.get(ty.element_type, i * vreg.shape[-2])
          ))
      tiles[i] = arith.AddIOp(iota, offset)
    tiles = np.broadcast_to(tiles[:, np.newaxis], tile_array_shape)
    ctx.replace(op, assemble(ty, layout_out, tiles))
  else:
    raise NotImplementedError("Unsupported dimension")


@_register_rule("tpu.gather")
def _tpu_gather_rule(ctx: RewriteContext, op: tpu.GatherOp,  # pylint: disable=missing-function-docstring
                     layout_in: VectorLayout, layout_out: VectorLayout):
  if (
      layout_in.implicit_dim is not None
      or layout_out.implicit_dim is not None
      or layout_in.offsets != layout_out.offsets
      or any(o != 0 and o is not REPLICATED for o in layout_in.offsets)
  ):
    raise NotImplementedError("Only 2D layouts supported")
  ty = ir.VectorType(op.result.type)
  dimension = ir.IntegerAttr(op.dimension).value
  in_tiles = disassemble(layout_in, op.source)
  out_tiles = np.empty_like(in_tiles, dtype=object)
  if dimension < ty.rank - 2:
    raise NotImplementedError("Unsupported dimension")
  width = TARGET_SHAPE[dimension - ty.rank]
  indices = np.asarray(ir.DenseI32ArrayAttr(op.attributes["indices"]),
                       dtype=np.int32)
  num_sections, rem = divmod(len(indices), width)
  if rem == 0:
    offsets = np.arange(num_sections, dtype=np.int32)[:, np.newaxis] * width
    segment_indices = indices.reshape(num_sections, width) - offsets
    if np.any(np.logical_or(segment_indices < 0, segment_indices >= width)):
      raise NotImplementedError("Cross-segment gather")
    if np.any(segment_indices != segment_indices[0]):
      raise NotImplementedError("Indices varying between segments")
    segment_indices = segment_indices[0]
  elif num_sections == 0:  # Only one vreg.
    segment_indices = np.pad(indices, (0, width - len(indices)),
                             mode="constant", constant_values=0)
  else:
    raise NotImplementedError("Not a multiple of target length")
  if dimension == ty.rank - 1:
    # TODO(b/265133497): Remove the broadcast once 2nd minor works.
    dyn_ix_ty = ir.VectorType.get(TARGET_SHAPE, ir.IntegerType.get_signless(32))
    dyn_ix_val = np.ascontiguousarray(
        np.broadcast_to(segment_indices, TARGET_SHAPE)
    )
    dyn_ix_ref = ctx.append_constant(
        ir.DenseIntElementsAttr.get(dyn_ix_val, type=dyn_ix_ty))
    all_sublanes = ir.DenseBoolArrayAttr.get([True] * TARGET_SHAPE.sublanes)
    dyn_ix = tpu.LoadOp(
        dyn_ix_ty, dyn_ix_ref, [ix_cst(0)] * 2, sublane_mask=all_sublanes)
    for idx, tile in np.ndenumerate(in_tiles):
      out_tiles[idx] = tpu.DynamicGatherOp(tile.type, tile, dyn_ix, 1)
  else:
    assert dimension == ty.rank - 2
    segment_indices_attr = ir.DenseI32ArrayAttr.get(segment_indices)
    for idx, tile in np.ndenumerate(in_tiles):
      out_tiles[idx] = tpu.GatherOp(tile.type, tile, segment_indices_attr, 0)
  return ctx.replace(op, assemble(ty, layout_out, out_tiles))


@_register_rule("tpu.repeat")
def _tpu_repeat_rule(ctx: RewriteContext, op: tpu.RepeatOp,  # pylint: disable=missing-function-docstring
                     layout_in: VectorLayout, layout_out: VectorLayout):
  if layout_in.implicit_dim is not None:
    raise NotImplementedError("Only 2D layouts supported")
  if layout_in != layout_out:
    raise NotImplementedError("Changing layout mid-repeat")
  if not layout_in.has_natural_topology or layout_in.offsets != (0, 0):
    raise NotImplementedError("Non-trivial layouts unsupported")
  src_ty = ir.VectorType(op.source.type)
  dim = op.dimension.value
  if dim != src_ty.rank - 1:
    raise NotImplementedError("Only repeats along the last dim supported")
  if src_ty.shape[-1] % TARGET_SHAPE[-1] != 0:
    raise NotImplementedError("Only free repeats supported")
  in_vregs = disassemble(layout_in, op.source)
  out_vregs = np.repeat(in_vregs, op.times.value, dim)
  return ctx.replace(op, assemble(op.result.type, layout_out, out_vregs))


@_register_rule("vector.extract_strided_slice")
def _vector_extract_strided_slice_rule(
    ctx: RewriteContext,  # pylint: disable=missing-function-docstring
    op: vector.ExtractStridedSliceOp,
    layout_in: VectorLayout,
    layout_out: VectorLayout,
):
  """Applies a vector layout to an ExtractStridedSliceOp."""
  operand = op.vector
  if not layout_in.has_natural_topology:
    raise NotImplementedError("Unsupported input layout")
  if layout_out != layout_in:
    raise NotImplementedError("Unsupported output layout")

  tiled_dims = tuple(ir.VectorType(operand.type).shape[-2:])
  if crem(tiled_dims, layout_in.tiling) != (0, 0):
    raise NotImplementedError(
        "Extract strides slices only works with "
        + "operands with sizes that are multiples of the native tiling."
    )

  # We currently only support zero-offset, tile-aligned slices. This implies the
  # output layout is merely a slice of the input layout, without needing to
  # modify physical any of the vregs' layouts.
  offsets = [
      ir.IntegerAttr(x).value for x in ir.ArrayAttr(op.attributes["offsets"])
  ]
  for offset in offsets[-2:]:
    if offset != 0:
      raise NotImplementedError("Only tile-aligned slices supported")

  slice_sizes = tuple(
      ir.IntegerAttr(x).value for x in ir.ArrayAttr(op.attributes["sizes"])
  )
  slice_tiled_shape = layout_in.tile_array_shape(slice_sizes)
  slices = tuple(
      slice(start, start + size)
      for (start, size) in zip(offsets, slice_tiled_shape)
  )
  input_tiles = disassemble(layout_in, operand)
  dst_tiles = input_tiles[slices]
  dst_ty = ir.VectorType(op.result.type)
  return ctx.replace(op, assemble(dst_ty, layout_out, dst_tiles))


@_register_rule("vector.broadcast")
def _vector_broadcast_rule(ctx: RewriteContext, op: vector.BroadcastOp,  # pylint: disable=missing-function-docstring
                           layout_in: Layout, layout_out: VectorLayout):
  dst_ty = ir.VectorType(op.result.type)
  dst_tiles_shape = layout_out.tile_array_shape(dst_ty.shape)
  if ir.VectorType.isinstance(op.source.type):
    src_ty = ir.VectorType(op.source.type)
    assert layout_in is not None
    if layout_in.implicit_dim != layout_out.implicit_dim:
      raise NotImplementedError("Changing implicit dims mid-broadcast")
    implicit_dim = layout_in.implicit_dim
    offsets_in = layout_in.offsets
    offsets_out = layout_out.offsets

    expand_rank = dst_ty.rank - src_ty.rank
    src_shape_paddded = [-1] * expand_rank + src_ty.shape
    dim_eq = [i == o for i, o in zip(src_shape_paddded, dst_ty.shape)]

    no_op = False
    if implicit_dim is None:
      for in_off, out_off, eq in zip(offsets_in, offsets_out, dim_eq[-2:]):
        if eq and in_off != out_off:
          raise NotImplementedError("Changing offsets mid-broadcast")
      no_op = (
          layout_in.has_natural_topology
          and layout_out.has_natural_topology
          and all(dim_eq[i] or offsets_in[i] is REPLICATED for i in (-1, -2))
      )
    elif implicit_dim is not None:
      if dim_eq[-1]:
        if offsets_in != offsets_out:
          raise NotImplementedError("Changing offsets mid-broadcast")
        no_op = True
      elif (
          implicit_dim == ImplicitDim.SECOND_MINOR
          and layout_in.offsets[1] is REPLICATED
      ):
        no_op = True
      elif (
          implicit_dim == ImplicitDim.MINOR
          and layout_in.offsets[0] is REPLICATED
      ):
        no_op = True

    src_tiles = disassemble(layout_in, op.source)
    dst_tiles = np.ndarray(dst_tiles_shape, dtype=object)
    if no_op:
      src_tiles = src_tiles.reshape((1,) * expand_rank + src_tiles.shape)
      for dst_idx in np.ndindex(dst_tiles.shape):
        src_idx = tuple(i if eq else 0 for i, eq in zip(dst_idx, dim_eq))
        dst_tiles[dst_idx] = src_tiles[src_idx]
    elif implicit_dim is None:
      if dim_eq[-1]:  # Sublane broadcast
        assert src_tiles.shape[-2] == 1
        offset = layout_in.offsets[-2]
        assert offset is not REPLICATED
        indices = ir.DenseI32ArrayAttr.get([offset] * TARGET_SHAPE.sublanes)
        everything = slice(None, None)
        for src_idx, tile in np.ndenumerate(src_tiles):
          src_idx_pad = (everything,) * expand_rank + src_idx
          dst_idx = tuple(
              i if eq else everything
              for i, eq in zip(src_idx_pad, dim_eq, strict=True)
          )
          dst_tiles[dst_idx] = tpu.GatherOp(tile.type, tile, indices, 0)
      elif dim_eq[-2]:  # Lane broadcast
        assert src_tiles.shape[-1] == 1
        offset = layout_in.offsets[-1]
        assert offset is not REPLICATED
        everything = slice(None, None)
        idx_ty = ir.VectorType.get(TARGET_SHAPE, i32())
        idx = arith.ConstantOp(
            idx_ty,
            ir.DenseElementsAttr.get_splat(
                idx_ty, ir.IntegerAttr.get(i32(), offset)
            ),
        )
        for src_idx, tile in np.ndenumerate(src_tiles):
          src_idx_pad = (everything,) * expand_rank + src_idx
          dst_idx = tuple(
              i if eq else everything
              for i, eq in zip(src_idx_pad, dim_eq, strict=True)
          )
          dst_tiles[dst_idx] = tpu.DynamicGatherOp(tile.type, tile, idx, 1)
      else:
        raise NotImplementedError
    else:
      raise NotImplementedError(layout_in)
    return ctx.replace(op, assemble(dst_ty, layout_out, dst_tiles))
  else:
    tile = vector.BroadcastOp(native_vreg_ty(op.source.type), op.source)
    dst_tiles = np.full(dst_tiles_shape, tile, dtype=object)
    return ctx.replace(op, assemble(dst_ty, layout_out, dst_tiles))


@_register_rule("vector.load")
def _vector_load_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext, op: vector.LoadOp,
    layout_in: Sequence[Layout], layout_out: VectorLayout):
  assert all(ip is None for ip in layout_in)
  memref_ty = ir.MemRefType(op.base.type)
  ty = ir.VectorType(op.result.type)
  target_ty = native_vreg_ty(ty.element_type)
  if layout_out.implicit_dim == ImplicitDim.MINOR:
    raise NotImplementedError
  is_1d = layout_out.implicit_dim is not None
  memref_tiling = get_memref_tiling(op.base)
  if layout_out.tiling != memref_tiling:
    # Now we can handle the case when tiling is (1, TARGET_SHAPE.lanes).
    # TODO(b/295393167): need to support strided load for bitwidth < 32.
    if layout_out.bitwidth != 32 or layout_out.tiling != (
        1,
        TARGET_SHAPE.lanes,
    ):
      raise NotImplementedError
  # TODO(apaszke): Check that loads are from vmem!
  indices = [get_int_const(v, "vector.load index") for v in op.indices]
  for i, n, extent in zip(indices, ty.shape, memref_ty.shape):
    if i + n > extent:
      raise ValueError("reading out of bounds")
  *_, ss, _ = layout_out.implicit_shape(ty.shape)
  sublane_stride = 1
  # The stride of load should be the number of sublanes in memref tile when
  # loaing a single sublane.
  if (
      layout_out.bitwidth == 32
      and layout_out.tiling
      == (
          1,
          TARGET_SHAPE.lanes,
      )
      and ss == 1
  ):
    sublane_stride = memref_tiling[0]
  tiling = TargetTuple(*layout_out.tiling)
  s, l = offsets = TargetTuple(*layout_out.offsets)
  check(l is not REPLICATED, "load replicated along lanes is unsupported")
  load_map = None
  if s is REPLICATED:
    check(ss == 1, "sublane-replicated load with of size > 1 is unsupported")
    if not layout_out.has_native_tiling:
      raise NotImplementedError
    batch_ixs = "".join(f"b{i}, " for i in range(memref_ty.rank - 2))
    load_map = ir.Attribute.parse(f"affine_map<({batch_ixs}i, j) -> (0, j)>")
    padding = arith.ConstantOp(
        ty.element_type, get_constant(ty.element_type, 0))
  # In the future it might be useful to place the data at an arbitrary
  # aligned offset, but for now we assume that no padding tiles precede the
  # first tile.
  if any((o or 0) > t for o, t in zip(offsets, tiling)):
    raise NotImplementedError
  if is_1d:
    *base_batch, base_l = indices
    base_s = 0
  else:
    *base_batch, base_s, base_l = indices
  tiles = np.ndarray(layout_out.tile_array_shape(ty.shape), dtype=object)
  vreg_slice = layout_out.vreg_slice
  for tile_ixs in np.ndindex(tiles.shape):
    if is_1d:
      *batch_ixs, lix = tile_ixs
      tile = (base_l + lix * vreg_slice.lanes - l,)
    else:
      *batch_ixs, six, lix = tile_ixs
      tile = (
          base_s + six * vreg_slice.sublanes - (s or 0),
          base_l + lix * vreg_slice.lanes - l,
      )
    indices = (*(b + i for b, i in zip(base_batch, batch_ixs)), *tile)
    assert indices[-1] + TARGET_SHAPE.lanes <= memref_ty.shape[-1]
    bounds = layout_out.tile_data_bounds(
        ty.shape, tile_ixs, allow_replicated=TargetTuple(True, False))
    indices_vs = list(map(ix_cst, indices))
    if bounds.mask_varies_along(SUBLANES):
      assert s is not REPLICATED  # Replicated loads should never go OOB
      tile = tpu.LoadOp(
          target_ty,
          op.base,
          indices_vs,
          bounds.get_sublane_mask(),
          sublane_stride=sublane_stride,
      )
    else:
      if load_map is not None:
        if layout_out.bitwidth != 32:
          raise NotImplementedError
        tile = vector.TransferReadOp(
            target_ty, op.base, indices_vs, load_map, padding)
      else:
        assert s is not REPLICATED
        sublane_mask = ir.DenseBoolArrayAttr.get(
            [True] * TARGET_SHAPE.sublanes)
        tile = tpu.LoadOp(
            target_ty,
            op.base,
            indices_vs,
            sublane_mask,
            sublane_stride=sublane_stride,
        )
    tiles[tile_ixs] = tile
  return ctx.replace(op, assemble(ty, layout_out, tiles))


@_register_rule("vector.store")
def _vector_store_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext, op: vector.StoreOp,
    layout_in: Sequence[Layout], layout_out: Layout):
  to_store_layout, *other_layouts = layout_in
  assert all(ip is None for ip in other_layouts)
  assert layout_out is None
  ty = ir.VectorType(op.valueToStore.type)
  if to_store_layout.implicit_dim == ImplicitDim.MINOR:
    raise NotImplementedError
  is_1d = to_store_layout.implicit_dim is not None
  memref_tiling = get_memref_tiling(op.base)
  if to_store_layout.tiling != memref_tiling:
    # Now we can handle the case when tiling is (1, TARGET_SHAPE.lanes).
    # TODO(b/295393167): need to support strided store for bitwidth < 32.
    if to_store_layout.bitwidth != 32 or to_store_layout.tiling != (
        1,
        TARGET_SHAPE.lanes,
    ):
      raise NotImplementedError
  base_indices = [get_int_const(v, "vector.store index") for v in op.indices]
  tiles = disassemble(to_store_layout, op.valueToStore)
  if is_1d:
    *base_batch, base_l = base_indices
    base_s = 0
    tiles = tiles.reshape(to_store_layout.implicit_shape(tiles.shape))
  else:
    *base_batch, base_s, base_l = base_indices
  sublane_offset, lane_offset = to_store_layout.offsets
  check(lane_offset is not REPLICATED and sublane_offset is not REPLICATED,
        "replicated layout disallowed in vector store")
  stored_shape = to_store_layout.implicit_shape(tuple(ty.shape))
  sublane_stride = 1
  # The stride of store should be the number of sublanes in memref tile when
  # store a single sublane.
  if to_store_layout.bitwidth == 32 and to_store_layout.tiling == (
      1,
      TARGET_SHAPE.lanes,
  ):
    sublane_stride = memref_tiling[0]
  vreg_slice = to_store_layout.vreg_slice
  for ixs, tile in np.ndenumerate(tiles):
    bounds = to_store_layout.tile_data_bounds(stored_shape, ixs)
    *batch_ixs, six, lix = ixs
    indices = (
        *(b + i for b, i in zip(base_batch, batch_ixs)),
        base_s + six * vreg_slice.sublanes - sublane_offset,
        base_l + lix * vreg_slice.lanes - lane_offset,
    )
    if is_1d:
      indices = (*indices[:-2], indices[-1])
    indices = list(map(ix_cst, indices))
    sublane_mask = bounds.get_sublane_mask()
    masks_subelements = bounds.mask_varies_along(SUBELEMENTS)
    if bounds.mask_varies_along(LANES) or masks_subelements:
      mask = bounds.get_vector_mask(ctx.hardware_generation)
      # Vmem stores don't support masking below 32-bit granularity, so we need
      # to load and blend explicitly if needed.
      if masks_subelements:
        data = tpu.LoadOp(tile.type, op.base, indices, sublane_mask)
        mask_is_a_bitmask = ir.IntegerType(
            ir.VectorType(mask.type).element_type
        ).width == 32
        if mask_is_a_bitmask:
          ones = arith.ConstantOp(
              mask.type,
              ir.DenseElementsAttr.get_splat(
                  mask.type, ir.IntegerAttr.get(i32(), 0xFFFFFFFF)
              ),
          )
          masked_tile = arith.AndIOp(mask, tpu.BitcastOp(mask.type, tile))
          mask_neg = arith.XOrIOp(ones, mask)
          masked_data = arith.AndIOp(mask_neg, tpu.BitcastOp(mask.type, data))
          updated = tpu.BitcastOp(
              tile.type, arith.OrIOp(masked_data, masked_tile))
        else:
          updated = arith.SelectOp(mask, tile, data)
        tpu.StoreOp(
            updated,
            op.base,
            indices,
            sublane_mask,
            sublane_stride=sublane_stride,
        )
      else:
        tpu.StoreOp(
            tile,
            op.base,
            indices,
            sublane_mask,
            mask=mask,
            sublane_stride=sublane_stride,
        )
    else:
      tpu.StoreOp(
          tile, op.base, indices, sublane_mask, sublane_stride=sublane_stride
      )
  return ctx.erase(op)


@_register_rule("vector.shape_cast")
def _vector_shape_cast_rule(ctx: RewriteContext, op: vector.ShapeCastOp,  # pylint: disable=missing-function-docstring
                            layout_in: VectorLayout, layout_out: VectorLayout):
  src_ty = ir.VectorType(op.source.type)
  dst_ty = ir.VectorType(op.result.type)
  layout_rank = layout_in.layout_rank
  no_op = False
  if (
      layout_in == layout_out
      and src_ty.shape[-layout_rank:] == dst_ty.shape[-layout_rank:]
  ):
    no_op = True
  elif (
      layout_in.implicit_dim is None
      and layout_out.implicit_dim == ImplicitDim.SECOND_MINOR
      and layout_in.has_native_tiling
      and layout_in.tiling == layout_out.tiling
      and layout_in.offsets == layout_out.offsets
      and src_ty.shape[-1] == dst_ty.shape[-1]
      and src_ty.shape[-2] == 1
  ):
    no_op = True
  elif (
      layout_in.implicit_dim is None
      and layout_out.implicit_dim == ImplicitDim.MINOR
      and layout_in.has_natural_topology
      and layout_in.tiling == layout_out.tiling
      and layout_in.offsets == layout_out.offsets
      and src_ty.shape == [*dst_ty.shape, 1]
  ):
    no_op = True
  elif (
      layout_in.implicit_dim == ImplicitDim.MINOR
      and layout_out.implicit_dim is None
      and layout_out.has_natural_topology
      and layout_in.tiling == layout_out.tiling
      and layout_in.offsets == layout_out.offsets
      and dst_ty.shape == [*src_ty.shape, 1]
  ):
    no_op = True
  elif (  # Fold or unfold sublane dim, but keeping a whole number of vregs.
      layout_in.implicit_dim is None
      and layout_out.implicit_dim is None
      and layout_in.offsets == layout_out.offsets == (0, 0)
      and layout_in.has_native_tiling
      and layout_in.tiling == layout_out.tiling
      and dst_ty.shape[-1] == src_ty.shape[-1]
      and dst_ty.shape[-2] % layout_in.tiling[-2] == 0
      and src_ty.shape[-2] % layout_in.tiling[-2] == 0
  ):
    no_op = True

  src_vregs = disassemble(layout_in, op.source)
  if no_op:
    dst_vregs = src_vregs.reshape(layout_out.tile_array_shape(dst_ty.shape))
  elif (
      dst_ty.shape[-2:] == [src_ty.shape[-1], 1]
      and layout_in.bitwidth == 32
      and layout_in.implicit_dim in {None, ImplicitDim.SECOND_MINOR}
      and layout_out.implicit_dim is None
      and layout_in.has_native_tiling
      and layout_in.tiling == layout_out.tiling
      and (layout_in.offsets[0] or 0) == 0
      and layout_in.offsets[1] == 0
      and layout_out.offsets[0] == 0
      # layout_out.offsets[1] can be anything, as we produce a replicated result
  ):
    # First, insert the new singleton lane dimension.
    dst_vregs = np.empty(
        layout_out.tile_array_shape((*src_ty.shape, 1)), dtype=object
    )
    if layout_in.implicit_dim == ImplicitDim.SECOND_MINOR:
      # Make the sublane dimension explicit.
      src_vregs = src_vregs.reshape(
          *src_vregs.shape[:-1], 1, src_vregs.shape[-1]
      )
      dst_vregs = dst_vregs.reshape(
          *dst_vregs.shape[:-2], 1, *dst_vregs.shape[-2:]
      )
    assert dst_vregs.shape[-1] == 1  # We're inserting a singleton dimension
    for *batch_idx, row_idx, col_idx in np.ndindex(dst_vregs.shape[:-1]):
      sublanes_in_lane, rem = divmod(TARGET_SHAPE.lanes, TARGET_SHAPE.sublanes)
      assert rem == 0
      if layout_in.offsets[0] is REPLICATED and row_idx != 0:
        continue  # All vregs along that dimension are the same.
      col_vreg = src_vregs[(
          *batch_idx,
          row_idx // TARGET_SHAPE.sublanes,
          col_idx // sublanes_in_lane,
      )]
      # BroadcastInSublanesOp requires the sublanes to be replicated.
      if layout_in.offsets[0] is not REPLICATED:
        sublane = row_idx % TARGET_SHAPE.sublanes
        indices = ir.DenseI32ArrayAttr.get([sublane] * TARGET_SHAPE.sublanes)
        col_vreg = tpu.GatherOp(col_vreg.type, col_vreg, indices, 0).result
      dst_vregs[(*batch_idx, row_idx, col_idx, 0)] = tpu.BroadcastInSublanesOp(
          col_vreg.type,
          col_vreg,
          (col_idx % sublanes_in_lane) * TARGET_SHAPE.sublanes,
      )
    if layout_in.offsets[0] is REPLICATED:
      # Broadcast the sublane vregs.
      dst_vregs[..., 1:, :, :] = dst_vregs[..., 0:1, :, :]
    # Now, permute the major axes of the vreg array.
    dst_vregs = dst_vregs.reshape(layout_out.tile_array_shape(dst_ty.shape))
  else:
    raise NotImplementedError(f"Unsupported vector.shape_cast: {op}")
  return ctx.replace(op, assemble(dst_ty, layout_out, dst_vregs))


@_register_rule("vector.contract")
def _vector_contract_rule(ctx: RewriteContext, op: vector.ContractionOp,  # pylint: disable=missing-function-docstring
                          layout_in: Sequence[Layout], layout_out: Layout):
  matmul_indexing_maps = ir.ArrayAttr.get([
      ir.Attribute.parse("affine_map<(i, j, k) -> (i, k)>"),
      ir.Attribute.parse("affine_map<(i, j, k) -> (k, j)>"),
      ir.Attribute.parse("affine_map<(i, j, k) -> (i, j)>"),
  ])
  matmul_indexing_maps_transposed = ir.ArrayAttr.get([
      ir.Attribute.parse("affine_map<(i, j, k) -> (i, k)>"),
      ir.Attribute.parse("affine_map<(i, j, k) -> (j, k)>"),
      ir.Attribute.parse("affine_map<(i, j, k) -> (i, j)>"),
  ])
  indexing_maps = op.attributes["indexing_maps"]
  if (indexing_maps != matmul_indexing_maps
      and indexing_maps != matmul_indexing_maps_transposed):
    raise NotImplementedError("non-matmul or unsupported indexing_maps")
  transpose_rhs = indexing_maps == matmul_indexing_maps_transposed
  matmul_iterator_types = ir.ArrayAttr.get([
      ir.Attribute.parse("#vector.iterator_type<parallel>"),
      ir.Attribute.parse("#vector.iterator_type<parallel>"),
      ir.Attribute.parse("#vector.iterator_type<reduction>"),
  ])
  if op.attributes["iterator_types"] != matmul_iterator_types:
    raise NotImplementedError("non-matmul iterator_types")
  transpose_lhs = False  # TODO(apaszke): Support that in the affine maps.
  return _matmul_rule(
      ctx, op, transpose_lhs, transpose_rhs, *layout_in, layout_out
  )


@_register_rule("tpu.matmul")
def _tpu_matmul_rule(ctx: RewriteContext, op: tpu.MatmulOp,
                     layout_in: Sequence[Layout], layout_out: VectorLayout):
  _matmul_rule(
      ctx, op, op.transpose_lhs.value, op.transpose_rhs.value,
      *layout_in, layout_out,
  )


def _matmul_rule(
    ctx: RewriteContext,
    op,
    transpose_lhs: bool,
    transpose_rhs: bool,
    layout_lhs: VectorLayout,
    layout_rhs: VectorLayout,
    layout_acc: VectorLayout,
    layout_out: VectorLayout,
):
  if transpose_lhs:
    raise NotImplementedError("transposed lhs not supported yet")
  layout_in = (layout_lhs, layout_rhs, layout_acc)
  if any(tuple((o or 0) for o in l.offsets) != (0, 0)
         for l in (*layout_in, layout_out)):
    raise NotImplementedError("unaligned layout in matmul")
  if any(l.implicit_dim is not None for l in layout_in):
    raise NotImplementedError("Unsupported matmul operand layout")
  if not all(l.has_native_tiling for l in layout_in):
    raise NotImplementedError("Unsupported matmul operand tiling")
  lhs_type = ir.VectorType(op.lhs.type)
  rhs_type = ir.VectorType(op.rhs.type)
  acc_type = ir.VectorType(op.acc.type)
  if type_bitwidth(acc_type.element_type) != 32:
    raise NotImplementedError("non-32-bit matmul result")
  if lhs_type.shape[0] % layout_lhs.tiling[0] != 0:
    raise NotImplementedError("layout matmul lhs")
  if rhs_type.shape[0] % 128 != 0 or rhs_type.shape[1] % 128 != 0:
    raise NotImplementedError("matmul rhs requires padding")
  layout_attr = op.attributes["out_layout"]
  lhs_col_ty = ir.VectorType.get(
      (lhs_type.shape[0], 128), lhs_type.element_type)
  acc_col_ty = ir.VectorType.get(
      (lhs_type.shape[0], 128), acc_type.element_type)
  lhs_tiles = disassemble(layout_lhs, op.lhs)
  acc_tiles = disassemble(layout_acc, op.acc)
  lhs_cols = [tpu.RollVectorsOp(lhs_col_ty, lhs_tiles[:, i])
              for i in range(lhs_tiles.shape[1])]
  for col in lhs_cols:
    col.attributes["out_layout"] = layout_attr
  rhs_tile_ty = ir.VectorType.get((128, 128), rhs_type.element_type)
  rhs_vregs = disassemble(layout_rhs, op.rhs)
  rhs_vregs_per_tile = 16 // layout_rhs.packing
  if transpose_rhs:
    nj, nk = cdiv(tuple(rhs_type.shape), (128, 128))
    rhs_tiles = rhs_vregs.reshape((nj, rhs_vregs_per_tile, nk, 1)).transpose(
        2, 0, 1, 3)
  else:
    nk, nj = cdiv(tuple(rhs_type.shape), (128, 128))
    rhs_tiles = rhs_vregs.reshape((nk, rhs_vregs_per_tile, nj, 1)).transpose(
        0, 2, 1, 3)

  precision = None
  if "precision" in op.attributes:
    precision = op.attributes["precision"]
  for j, k in np.ndindex((nj, nk)):
    rhs_tile = rhs_tiles[k, j]
    assert rhs_tile.shape == (rhs_vregs_per_tile, 1)
    rhs_rolled_tile = tpu.RollVectorsOp(rhs_tile_ty, list(rhs_tile.flat))
    rhs_rolled_tile.attributes["out_layout"] = layout_attr
    acc_col = tpu.RollVectorsOp(acc_col_ty, acc_tiles[:, j])
    acc_col.attributes["out_layout"] = layout_attr
    new_acc_col = tpu.MatmulOp(
        acc_col_ty, lhs_cols[k], rhs_rolled_tile, acc_col,
        transpose_lhs=transpose_lhs,
        transpose_rhs=transpose_rhs,
        precision=precision,
    )
    new_acc_tiles = tpu.UnrollVectorsOp([v.type for v in acc_tiles[:, j]],
                                        new_acc_col)
    new_acc_tiles.attributes["in_layout"] = layout_attr
    acc_tiles[:, j] = new_acc_tiles.results
  return ctx.replace(op, assemble(op.result.type, layout_out, acc_tiles))


@_register_rule("vector.multi_reduction")
def _vector_multi_reduction_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext, op: vector.MultiDimReductionOp,
    layout_in: Sequence[Layout], layout_out: Layout):
  dims = ir.ArrayAttr(op.attributes["reduction_dims"])
  if len(dims) != 1:
    raise NotImplementedError("only 1d reductions supported")
  (dim_attr,) = dims
  dim = ir.IntegerAttr(dim_attr).value
  src_type = ir.VectorType(op.source.type)
  src_layout, acc_layout = layout_in
  try:
    res_type = ir.VectorType(op.result.type)
  except ValueError:
    raise NotImplementedError("Can only reduce into vectors") from None
  assert layout_out is not None  # Shouldn't be None since result is a vector

  # Make sure that the accumulator is a splat of the neutral value
  if acc_layout.offsets != (REPLICATED, REPLICATED):
    raise NotImplementedError("only replicated accumulator supported")
  acc_tile = disassemble(acc_layout, op.acc).flat[0]
  acc_def = acc_tile.owner.opview
  if not isinstance(acc_def, arith.ConstantOp):
    raise NotImplementedError("only constant accumulator supported")
  val, _ = get_splat_value(acc_def.value)
  if src_type.element_type != ir.F32Type.get():
    raise NotImplementedError(
        f"only fp32 reductions supported, but got {src_type}")
  if op.attributes["kind"] == ir.Attribute.parse("#vector.kind<add>"):
    neutral = ir.FloatAttr.get_f32(0.)
  elif op.attributes["kind"] == ir.Attribute.parse("#vector.kind<maxf>"):
    neutral = ir.FloatAttr.get_f32(-math.inf)
  else:
    raise NotImplementedError(op.attributes["kind"])
  if val != neutral.value:
    raise NotImplementedError("only neutral accumulator supported")

  if src_layout.implicit_dim is None and src_layout.has_natural_topology:
    sublane_offset, lane_offset = src_layout.offsets
    check(dim >= 0, "negative reduction dimension unsupported")
    if dim < src_type.rank - 2:
      raise NotImplementedError("reductions over non-layout dims")
    elif dim == src_type.rank - 2:
      reduce_over = SUBLANES
      allow_replicated = TargetTuple(False, True)
      vdim = 0
      if sublane_offset is REPLICATED:
        # TODO(apaszke): Note that it is just scaling!
        raise NotImplementedError("reductions over replicated axes")
      if layout_out != VectorLayout(
          32, (REPLICATED, lane_offset), TARGET_SHAPE, ImplicitDim.SECOND_MINOR
      ):
        raise NotImplementedError(f"unexpected destination layout {layout_out}")
    elif dim == src_type.rank - 1:
      reduce_over = LANES
      allow_replicated = TargetTuple(True, False)
      vdim = 1
      if lane_offset is REPLICATED:
        # TODO(apaszke): Note that it is just scaling!
        raise NotImplementedError("reductions over replicated axes")
      if layout_out != VectorLayout(
          32, (sublane_offset, REPLICATED), TARGET_SHAPE, ImplicitDim.MINOR
      ):
        raise NotImplementedError(f"unexpected destination layout {layout_out}")
    source_tiles = disassemble(src_layout, op.source)
    result_tiles = np.empty(
        layout_out.tile_array_shape(res_type.shape), dtype=object)
    if op.attributes["kind"] == ir.Attribute.parse("#vector.kind<maxf>"):
      tpu_kind = ir.Attribute.parse("#tpu.reduction_kind<max>")
      pointwise = arith.MaximumFOp
    elif op.attributes["kind"] == ir.Attribute.parse("#vector.kind<add>"):
      tpu_kind = ir.Attribute.parse("#tpu.reduction_kind<sum>")
      pointwise = arith.AddFOp
    else:
      raise NotImplementedError(op.attributes["kind"])
    src_shape = tuple(src_type.shape)
    for ixs in np.ndindex(source_tiles.shape):
      data_bounds = src_layout.tile_data_bounds(
          src_shape, ixs, allow_replicated=allow_replicated)
      tile = mask_oob(
          source_tiles[ixs], data_bounds, neutral, ctx.hardware_generation)
      *batch_ix, six, lix = ixs
      if reduce_over == LANES:
        outer = (*batch_ix, six)
        reduced_ix = lix
        last_reduced_ix = source_tiles.shape[-1] - 1
      else:
        assert reduce_over == SUBLANES
        outer = (*batch_ix, lix)
        reduced_ix = six
        last_reduced_ix = source_tiles.shape[-2] - 1
      if reduced_ix == 0:
        new_acc = tile
      else:
        new_acc = pointwise(result_tiles[outer], tile)
      if reduced_ix == last_reduced_ix:
        new_acc = tpu.AllReduceOp(
            new_acc, ir.IntegerAttr.get(i64(), vdim), tpu_kind)
      result_tiles[outer] = new_acc
    return ctx.replace(op, assemble(op.result.type, layout_out, result_tiles))
  raise NotImplementedError(f"unsupported layout: {src_layout}")


@_register_rule("vector.transpose")
def _vector_transpose_rule(  # pylint: disable=missing-function-docstring
    ctx: RewriteContext, op: vector.TransposeOp,
    layout_in: VectorLayout, layout_out: VectorLayout):
  if layout_in.implicit_dim is not None or layout_in != layout_out:
    raise NotImplementedError("Unsupported 2D layouts")
  src_ty = ir.VectorType(op.vector.type)
  dst_ty = ir.VectorType(op.result.type)
  rank = src_ty.rank
  src_vregs = disassemble(layout_in, op.vector)
  permutation = [
      ir.IntegerAttr(attr).value
      for attr in ir.ArrayAttr(op.attributes["transp"])
  ]
  batch_perm, tile_perm = permutation[:-2], permutation[-2:]
  if set(batch_perm) != set(range(len(batch_perm))):
    raise NotImplementedError("Unsupported major permutation")
  if set(tile_perm) != {rank - 2, rank - 1}:
    raise NotImplementedError("Unsupported minor permutation")
  src_vregs_perm = src_vregs.transpose(*batch_perm, rank - 2, rank - 1)
  if tile_perm == [rank - 2, rank - 1]:
    return ctx.replace(op, assemble(dst_ty, layout_out, src_vregs_perm))
  if layout_in.offsets != (0, 0) or not layout_in.has_native_tiling:
    raise NotImplementedError("Non-native or offset layout unsupported")
  transpose_unit_size = TARGET_SHAPE.lanes
  for s in src_ty.shape[-2:]:
    if s % transpose_unit_size != 0:
      raise NotImplementedError("Padded transpose")
  if ctx.hardware_generation < 4 and layout_in.bitwidth != 32:
    raise NotImplementedError("TPUs before v4 only support 32-bit transposes")
  dst_vregs = np.empty(layout_out.tile_array_shape(dst_ty.shape), dtype=object)
  packing = layout_in.packing
  # Note that we checked for native tiling above.
  vregs_per_tile = transpose_unit_size // layout_in.tiling[0]
  minor_perm = ir.ArrayAttr.get([ir.IntegerAttr.get(i64(), i) for i in (1, 0)])
  tile_ty = ir.VectorType.get((transpose_unit_size,) * 2, src_ty.element_type)
  batch_tile_ty_in = ir.VectorType.get(
      (transpose_unit_size, transpose_unit_size * packing), src_ty.element_type
  )
  batch_tile_ty_out = ir.VectorType.get(
      (transpose_unit_size * packing, transpose_unit_size), src_ty.element_type
  )
  # For packed types, we can increase the XLU throughput by batching together
  # multiple tiles. At the moment we always batch along columns, with the
  # reasoning being that if all the tiles are fed into the MXU, then it's better
  # if we end up with results that contribute to the same contraction.
  can_batch = layout_in.bitwidth == 16
  def do_transpose(
      batch_idx,
      src_row: int,
      src_col: int,
      src_col_end: int,
      tile_ty_in: ir.VectorType,
      tile_ty_out: ir.VectorType,
  ):
    src_row_slice = slice(
        src_row * vregs_per_tile, (src_row + 1) * vregs_per_tile
    )
    src_col_slice = slice(src_col, src_col_end)
    dst_row_slice = slice(
        src_col * vregs_per_tile, src_col_end * vregs_per_tile
    )
    dst_col_slice = slice(src_row, src_row + 1)
    src_tile_vregs = src_vregs_perm[
        (*batch_idx, src_row_slice, src_col_slice)
    ]
    src_tile = assemble(tile_ty_in, layout_in, src_tile_vregs)
    dst_tile = vector.TransposeOp(tile_ty_out, src_tile, minor_perm)
    dst_tile.attributes["out_layout"] = ir.StringAttr.get(
        print_layout(layout_out)
    )
    dst_tile_vregs = tpu.UnrollVectorsOp(
        [v.type for v in src_tile_vregs.flat], dst_tile)
    dst_tile_vregs = np.asarray(dst_tile_vregs.results, dtype=object)
    dst_tile_vregs = dst_tile_vregs.reshape(
        layout_out.tile_array_shape(tile_ty_out.shape))
    dst_vregs[(*batch_idx, dst_row_slice, dst_col_slice)] = dst_tile_vregs

  for batch_idx in np.ndindex(tuple(dst_ty.shape[:-2])):
    for src_row in range(src_ty.shape[-2] // transpose_unit_size):
      num_col_tiles = src_ty.shape[-1] // transpose_unit_size
      if can_batch:
        num_batch_tiles = num_col_tiles // 2
        for src_col in range(num_batch_tiles):
          do_transpose(
              batch_idx, src_row,
              src_col * 2, (src_col + 1) * 2,
              batch_tile_ty_in, batch_tile_ty_out,
          )
        if num_col_tiles % 2 == 1:
          do_transpose(batch_idx, src_row, num_col_tiles - 1, num_col_tiles,
                       tile_ty, tile_ty)
      else:
        for src_col in range(src_ty.shape[-1] // transpose_unit_size):
          do_transpose(batch_idx, src_row, src_col, src_col + 1,
                       tile_ty, tile_ty)
  assert all(t is not None for t in dst_vregs.flat), dst_vregs
  return ctx.replace(op, assemble(dst_ty, layout_out, dst_vregs))


################################################################################
# MLIR helpers
################################################################################

OpOrValue = Union[ir.Operation, ir.OpView, ir.Value]


def get_int_const(v: ir.Value, what: str) -> int:
  op = v.owner.opview
  if not isinstance(op, arith.ConstantOp):
    raise ValueError(f"only constant {what} supported {type(op)}")
  try:
    return ir.IntegerAttr(op.value).value
  except ValueError:
    raise ValueError(f"{what} should be an integer constant") from None


def check(cond: bool, msg: str):
  if not cond:
    raise NotImplementedError(msg)


@overload
def cdiv(l: int, r: int) -> int:
  ...


@overload
def cdiv(l: tuple[int, int], r: tuple[int, int]) -> tuple[int, int]:
  ...


def cdiv(l, r):
  if isinstance(l, tuple):
    assert isinstance(r, tuple)
    return tuple(cdiv(ll, rr) for ll, rr in zip(l, r))  # pytype: disable=wrong-arg-types
  return (l + r - 1) // r


def crem(l: Any, r: Any) -> Any:
  if isinstance(l, tuple):
    assert isinstance(r, tuple)
    return tuple(crem(ll, rr) for ll, rr in zip(l, r))
  return l % r


def is_scalar(ty: ir.Type) -> bool:
  return (
      ir.IntegerType.isinstance(ty)
      or ir.IndexType.isinstance(ty)
      or ir.F32Type.isinstance(ty)
  )


def try_cast(x: Any, ty: ir.Type) -> Any:
  try:
    return ty(x)
  except ValueError:
    return None


def assemble(ty: ir.Type, layout: VectorLayout,
             vals: np.ndarray) -> ir.Operation:
  ty = ir.VectorType(ty)
  assert vals.shape == layout.tile_array_shape(ty.shape), (
      vals.shape, layout.tile_array_shape(ty.shape))
  op = tpu.RollVectorsOp(ty, list(vals.flat))
  op.attributes["out_layout"] = ir.ArrayAttr.get(
      [ir.StringAttr.get(print_layout(layout))]
  )
  return op


def disassemble(layout: VectorLayout, val: ir.Value) -> np.ndarray:
  """Disassemble an MLIR vector into an ndarray of native vectors.

  Args:
    layout: The layout of val. Used to determine the unrolling into
      native-shaped vectors.
    val: Value to disassemble.

  Returns:
    An ndarray of MLIR values representing the tiling of val given by layout.
  """
  vty = ir.VectorType(val.type)
  tiles_shape = layout.tile_array_shape(vty.shape)
  op = val.owner
  res_idx = ir.OpResult(val).result_number
  arr_attr = ir.ArrayAttr(op.attributes["out_layout"])
  def_layout = parse_layout(arr_attr[res_idx], vty)
  assert type(def_layout) is type(layout)
  assert def_layout.generalizes(layout)
  def_layout_shape = def_layout.tile_array_shape(vty.shape)
  if isinstance(op.opview, tpu.RollVectorsOp):
    tile_vals = op.operands
  # TODO(apaszke): Clean this up. We don't need it for transpose.
  elif isinstance(op.opview, vector.ContractionOp):
    num_vectors = np.prod(tiles_shape)
    u = tpu.UnrollVectorsOp(
        [native_vreg_ty(vty.element_type)] * num_vectors, val
    )
    tile_vals = u.results
  else:
    raise NotImplementedError(op)
  tile_vals_arr = np.array(tile_vals, dtype=object).reshape(def_layout_shape)
  assert tiles_shape == def_layout_shape, (def_layout, layout, vty)
  return tile_vals_arr


def get_splat_value(attr: ir.Attribute) -> tuple[Any, ir.Type]:
  if fv := try_cast(attr, ir.DenseFPElementsAttr):
    check(fv.is_splat, "Expected a splat")
    el_ty = ir.ShapedType(fv.type).element_type
    return fv[0], el_ty
  if fv := try_cast(attr, ir.DenseIntElementsAttr):
    check(fv.is_splat, "Expected a splat")
    el_ty = ir.ShapedType(fv.type).element_type
    return fv[0], el_ty
  raise NotImplementedError(f"Not a splat? Got {attr}")


def mask_oob(
    value: ir.Value,
    bounds: VRegDataBounds,
    neutral: ir.Attribute,
    hardware_generation: int,
) -> ir.Value:
  """Masks all values outside of bounds.

  Arguments:
    value: A rank 2 MLIR vector to be masked.
    bounds: A TargetTuple of slices specifying a rectangular subregion of value
      that should be preserved during masking.
    neutral: A scalar attribute specifying the value that will be inserted
      for all values outside of specified bounds.
    hardware_generation: The target TPU generation.

  Returns:
    An MLIR value of the same type as the value argument, with all entries
    outside of bounds replaced by neutral.
  """
  assert tuple(ir.VectorType(value.type).shape) == TARGET_SHAPE
  if bounds.complete:
    return value
  mask = bounds.get_vector_mask(hardware_generation)
  assert ir.IntegerType(ir.VectorType(mask.type).element_type).width == 1
  neutral_vec_ty = ir.VectorType.get(TARGET_SHAPE, neutral.type)
  neutral_vec = arith.ConstantOp(
      neutral_vec_ty, ir.DenseElementsAttr.get_splat(neutral_vec_ty, neutral))
  return arith.SelectOp(mask, value, neutral_vec).result


def ix_cst(v: int) -> arith.ConstantOp:
  index = ir.IndexType.get()
  return arith.ConstantOp(index, ir.IntegerAttr.get(index, v)).result


def native_vreg_ty(el_ty: ir.Type) -> ir.Type:
  bitwidth = type_bitwidth(el_ty)
  if bitwidth == 32:
    return ir.VectorType.get(TARGET_SHAPE, el_ty)
  else:
    return ir.VectorType.get((*TARGET_SHAPE, 32 // bitwidth), el_ty)


def native_tiling(bitwidth: int) -> tuple[int, int]:
  packing = 32 // bitwidth
  return (TARGET_SHAPE[0] * packing, TARGET_SHAPE[1])


def type_bitwidth(ty: ir.Type) -> int:
  if ir.IntegerType.isinstance(ty):
    width = ir.IntegerType(ty).width
    if width == 1:
      return 32  # We store only one i1 per vreg element.
    return width
  elif ty == ir.F32Type.get():
    return 32
  elif ty == ir.BF16Type.get():
    return 16
  raise NotImplementedError(ty)


def get_constant(ty: ir.Type, value: Union[int, float]) -> ir.Attribute:
  if ir.IntegerType.isinstance(ty):
    return ir.IntegerAttr.get(ty, value)
  elif ty == ir.IndexType.get():
    return ir.IntegerAttr.get(ty, value)
  elif ty == ir.BF16Type.get():
    return ir.FloatAttr.get_bf16(value)
  elif ty == ir.F32Type.get():
    return ir.FloatAttr.get_f32(value)
  raise NotImplementedError(ty)


def i64() -> ir.IntegerType:
  return ir.IntegerType.get_signless(64)


def i32() -> ir.IntegerType:
  return ir.IntegerType.get_signless(32)


def get_memref_tiling(value: ir.Value) -> tuple[int, int]:
  """Returns the first-level tiling of a (packed and tiled) memref value."""
  definition = value.owner
  if isinstance(definition, ir.Operation):
    if isinstance(definition.opview, tpu.EraseLayoutOp):
      value = definition.opview.operand
  ty = ir.MemRefType(value.type)
  mem_layout = ty.layout
  if not tpu.private_is_tiled_layout(mem_layout):
    raise RuntimeError("Expected a tiled memref")
  first_tiles, *other_tiles = tpu.private_get_tiles(mem_layout)
  bitwidth = type_bitwidth(ty.element_type)
  packing = 32 // bitwidth
  if len(first_tiles) == 1:
    tile_size = first_tiles[0]
    if tile_size % (TARGET_SHAPE.lanes * packing) != 0:
      raise NotImplementedError
    if bitwidth == 32:
      if other_tiles:
        raise NotImplementedError
    elif bitwidth < 32:
      if other_tiles != [(TARGET_SHAPE.lanes,), (packing, 1)]:
        raise NotImplementedError(other_tiles)
    return (1, tile_size)
  elif len(first_tiles) == 2:
    if bitwidth == 32:
      if other_tiles:
        raise NotImplementedError
      return first_tiles
    elif bitwidth < 32:
      packing = 32 // bitwidth
      if (
          len(other_tiles) != 1
          or len(other_tiles[0]) != 2
          or other_tiles[0] != (packing, 1)
      ):
        raise NotImplementedError
      return first_tiles
  raise NotImplementedError((first_tiles, *other_tiles))
