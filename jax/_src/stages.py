# Copyright 2022 The JAX Authors.
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
"""
Interfaces to JAX's compilation steps, and utilities for conforming to them.

This module defines a set of public-facing types that reflect the output of
intermediate stages in the process of compilation. Currently there are two
stages modeled: lowering (which produces compiler input), and compilation
(which produces compiler output).

It also defines some internal-facing types to guide what JAX can present in
this common form: an internal ``Lowering`` suffices to back a public-facing
``Lowered`` and an internal ``Executable`` suffices to back a public-facing
``Compiled``.

Finally, this module defines a couple more classes to commonly adapt our
various internal XLA-backed lowerings and executables into the lowering and
executable protocols described above.
"""
from __future__ import annotations

import warnings

from dataclasses import dataclass
import io
import pickle
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple

import jax

from jax._src import core
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib import xla_client as xc


source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


xla_extension = xc._xla
map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


# -- Internal protocols

class Executable(Protocol):
  """Protocol for executables, which a user-facing ``Compiled`` encapsulates."""

  def call(self, *args_flat) -> Sequence[Any]:
    """Execute on the flat list of arguments, returning flat outputs."""
    # TODO(frostig): improve annotation (sequences of arrays/buffers)
    raise NotImplementedError

  def input_shardings(self) -> Sequence[jax.sharding.XLACompatibleSharding]:
    """Flat sequence of input shardings.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    raise NotImplementedError

  def output_shardings(self) -> Sequence[jax.sharding.XLACompatibleSharding]:
    """Flat sequence of output shardings.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    raise NotImplementedError

  def as_text(self) -> str:
    """A human-readable text representation of this executable.

    Intended for visualization and debugging purposes. This need not be a valid
    nor reliable serialization. It is relayed directly to external callers.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    raise NotImplementedError

  def cost_analysis(self) -> Any:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it need not be consistent across versions of JAX
    and jaxlib, or even across invocations. It is relayed directly to external
    callers.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    # TODO(frostig): improve annotation (arbitrary pytree)
    raise NotImplementedError

  def memory_analysis(self) -> Any:
    """A summary of estimated memory requirements.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it need not be consistent across versions of JAX
    and jaxlib, or even across invocations. It is relayed directly to external
    callers.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    # TODO(frostig): improve annotation (arbitrary pytree)
    raise NotImplementedError

  def runtime_executable(self) -> Any:
    """An arbitrary object representation of this executable.

    Intended for debugging purposes. This need not be a valid nor reliable
    serialization. It is relayed directly to external callers, with no
    guarantee on type, structure, or consistency across invocations.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend or
    compiler.
    """
    raise NotImplementedError

  def create_cpp_call(self, no_kwargs, in_tree, out_tree) -> Any:
    """Optionally constructs a fast c++ dispatcher."""
    return None


class Lowering(Protocol):
  """Protocol for lowerings, which a user-facing ``Lowered`` encapsulates."""

  def compile(self, compiler_options=None) -> Executable:
    """Compile and return a corresponding ``Executable``."""
    raise NotImplementedError

  def as_text(self, dialect: Optional[str] = None) -> str:
    """A human-readable text representation of this lowering.

    Intended for visualization and debugging purposes. This need not be a valid
    nor reliable serialization. It is relayed directly to external callers.
    """
    raise NotImplementedError

  def compiler_ir(self, dialect: Optional[str] = None) -> Any:
    """An arbitrary object representation of this lowering.

    Intended for debugging purposes. This need not be a valid nor reliable
    serialization. It is relayed directly to external callers, with no
    guarantee on type, structure, or consistency across invocations.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend or
    compiler.

    Args:
      dialect: Optional string specifying a representation dialect
      (e.g. "stablehlo")
    """
    raise NotImplementedError

  def cost_analysis(self) -> Any:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it need not be consistent across versions of JAX
    and jaxlib, or even across invocations. It is relayed directly to external
    callers.

    This function estimates execution cost in the absence of compiler
    optimizations, which may drastically affect the cost. For execution cost
    estimates after optimizations, compile this lowering and see
    ``Compiled.cost_analysis``.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    # TODO(frostig): improve annotation (arbitrary pytree)
    raise NotImplementedError

# -- Internal adapters from XLA-related objects to the above protocols

class XlaExecutable(Executable):

  def xla_extension_executable(self) -> xc.LoadedExecutable:
    raise NotImplementedError("must override")

  def call(self, *args_flat) -> Sequence[Any]:
    raise NotImplementedError("must override")

  def input_shardings(self) -> Sequence[jax.sharding.XLACompatibleSharding]:
    raise NotImplementedError(
        "compiled executable carries no input sharding information")

  def output_shardings(self) -> Sequence[jax.sharding.XLACompatibleSharding]:
    raise NotImplementedError(
        "compiled executable carries no output sharding information")

  def as_text(self) -> str:
    xla_ext_exe = self.xla_extension_executable()
    err_msg = ("text view unsupported on current XLA backend: "
               f"{type(xla_ext_exe)}")
    if not hasattr(xla_ext_exe, "hlo_modules"):
      raise NotImplementedError(err_msg)
    try:
      return "\n\n".join([m.to_string() for m in xla_ext_exe.hlo_modules()])
    except xla_extension.XlaRuntimeError as e:
      msg, *_ = e.args
      if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
        raise NotImplementedError(err_msg) from e
      else:
        raise

  def cost_analysis(self) -> List[Dict[str, float]]:
    xla_ext_exe = self.xla_extension_executable()
    err_msg = ("cost analysis unsupported on current XLA backend: "
               f"{type(xla_ext_exe)}")
    # TODO(b/259255524): Unify/merge the two cost_analysis calls below.
    if hasattr(xla_ext_exe, "client"):
      try:
        return [
            xla_extension.hlo_module_cost_analysis(xla_ext_exe.client, m)
            for m in xla_ext_exe.hlo_modules()
        ]
      except xla_extension.XlaRuntimeError as e:
        msg, *_ = e.args
        if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
          raise NotImplementedError(err_msg) from e
        else:
          raise
    elif hasattr(xla_ext_exe, "cost_analysis"):
      try:
        return xla_ext_exe.cost_analysis()
      except xla_extension.XlaRuntimeError as e:
        msg, *_ = e.args
        if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
          raise NotImplementedError(err_msg) from e
        else:
          raise
    else:
      raise NotImplementedError(err_msg)

  def memory_analysis(self) -> Any:
    xla_ext_exe = self.xla_extension_executable()
    err_msg = ("memory analysis unsupported on current XLA backend: "
               f"{type(xla_ext_exe)}")
    if not hasattr(xla_ext_exe, "get_compiled_memory_stats"):
      raise NotImplementedError(err_msg)
    try:
      return xla_ext_exe.get_compiled_memory_stats()
    except xla_extension.XlaRuntimeError as e:
      msg, *_ = e.args
      if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
        raise NotImplementedError(err_msg) from e
      else:
        raise

  def runtime_executable(self) -> Any:
    return self.xla_extension_executable()


class XlaLowering(Lowering):
  """Adapts our various internal XLA-backed computations into a ``Lowering``."""

  compile_args: Dict[str, Any]

  def hlo(self) -> xc.XlaComputation:
    """Return an HLO representation of this computation."""
    raise NotImplementedError("must override")

  def mhlo(self) -> ir.Module:
    """Return an MHLO representation of this computation."""
    module_str = xla_extension.mlir.stablehlo_to_mhlo(
        mlir.module_to_bytecode(self.stablehlo()))
    with self.stablehlo().context:
      return ir.Module.parse(module_str)

  def stablehlo(self) -> ir.Module:
    """Return a StableHLO representation of this computation."""
    raise NotImplementedError("must override")

  def compile(self, compiler_options=None) -> Executable:
    raise NotImplementedError("must override")

  def as_text(self, dialect: Optional[str] = None) -> str:
    if dialect is None:
      dialect = "stablehlo"
    if dialect == "mhlo":
      return str(self.mhlo())
    elif dialect == "stablehlo":
      return str(self.stablehlo())
    elif dialect == "hlo":
      return self.hlo().as_hlo_text()
    else:
      raise ValueError(f"unknown dialect: {dialect}")

  def compiler_ir(self, dialect: Optional[str] = None) -> Any:
    if dialect is None:
      dialect = "stablehlo"
    if dialect == "mhlo":
      return self.mhlo()
    elif dialect == "stablehlo":
      return self.stablehlo()
    elif dialect == "hlo":
      return self.hlo()
    else:
      raise ValueError(f"unknown dialect: {dialect}")

  def cost_analysis(self) -> Dict[str, float]:
    raise NotImplementedError("must override")


# -- Public-facing API, plus helpers

@dataclass
class ArgInfo:
  aval: core.AbstractValue
  donated: bool


class Stage:
  args_info: Any  # PyTree of ArgInfo

  @property
  def in_tree(self) -> tree_util.PyTreeDef:
    """Tree structure of the pair (positional arguments, keyword arguments)."""
    return tree_util.tree_structure(self.args_info)

  @property
  def in_avals(self):
    """Tree of input avals."""
    return tree_util.tree_map(lambda x: x.aval, self.args_info)

  @property
  def donate_argnums(self):
    """Flat tuple of donated argument indices."""
    return tuple(
        i for i, x in enumerate(tree_util.tree_leaves(self.args_info))
        if x.donated)


def make_args_info(in_tree, in_avals, donate_argnums):
  donate_argnums = frozenset(donate_argnums)
  flat_avals, _ = tree_util.tree_flatten(in_avals)  # todo: remove
  return in_tree.unflatten([
      ArgInfo(aval, i in donate_argnums)
      for i, aval in enumerate(flat_avals)])

class CompiledCallParams(NamedTuple):
  executable: Executable
  no_kwargs: bool
  in_tree: tree_util.PyTreeDef
  out_tree: tree_util.PyTreeDef


class Compiled(Stage):
  """Compiled representation of a function specialized to types/values.

  A compiled computation is associated with an executable and the
  remaining information needed to execute it. It also provides a
  common API for querying properties of compiled computations across
  JAX's various compilation paths and backends.
  """
  __slots__ = ["args_info", "out_tree", "_executable", "_no_kwargs"]

  args_info: Any                # PyTree of ArgInfo
  out_tree: tree_util.PyTreeDef
  _executable: Executable
  _no_kwargs: bool

  def __init__(self, executable, args_info, out_tree, no_kwargs=False):
    self._executable = executable
    self._no_kwargs = no_kwargs
    self.args_info = args_info
    self.out_tree = out_tree
    self._params = CompiledCallParams(self._executable, self._no_kwargs,
                                      self.in_tree, self.out_tree)
    self._cpp_call = self._executable.create_cpp_call(self._no_kwargs,
                                                      self.in_tree,
                                                      self.out_tree)

  def compiler_ir(self):
    """Post-compilation IR.

    Compilation typically involves code transformation and
    optimization. This method exists to reflect the compiler's
    representation of the program after such passes, whenever
    possible.
    """
    # TODO(frostig): remove (deprecated)
    warnings.warn(
        "compiler_ir() is deprecated, consider runtime_executable() instead",
        DeprecationWarning)
    exe = self.runtime_executable()
    return exe.hlo_modules() if exe is not None else None

  def as_text(self) -> Optional[str]:
    """A human-readable text representation of this executable.

    Intended for visualization and debugging purposes. This is not a valid nor
    reliable serialization.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    try:
      return self._executable.as_text()
    except NotImplementedError:
      return None

  def cost_analysis(self) -> Optional[Any]:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    # TODO(frostig): improve annotation (basic pytree of arbitrary structure)
    try:
      return self._executable.cost_analysis()
    except NotImplementedError:
      return None

  def memory_analysis(self) -> Optional[Any]:
    """A summary of estimated memory requirements.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    # TODO(frostig): improve annotation (basic pytree of arbitrary structure)
    try:
      return self._executable.memory_analysis()
    except NotImplementedError:
      return None

  def runtime_executable(self) -> Optional[Any]:
    """An arbitrary object representation of this executable.

    Intended for debugging purposes. This is not valid nor reliable
    serialization. The output has no guarantee of consistency across
    invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    return self._executable.runtime_executable()

  @property
  def input_shardings(self):  # PyTree[sharding.XLACompatibleSharding]
    shardings_flat = self._executable.input_shardings()
    return tree_util.tree_unflatten(self.in_tree, shardings_flat)  # pytype: disable=attribute-error

  @property
  def output_shardings(self):  # PyTree[sharding.XLACompatibleSharding]
    shardings_flat = self._executable.output_shardings()
    return tree_util.tree_unflatten(self.out_tree, shardings_flat)  # pytype: disable=attribute-error

  @staticmethod
  def call(*args, **kwargs):
    # This is because `__call__` passes in `self._params` as the first argument.
    # Instead of making the call signature `call(params, *args, **kwargs)`
    # extract it from args because `params` can be passed as a kwarg by users
    # which might confict here.
    params = args[0]
    args = args[1:]
    if jax.config.jax_dynamic_shapes:
      raise NotImplementedError
    if params.no_kwargs and kwargs:
      kws = ', '.join(kwargs.keys())
      raise NotImplementedError(
          "function was compiled by a transformation that does not support "
          f"keyword arguments, but called with keyword arguments: {kws}")
    args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
    if in_tree != params.in_tree:
      # TODO(frostig): provide more info about the source function
      # and transformation
      raise TypeError(
          f"function compiled for {params.in_tree}, called with {in_tree}")
    try:
      out_flat = params.executable.call(*args_flat)
    except TypeError as e:
      # We can't transform ahead-of-time compiled calls, since we've
      # lowered and compiled for a fixed function signature, and JAX
      # transformations change signatures. We interpret a Tracer
      # argument as an indication of a transformation attempt. We
      # could check this before the executable call, but we'd rather
      # avoid isinstance checks on the call path. Seeing a TypeError
      # might mean that arguments have JAX-invalid types, which in
      # turn might mean some are Tracers.
      for arg in args_flat:
        if isinstance(arg, core.Tracer):
          raise TypeError(
              "Cannot apply JAX transformations to a function lowered and "
              "compiled for a particular signature. Detected argument of "
              f"Tracer type {type(arg)}.") from e
      else:
        raise
    outs = tree_util.tree_unflatten(params.out_tree, out_flat)
    return outs, out_flat, args_flat

  def __call__(self, *args, **kwargs):
    if self._cpp_call is not None:
      return self._cpp_call(*args, **kwargs)

    outs, _, _ = Compiled.call(self._params, *args, **kwargs)
    return outs

  def serialize(self) -> bytes:
    """Serializes to bytes in a format with no stability garuntees."""

    unloaded_executable = getattr(self._executable,
                                  '_unloaded_executable', None)
    if unloaded_executable is None:
      raise ValueError("Compilation does not support serialization")
    args_info_flat, in_tree = jax.tree_util.tree_flatten(self.args_info)

    with io.BytesIO() as file:
      _JaxPjrtPickler(file).dump(
          (unloaded_executable, args_info_flat,
           self._no_kwargs, in_tree, self.out_tree))
      return file.getvalue()

  def load(self, backend=None) -> Compiled:
    """Loads a Compiled object onto a concrete backend."""
    raise NotImplementedError

  @classmethod
  def deserialize(cls, blob: bytes) -> _DeserializedCompiled:
    """Deserializes a Compiled object.

    The deserialized object is unusable and must be loaded via load().
    """
    return _DeserializedCompiled(blob)


class Lowered(Stage):
  """Lowering of a function specialized to argument types and values.

  A lowering is a computation ready for compilation. This class
  carries a lowering together with the remaining information needed to
  later compile and execute it. It also provides a common API for
  querying properties of lowered computations across JAX's various
  lowering paths (:func:`~jax.jit`, :func:`~jax.pmap`, etc.).
  """
  __slots__ = ["args_info", "out_tree", "_lowering", "_no_kwargs"]

  args_info: Any                # PyTree of ArgInfo
  out_tree: tree_util.PyTreeDef
  _lowering: XlaLowering
  _no_kwargs: bool

  def __init__(
      self,
      lowering: XlaLowering,
      args_info,  # PyTree of ArgInfo
      out_tree: tree_util.PyTreeDef,
      no_kwargs: bool = False):
    self._lowering = lowering
    self._no_kwargs = no_kwargs
    self.args_info = args_info
    self.out_tree = out_tree

  @classmethod
  def from_flat_info(cls,
                     lowering: XlaLowering,
                     in_tree: tree_util.PyTreeDef,
                     in_avals,
                     donate_argnums: Tuple[int, ...],
                     out_tree: tree_util.PyTreeDef,
                     no_kwargs: bool = False):
    """Initialize from flat info (``in_avals`` etc.) and an input PyTreeDef.

    Args:
      in_tree: The ``PyTreeDef`` of (args, kwargs).
      out_tree: The ``PyTreeDef`` of the outputs.
      no_kwargs: If ``True`` the transformation, and the
        ``Compiled`` returned from this object will not support keyword
        arguments (an error will be raised if some are provided).
    """
    return cls(
        lowering,
        make_args_info(in_tree, in_avals, donate_argnums),
        out_tree,
        no_kwargs=no_kwargs)

  def compile(self, compiler_options=None) -> Compiled:
    """Compile, returning a corresponding ``Compiled`` instance."""
    from jax._src.interpreters import pxla

    kw = {"compiler_options": compiler_options}

    if isinstance(self._lowering, pxla.MeshComputation):
      kw.update(
          _allow_propagation_to_outputs=[
              pxla._is_unspecified(o)
              for o in self._lowering.compile_args["out_shardings"]
          ]
      )

    return Compiled(
        self._lowering.compile(**kw),  # pytype: disable=wrong-keyword-args
        self.args_info,
        self.out_tree,
        no_kwargs=self._no_kwargs,
    )

  def as_text(self, dialect: Optional[str] = None) -> str:
    """A human-readable text representation of this lowering.

    Intended for visualization and debugging purposes. This need not be a valid
    nor reliable serialization. It is relayed directly to external callers.

    Args:
      dialect: Optional string specifying a lowering dialect (e.g. "stablehlo")
    """
    return self._lowering.as_text(dialect)

  def compiler_ir(self, dialect: Optional[str] = None) -> Optional[Any]:
    """An arbitrary object representation of this lowering.

    Intended for debugging purposes. This is not a valid nor reliable
    serialization. The output has no guarantee of consistency across
    invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.

    Args:
      dialect: Optional string specifying a lowering dialect (e.g. "stablehlo")
    """
    try:
      return self._lowering.compiler_ir(dialect)
    except NotImplementedError:
      return None

  def cost_analysis(self) -> Optional[Any]:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    # TODO(frostig): improve annotation (basic pytree of arbitrary structure)
    try:
      return self._lowering.cost_analysis()
    except NotImplementedError:
      return None


class Wrapped(Protocol):
  """A function ready to be specialized, lowered, and compiled.

  This protocol reflects the output of functions such as
  ``jax.jit``. Calling it results in JIT (just-in-time) lowering,
  compilation, and execution. It can also be explicitly lowered prior
  to compilation, and the result compiled prior to execution.
  """

  def __call__(self, *args, **kwargs):
    """Executes the wrapped function, lowering and compiling as needed."""
    raise NotImplementedError

  def lower(self, *args, **kwargs) -> Lowered:
    """Lower this function explicitly for the given arguments.

    A lowered function is staged out of Python and translated to a
    compiler's input language, possibly in a backend-dependent
    manner. It is ready for compilation but not yet compiled.

    Returns:
      A ``Lowered`` instance representing the lowering.
    """
    raise NotImplementedError


class _JaxPjrtPickler(pickle.Pickler):
  device_types = (xc.Device,)
  client_types = (xc.Client,)

  def persistent_id(self, obj):
    if isinstance(obj, xc.LoadedExecutable):
      return ('exec', obj.client.serialize_executable(obj),
              obj.compile_options())
    if isinstance(obj, xc._xla.Executable):
      return ('exec', obj.serialize(), obj.compile_options())
    if isinstance(obj, self.device_types):
      return ('device', obj.id)
    if isinstance(obj, self.client_types):
      return ('client',)


class _JaxPjrtUnpickler(pickle.Unpickler):

  def __init__(self, file, backend):
    super().__init__(file)
    self.backend = backend
    self.devices_by_id = {d.id: d for d in backend.devices()}

  def persistent_load(self, pid):
    if pid[0] == 'exec':
      return self.backend.deserialize_executable(pid[1], pid[2])
    if pid[0] == 'device':
      return self.devices_by_id[pid[1]]
    if pid[0] == 'client':
      return self.backend
    raise pickle.UnpicklingError


class _DeserializedCompiled:
  """Represents a Compiled object that has been deserialized and not loaded.

  Because of the fact that we can only currently deserialize using the concrete
  backend with physical devices attached, this is not a Compiled() object itself
  and can only be loaded. When that restriction is lifted, this class can be
  deleted in favor of Compiled.
  """

  def __init__(self, blob):
    self.blob = blob

  def load(self, backend=None) -> Compiled:
    """Loads a Compiled object onto a concrete backend."""
    if backend is None or isinstance(backend, str):
      backend = jax.devices(backend)[0].client
    (unloaded_executable, args_info_flat,
     no_kwargs, in_tree, out_tree) = _JaxPjrtUnpickler(io.BytesIO(self.blob), backend).load()

    args_info = in_tree.unflatten(args_info_flat)

    loaded_compiled_obj = unloaded_executable.load()

    return Compiled(
        loaded_compiled_obj, args_info, out_tree, no_kwargs=no_kwargs)
