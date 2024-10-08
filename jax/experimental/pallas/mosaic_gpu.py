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

"""Experimental GPU backend for Pallas targeting H100.

These APIs are highly unstable and can change weekly. Use at your own risk.
"""

from jax._src.pallas.mosaic_gpu.core import Barrier
from jax._src.pallas.mosaic_gpu.core import GPUBlockSpec
from jax._src.pallas.mosaic_gpu.core import GPUCompilerParams
from jax._src.pallas.mosaic_gpu.core import GPUMemorySpace
from jax._src.pallas.mosaic_gpu.core import TilingTransform
from jax._src.pallas.mosaic_gpu.core import TransposeTransform
from jax._src.pallas.mosaic_gpu.core import WGMMAAccumulatorRef as ACC
from jax._src.pallas.mosaic_gpu.core import reshape_ref
from jax._src.pallas.mosaic_gpu.primitives import copy_gmem_to_smem
from jax._src.pallas.mosaic_gpu.primitives import copy_smem_to_gmem
from jax._src.pallas.mosaic_gpu.primitives import wait_barrier
from jax._src.pallas.mosaic_gpu.primitives import wait_smem_to_gmem
from jax._src.pallas.mosaic_gpu.primitives import wgmma
from jax._src.pallas.mosaic_gpu.primitives import wgmma_wait

from jax.experimental.mosaic.gpu.utils import tile_shape

GMEM = GPUMemorySpace.GMEM
SMEM = GPUMemorySpace.SMEM
