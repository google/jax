# Copyright 2018 The JAX Authors.
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

"""Interface and utility functions to XLA.

This module wraps the XLA client(s) and builders to standardize their interfaces
and provide some automatic type mapping logic for converting between Numpy and
XLA. There are also a handful of related casting utilities.
"""

from collections.abc import Mapping
import dataclasses
from functools import partial, lru_cache
import glob
import importlib
import json
import logging
import os
import pathlib
import platform as py_platform
import pkgutil
import sys
import threading
from typing import Any, Callable, Optional, Union
import warnings

from jax._src import config
from jax._src import distributed
from jax._src.cloud_tpu_init import maybe_import_libtpu
from jax._src.lib import cuda_versions
from jax._src.lib import xla_client
from jax._src.lib import xla_extension
from jax._src.lib import xla_extension_version
from jax._src import traceback_util
from jax._src import util

logger = logging.getLogger(__name__)

jax_plugins: Optional[Any]
try:
  import jax_plugins  # type: ignore
except ModuleNotFoundError:
  jax_plugins = None
except ImportError as e:
  logger.error("Failed to import jax_plugins: %s", e)
  jax_plugins = None

traceback_util.register_exclusion(__file__)


XlaBackend = xla_client.Client


# TODO(phawkins): Remove jax_xla_backend.
_XLA_BACKEND = config.DEFINE_string(
    'jax_xla_backend', '',
    'Deprecated, please use --jax_platforms instead.')
BACKEND_TARGET = config.DEFINE_string(
    'jax_backend_target',
    os.getenv('JAX_BACKEND_TARGET', '').lower(),
    'Either "local" or "rpc:address" to connect to a remote service target.')
# TODO(skye): warn when this is used once we test out --jax_platforms a bit
_PLATFORM_NAME = config.DEFINE_string(
    'jax_platform_name',
    os.getenv('JAX_PLATFORM_NAME', '').lower(),
    'Deprecated, please use --jax_platforms instead.')
CUDA_VISIBLE_DEVICES = config.DEFINE_string(
    'jax_cuda_visible_devices', 'all',
    'Restricts the set of CUDA devices that JAX will use. Either "all", or a '
    'comma-separate list of integer device IDs.')
_ROCM_VISIBLE_DEVICES = config.DEFINE_string(
    'jax_rocm_visible_devices', 'all',
    'Restricts the set of ROCM devices that JAX will use. Either "all", or a '
    'comma-separate list of integer device IDs.')

_USE_MOCK_GPU_CLIENT = config.DEFINE_bool(
    name="use_mock_gpu_client",
    default=False,
    help="If True, use a mock GPU client instead of a real one.",
)

_MOCK_NUM_GPUS = config.DEFINE_integer(
    name="mock_num_gpus",
    default=1,
    help="Mock GPU client number of gpus.",
)


# Backends


def _get_tpu_library_path() -> Optional[str]:
  path_from_env = os.getenv("TPU_LIBRARY_PATH")
  if path_from_env is not None:
    return path_from_env

  libtpu_module = maybe_import_libtpu()
  if libtpu_module is not None:
    if hasattr(libtpu_module, "get_library_path"):
      # TODO(b/305803029): temporarily calls configure_library_path because the
      # tpu_tracer still depends on it. The tpu_tracer dependenecy on it will be
      # removed in the next two weeks.
      libtpu_module.configure_library_path()
      return libtpu_module.get_library_path()
    else:
      # TODO(b/305803029): Remove this branch around 01/2024 after the oldest
      # supported TPU has get_library_path.
      libtpu_module.configure_library_path()
      return os.getenv("TPU_LIBRARY_PATH", None)

  return None


def tpu_client_timer_callback(timer_secs: float) -> Optional[xla_client.Client]:
  def _log_warning():
    warnings.warn(
      f'TPU backend initialization is taking more than {timer_secs} seconds. '
      'Did you run your code on all TPU hosts? '
      'See https://jax.readthedocs.io/en/latest/multi_process.html '
      'for more information.')

  # Will log a warning after `timer_secs`.
  t = threading.Timer(timer_secs, _log_warning)
  t.start()

  try:
    if xla_extension_version >= 205:
      client = xla_client.make_tpu_client(_get_tpu_library_path())  # type: ignore
    else:
      libtpu_module = maybe_import_libtpu()
      if libtpu_module is not None:
        libtpu_module.configure_library_path()
      client = xla_client.make_tpu_client()  # type: ignore
  finally:
    t.cancel()

  return client


# Backends
#
# We have no particular opinion about how "backends" relate to "devices". For
# example, there could be multiple backends that provide the same kind of
# device.

BackendFactory = Callable[[], Optional[xla_client.Client]]

@dataclasses.dataclass
class BackendRegistration:
  factory: BackendFactory

  # Priority of this backend when choosing a default backend. Higher = more
  # preferred.
  priority: int

  # If this backend fails to initialize, should we log a user-visible warning?
  # For plugins (e.g., TPU) we usually want a visible failure, because why
  # install a plugin if you don't intend it to be used?
  fail_quietly: bool = False

  # Is this plugin experimental? If a plugin is deemed experimental, we issue
  # a warning when it is initialized. This is mostly to set user expectations
  # correctly: we don't want users to think that JAX is buggy because of a
  # a buggy plugin.
  experimental: bool = False

_backend_factories: dict[str, BackendRegistration] = {}
_default_backend: Optional[xla_client.Client] = None
_backends : dict[str, xla_client.Client] = {}
_backend_errors : dict[str, str] = {}
_backend_lock = threading.Lock()
_plugins_registered: bool = False
_plugin_lock = threading.Lock()

# The set of known non-experimental plugins.
#
# If a plugin passes the JAX test suite, it can be added to the allowlist below.
# Send a PR if you would like to be added.
#
# It is fine for a plugin not to implement every feature that JAX uses, provided
# that a reasonable feature set is implemented and the plugin fails gracefully
# for unimplemented features. Wrong outputs are not acceptable.
_nonexperimental_plugins: set[str] = set()

def register_backend_factory(name: str, factory: BackendFactory, *,
                             priority: int = 0,
                             fail_quietly: bool = True,
                             experimental: bool = False) -> None:
  with _backend_lock:
    if name in _backends:
      raise RuntimeError(f"Backend {name} already initialized")
  _backend_factories[name] = BackendRegistration(
    factory, priority, fail_quietly, experimental)


register_backend_factory(
    "cpu", xla_client.make_cpu_client, priority=0, fail_quietly=False
)


def _check_cuda_versions():
  # TODO(phawkins): remove the test for None cuda_versions after jaxlib 0.4.17
  # is the minimum.
  if cuda_versions is None:
    return

  def _version_check(name, get_version, get_build_version,
                     scale_for_comparison=1):
    build_version = get_build_version()
    try:
      version = get_version()
    except Exception as e:
      raise RuntimeError(f"Unable to load {name}.") from e
    if build_version // scale_for_comparison > version // scale_for_comparison:
      raise RuntimeError(
          f"Found {name} version {version}, but JAX was built against version "
          f"{build_version}, which is newer. The copy of {name} that is "
          "installed must be at least as new as the version against which JAX "
          "was built."
      )

  _version_check("CUDA", cuda_versions.cuda_runtime_get_version,
                 cuda_versions.cuda_runtime_build_version)
  _version_check(
      "cuDNN",
      cuda_versions.cudnn_get_version,
      cuda_versions.cudnn_build_version,
      # NVIDIA promise both backwards and forwards compatibility for cuDNN patch
      # versions: https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#api-compat
      scale_for_comparison=100,
  )
  _version_check("cuFFT", cuda_versions.cufft_get_version,
                 cuda_versions.cufft_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100)
  _version_check("cuSOLVER", cuda_versions.cusolver_get_version,
                 cuda_versions.cusolver_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100)
  _version_check("cuPTI", cuda_versions.cupti_get_version,
                 cuda_versions.cupti_build_version)
  # TODO(phawkins): ideally we'd check cublas and cusparse here also, but their
  # "get version" APIs require initializing those libraries, which we don't want
  # to do here.


def make_gpu_client(
    *, platform_name: str, visible_devices_flag: config.FlagHolder[str]
) -> xla_client.Client:
  visible_devices = visible_devices_flag.value
  allowed_devices = None
  if visible_devices != "all":
    allowed_devices = {int(x) for x in visible_devices.split(",")}

  if platform_name == "cuda":
    _check_cuda_versions()

  if xla_extension_version <= 199:
    return xla_client.make_gpu_client(
        distributed_client=distributed.global_state.client,
        node_id=distributed.global_state.process_id,
        num_nodes=distributed.global_state.num_processes,
        platform_name=platform_name,
        allowed_devices=allowed_devices,
    )
  use_mock_gpu_client = _USE_MOCK_GPU_CLIENT.value
  num_nodes = (
      _MOCK_NUM_GPUS.value
      if use_mock_gpu_client
      else distributed.global_state.num_processes
  )

  return xla_client.make_gpu_client(
      distributed_client=distributed.global_state.client,
      node_id=distributed.global_state.process_id,
      num_nodes=num_nodes,
      platform_name=platform_name,
      allowed_devices=allowed_devices,
      mock=use_mock_gpu_client,  # type: ignore[call-arg]
  )


if hasattr(xla_client, "make_gpu_client"):
  register_backend_factory(
      "cuda",
      partial(
          make_gpu_client,
          platform_name="cuda",
          visible_devices_flag=CUDA_VISIBLE_DEVICES,
      ),
      priority=200,
      fail_quietly=True,
  )
  register_backend_factory(
      "rocm",
      partial(
          make_gpu_client,
          platform_name="rocm",
          visible_devices_flag=_ROCM_VISIBLE_DEVICES,
      ),
      priority=200,
      fail_quietly=True,
  )


if hasattr(xla_client, "make_tpu_client"):
  # TODO(phawkins,skyewm): switch TPU plugin to use the PJRT plugin mechanism,
  # and then fail loudly on initialization failure.
  register_backend_factory(
    'tpu', partial(tpu_client_timer_callback, timer_secs=60.0), priority=300,
    fail_quietly=True)


def _get_pjrt_plugin_names_and_library_paths(
    plugins_from_env: str,
) -> dict[str, str]:
  """Gets the names and library paths of PJRT plugins to load from env var.

  Args:
    plugins_from_env: plugin name and paths from env var. It is in the format
      of 'name1:path1,name2:path2' ('name1;path1,name2;path2' for windows).

  Returns:
    A dict of {plugin_name: library path} for the PJRT plugins to load.
  """
  if not plugins_from_env:
    return {}

  pjrt_plugins = {}
  for plugin in plugins_from_env.split(','):
    try:
      name, library_path = plugin.split(os.path.pathsep)
      pjrt_plugins[name] = library_path
    except ValueError:
      logger.warning(
          'invalid value %s in env var PJRT_NAMES_AND_LIBRARY_PATHS %s',
          plugin,
          plugins_from_env,
      )
  return pjrt_plugins


def _get_pjrt_plugin_config(
    json_path: str,
) -> tuple[
    str, Optional[Mapping[str, Union[str, int, list[int], float, bool]]]
]:
  """Gets PJRT plugin configuration from a json file.

  The json file needs to have a "library_path" field for the plugin library
  path. It can have an optional "create_option" field for the options used when
  creating a PJRT plugin client. The value of "create_option" is key-value
  pairs. Please see xla_client._NameValueMapping for the supported types of
  values.
  """
  with open(json_path) as f:
    config = json.load(f)
  if 'library_path' not in config.keys():
    raise ValueError(
        'PJRT plugin config file should contain "library_path" field.'
    )
  return (config['library_path'], config.get('create_options'))

def discover_pjrt_plugins() -> None:
  """Discovers plugins in the namespace package `jax_plugins` and import them.

  There are two methods used to discover plugin modules. They are intended
  to be used together by implementors in order to cover all packaging and
  development cases:

  1. Define a globally unique module under the `jax_plugins` namespace
     package (i.e. just create a `jax_plugins` directory and define your
     module below it).
  2. If building a package via pyproject.toml or setup.py, advertise your
     plugin module name by including an entry-point under the `jax_plugins`
     group which points to your full module name.

  During Jax startup, Jax will load each module discovered in such a way and
  call its `initialize()` function. It is expected that this function should
  register its concrete plugin name/implementations via call(s) to
  `jax._src.xla_bridge.register_plugin(name, priority=, library_paty=,
  options=)`. Since `initialize()` functions are called for all installed
  plugins, they should avoid doing expensive, non-registration related work.

  TODO: We should provide a variant of `register_plugin` which allows the
  library_path and options to be resolved via a callback. This would enable
  light-weight plugin registration in cases where options need to be derived
  from heavy-weight system initialization.
  """
  plugin_modules = set()
  # Scan installed modules under |jax_plugins|. Note that not all packaging
  # scenarios are amenable to such scanning, so we also use the entry-point
  # method to seed the list.
  if jax_plugins:
    for _, name, _ in pkgutil.iter_modules(
        jax_plugins.__path__, jax_plugins.__name__ + '.'
    ):
      logger.debug("Discovered path based JAX plugin: %s", name)
      plugin_modules.add(name)
  else:
    logger.debug("No jax_plugins namespace packages available")

  # Augment with advertised entrypoints.
  if sys.version_info < (3, 10):
    # Use the backport library because it provides a forward-compatible
    # implementation.
    from importlib_metadata import entry_points
  else:
    from importlib.metadata import entry_points

  for entry_point in entry_points(group="jax_plugins"):
    logger.debug("Discovered entry-point based JAX plugin: %s",
                 entry_point.value)
    plugin_modules.add(entry_point.value)

  # Now load and initialize them all.
  for plugin_module_name in plugin_modules:
    logger.debug("Loading plugin module %s", plugin_module_name)
    plugin_module = None
    try:
      plugin_module = importlib.import_module(plugin_module_name)
    except ModuleNotFoundError:
      logger.warning("Jax plugin configuration error: Plugin module %s "
                     "does not exist", plugin_module_name)
    except ImportError:
      logger.exception("Jax plugin configuration error: Plugin module %s "
                       "could not be loaded")

    if plugin_module:
      try:
        plugin_module.initialize()
      except:
        logger.exception("Jax plugin configuration error: Exception when "
                         "calling %s.initialize()", plugin_module_name)


# TODO(b/261345120): decide on a public name and expose a public method which is
# an alias of this method.
def register_plugin(
    plugin_name: str,
    *,
    priority: int = 400,
    library_path: Optional[str] = None,
    options: Optional[Mapping[str, Union[str, int, list[int], float, bool]]] = None,
) -> None:
  """Registers a backend factory for the PJRT plugin.

  Args:
    plugin_name: the name of the plugin.
    priority: the priority this plugin should be registered in jax backends.
      Default to be 400.
    library_path: Optional. The full path to the .so file of the plugin.
      Required when the plugin is dynamically linked.
    options: Optional. It is used when creating a PJRT plugin client.
  """
  def factory():
    if xla_extension_version >= 183:
      if not xla_client.pjrt_plugin_initialized(plugin_name):
        xla_client.initialize_pjrt_plugin(plugin_name)

    if distributed.global_state.client is None:
      return xla_client.make_c_api_client(plugin_name, options, None)
    distribute_options = {
        'node_id': distributed.global_state.process_id,
        'num_nodes': distributed.global_state.num_processes,
    }
    if options is not None:
      distribute_options.update(options)
    return xla_client.make_c_api_client(
        plugin_name, distribute_options, distributed.global_state.client
    )


  logger.debug(
      'registering PJRT plugin %s from %s', plugin_name, library_path
  )
  experimental = plugin_name not in _nonexperimental_plugins
  register_backend_factory(plugin_name, factory, priority=priority,
                           fail_quietly=False, experimental=experimental)
  if library_path is not None:
    if xla_extension_version >= 198:
      c_api = xla_client.load_pjrt_plugin_dynamically(plugin_name, library_path)  # type: ignore
      if xla_extension_version >= 203:
        xla_client.profiler.register_plugin_profiler(c_api)
      return c_api
    else:
      xla_client.load_pjrt_plugin_dynamically(plugin_name, library_path)
  return None


def register_pjrt_plugin_factories_from_env() -> None:
  """Registers backend factories for PJRT plugins.

  A backend factory will be registered for every PJRT plugin in the input
  string, in the format of 'name1:path1,name2:path2' ('name1;path1,name2;path2'
  for windows). The path can be a path to the plugin library or a path to the
  plugin configuration json file. The json file needs to have a "library_path"
  field for the plugin library path. It can have an optional "create_option"
  field for the options used when creating a PJRT plugin client. The value of
  "create_option" is key-value pairs. Please see xla_client._NameValueMapping
  for the supported types of values.

  TPU PJRT plugin will be loaded and registered separately in make_tpu_client.
  """
  pjrt_plugins = _get_pjrt_plugin_names_and_library_paths(
      os.getenv('PJRT_NAMES_AND_LIBRARY_PATHS', '')
  )
  for plugin_name, path in pjrt_plugins.items():
    if path.endswith('.json'):
      library_path, options = _get_pjrt_plugin_config(path)
    else:
      library_path = path
      options = None
    logger.debug(
        'registering PJRT plugin %s from %s', plugin_name, library_path
    )
    register_plugin(plugin_name, library_path=library_path, options=options)


_platform_aliases = {
  "cuda": "gpu",
  "rocm": "gpu",
}

_alias_to_platforms: dict[str, list[str]] = {}
for _platform, _alias in _platform_aliases.items():
  _alias_to_platforms.setdefault(_alias, []).append(_platform)


def is_known_platform(platform: str) -> bool:
  # A platform is valid if there is a registered factory for it. It does not
  # matter if we were unable to initialize that platform; we only care that
  # we've heard of it and it isn't, e.g., a typo.
  return (platform in _backend_factories.keys() or
          platform in _platform_aliases.keys())


def canonicalize_platform(platform: str) -> str:
  """Replaces platform aliases with their concrete equivalent.

  In particular, replaces "gpu" with either "cuda" or "rocm", depending on which
  hardware is actually present. We want to distinguish "cuda" and "rocm" for
  purposes such as MLIR lowering rules, but in many cases we don't want to
  force users to care.
  """
  platforms = _alias_to_platforms.get(platform, None)
  if platforms is None:
    return platform

  b = backends()
  for p in platforms:
    if p in b.keys():
      return p
  raise RuntimeError(f"Unknown backend: '{platform}' requested, but no "
                     f"platforms that are instances of {platform} are present. "
                     "Platforms present are: " + ",".join(b.keys()))


def expand_platform_alias(platform: str) -> list[str]:
  """Expands, e.g., "gpu" to ["cuda", "rocm"].

  This is used for convenience reasons: we expect cuda and rocm to act similarly
  in many respects since they share most of the same code.
  """
  return _alias_to_platforms.get(platform, [platform])

def is_gpu(platform):
  return platform in ("cuda", "rocm")

def backends() -> dict[str, xla_client.Client]:
  global _backends
  global _backend_errors
  global _default_backend
  global _plugins_registered

  # Needs a separate lock because register_backend_factory (called from
  # register_plugin) requries to hold _backend_lock.
  with _plugin_lock:
    if not _plugins_registered:
      # Plugins in the namespace package `jax_plugins` or have an entry-point
      # under the `jax_plugins` group will be imported.
      discover_pjrt_plugins()
      # Registers plugins names and paths set in env var
      # PJRT_NAMES_AND_LIBRARY_PATHS, in the format of 'name1:path1,name2:path2'
      # ('name1;path1,name2;path2' for windows).
      register_pjrt_plugin_factories_from_env()
      _plugins_registered = True

  with _backend_lock:
    if _backends:
      return _backends
    if jax_platforms := config.jax_platforms.value:
      platforms = []
      # Allow platform aliases in the list of platforms.
      for platform in jax_platforms.split(","):
        platforms.extend(expand_platform_alias(platform))
      priorities = range(len(platforms), 0, -1)
      # If the user specified a list of platforms explicitly, always fail
      # loudly.
      fail_quietly_list = [False] * len(platforms)
      platform_registrations = list(
        zip(platforms, priorities, fail_quietly_list))
    else:
      platform_registrations = list(
          (platform, registration.priority, registration.fail_quietly)
          for platform, registration
          in _backend_factories.items()
      )
    default_priority = -1000
    for platform, priority, fail_quietly in platform_registrations:
      try:
        backend = _init_backend(platform)
        _backends[platform] = backend

        if priority > default_priority:
          _default_backend = backend
          default_priority = priority
      except Exception as err:
        err_msg = f"Unable to initialize backend '{platform}': {err}"
        if fail_quietly:
          _backend_errors[platform] = str(err)
          logger.info(err_msg)
        else:
          if config.jax_platforms.value:
            err_msg += " (set JAX_PLATFORMS='' to automatically choose an available backend)"
          else:
            err_msg += " (you may need to uninstall the failing plugin package, or set JAX_PLATFORMS=cpu to skip this backend.)"
          raise RuntimeError(err_msg)

    assert _default_backend is not None
    if not config.jax_platforms.value:
      _suggest_missing_backends()
    return _backends


# Code to suggest plugins that should be installed.
#
# Plugin vendors are welcome to add code to this list, assuming there's a
# lightweight way to determine if hardware is present without requiring
# the relevant plugin be installed.

_GOOGLE_PCI_VENDOR_ID = '0x1ae0'
_TPU_PCI_DEVICE_IDS = [
    # TPU v2, v3
    '0x0027',
    # TPU v4
    '0x005e',
    # TPU v5e
    '0x0063',
    # Testing only
    '0x0056',
    '0x0062',
]

def _num_available_tpu_chips() -> int:
  """Returns the number of TPU chips attached through PCI."""
  num_chips = 0
  for vendor_path in glob.glob('/sys/bus/pci/devices/*/vendor'):
    vendor_id = pathlib.Path(vendor_path).read_text().strip()
    if vendor_id != _GOOGLE_PCI_VENDOR_ID:
      continue

    device_path = os.path.join(os.path.dirname(vendor_path), 'device')
    device_id = pathlib.Path(device_path).read_text().strip()
    if device_id in _TPU_PCI_DEVICE_IDS:
      num_chips += 1

  return num_chips

def _suggest_missing_backends():
  if py_platform.system() != "Linux":
    # If you're not using Linux (or WSL2), we don't have any suggestions at the
    # moment.
    return

  assert _default_backend is not None
  default_platform = _default_backend.platform
  nvidia_gpu_devices = [
    "/dev/nvidia0",
    "/dev/dxg",  # WSL2
  ]
  if ("cuda" not in _backends and
      any(os.path.exists(d) for d in nvidia_gpu_devices)):
    if hasattr(xla_extension, "GpuAllocatorConfig") and "cuda" in _backend_errors:
      err = _backend_errors["cuda"]
      logger.warning(f"CUDA backend failed to initialize: {err} (Set "
                     "TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)")
    else:
      logger.warning("An NVIDIA GPU may be present on this machine, but a "
                     "CUDA-enabled jaxlib is not installed. Falling back to "
                     f"{default_platform}.")
  elif "tpu" not in _backends and _num_available_tpu_chips() > 0:
    logger.warning("A Google TPU may be present on this machine, but either a "
                    "TPU-enabled jaxlib or libtpu is not installed. Falling "
                    f"back to {default_platform}.")


def _clear_backends() -> None:
  global _backends
  global _backend_errors
  global _default_backend

  logger.info("Clearing JAX backend caches.")
  with _backend_lock:
    _backends = {}
    _backend_errors = {}
    _default_backend = None

  get_backend.cache_clear()


def _init_backend(platform: str) -> xla_client.Client:
  registration = _backend_factories.get(platform, None)
  if registration is None:
    raise RuntimeError(
        f"Backend '{platform}' is not in the list of known backends: "
        f"{list(_backend_factories.keys())}.")

  if registration.experimental:
    logger.warning(f"Platform '{platform}' is experimental and not all JAX "
                   "functionality may be correctly supported!")
  logger.debug("Initializing backend '%s'", platform)
  backend = registration.factory()
  # TODO(skye): consider raising more descriptive errors directly from backend
  # factories instead of returning None.
  if backend is None:
    raise RuntimeError(f"Could not initialize backend '{platform}'")
  if backend.device_count() == 0:
    raise RuntimeError(f"Backend '{platform}' provides no devices.")
  util.distributed_debug_log(("Initialized backend", backend.platform),
                             ("process_index", backend.process_index()),
                             ("device_count", backend.device_count()),
                             ("local_devices", backend.local_devices()))
  logger.debug("Backend '%s' initialized", platform)
  return backend


def _get_backend_uncached(
    platform: Union[None, str, xla_client.Client] = None
) -> xla_client.Client:
  # TODO(mattjj,skyewm): remove this input polymorphism after we clean up how
  # 'backend' values are handled
  if platform is not None and not isinstance(platform, str):
    return platform

  platform = (platform or _XLA_BACKEND.value or _PLATFORM_NAME.value or None)

  bs = backends()
  if platform is not None:
    platform = canonicalize_platform(platform)
    backend = bs.get(platform, None)
    if backend is None:
      if platform in _backend_errors:
        raise RuntimeError(f"Backend '{platform}' failed to initialize: "
                           f"{_backend_errors[platform]}. "
                           f'Available backends are {list(bs)}')
      raise RuntimeError(f"Unknown backend {platform}")
    return backend
  else:
    assert _default_backend is not None
    return _default_backend


@lru_cache(maxsize=None)  # don't use util.memoize because there is no X64 dependence.
def get_backend(
    platform: Union[None, str, xla_client.Client] = None
) -> xla_client.Client:
  return _get_backend_uncached(platform)


def get_device_backend(
    device: Optional[xla_client.Device] = None,
) -> xla_client.Client:
  """Returns the Backend associated with `device`, or the default Backend."""
  if device is not None:
    return device.client
  return get_backend()


def device_count(
    backend: Optional[Union[str, xla_client.Client]] = None
) -> int:
  """Returns the total number of devices.

  On most platforms, this is the same as :py:func:`jax.local_device_count`.
  However, on multi-process platforms where different devices are associated
  with different processes, this will return the total number of devices across
  all processes.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    Number of devices.

  """
  return int(get_backend(backend).device_count())


def local_device_count(
    backend: Optional[Union[str, xla_client.Client]] = None
) -> int:
  """Returns the number of devices addressable by this process."""
  return int(get_backend(backend).local_device_count())


def devices(
    backend: Optional[Union[str, xla_client.Client]] = None
) -> list[xla_client.Device]:
  """Returns a list of all devices for a given backend.

  .. currentmodule:: jaxlib.xla_extension

  Each device is represented by a subclass of :class:`Device` (e.g.
  :class:`CpuDevice`, :class:`GpuDevice`). The length of the returned list is
  equal to ``device_count(backend)``. Local devices can be identified by
  comparing :attr:`Device.process_index` to the value returned by
  :py:func:`jax.process_index`.

  If ``backend`` is ``None``, returns all the devices from the default backend.
  The default backend is generally ``'gpu'`` or ``'tpu'`` if available,
  otherwise ``'cpu'``.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    List of Device subclasses.
  """
  return get_backend(backend).devices()


def default_backend() -> str:
  """Returns the platform name of the default XLA backend."""
  return get_backend(None).platform

@lru_cache
def local_devices(process_index: Optional[int] = None,
                  backend: Optional[Union[str, xla_client.Client]] = None,
                  host_id: Optional[int] = None) -> list[xla_client.Device]:
  """Like :py:func:`jax.devices`, but only returns devices local to a given process.

  If ``process_index`` is ``None``, returns devices local to this process.

  Args:
    process_index: the integer index of the process. Process indices can be
      retrieved via ``len(jax.process_count())``.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    List of Device subclasses.
  """
  if host_id is not None:
    warnings.warn(
        "The argument to jax.local_devices has been renamed from `host_id` to "
        "`process_index`. This alias will eventually be removed; please update "
        "your code.")
    process_index = host_id
  if process_index is None:
    process_index = get_backend(backend).process_index()
  if not (0 <= process_index < process_count()):
    raise ValueError(f"Unknown process_index {process_index}")
  return [d for d in devices(backend) if d.process_index == process_index]


def process_index(
    backend: Optional[Union[str, xla_client.Client]] = None
) -> int:
  """Returns the integer process index of this process.

  On most platforms, this will always be 0. This will vary on multi-process
  platforms though.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    Integer process index.
  """
  return get_backend(backend).process_index()


# TODO: remove this sometime after jax 0.2.13 is released
def host_id(backend: Optional[Union[str, xla_client.Client]] = None) -> int:
  warnings.warn(
      "jax.host_id has been renamed to jax.process_index. This alias "
      "will eventually be removed; please update your code.")
  return process_index(backend)


@lru_cache
def process_count(
    backend: Optional[Union[str, xla_client.Client]] = None
) -> int:
  """Returns the number of JAX processes associated with the backend."""
  return max(d.process_index for d in devices(backend)) + 1


# TODO: remove this sometime after jax 0.2.13 is released
def host_count(backend: Optional[Union[str, xla_client.Client]] = None) -> int:
  warnings.warn(
      "jax.host_count has been renamed to jax.process_count. This alias "
      "will eventually be removed; please update your code.")
  return process_count(backend)


# TODO: remove this sometime after jax 0.2.13 is released
def host_ids(
    backend: Optional[Union[str, xla_client.Client]] = None
) -> list[int]:
  warnings.warn(
      "jax.host_ids has been deprecated; please use range(jax.process_count()) "
      "instead. jax.host_ids will eventually be removed; please update your "
      "code.")
  return list(range(process_count(backend)))


def using_pjrt_c_api(backend=None):
  return "PJRT C API" in get_backend(backend).platform_version


# TODO(parkers): Get rid of this in favor of a generic way to get topologies.
def make_pjrt_tpu_topology(topology_name='', **kwargs):
  if not xla_client.pjrt_plugin_loaded("tpu"):
    library_path = _get_tpu_library_path()
    if library_path is None:
      raise RuntimeError(
          "JAX TPU support not installed; cannot generate TPU topology. See"
          " https://github.com/google/jax#installation")
    xla_client.load_pjrt_plugin_dynamically("tpu", library_path)
  assert xla_client.pjrt_plugin_loaded("tpu")
  if not xla_client.pjrt_plugin_initialized("tpu"):
    xla_client.initialize_pjrt_plugin("tpu")
  return xla_client.make_tfrt_tpu_c_api_device_topology(
      topology_name, **kwargs
  )
