# Copyright 2018 Google LLC
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

# TODO(phawkins): this file triggers a pytype bug.
# pytype: skip-file

import contextlib
import functools
import itertools
import os
import sys
import threading
from typing import Any, List, Callable, NamedTuple, Optional
import warnings

from jax import lib
from jax.lib import jax_jit

def bool_env(varname: str, default: bool) -> bool:
  """Read an environment variable and interpret it as a boolean.

  True values are (case insensitive): 'y', 'yes', 't', 'true', 'on', and '1';
  false values are 'n', 'no', 'f', 'false', 'off', and '0'.

  Args:
    varname: the name of the variable
    default: the default boolean value
  Raises: ValueError if the environment variable is anything else.
  """
  val = os.getenv(varname, str(default))
  val = val.lower()
  if val in ('y', 'yes', 't', 'true', 'on', '1'):
    return True
  elif val in ('n', 'no', 'f', 'false', 'off', '0'):
    return False
  else:
    raise ValueError("invalid truth value %r for environment %r" % (val, varname))

def int_env(varname: str, default: int) -> int:
  """Read an environment variable and interpret it as an integer."""
  return int(os.getenv(varname, default))


class Config:
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self):
    self.values = {}
    self.meta = {}
    self.FLAGS = NameSpace(self.read, self.update)
    self.use_absl = False
    self._contextmanager_flags = set()
    self._update_hooks = {}

    self.omnistaging_enabled = True  # TODO(mattjj): remove this

  def update(self, name, val):
    if self.use_absl:
      setattr(self.absl_flags.FLAGS, name, val)
    else:
      self.check_exists(name)
      if name not in self.values:
        raise Exception("Unrecognized config option: {}".format(name))
      self.values[name] = val

    hook = self._update_hooks.get(name, None)
    if hook:
      hook(val)

  def read(self, name):
    if name in self._contextmanager_flags:
      raise AttributeError(
          "For flags with a corresponding contextmanager, read their value "
          f"via e.g. `config.{name}` rather than `config.FLAGS.{name}`.")
    return self._read(name)

  def _read(self, name):
    if self.use_absl:
      return getattr(self.absl_flags.FLAGS, name)
    else:
      self.check_exists(name)
      return self.values[name]

  def add_option(self, name, default, opt_type, meta_args, meta_kwargs,
                 update_hook=None):
    if name in self.values:
      raise Exception("Config option {} already defined".format(name))
    self.values[name] = default
    self.meta[name] = (opt_type, meta_args, meta_kwargs)
    if update_hook:
      self._update_hooks[name] = update_hook
      update_hook(default)

  def check_exists(self, name):
    if name not in self.values:
      raise AttributeError("Unrecognized config option: {}".format(name))

  def DEFINE_bool(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, bool, args, kwargs, update_hook=update_hook)

  def DEFINE_integer(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, int, args, kwargs, update_hook=update_hook)

  def DEFINE_string(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, str, args, kwargs, update_hook=update_hook)

  def DEFINE_enum(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, 'enum', args, kwargs,
                    update_hook=update_hook)

  def config_with_absl(self):
    # Run this before calling `app.run(main)` etc
    import absl.flags as absl_FLAGS  # noqa: F401
    from absl import app, flags as absl_flags

    self.use_absl = True
    self.absl_flags = absl_flags
    absl_defs = { bool: absl_flags.DEFINE_bool,
                  int:  absl_flags.DEFINE_integer,
                  str:  absl_flags.DEFINE_string,
                  'enum': absl_flags.DEFINE_enum }

    for name, val in self.values.items():
      flag_type, meta_args, meta_kwargs = self.meta[name]
      absl_defs[flag_type](name, val, *meta_args, **meta_kwargs)

    app.call_after_init(lambda: self.complete_absl_config(absl_flags))

  def complete_absl_config(self, absl_flags):
    for name, _ in self.values.items():
      self.update(name, getattr(absl_flags.FLAGS, name))

  def parse_flags_with_absl(self):
    global already_configured_with_absl
    if not already_configured_with_absl:
      # Extract just the --jax... flags (before the first --) from argv. In some
      # environments (e.g. ipython/colab) argv might be a mess of things
      # parseable by absl and other junk.
      jax_argv = itertools.takewhile(lambda a: a != '--', sys.argv)
      jax_argv = ['', *(a for a in jax_argv if a.startswith('--jax'))]

      import absl.flags
      self.config_with_absl()
      absl.flags.FLAGS(jax_argv, known_only=True)
      self.complete_absl_config(absl.flags)
      already_configured_with_absl = True

      if not FLAGS.jax_omnistaging:
        raise Exception(
          "Disabling of omnistaging is no longer supported in JAX version 0.2.12 and higher: "
          "see https://github.com/google/jax/blob/main/design_notes/omnistaging.md.\n"
          "To remove this warning, unset the JAX_OMNISTAGING environment variable.")

  def enable_omnistaging(self):
    warnings.warn(
        "enable_omnistaging() is a no-op in JAX versions 0.2.12 and higher;\n"
        "see https://github.com/google/jax/blob/main/design_notes/omnistaging.md")

  def disable_omnistaging(self):
    raise Exception(
      "Disabling of omnistaging is no longer supported in JAX version 0.2.12 and higher: "
      "see https://github.com/google/jax/blob/main/design_notes/omnistaging.md.")

  def define_bool_state(
    self, name: str, default: bool, help: str, *,
    update_global_hook: Optional[Callable[[bool], None]] = None,
    update_thread_local_hook: Optional[Callable[[Optional[bool]], None]] = None):
    """Set up thread-local state and return a contextmanager for managing it.

    This function is a convenience wrapper. It defines a flag, environment
    variable, and corresponding thread-local state, which can be managed via the
    contextmanager it returns.

    The thread-local state value can be read via the ``config.<option_name>``
    attribute, where ``config`` is the singleton ``Config`` instance.

    Args:
      name: string, converted to lowercase to define the name of the config
        option (and absl flag). It is converted to uppercase to define the
        corresponding shell environment variable.
      default: boolean, a default value for the option.
      help: string, used to populate the flag help information as well as the
        docstring of the returned context manager.
      update_global_hook: a optional callback that is called with the updated
        value of the global state when it is altered or set initially.
      update_thread_local_hook: a optional callback that is called with the
        updated value of the thread-local state when it is altered or set
        initially.

    Returns:
      A contextmanager to control the thread-local state value.

    Example:

      enable_foo = config.define_bool_state(
          name='jax_enable_foo',
          default=False,
          help='Enable foo.')

      # Now the JAX_ENABLE_FOO shell environment variable and --jax_enable_foo
      # command-line flag can be used to control the process-level value of
      # the configuration option, in addition to using e.g.
      # ``config.update("jax_enable_foo", True)`` directly. We can also use a
      # context manager:

      with enable_foo(True):
        ...

    The value of the thread-local state or flag can be accessed via
    ``config.jax_enable_foo``. Reading it via ``config.FLAGS.jax_enable_foo`` is
    an error.
    """
    name = name.lower()
    self.DEFINE_bool(name, bool_env(name.upper(), default), help,
                     update_hook=update_global_hook)
    self._contextmanager_flags.add(name)

    def get_state(self):
      val = getattr(_thread_local_state, name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    return _StateContextManager(name, help, update_thread_local_hook)

  def define_enum_state(
      self, name: str, enum_values: List[str], default: Optional[str],
      help: str, update_global_hook: Optional[Callable[[str], None]] = None,
      update_thread_local_hook: Optional[Callable[[Optional[str]], None]] \
        = None):
    """Set up thread-local state and return a contextmanager for managing it.
    Args:
      name: string, converted to lowercase to define the name of the config
        option (and absl flag). It is converted to uppercase to define the
        corresponding shell environment variable.
      enum_values: list of strings representing the possible values for the
        option.
      default: optional string, default value.
      help: string, used to populate the flag help information as well as the
        docstring of the returned context manager.
    Returns:
      A contextmanager to control the thread-local state value.
    See docstring for ``define_bool_state``.
    """
    name = name.lower()
    default = os.getenv(name.upper(), default)
    if default is not None and default not in enum_values:
      raise ValueError(f"Invalid value \"{default}\" for JAX flag {name}")
    self.DEFINE_enum(name, default,
                     enum_values=enum_values, help=help,
                     update_hook=update_global_hook)
    self._contextmanager_flags.add(name)

    def get_state(self):
      val = getattr(_thread_local_state, name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    def validate(new_val):
      if (new_val is not None and
          (type(new_val) is not str or new_val not in enum_values)):
        raise ValueError(f"new enum value must be None or in {enum_values}, "
                         f"got {new_val} of type {type(new_val)}.")

    return _StateContextManager(name, help, update_thread_local_hook, validate)

  def define_string_state(
      self, name: str, default: Optional[str], help: str,
      update_global_hook: Optional[Callable[[str], None]] = None,
      update_thread_local_hook: Optional[Callable[[Optional[str]], None]] = None):
    """Set up thread-local state and return a contextmanager for managing it.

    See docstring for ``define_bool_state``.

    Args:
      name: string, converted to lowercase to define the name of the config
        option (and absl flag). It is converted to uppercase to define the
        corresponding shell environment variable.
      default: string, a default value for the option.
      help: string, used to populate the flag help information as well as the
        docstring of the returned context manager.
      update_global_hook: an optional callback that is called with the updated
        value of the global state when it is altered or set initially.
      update_thread_local_hook: an optional callback that is called with the
        updated value of the thread-local state when it is altered or set
        initially.

    Returns:
      A contextmanager to control the thread-local state value.
    """
    name = name.lower()
    default = os.getenv(name.upper(), default)
    self.DEFINE_string(name, default, help=help,
                       update_hook=update_global_hook)
    self._contextmanager_flags.add(name)

    def get_state(self):
      val = getattr(_thread_local_state, name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    def validate(new_val):
      if new_val is not None and not isinstance(new_val, str):
        raise ValueError(f"new string config value must be None or of type str,"
                         f" got {new_val} of type {type(new_val)}.")

    return _StateContextManager(name, help, update_thread_local_hook, validate)

  def _trace_context(self):
    """Returns a tuple of configuration values that affect tracing.

    These values are included in the cache key for linear_util.cache.

    Values included in this set should also most likely be included in
    the C++ JIT state, which is handled separately."""
    return (self.x64_enabled, self.jax_numpy_rank_promotion,
            self.jax_default_matmul_precision)

class _StateContextManager:
  def __init__(self, name, help, update_thread_local_hook,
               validate_new_val_hook: Optional[Callable[[Any], None]] = None):
    self._name = name
    self.__name__ = name[4:] if name.startswith('jax_') else name
    self.__doc__ = f"Context manager for `{name}` config option.\n\n{help}"
    self._update_thread_local_hook = update_thread_local_hook
    self._validate_new_val_hook = validate_new_val_hook

  @contextlib.contextmanager
  def __call__(self, new_val):
    if self._validate_new_val_hook:
      self._validate_new_val_hook(new_val)
    prev_val = getattr(_thread_local_state, self._name, unset)
    setattr(_thread_local_state, self._name, new_val)
    if self._update_thread_local_hook:
      self._update_thread_local_hook(new_val)
    try:
      yield
    finally:
      if prev_val is unset:
        delattr(_thread_local_state, self._name)
        if self._update_thread_local_hook:
          self._update_thread_local_hook(None)
      else:
        setattr(_thread_local_state, self._name, prev_val)
        if self._update_thread_local_hook:
          self._update_thread_local_hook(prev_val)

  def _add_hooks(self, update_global_hook, update_thread_local_hook):
    """Private method that adds hooks to an existing context-manager.

    Used to avoid cyclic import dependencies."""
    self._update_thread_local_hook = update_thread_local_hook
    config._update_hooks[self._name] = update_global_hook
    update_global_hook(config._read(self._name))


_thread_local_state = threading.local()

class _Unset: pass
unset = _Unset()

class NameSpace:
  def __init__(self, getter, setter):
    # must use super because we override this class's __setattr__, see
    # https://docs.python.org/3/reference/datamodel.html#object.__setattr__
    super().__setattr__('_getter', getter)
    super().__setattr__('_setter', setter)

  def __getattr__(self, name):
    return self._getter(name)

  def __setattr__(self, name, val):
    self._setter(name, val)


config = Config()
flags = config
FLAGS = flags.FLAGS

already_configured_with_absl = False


# The C++ JIT maintains its own copy of several configuration items as
# a global/thread-local state. These methods allow updates to part of the
# state when a configuration value changes.

class GlobalJitState(NamedTuple):
  numpy_rank_promotion: Optional[str] = None
  default_matmul_precision: Optional[Any] = None


def update_global_jit_state(**kw):
  gs = jax_jit.global_state()
  context = gs.extra_jit_context or GlobalJitState()
  gs.extra_jit_context = context._replace(**kw)


class ThreadLocalJitState(NamedTuple):
  dynamic_trace_state: Optional[Any] = None
  numpy_rank_promotion: Optional[str] = None
  default_matmul_precision: Optional[Any] = None


def update_thread_local_jit_state(**kw):
  tls = jax_jit.thread_local_state()
  context = tls.extra_jit_context or ThreadLocalJitState()
  tls.extra_jit_context = context._replace(**kw)


# TODO(mattjj): remove all uses of this flag
flags.DEFINE_bool(
    'jax_omnistaging',
    bool_env('JAX_OMNISTAGING', True),
    help=('Deprecated. Setting this flag to False raises an error. Setting it '
          'to True has no effect.'),
)

flags.DEFINE_integer(
    'jax_tracer_error_num_traceback_frames',
    int_env('JAX_TRACER_ERROR_NUM_TRACEBACK_FRAMES', 5),
    help='Set the number of stack frames in JAX tracer error messages.'
)

flags.DEFINE_bool(
    'jax_host_callback_inline',
    bool_env('JAX_HOST_CALLBACK_INLINE', False),
    help='Inline the host_callback, if not in a staged context.'
)
flags.DEFINE_integer(
    'jax_host_callback_max_queue_byte_size',
    int_env('JAX_HOST_CALLBACK_MAX_QUEUE_BYTE_SIZE', int(256 * 1e6)),
    help=('The size in bytes of the buffer used to hold outfeeds from each '
          'device. When this capacity is reached consuming outfeeds from the '
          'device is paused, thus potentially pausing the device computation, '
          'until the Python callback consume more outfeeds.'),
    lower_bound=int(16 * 1e6)
)
flags.DEFINE_bool(
    'jax_host_callback_outfeed',
    bool_env('JAX_HOST_CALLBACK_OUTFEED', False),
    help=(
        'Use outfeed implementation for host_callback, even on CPU and GPU. '
        'If false, use the CustomCall implementation. '
        'Has no effect on TPU, since only the outfeed mechanism is implemented.'
    )
)

enable_checks = config.define_bool_state(
    name='jax_enable_checks',
    default=False,
    help='Turn on invariant checking for JAX internals. Makes things slower.')

check_tracer_leaks = config.define_bool_state(
    name='jax_check_tracer_leaks',
    default=False,
    help=('Turn on checking for leaked tracers as soon as a trace completes. '
          'Enabling leak checking may have performance impacts: some caching '
          'is disabled, and other overheads may be added.'))
checking_leaks = functools.partial(check_tracer_leaks, True)

debug_nans = config.define_bool_state(
    name='jax_debug_nans',
    default=False,
    help=('Add nan checks to every operation. When a nan is detected on the '
          'output of a jit-compiled computation, call into the un-compiled '
          'version in an attempt to more precisely identify the operation '
          'which produced the nan.'))

debug_infs = config.define_bool_state(
    name='jax_debug_infs',
    default=False,
    help=('Add inf checks to every operation. When an inf is detected on the '
          'output of a jit-compiled computation, call into the un-compiled '
          'version in an attempt to more precisely identify the operation '
          'which produced the inf.'))

log_compiles = config.define_bool_state(
    name='jax_log_compiles',
    default=False,
    help=('Log a message each time every time `jit` or `pmap` compiles an XLA '
          'computation. Logging is performed with `absl.logging`. When this '
          'option is set, the log level is WARNING; otherwise the level is '
          'DEBUG.'))

distributed_debug = config.define_bool_state(
    name='jax_distributed_debug',
    default=False,
    help=('Enable logging useful for debugging multi-process distributed '
          'computations. Logging is performed with `absl.logging` at WARNING '
          'level.'))

enable_custom_prng = config.define_bool_state(
    name='jax_enable_custom_prng',
    default=False,
    help=('Enables an internal upgrade that allows one to define custom '
          'pseudo-random number generator implementations. This will '
          'be enabled by default in future versions of JAX, at which point '
          'disabling it will be considered deprecated. In a version '
          'after that the flag will be removed altogether.'))

hlo_source_file_canonicalization_regex = config.define_string_state(
    name='jax_hlo_source_file_canonicalization_regex',
    default=None,
    help=('Used to canonicalize the source_path metadata of HLO instructions '
          'by removing the given regex. If set, re.sub() is called on each '
          'source_file with the given regex, and all matches are removed. '
          'This can be used to avoid spurious cache misses when using the '
          'persistent compilation cache, which includes HLO metadata in the '
          'cache key.'))

def _update_x64_global(val):
  lib.jax_jit.global_state().enable_x64 = val

def _update_x64_thread_local(val):
  lib.jax_jit.thread_local_state().enable_x64 = val

enable_x64 = config.define_bool_state(
    name='jax_enable_x64',
    default=False,
    help='Enable 64-bit types to be used',
    update_global_hook=_update_x64_global,
    update_thread_local_hook=_update_x64_thread_local)

# TODO(phawkins): remove after fixing users of FLAGS.x64_enabled.
config._contextmanager_flags.remove("jax_enable_x64")

Config.x64_enabled = Config.jax_enable_x64  # type: ignore

def _update_disable_jit_global(val):
  lib.jax_jit.global_state().disable_jit = val

def _update_disable_jit_thread_local(val):
  lib.jax_jit.thread_local_state().disable_jit = val

disable_jit = config.define_bool_state(
    name='jax_disable_jit',
    default=False,
    help=('Disable JIT compilation and just call original Python.'),
    update_global_hook=_update_disable_jit_global,
    update_thread_local_hook=_update_disable_jit_thread_local)


numpy_rank_promotion = config.define_enum_state(
    name='jax_numpy_rank_promotion',
    enum_values=['allow', 'warn', 'raise'],
    default='allow',
    help=('Control NumPy-style automatic rank promotion broadcasting '
          '("allow", "warn", or "raise").'),
    update_global_hook=lambda val: \
      update_global_jit_state(numpy_rank_promotion=val),
    update_thread_local_hook=lambda val: \
      update_thread_local_jit_state(numpy_rank_promotion=val))

default_matmul_precision = config.define_enum_state(
    name='jax_default_matmul_precision',
    enum_values=['bfloat16', 'tensorfloat32', 'float32'],
    default='float32',
    help=('Control the default matmul and conv precision for 32bit inputs.\n\n'

          'Some platforms, like TPU, offer configurable precision levels for '
          'matrix multiplication and convolution computations, trading off '
          'accuracy for speed. The precision can be controlled for each '
          'operation; for example, see the :func:`jax.lax.conv_general_dilated` '
          'and :func:`jax.lax.dot` docstrings. But it can be useful to control '
          'the default behavior obtained when an operation is not given a '
          'specific precision.\n\n'

          'This option can be used to control the default precision '
          'level for computations involved in matrix multiplication and '
          'convolution on 32bit inputs. The levels roughly describe the '
          "precision at which scalar products are computed. The 'bfloat16' "
          "option is the fastest and least precise; 'float32' is similar to "
          "full float32 precision; 'tensorfloat32' is intermediate.\n\n"),
    update_global_hook=lambda val: \
      update_global_jit_state(default_matmul_precision=val),
    update_thread_local_hook=lambda val: \
      update_thread_local_jit_state(default_matmul_precision=val))

traceback_filtering = config.define_enum_state(
    name = 'jax_traceback_filtering',
    enum_values=["off", "tracebackhide", "remove_frames", "auto"],
    default="auto",
    help="Controls how JAX filters internal frames out of tracebacks.\n\n"
         "Valid values are:\n"
         " * \"off\": disables traceback filtering.\n"
         " * \"auto\": use \"tracebackhide\" if running under a sufficiently "
         "new IPython, or \"remove_frames\" otherwise.\n"
         " * \"tracebackhide\": adds \"__tracebackhide__\" annotations to "
         " hidden stack frames, which some traceback printers support.\n"
         " * \"remove_frames\": removes hidden frames from tracebacks, and adds "
         " the unfiltered traceback as a __cause__ of the exception.\n")
