# Copyright 2020 The JAX Authors.
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
"""Primitives for calling Python functions on the host from JAX accelerator code.

.. warning::
  The host_callback APIs are deprecated as of March 20, 2024.
  The functionality is subsumed by the
  `new JAX external callbacks <https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html>`_
  See https://github.com/google/jax/issues/20385.

This module introduces the host callback functions :func:`call`,
:func:`id_tap`, and :func:`id_print`, that send their arguments from the device
to the host and invoke user-defined Python functions on the host, optionally
returning results back to the device computation.

We show below how these functions can be used. We start with :func:`call`,
and we discuss examples of calling from JAX to arbitrary Python functions
on the CPU, e.g., to use NumPy CPU custom kernels. Then we
show uses of :func:`id_tap` and :func:`id_print`, which have the restriction
that they cannot return values from the host to the device.
These primitives are generally faster
because they are executed asynchronously with the device code.
In particular, they can be used to tap into and to debug JAX code.

Using :func:`call` to call a host function and return results to device
-----------------------------------------------------------------------

Use :func:`call` to invoke a computation on the host and return
NumPy arrays to the device computation.
Host computation is useful, e.g., when a device computation needs some data
that requires I/O on the host, or it needs a library that is available on the
host and you do not want to code it in JAX.
For example, eigen decomposition for general matrices in JAX does not work on TPU.
We can call the Numpy implementation from any JAX accelerator computation,
using a host computation::

  # This function runs on the host
  def host_eig(m: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(m)

  # This function is used in JAX
  def device_fun(m):
    # We send "m" to the host, asking it to call "host_eig" and return the result.
    # We have to specify the result shape and dtype, either in the form of an
    # example return value or any object that has `shape` and `dtype` attributes,
    # e.g., a NumPy array or a `jax.ShapeDtypeStruct`.
    return hcb.call(host_eig, m,
                    # Given an input of shape (..., d, d), eig output has shape (..., d)
                    result_shape=jax.ShapeDtypeStruct(m.shape[:-1], m.dtype))


The :func:`call` function and the Python host function both take a single argument
and return a single result, but those can be pytrees. Note that we must tell
the :func:`call` what shape and dtype to expect from the host invocation, using
the ``result_shape`` keyword argument.
This is important because the device code is compiled with that expectation.
There will be an error raised at runtime if the actual invocation produces a
different result shape. In general, **such errors and also exceptions raised
by the host computation may be difficult to debug**. See the Debugging section
below.
This is a problem for :func:`call` but not for :func:`id_tap` because for the
latter the device code does not expect a returned value.

The :func:`call` API can be used inside a jit or pmap computation or inside
cond/scan/while control flow. When used inside :func:`jax.pmap`, there will be
separate calls to the host from each of the participating devices::

  def host_sin(x, *, device):
    # The ``device`` argument is passed due to ``call_with_device=True`` below.
    print(f"Invoking host_sin with {x.shape} on {device}")
    return np.sin(x)

  # Use pmap to run the computation on two devices
  jax.pmap(lambda x: hcb.call(host_sin, x,
                              result_shape=x,
                              # Ask that the `host_sin` function be passed `device=dev`
                              call_with_device=True))(
           np.ones((2, 4), dtype=np.float32))

  # prints (in arbitrary order)
  # Invoking host_sin with (4,) on cpu:0
  # Invoking host_sin with (4,) on cpu:1

Note that :func:`call` does not support any JAX transformations, but as we
show below one can make use of the
existing support for `Custom differentiation in JAX <https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_.

Using :func:`id_tap` to call a Python function on the host, with no returned values
-----------------------------------------------------------------------------------

The :func:`id_tap` and :func:`id_print` are special cases of :func:`call`, when
you just want the side effects of your Python callback. These functions have
the advantage that once the arguments have been sent to the host, the device
computation can proceed without waiting for the Python callback to return.
For :func:`id_tap` you can specify your Python callback to be called, while
:func:`id_print` uses a built-in callback that prints the arguments to
`stdout` on the host.
The Python function passed
to :func:`id_tap` takes two positional arguments (the value tapped
from the device computation along with a ``transforms`` tuple,
described below). Optionally, the function may be passed a keyword argument
``device`` with the Device from which the value was tapped.

A few examples::

  def host_func(arg, transforms):
     ...do something with arg...

  # calls host_func(2x, []) on host
  id_tap(host_func, 2 * x)

  # calls host_func((2x, 3x), [])
  id_tap(host_func, (2 * x, 3 * x))  # The argument can be a pytree

  # calls host_func(2x, [], device=jax.devices()[0])
  id_tap(host_func, 2 * x, tap_with_device=True)  # Pass the device to the tap

  # calls host_func(2x, [], what='activation')
  id_tap(functools.partial(host_func, what='activation'), 2 * x)

  # calls host_func(dict(x=x, y=y), what='data')
  id_tap(lambda tap, transforms: host_func(tap, what='data'), dict(x=x, y=y))

The above examples can all be adapted to use :func:`id_print` instead, with
the difference that :func:`id_print` prints on the host the positional argument,
along with any additional kwargs and the automatic kwarg ``transforms``.

Using :func:`barrier_wait` to wait until all callbacks have executed
--------------------------------------------------------------------

If your Python callbacks have side-effects you may need to wait until the
computation has finished to ensure that the side-effects have been observed.
You can use the :func:`barrier_wait` function for that purpose::

   accumulator = []
   def host_log(arg, transforms):
     # We just record the arguments in a list
     accumulator.append(arg)


   def device_fun(x):
     id_tap(host_log, x)
     id_tap(host_log, 2. * x)

   jax.jit(device_fun)(1.)
   jax.jit(device_fun)(1.)

   # At this point, we have started two computations, each with two
   # taps, but they may not have yet executed.
   barrier_wait()
   # Now we know that all the computations started before `barrier_wait`
   # on all devices, have finished, and all the callbacks have finished
   # executing.

Note that :func:`barrier_wait` will start one
tiny computation with one tap on each of the `jax.local_devices()` and
will wait for all these taps to be received.

An alternative to using :func:`barrier_wait` is to just wait for the end
of the computation, if all the callbacks are :func:`call`::

   accumulator = p[]
   def host_log(arg):
     # We just record the arguments in a list
     accumulator.append(arg)
     return 0.  #  return something


   def device_fun(c):
     y = call(host_log, x, result_shape=jax.ShapeDtypeStruct((), np.float32))
     z = call(host_log, 2. * x, result_shape=jax.ShapeDtypeStruct((), np.float32))
     return y + z  # return something that uses both results

   res1 = jax.jit(device_fun)(1.)
   res2 = jax.jit(device_fun)(1.)
   res1.block_until_ready()
   res2.block_until_ready()

Behavior under parallelization transformations
----------------------------------------------

In presence of :func:`jax.pmap` the code will run on multiple devices and
each device will tap its values independently.
It may be helpful to use the ``tap_with_device`` option for :func:`id_print`
or :func:`id_tap`, so that you see which device is sending which data::

  jax.pmap(power3, devices=jax.local_devices()[:2])(np.array([3., 4.])
  # device=cpu:0 what=x,x^2: (3., 9.)  # from the first device
  # device=cpu:1 what=x,x^2: (4., 16.)  # from the second device

When using :func:`jax.pmap` with multiple devices on multiple hosts, every
host will receive callbacks from all of its local devices, with an operand
that corresponds to each device slice. For a
:func:`call`, the callback must return to each device only the slice of the
result that pertains to the corresponding device.

When using the experimental :func:`pjit.pjit` the code will run on multiple
devices on different shards of the input. The current implementation of
host callbacks will ensure that a single device will collect and outfeed
the entire operand, in a single callback. The callback function is supposed
to return the entire array, which will then be sent in a single infeed to the
same device that issued the outfeed. This device is then responsible for
sending the required shards to the other devices::

  with jax.sharding.Mesh(jax.local_devices()[:2], ["d"]):
    pjit.pjit(power3, in_shardings=(P("d"),),
              out_shardings=(P("d"),))(np.array([3., 4.]))

  # device=TPU:0 what=x,x^2: ( [3., 4.],
  #                            [9., 16.] )

Note that the collection of the operand on one device may result in OOM if
the operand was sharded across devices.

When using :func:`pjit.pjit` with multiple devices on multiple hosts, only
the host for the device 0 (w.r.t. the mesh) will receive the callback, with
the operand collected
from all participating devices on all hosts. For a :func:`call`, the callback
must return the entire array for all devices on all hosts.

Behavior under JAX autodiff transformations
-------------------------------------------

When used under a JAX autodiff transformation, the host callback functions
operate on the primal values only. Consider the following example::

    def power3(x):
      y = x * x
      # Print both 'x' and 'x^2'. Must pack as a tuple.
      hcb.id_print((x, y), what="x,x^2")
      return y * x

    power3(3.)
    # what: x,x^2 : (3., 9.)

(You can see these examples tested in `host_callback_test.HostCallbackTapTest.test_tap_transforms`.)

When used under :func:`jax.jvp` there will be one callback with the primal
values only::

    jax.jvp(power3, (3.,), (0.1,))
    # what: x,x^2 : (3., 9.)

Similarly for :func:`jax.grad`, we get a callback from the forward computation
only::

    jax.grad(power3)(3.)
    # what: x,x^2 : (3., 9.)

If you want to invoke the callback on the tangents during a :func:`jax.jvp`,
you can use a custom_jvp. For example, you can define a function that does
nothing interesting except that its custom_jvp will print the tangents::

    @jax.custom_jvp
    def print_tangents(arg):
      return None

    @print_tangents.defjvp
    def print_tangents_jvp(primals, tangents):
      arg_dot, = tangents
      hcb.id_print(arg_dot, what="tangents")
      return primals, tangents

Then you use this function in the places where you want to tap the tangents::

    def power3_with_tangents(x):
      y = x * x
      # Print both 'x' and 'x^2'. Must pack as a tuple.
      hcb.id_print((x, y), what="x,x^2")
      print_tangents((x, y))
      return y * x

    jax.jvp(power3_with_tangents, (3.,), (0.1,))
    # what: x,x^2 : (3., 9.)
    # what: tangents : (0.1, 0.6)

You can do a similar thing for the cotangents during :func:`jax.grad`. This
time you must be careful to use in the rest of the computation the values whose
cotangents you want to tap. Hence we make the ``print_cotangents`` return
its argument::

    @jax.custom_vjp
    def print_cotangents(arg):
      # Must return the argument for which we want the cotangent.
      return arg

    # f_fwd: a -> (b, residual)
    def print_cotangents_fwd(arg):
      return print_cotangents(arg), None
    # f_bwd: (residual, CT b) -> [CT a]
    def print_cotangents_bwd(residual, ct_b):
      hcb.id_print(ct_b, what="cotangents", output_stream=testing_stream)
      return ct_b,

    print_cotangents.defvjp(print_cotangents_fwd, print_cotangents_bwd)

    def power3_with_cotangents(x):
      y = x * x
      # Print both 'x' and 'x^2'. Must pack as a tuple.
      hcb.id_print((x, y), what="x,x^2", output_stream=testing_stream)
      (x1, y1) = print_cotangents((x, y))
      # Must use the output of print_cotangents
      return y1 * x1

    jax.grad(power3_with_cotangents)(3.)
    # what: x,x^2 : (3., 9.)
    # what: cotangents : (9., 3.)

If you use :func:`ad_checkpoint.checkpoint` to rematerialize the residuals
for the backward pass, then the callbacks from the primal computation will
be called twice::

    jax.grad(lambda x: power3(ad_checkpoint.checkpoint(power3)(x)))(3.)
    # what: x,x^2 : (3., 9.)
    # what: x,x^2 : (27., 729.)
    # what: x,x^2 : (3., 9.)

The callbacks are, in order from: the primal computation of the inner ``power3``,
the primal computation of the outer ``power3``, and the rematerialization
of the residuals for the inner ``power3``.


Behavior under jax.vmap
-----------------------

The host callback functions :func:`id_print` and :func:`id_tap` support the
vectorization transformation :func:`jax.vmap`.

For :func:`jax.vmap` the arguments to the callback are batched,
and the callback function is
passed an additional special ``transforms`` containing a list of transformation descriptors
in the form ``("batch", {"batch_dims": ...})``, where ``...``` denotes the
batched dimensions for the tapped values (one entry per argument, `
`None`` denotes an argument that was broadcast).

  jax.vmap(power3)(np.array([2., 3.]))
  # transforms: [('batch', {'batch_dims': (0, 0)})] what: x,x^2 : ([2., 3.], [4., 9.])

See documentation for :func:`id_tap`, :func:`id_print`, and :func:`call`.

For more usage example, see tests/host_callback_test.py.

Using :func:`call` to call a TensorFlow function, with reverse-mode autodiff support
------------------------------------------------------------------------------------

Another possible use for host computation is to invoke a library written for
another framework, such as TensorFlow.
In this case it becomes interesting to support JAX autodiff for host callbacks
by deferring to the autodiff mechanism in TensorFlow,
using the :func:`jax.custom_vjp` mechanism.

This is relatively easy to do, once one understands both the JAX custom VJP
and the TensorFlow autodiff mechanisms.
The code for how this can be done is shown in the ``call_tf_full_ad``
function in `host_callback_to_tf_test.py <https://github.com/google/jax/blob/main/tests/host_callback_to_tf_test.py>`_.
This example supports arbitrary higher-order differentiation as well.

Note that if you just want to call TensorFlow functions from JAX, you can also
use the `jax2tf.call_tf function <https://github.com/google/jax/blob/main/jax/experimental/jax2tf/call_tf.py>`_.

Using :func:`call` to call a JAX function on another device, with reverse-mode autodiff support
------------------------------------------------------------------------------------------------

It should not be surprising that we can use host computation to invoke a JAX
computation on another device. The arguments are sent from the accelerator to
the host, and then to the outside device on which the JAX host
computation will run, and then the results are sent back to the original accelerator.

The code for how this can be done is shown in the ``call_jax_other_device function``
in `host_callback_test.py <https://github.com/google/jax/blob/main/tests/host_callback_test.py>`_.

Low-level details and debugging
-------------------------------

The host callback functions will be executed for each device in the order in
which the send operations were performed on the device.

The host callback functions for multiple devices may be interleaved.
The data from the devices is received by separate threads managed by the JAX
runtime (one thread per device). The runtime maintains a buffer of
configurable size (see the flag ``--jax_host_callback_max_queue_byte_size``).
When the buffer is full, all the receiving threads are paused
which eventually pauses the computation on devices. The runtime has one
additional thread for each device to invoke the Python user functions with the
received data. If the processing of the callbacks is slow, it may actually
lead to the runtime buffer filling up, and eventually pausing the computation
on the devices when they need to send something.
For more details on the outfeed receiver runtime mechanism see
`runtime code
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/outfeed_receiver.cc>`_.

In order to pause the execution until all data from computations already
started on devices has arrived and has been processed, use :func:`barrier_wait`.

Exceptions from the user-defined callback functions are logged along with their
stack traces, but the receiving threads are not stopped. Instead the last
exception is recorded and the subsequent :func:`barrier_wait` will
raise :exc:`CallbackException` if any exception had occurred
in one of the tap functions. This exception will include the text and the
stack trace of the last exception encountered.

One further complication arises for callback functions that must return
results to the call origin device, such as :func:`call()`. This is handled
differently on CPU/GPU devices compared to TPU devices.

On CPU/GPU devices, in order to avoid the device computation
being stuck waiting for a result that will never arrive, in case of any
error during the processing of the callback (whether raised by the user-code
itself or due to a mismatch of the returned value and the expected return_shape)
we send the device a "fake" result of shape ``int8[12345]``.
This will make the device
computation abort because the received data is different than the one that
it expects. On CPU the runtime will crash with a distinctive error message:

```
Check failed: buffer->length() == buffer_length (12345 vs. ...)
```

On GPU, the failure is more user-friendly and will be surfaced to the Python
program as:

```
RET_CHECK failure ... Mismatch between infeed source buffer shape s8[12345] ...
```

To debug the underlying cause for these messages, see the Debugging section.

On TPU devices, there is currently no shape check for infeed, so we take the
safer route of not sending this fake result in case of errors. This means
that the computation will hang, and no exception will be raised (but any
exceptions in the callback functions will still appear in the logs).

The current implementation uses the outfeed mechanism provided by XLA. The
mechanism itself is quite primitive in the sense that a receiver must know
exactly the shape of each incoming packet, and how many packets are expected.
This makes it hard to use for multiple kinds of data in the same computation,
and it is practically impossible to use it under conditionals or in loops
of non-constant iteration count. Furthermore, code that uses the outfeed
mechanism directly cannot be transformed by JAX. All these limitations are
addressed by the host callback functions. The tapping API introduced here
makes it easy to share the outfeed mechanism for multiple purposes, while
supporting all transformations.

**Note that after you have used the host callback functions, you cannot
use lax.outfeed directly**. You may want to :func:`stop_outfeed_receiver`
if you later need to use lax.outfeed.

Since the actual calls to your callback functions are made from the C++
receiver, it may be hard to debug the calls. In particular, the stack trace
will not include the calling code. You can use the flag
``jax_host_callback_inline`` (or the environment variable
``JAX_HOST_CALLBACK_INLINE``) to ensure that the calls to the callbacks are
inlined. This works only if the calls are outside a staging context
(:func:`~jax.jit` or a control-flow primitive).

The C++ `receiver
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/outfeed_receiver.cc>`_
is started automatically on the first call to :func:`id_tap`. In order to stop
it properly, upon start an ``atexit`` handler is registered to call
:func:`barrier_wait` with the logging name "at_exit".

There are a few environment variables that you can use to turn on logging
for the C++ outfeed `receiver backend
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/outfeed_receiver.cc>`_.

  * ``TF_CPP_MIN_LOG_LEVEL=0``: will turn on INFO logging, needed for all below.
  * ``TF_CPP_MIN_VLOG_LEVEL=3``: will make all VLOG logging up to level 3 behave
    like INFO logs. This may be too much, but you will see which modules are
    logging relevant info, and then you can select which modules to log from.
  * ``TF_CPP_VMODULE=<module_name>=3`` (the module name can be either C++ or
    Python, without the extension).

You should also use the ``--verbosity=2`` flag so that you see the logs
from Python.

For example, you can try to enable logging in the ``host_callback`` module:
``TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=host_callback=3 python tests/host_callback_test.py --verbosity=2 HostCallbackIdTapTest.test_tap_jit_simple``

If you want to enable logging in lower-level implementation modules try:
``TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=outfeed_receiver=3,host_callback=3,outfeed_receiver_py=3,outfeed_thunk=3,infeed_thunk=3,cpu_transfer_manager=3,cpu_runtime=3,xfeed_manager=3,pjrt_client=3 python tests/host_callback_test.py --verbosity=2 HostCallbackIdTapTest.test_tap_jit_simple``

(For bazel tests use --test_arg=--vmodule=...

Still to do:
  * More performance tests.
  * Explore implementation with outside compilation for TPU.
  * Explore implementation with XLA CustomCall for CPU and GPU.

"""

from __future__ import annotations

import atexit
import enum
from collections.abc import Callable, Sequence
import functools
import itertools
import logging
import math
import threading
import traceback
from typing import Any, cast

import jax
from jax._src import api
from jax._src import core
from jax._src import config
from jax import custom_derivatives
from jax._src import dtypes
from jax import lax
from jax.experimental import pjit
from jax.experimental import io_callback
from jax._src.interpreters import ad, batching, pxla
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import xla
from jax._src import ad_checkpoint
from jax._src import compiler
from jax._src import dispatch
from jax._src import pretty_printer as pp
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import tree_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client
from jax._src.lib import xla_extension
from jax._src.lib.mlir.dialects import hlo

import numpy as np


_HOST_CALLBACK_INLINE = config.bool_flag(
    'jax_host_callback_inline',
    config.bool_env('JAX_HOST_CALLBACK_INLINE', False),
    help='Inline the host_callback, if not in a staged context.'
)
_HOST_CALLBACK_MAX_QUEUE_BYTE_SIZE = config.int_flag(
    'jax_host_callback_max_queue_byte_size',
    config.int_env('JAX_HOST_CALLBACK_MAX_QUEUE_BYTE_SIZE', int(256 * 1e6)),
    help=('The size in bytes of the buffer used to hold outfeeds from each '
          'device. When this capacity is reached consuming outfeeds from the '
          'device is paused, thus potentially pausing the device computation, '
          'until the Python callback consume more outfeeds.'),
    lower_bound=int(16 * 1e6)
)
_HOST_CALLBACK_OUTFEED = config.bool_flag(
    'jax_host_callback_outfeed',
    config.bool_env('JAX_HOST_CALLBACK_OUTFEED', False),
    help=(
        'Use outfeed implementation for host_callback, even on CPU and GPU. '
        'If false, use the CustomCall implementation. '
        'Has no effect on TPU, since only the outfeed mechanism is implemented.'
    )
)
_HOST_CALLBACK_LEGACY = config.bool_flag(
    'jax_host_callback_legacy',
    config.bool_env('JAX_HOST_CALLBACK_LEGACY', True),
    help=(
        'Use old implementation of host_callback, documented in the module docstring.'
        'If False, use the jax.experimental.io_callback implementation. '
        'See https://github.com/google/jax/issues/20385.'
    )
)

logger = logging.getLogger(__name__)


def _use_outfeed(platform: str) -> bool:
  return (platform in ("tpu", "gpu", "cuda", "rocm") or
          _HOST_CALLBACK_OUTFEED.value)


def _raise_if_using_outfeed_with_pjrt_c_api(backend: xb.XlaBackend):
  """Should be called whenever outfeed (or infeed) will be used."""
  if xb.using_pjrt_c_api(backend):
    raise NotImplementedError(
        "host_callback functionality isn't supported with PJRT C API. "
        "See https://jax.readthedocs.io/en/latest/debugging/index.html and "
        "https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html"
        " for alternatives. Please file a feature request at "
        "https://github.com/google/jax/issues if none of the alternatives are "
        "sufficient.")


xops = xla_client._xla.ops

XlaOp = xla_client.XlaOp
XlaShape = xla_client.Shape
XlaBuilder = xla_client.XlaBuilder
XlaDevice = xla_client.Device
XlaLocalClient = xla_client.Client
DType = Any

class CallbackFlavor(enum.Enum):
  """Specifies which flavor of callback to use under JAX_HOST_CALLBACK_LEGACY=False.

  See https://github.com/google/jax/issues/20385.
  """
  IO_CALLBACK = 1  # uses jax.experimental.io_callback
  PURE = 2  # uses jax.pure_callback
  DEBUG = 3  # uses jax.debug.callback, valid only when there are no results


def _deprecated_id_tap(tap_func,
           arg,
           *,
           result=None,
           tap_with_device=False,
           device_index=0,
           callback_flavor=CallbackFlavor.IO_CALLBACK,
           **kwargs):
  """Host-callback tap primitive, like identity function with a call to ``tap_func``.

  .. warning::
    The host_callback APIs are deprecated as of March 20, 2024.
    The functionality is subsumed by the
    `new JAX external callbacks <https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html>`_
    See https://github.com/google/jax/issues/20385.

  ``id_tap`` behaves semantically like the identity function but has the
  side-effect that a user-defined Python function is called with the runtime
  value of the argument.

  Args:
    tap_func: tap function to call like ``tap_func(arg, transforms)``, with
      ``arg`` as described below and where ``transforms`` is the sequence of
      applied JAX transformations in the form ``(name, params)``. If the
      `tap_with_device` optional argument is True, then the invocation also
      includes the device from which the value is tapped as a keyword argument:
      ``tap_func(arg, transforms, device=dev)``.
    arg: the argument passed to the tap function, can be a pytree of JAX
      types.
    result: if given, specifies the return value of ``id_tap``. This value is
      not passed to the tap function, and in fact is not sent from the device to
      the host. If the ``result`` parameter is not specified then the return
      value of ``id_tap`` is ``arg``.
    tap_with_device: if True then the tap function is invoked with the
      device from which the tap originates as a keyword argument.
    device_index: specifies from which device the tap function is invoked in a
      SPMD program. Works only when using the outfeed implementation mechanism,
      i.e., does not work on CPU unless --jax_host_callback_outfeed=True.
    callback_flavor: if running with `JAX_HOST_CALLBACK_LEGACY=False` specifies
       the flavor of callback to use.
       See https://github.com/google/jax/issues/20385.

  Returns:
    ``arg``, or ``result`` if given.

  The order of execution is by data dependency: after all the arguments and
  the value of ``result`` if present, are computed and before the returned
  value is used. At least one of the returned values of ``id_tap`` must be
  used in the rest of the computation, or else this operation has no effect.

  Tapping works even for code executed on accelerators and even for code under
  JAX transformations.

  For more details see the :mod:`jax.experimental.host_callback` module documentation.
  """
  if kwargs:
    msg = (
        "Support for **kwargs in ``id_tap`` has been removed. Instead, "
        "pre-apply keyword arguments, either by using a closure or by passing "
        "``functools.partial(tap_func, **kwargs)``.")
    raise TypeError(msg)

  if result is not None:
    flat_results, _ = tree_util.tree_flatten(result)
    for r in flat_results:
      dispatch.check_arg(r)

  call_res = _call(
      tap_func,
      arg,
      call_with_device=tap_with_device,
      result_shape=None,
      identity=True,
      device_index=device_index,
      callback_flavor=callback_flavor)

  if result is not None:
    return result
  else:
    return call_res


def _deprecated_id_print(arg,
             *,
             result=None,
             tap_with_device=False,
             device_index=0,
             output_stream=None,
             threshold=None,
             callback_flavor=CallbackFlavor.IO_CALLBACK,
             **kwargs):
  """Like :func:`id_tap` with a printing tap function.

  .. warning::
    The host_callback APIs are deprecated as of March 20, 2024.
    The functionality is subsumed by the
    `new JAX external callbacks <https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html>`_
    See https://github.com/google/jax/issues/20385.

   On each invocation of the printing tap, the ``kwargs`` if present
   will be printed first (sorted by keys). Then arg will be printed,
   with the arrays stringified with ``numpy.array2string``.

   See the :func:`id_tap` documentation.

   Additional keyword arguments:

   * ``tap_with_device`` if True, will print also the device from which
     the value originates.
   * ``output_stream`` if given then it will be used instead of the
     built-in ``print``. The string will be passed as
     ``output_stream.write(s)``.
   * ``threshold`` is passed to ``numpy.array2string``.
   * ``callback_flavor``: if running with `JAX_HOST_CALLBACK_LEGACY=False` specifies
       the flavor of callback to use.
       See https://github.com/google/jax/issues/20385.

  For more details see the :mod:`jax.experimental.host_callback` module documentation.
  """
  printer = functools.partial(_print_tap_func,
                              output_stream=output_stream,
                              threshold=threshold, **kwargs)
  return _deprecated_id_tap(
      printer,
      arg,
      result=result,
      tap_with_device=tap_with_device,
      device_index=device_index,
      callback_flavor=callback_flavor)


def _deprecated_call(callback_func: Callable, arg, *,
         result_shape=None,
         call_with_device=False,
         device_index=0,
         callback_flavor=CallbackFlavor.IO_CALLBACK):
  """Make a call to the host, and expect a result.

  .. warning::
    The host_callback APIs are deprecated as of March 20, 2024.
    The functionality is subsumed by the
    `new JAX external callbacks <https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html>`_
    See https://github.com/google/jax/issues/20385.

  Args:
    callback_func: The Python function to invoke on the host as
      ``callback_func(arg)``. If the ``call_with_device`` optional argument is True,
      then the invocation also includes the ``device`` kwarg with the device
      from which the call originates: ``callback_func(arg, device=dev)``. This function
      must return a pytree of numpy ndarrays.

    arg: the argument passed to the callback function, can be a pytree of JAX
      types.

    result_shape: a value that describes the expected shape and dtype of the
      result. This can be a numeric scalar, from which a shape and dtype are
      obtained, or an object that has ``.shape`` and ``.dtype`` attributes.
      If the result of the callback is a pytree, then ``result_shape`` should
      also be a pytree with the same structure. In particular, ``result_shape``
      can be `()` or `None` if the function does not have any results.
      The device code containing ``call`` is compiled with the expected result shape and dtype,
      and an error will be raised at runtime if the actual ``callback_func``
      invocation returns a different kind of result.

    call_with_device: if True then the callback function is invoked with the
      device from which the call originates as a keyword argument.

    device_index: specifies from which device the tap function is invoked in a
      SPMD program. Works only when using the outfeed implementation mechanism,
      i.e., does not work on CPU unless --jax_host_callback_outfeed=True.
    callback_flavor: if running with `JAX_HOST_CALLBACK_LEGACY=False` specifies
       the flavor of callback to use.
       See https://github.com/google/jax/issues/20385.

  Returns:
    the result of the ``callback_func`` invocation.

  For more details see the :mod:`jax.experimental.host_callback` module documentation.
  """
  if (not _HOST_CALLBACK_LEGACY.value and
      callback_flavor is CallbackFlavor.DEBUG and
      result_shape is not None):
    raise NotImplementedError(
        "When using JAX_HOST_CALLBACK_LEGACY=False you can use the `DEBUG` "
        "flavor of callback only when the `result_shape` is None. "
        "See https://github.com/google/jax/issues/20385."
    )
  return _call(callback_func, arg, result_shape=result_shape,
               call_with_device=call_with_device, identity=False,
               device_index=device_index, callback_flavor=callback_flavor)


# We need the wrapper function to have hash and equality defined since it is
# used as a primitive keyword argument, and we want a compilation cache hit if
# the user uses the same function twice.
class _CallbackWrapper:
  def __init__(self, callback_func, identity, call_with_device):
    self.callback_func = callback_func
    self.identity = identity
    self.call_with_device = call_with_device
    if not _HOST_CALLBACK_LEGACY.value and call_with_device:
      raise NotImplementedError(
          "When using JAX_HOST_CALLBACK_LEGACY=False, the host_callback APIs"
          " do not support `tap_with_device` and `call_with_device`. "
          "See https://github.com/google/jax/issues/20385.")

  def __hash__(self):
    return hash((self.callback_func, self.identity, self.call_with_device))

  def __eq__(self, other):
    return (self.callback_func == other.callback_func and
            self.identity == other.identity and
            self.call_with_device == other.call_with_device)

  def __call__(self, *args, **kwargs):
    if _HOST_CALLBACK_LEGACY.value:
      return self._call_legacy(*args, **kwargs)
    else:
      if self.identity:
        # For id_tap, we pass empty transforms, for backwards compatibility
        return self.callback_func(args[0], ())
      return self.callback_func(*args, **kwargs)

  def _call_legacy(self, arg, device, transforms):
    if self.identity:
      # For id_tap, we pass the transforms, for backwards compatibility
      if self.call_with_device:
        return self.callback_func(arg, transforms, device=device)
      else:
        return self.callback_func(arg, transforms)
    else:
      if self.call_with_device:
        return self.callback_func(arg, device=device)
      else:
        return self.callback_func(arg)


# Helper function to implement both `call` and `id_tap`. The two cases are
# differentiated by the `identity` flag.
def _call(callback_func: Callable,
          arg,
          *,
          result_shape=None,
          call_with_device=False,
          device_index=0,
          identity=False,
          callback_flavor=CallbackFlavor.IO_CALLBACK):
  if _HOST_CALLBACK_LEGACY.value:
    # Lazy initialization
    _initialize_outfeed_receiver(
        max_callback_queue_size_bytes=_HOST_CALLBACK_MAX_QUEUE_BYTE_SIZE.value)
  api.check_callable(callback_func)
  flat_args, arg_treedef = tree_util.tree_flatten(arg)
  for arg_ in flat_args:
    dispatch.check_arg(arg_)
  # See definition of outside_call_p for what parameters it takes
  params: dict[str, Any] = {}
  # TODO: wrap function
  params["callback"] = _CallbackWrapper(callback_func, identity,
                                        call_with_device)
  params["identity"] = identity
  params["arg_treedef"] = arg_treedef
  params["device_index"] = device_index

  if not identity:
    # Turn abstract values into ShapesDtypeStruct
    flat_results_shape, result_treedef = tree_util.tree_flatten(result_shape)
    try:
      flat_results_aval = [core.ShapedArray(np.shape(r), dtypes.dtype(r, canonicalize=True))
                           for r in flat_results_shape]
    except Exception:
      msg = ("result_shape should be a pytree of values with structure "
             "matching the expected result of the callback function. The "
             "values must be either numeric scalars, or must have 'shape' and "
             f"'dtype' attributes. Got {result_shape}")
      raise ValueError(msg)

    params["result_treedef"] = result_treedef
    params["flat_results_aval"] = tuple(flat_results_aval)

  if _HOST_CALLBACK_LEGACY.value:
    flat_results = outside_call_p.bind(*flat_args, **params)
    return result_treedef.unflatten(flat_results) if not identity else arg_treedef.unflatten(flat_results)
  else:
    callback_device = jax.local_devices()[device_index]
    sharding = jax.sharding.SingleDeviceSharding(callback_device)
    callback_func = _CallbackWrapper(callback_func, identity,
                                     call_with_device)
    if callback_flavor is CallbackFlavor.DEBUG:
      assert identity
      jax.debug.callback(callback_func, arg)
      return arg
    elif callback_flavor is CallbackFlavor.PURE:
      call_res = jax.pure_callback(callback_func, result_shape, arg,
                                   sharding=sharding)
    else:
      call_res = io_callback(callback_func, result_shape, arg,
                             sharding=sharding,
                             ordered=True)
    return call_res if not identity else arg


# We need the lock for when we use the CustomCall implementation of callbacks.
# The outfeed implementation is driven by a single thread from C++.
_print_tap_lock = threading.Lock()


def _print_tap_func(
    arg, transforms, *, device=None,
    output_stream=None, threshold=1024, **kwargs):
  """The consumer for id_print.

  We provide this as a simple tapping function for printing.
  This is **experimental** and may not want to add many features to it;
  it should be easy for the user to roll their own printing function.

  Args:
    device: the device from which the value originates (only if
      ``tap_with_device`` was used for :func:`id_print`).
    output_stream: a function whose `write` method is called with the strings to
      be output.
    threshold: the value of numpy.array2string threshold parameter.
    **kwargs: all other keyword args are printed before printing `arg`.
  """
  def emit_str(s: str):
    if output_stream is not None:
      output_stream.write(s + "\n")
    else:
      print(s)

  if transforms:
    kwargs['transforms'] = [(name, params) if params else name
                            for name, params in transforms]
  if device is not None:
    kwargs['device'] = device
  kv_pairs = " ".join([
      f"{k}: {v}" for k, v in sorted(kwargs.items())
  ])

  def pp_val(arg) -> pp.Doc:
    if isinstance(arg, tuple):
      return pp.group(pp.concat([
        pp.text("( "),
        pp.nest(2, pp.join(pp.brk(), [pp_val(e) for e in arg])),
        pp.text(" )")
      ]))
    elif isinstance(arg, list):
      return pp.group(pp.concat([
        pp.text("[ "),
        pp.nest(2, pp.join(pp.brk(), [pp_val(e) for e in arg])),
        pp.text(" ]")
      ]))
    elif isinstance(arg, dict):
      return pp.group(pp.concat([
        pp.text("{ "),
        pp.nest(2, pp.join(pp.brk(), [
          pp.text(f"{k}=") + pp_val(v) for k, v in sorted(arg.items())
        ])),
        pp.text(" }")
      ]))
    elif isinstance(arg, np.ndarray):
      return pp.text(np.array2string(arg, threshold=threshold))
    else:
      return pp.text(str(arg))

  with _print_tap_lock:
    if kv_pairs:
      emit_str(kv_pairs)
    emit_str(str(pp_val(arg)))


def _values_to_avals(vals) -> Sequence[core.ShapedArray]:
  return tuple(core.raise_to_shaped(core.get_aval(v)) for v in vals)

### The outside_call primitive
"""
This primitive is used to implement the `call` and `id_tap` functions.
It takes several positional arguments that are the flattened
according to `arg_treedef`.
The result of the primitive is computed based on the `identity` parameter,
as follows:

  * if `identity` is True, then the results are the same as the
  positional arguments of the primitive (except perhaps the last couple of
  arguments, see `has_token`). In this case, `result_treedef` and
  `flat_results_aval` are ignored, and `args_treedef` describes the result also.
  * if `identity` is False, then the results are those from
  the call to the outside computation:

     flatten(callback(arg_treedef.unflatten(args), device=...))

   In this case, the callback results must match `result_treedef`
   and `flat_results_aval`.

It takes the following parameters:

  * callback: the function to invoke with the unflattened arguments,
    the device and the transforms: `callback(arrays, device, transforms)`
  * arg_treedef: the treedef for the argument.
  * identity: see description above.
  * result_treedef, flat_results_aval: describes the expected result of the
    callback. Only used when not `identity`.
  * transforms: a tuple of the transformations that have been applied. Each
    element of the tuple is itself a tuple with the first element the name
    of the transform. The remaining elements depend on the transform. For
    example, for `batch`, the parameters are the dimensions that have been
    batched, and for `mask` the logical shapes. These are unpacked by
    _outside_call_run_callback before passing to the user function.
  * has_token: a boolean, when True it means that the last positional argument
    is the current token. In this case, the result of the primitive is
    going to be the non-token positional arguments, along with the updated
    token. The tokens and this parameter are added after all the JAX
    transformations, just before staging XLA.
  * device_index: an integer, denotes from which device the invocation is from.
    Works only when using the outfeed implementation mechanism, i.e., does
    not work on CPU unless --jax_host_callback_outfeed=True.
"""
outside_call_p = core.Primitive("outside_call")
outside_call_p.multiple_results = True
core.outfeed_primitives.add(outside_call_p)


def _outside_call_abstract_eval(*args_a: pe.AbstractValue,
                                identity, **params) -> Sequence[pe.AbstractValue]:
  if identity:
    # Do some validation here
    assert "result_treedef" not in params
    assert "flat_results_aval" not in params
    return args_a
  assert params["device_index"] is not None
  assert params["result_treedef"] is not None
  assert params["flat_results_aval"] is not None
  flat_results_aval = params["flat_results_aval"]
  if "has_token" in params and params["has_token"]:
    assert len(args_a) >= 2
    return flat_results_aval + args_a[-2:]
  else:
    return flat_results_aval


outside_call_p.def_abstract_eval(_outside_call_abstract_eval)


def _outside_call_impl(*args, **params):
  assert "has_token" not in params
  if _HOST_CALLBACK_INLINE.value:
    device_index = params["device_index"]
    device = xb.devices()[device_index]
    results = _outside_call_run_callback(args, device, send_infeed=False, **params)
    return results
  else:
    # We use the jitted-version of the primitive even for eager execution, both
    # so that we do not duplicate logic, but also so that all outfeed is received
    # by the outfeed_listeners, in the same thread from a given device. If we were
    # to process the tap here, it would be coming from the main thread. Also,
    # even in eager execution some primitives, such as while, are compiled.
    # It would be confusing to process a sequence "id_tap; while" in two
    # different threads.
    return dispatch.apply_primitive(outside_call_p, *args, **params)


outside_call_p.def_impl(_outside_call_impl)


def _with_sharding_proto(builder, sharding_proto, op_fn, *args, **kwargs):
  """Builds op_fn(*args, **kwargs) with sharding annotation."""
  builder.set_sharding(sharding_proto)
  try:
    return op_fn(*args, **kwargs)
  finally:
    builder.clear_sharding()


def _outside_call_translation_rule(ctx,
                                   avals_in,
                                   avals_out,
                                   *args_op: XlaOp,
                                   has_token,
                                   identity,
                                   device_index,
                                   flat_results_aval=(),
                                   **params):
  # We expect the current tokens at the end, inserted by _rewrite_jaxpr.
  assert has_token
  use_outfeed = _use_outfeed(ctx.platform)
  assert use_outfeed, 'Should be using MLIR path for `CustomCall` lowering'
  current_token = args_op[-2]
  current_itoken = args_op[-1]
  comp = ctx.builder
  assert comp.get_shape(current_token).is_token() and comp.get_shape(current_itoken).is_token(), (
      "The last two arguments must be tokens")

  args_to_outfeed = args_op[:-2]
  # Some platforms refuse to infeed empty arrays. We generate constants
  # instead.
  non_empty_flat_results_aval = list(filter(lambda aval: not (_aval_is_empty(aval)),
                                            flat_results_aval))
  need_callback_results_on_device = (not identity and
                                     len(non_empty_flat_results_aval) > 0)
  send_infeed = use_outfeed and need_callback_results_on_device
  generated_infeed = False  # Keep track if we emitted an infeed op

  _raise_if_using_outfeed_with_pjrt_c_api(xb.get_backend(ctx.platform))
  callback_id = _register_callback(
      functools.partial(
          _outside_call_run_callback,
          send_infeed=send_infeed,
          identity=identity,
          flat_results_aval=flat_results_aval,
          **params))
  next_token = _callback_handler_data.receiver.add_outfeed(
      comp, current_token, callback_id, args_to_outfeed, device_index)
  if identity:
    results = list(args_to_outfeed)
    next_itoken = current_itoken
  else:
    empty_results = [
        xops.ConstantLiteral(comp, np.zeros(aval.shape, aval.dtype))
        for aval in flat_results_aval
        if _aval_is_empty(aval)
    ]
    if non_empty_flat_results_aval:
      assert need_callback_results_on_device
      after_outfeed_itoken = xops.AfterAll(comp, [current_itoken, next_token])
      # We shard the infeed as AssignedDevice(device_index). This must match the
      # outfeed (from outfeed_receiver.cc). Since `lax.infeed` does not support
      # this kind of sharding, we use a custom translation for infeed.
      array_sharding_proto = xla_client.OpSharding()
      array_sharding_proto.type = xla_client.OpSharding.Type.MAXIMAL
      array_sharding_proto.tile_assignment_dimensions = [1]
      array_sharding_proto.tile_assignment_devices = [device_index]

      token_sharding_proto = xla_client.OpSharding()
      token_sharding_proto.type = xla_client.OpSharding.Type.REPLICATED
      infeed_sharding_proto = xla.tuple_sharding_proto(
          [array_sharding_proto] * len(non_empty_flat_results_aval) +
          [token_sharding_proto])

      shape = [
          shape.with_major_to_minor_layout_if_absent()
          for x in non_empty_flat_results_aval
          for shape in xla.aval_to_xla_shapes(x)
      ]

      build_infeed = functools.partial(xops.InfeedWithToken,
                                        after_outfeed_itoken,
                                        xla_client.Shape.tuple_shape(shape))
      outs_and_token = _with_sharding_proto(comp, infeed_sharding_proto,
                                            build_infeed)
      outs = xops.GetTupleElement(outs_and_token, 0)
      next_itoken = xops.GetTupleElement(outs_and_token, 1)
      non_empty_results = [
          xops.GetTupleElement(outs, i)
          for i in range(len(non_empty_flat_results_aval))
      ]
      generated_infeed = True
      results = [
          empty_results.pop(0)
          if _aval_is_empty(result_aval) else non_empty_results.pop(0)
          for result_aval in flat_results_aval
      ]
    else:
      results = empty_results
      next_itoken = current_itoken

  assert generated_infeed == send_infeed, (
      f"generated_infeed ({generated_infeed}) != send_infeed ({send_infeed})")
  assert identity or len(results) == len(flat_results_aval), (
      f"got {len(results)} but expected {len(flat_results_aval)}. "
      f"identity = {identity}")
  return results + [next_token, next_itoken]


xla.register_translation(outside_call_p, _outside_call_translation_rule)


def _outside_call_lowering(ctx: mlir.LoweringRuleContext,
                           *args,
                           has_token: bool,
                           identity: bool,
                           device_index: int,
                           flat_results_aval=(),
                           **params):
  """MLIR Lowering for `CustomCall`-based HCB."""
  if len(ctx.module_context.platforms) > 1:
    raise NotImplementedError("multi-platform lowering for host_callback")
  platform = ctx.module_context.platforms[0]
  use_outfeed = _use_outfeed(platform)
  if use_outfeed:
    # Fall back to XLA path if we are using the outfeed
    # TODO(sharadmv): update to use MLIR for this path as well and delete
    #                 XLA lowering
    return mlir.xla_fallback_lowering(outside_call_p)(
        ctx,
        *args,
        has_token=has_token,
        identity=identity,
        flat_results_aval=flat_results_aval,
        device_index=device_index,
        **params)
  else:
    # TODO(necula): It seems that on CPU, with custom call, the device_index
    # does not work, and the callback is always run on device_index=0
    if (device_index != 0 and "cpu" in ctx.module_context.platforms):
      raise ValueError(
          "The device_index feature on CPU works only when using outfeed.")
  # We expect the current tokens at the end, inserted by _rewrite_jaxpr.
  assert has_token
  current_token = args[-2]
  current_itoken = args[-1]
  assert current_token.type == hlo.TokenType.get(), "The last two arguments must be tokens"
  assert current_itoken.type == hlo.TokenType.get(), "The last two arguments must be tokens"

  args_to_outfeed = args[:-2]
  # TODO(necula): this is a weak attempt to get the device. This works
  # inside pmap, but does not work when we just execute on a single device,
  # because in such executions we always get replica_id == 0.
  replica_id = hlo.ReplicaIdOp()
  callback_operands = [replica_id, *args_to_outfeed]
  callback_operand_avals = [
      core.ShapedArray((), np.uint32), *ctx.avals_in[:-2]]
  if identity:
    callback_flat_results_aval = []
  else:
    callback_flat_results_aval = [*flat_results_aval]

  def wrapped_callback(*args):
    replica_id, *arrays = args
    result_arrays = _outside_call_run_callback(
        arrays,
        xb.local_devices()[replica_id],
        send_infeed=False,
        # The same parameters as outside_call_p
        identity=identity,
        flat_results_aval=flat_results_aval,
        **params)
    if identity:
      # For identity, we do not pass the any results back to the device
      result_arrays = ()
    return result_arrays

  if isinstance(
      ctx.module_context.axis_context,
      (sharding_impls.SPMDAxisContext, sharding_impls.ShardingContext),
  ):
    # Apply maximal sharding so pjit only executes the callback on device device_index.
    sharding = xla_client.OpSharding()
    sharding.type = xla_client.OpSharding.Type.MAXIMAL
    sharding.tile_assignment_dimensions = [1]
    sharding.tile_assignment_devices = [device_index]
  else:
    sharding = None
  results, next_token, keep_alive = mlir.emit_python_callback(ctx,
      wrapped_callback, current_token, callback_operands,
      callback_operand_avals, callback_flat_results_aval,  # type: ignore[arg-type]
      has_side_effect=True, sharding=sharding)
  _callback_handler_data.keep_alives.append(keep_alive)
  # We must put the two tokens at the end
  if identity:
    results = list(args_to_outfeed)
  next_itoken = current_itoken

  assert identity or len(results) == len(flat_results_aval), (
      f"got {len(results)} but expected {len(flat_results_aval)}. "
      f"identity = {identity}")
  return list(results) + [next_token, next_itoken]

mlir.register_lowering(outside_call_p, _outside_call_lowering, platform="cpu")

def _outside_call_run_callback(
    arrays, device, *,
    send_infeed=True,
    # The same parameters as outside_call_p
    callback, arg_treedef,
    identity, result_treedef=None, flat_results_aval=None,
    transforms=(), has_token=False):
  """Performs the callback:
       callback(arg, device, transforms)

  Called during the device computation once we have the argument, either from
  an inlined callback or from an XLA computation outfeed.

  Returns the flat list of result arrays. If `send_infeed` then it will also send
  the flat list of results to the device.
  """

  def _unpack_transforms(transforms) -> tuple[tuple[str, dict[str, Any]], ...]:
    def _unpack_transform(name, *params):
      if name == "batch":
        return name, dict(batch_dims=params[0])
      elif name == "mask":
        return name, dict(logical_shapes=5)
      else:
        assert not params, f"{name}, {params}"
        return name, {}

    return tuple(_unpack_transform(*t) for t in transforms)

  try:
    arg = api.tree_unflatten(arg_treedef, arrays)
    unpacked_transforms = _unpack_transforms(transforms)
    logger.debug(
      "Outside call invoking call_func %s, device=%s, transforms=%s",
      callback, device, unpacked_transforms
    )
    res = callback(arg, device, unpacked_transforms)
    if identity:
      return tuple(arrays)

    else:  # Check the type of the callback results
      assert result_treedef is not None
      assert flat_results_aval is not None
      actual_flat_results, actual_result_treedef = tree_util.tree_flatten(res)
      if actual_result_treedef != result_treedef:
        msg = (f"Callback func {callback} should have returned a result "
               f"with pytree {result_treedef} but returned "
               f"{actual_result_treedef}")
        raise TypeError(msg)

      canonical_flat_results = tuple(util.safe_map(xla.canonicalize_dtype, actual_flat_results))
      actual_flat_results_aval = _values_to_avals(canonical_flat_results)
      logger.debug(
        "Outside call %s result %s. Sending to infeed for device %s.",
        callback, flat_results_aval, device,
        )

      if not all(ea.strip_weak_type() == ra.strip_weak_type()
                 for ea, ra in util.safe_zip(flat_results_aval,
                                             actual_flat_results_aval)):
        msg = (f"Callback func {callback} should have returned a result "
               "with abstract values "
               f"{result_treedef.unflatten(flat_results_aval)} "
               f"but returned {actual_result_treedef.unflatten(actual_flat_results_aval)}")
        raise TypeError(msg)

      if send_infeed:
        # Do not send the 0-sized arrays
        non_empty_canonical_flat_results = tuple(filter(lambda r: not _aval_is_empty(r),
                                                        canonical_flat_results))
        device.transfer_to_infeed(non_empty_canonical_flat_results)
      return canonical_flat_results

  except Exception as e:
    logger.error("Outside call %s threw exception %s.", callback, e)
    if send_infeed:
      # Prepare some results to send in case of error. We are sending something
      # with a distinctive shape (int8[12345]), one that is unlikely to be what the device
      # expects. This should have the effect to abort the device computation,
      # with an error message that we recognize. On TPU there seem to be no
      # such check, and if we send anything at all the device computation will
      # use some garbage data. So, on TPU we prefer to not send anything and let
      # the computation hang.
      # TODO: implement a proper error handling for TPU
      if device.platform != "tpu":
        canonical_flat_results = [xla.canonicalize_dtype(np.arange(12345, dtype=np.int8))]
        logger.debug("Outside call consumer %s exception %s. Sending to infeed the error result.",
                     callback, e)
        device.transfer_to_infeed(tuple(canonical_flat_results))
      else:
        logger.debug("Outside call consumer %s exception %s. On TPU we do not send infeed.",
                     callback, e)
    raise e  # Let the exception propagate


def _add_transform(params: dict, name: str, *transform_params) -> dict:
  """Adds the `transform` to the params["transforms"].

  Uses a tuple representation internally, will be unpacked before the
  callback by _ConsumerCallable.
  """
  new_transform = (name, *transform_params)
  return dict(
      params, transforms=(params.get("transforms", ()) + (new_transform,)))


def _aval_is_empty(aval) -> bool:
  return math.prod(aval.shape) == 0

def _instantiate_zeros(tan, arg):
  del arg
  return ad.instantiate_zeros(tan)

def _outside_call_jvp_rule(primals, tangents, **params):
  assert "has_token" not in params
  if not params["identity"]:
    raise NotImplementedError("JVP rule is implemented only for id_tap, not for call.")
  out_primals_tapped = outside_call_p.bind(*primals, **params)
  return tuple(out_primals_tapped), tangents


ad.primitive_jvps[outside_call_p] = _outside_call_jvp_rule

def _outside_call_transpose_rule(cts, *args, **params):
  if not params["identity"]:
    raise NotImplementedError("differentiation rules are implemented only for id_tap, not for call.")
  assert "has_token" not in params
  assert len(cts) == len(args)
  cts_instantiated = tuple(map(_instantiate_zeros, cts, args))

  # The args have been prepared by the id_tap_jvp_rule: tapped_primals, tapped_tangents, rest_primals, rest_tangents
  transforms = params.get("transforms", ())
  if not transforms or transforms[-1] != ("jvp",):
    # TODO: I should understand better when can this happen. It seems to arise
    # in scan.
    return outside_call_p.bind(
        *cts_instantiated,
        **_add_transform(params, "transpose"))

  assert False


ad.primitive_transposes[outside_call_p] = _outside_call_transpose_rule


def _outside_call_batching_rule(batched_args, batch_dims, **params):
  if not params["identity"]:
    raise NotImplementedError("batching rules are implemented only for id_tap, not for call.")
  assert "has_token" not in params
  new_params = _add_transform(params, "batch", batch_dims)
  res = outside_call_p.bind(*batched_args, **new_params)
  return res, batch_dims


batching.primitive_batchers[outside_call_p] = _outside_call_batching_rule

####
#### Jaxpr rewriting logic to thread the tokens through stateful primitives.
####


def _rewrite_closed_jaxpr(cjaxpr: core.ClosedJaxpr, has_input_token: bool,
                          has_output_token: bool) -> core.ClosedJaxpr:
  """Rewrites a ClosedJaxpr to thread the token, if needed."""
  new_jaxpr = _rewrite_jaxpr(cjaxpr.jaxpr, has_input_token, has_output_token)
  return core.ClosedJaxpr(new_jaxpr, cjaxpr.consts)


def _rewrite_jaxpr(jaxpr: core.Jaxpr, has_input_token: bool,
                   has_output_token: bool) -> core.Jaxpr:
  """Rewrite a Jaxpr to thread the token, if needed."""
  assert has_input_token or not has_output_token

  if not has_input_token and not core.jaxpr_uses_outfeed(jaxpr):
    return jaxpr

  mk_new_var = core.gensym()

  eqns: list[core.JaxprEqn] = []
  # store the incoming tokens
  last_token_var = mk_new_var(core.abstract_token)
  last_itoken_var = mk_new_var(core.abstract_token)
  if has_input_token:
    invars = jaxpr.invars + [last_token_var, last_itoken_var]
  else:
    invars = jaxpr.invars
    # We need tokens but none is given in input; make one depending on all invars
    eqns.append(
        core.new_jaxpr_eqn(jaxpr.invars, [last_token_var],
                           lax.create_token_p, {}, core.no_effects, source_info_util.current()))
    eqns.append(
        core.new_jaxpr_eqn(jaxpr.invars, [last_itoken_var],
                           lax.create_token_p, {}, core.no_effects, source_info_util.current()))

  for eqn in jaxpr.eqns:
    if not core.primitive_uses_outfeed(eqn.primitive, eqn.params):
      eqns.append(eqn)
    else:
      output_token_var = mk_new_var(last_token_var.aval)
      output_itoken_var = mk_new_var(last_itoken_var.aval)
      _rewrite_eqn(eqn, eqns, last_token_var, output_token_var,
                   last_itoken_var, output_itoken_var, mk_new_var)
      last_token_var = output_token_var
      last_itoken_var = output_itoken_var

  outvars = jaxpr.outvars + ([last_token_var, last_itoken_var] if has_output_token else [])
  new_jaxpr = core.Jaxpr(jaxpr.constvars, invars, outvars, eqns, jaxpr.effects)
  return new_jaxpr


def _rewrite_eqn(eqn: core.JaxprEqn, eqns: list[core.JaxprEqn],
                 input_token_var: core.Var, output_token_var: core.Var,
                 input_itoken_var: core.Var, output_itoken_var: core.Var,
                 mk_new_var: Callable[[core.AbstractValue], core.Var]):
  """Rewrite an `eqn` and append equations to `eqns`.

  This is only called if the current primitive uses outfeed.
  Assume that the current token is in `input_token_var` and the resulting
  token must end in `output_token_var`.

  Append the result of rewriting to `eqns`.
  """
  if eqn.primitive is outside_call_p:
    assert "has_token" not in eqn.params
    eqns.append(eqn.replace(invars=eqn.invars + [input_token_var, input_itoken_var],
                            outvars=eqn.outvars + [output_token_var, output_itoken_var],
                            params=dict(eqn.params, has_token=True)))
  elif eqn.primitive is lax.while_p:
    cond_jaxpr, _, body_jaxpr, _ = util.split_dict(
        eqn.params,
        ["cond_jaxpr", "cond_nconsts", "body_jaxpr", "body_nconsts"])
    if core.jaxpr_uses_outfeed(cond_jaxpr.jaxpr):
      _rewrite_while_outfeed_cond(eqn, eqns, input_token_var, output_token_var,
                                  input_itoken_var, output_itoken_var,
                                  mk_new_var)
      return

    eqns.append(
        eqn.replace(
            invars=eqn.invars + [input_token_var, input_itoken_var],
            outvars=eqn.outvars + [output_token_var, output_itoken_var],
            params=dict(
                eqn.params,
                body_jaxpr=_rewrite_closed_jaxpr(body_jaxpr, True, True),
                cond_jaxpr=_rewrite_closed_jaxpr(cond_jaxpr, True, False))))
  elif eqn.primitive is lax.cond_p:
    branches, = util.split_dict(eqn.params, ["branches"])
    index, *operands = eqn.invars
    new_invars = [index, *operands, input_token_var, input_itoken_var]
    eqns.append(
        eqn.replace(
            invars=new_invars, outvars=eqn.outvars + [output_token_var, output_itoken_var],
            params=dict(
                eqn.params,
                branches=tuple(
                    _rewrite_closed_jaxpr(jaxpr, True, True)
                    for jaxpr in branches))))
  elif eqn.primitive is lax.scan_p:
    num_consts, num_carry, carry_jaxpr, linear, _, _, _, _ = util.split_dict(
        eqn.params,
        ["num_consts", "num_carry", "jaxpr", "linear", "reverse", "length",
         "unroll", "_split_transpose"])
    # We add the tokens right at the end of carry
    nr_const_and_carry = num_consts + num_carry
    new_invars = eqn.invars[0:nr_const_and_carry] + [
        input_token_var, input_itoken_var] + eqn.invars[nr_const_and_carry:]
    new_jaxpr = _rewrite_closed_jaxpr(carry_jaxpr, True, True)
    # The rewrite has put the token at end, it has to be at end of carry
    new_jaxpr_invars = new_jaxpr.jaxpr.invars
    new_jaxpr_invars = (
        new_jaxpr_invars[0:nr_const_and_carry] + new_jaxpr_invars[-2:] +
        new_jaxpr_invars[nr_const_and_carry:-2])
    new_jaxpr = new_jaxpr.replace(jaxpr=new_jaxpr.jaxpr.replace(invars=new_jaxpr_invars))

    new_jaxpr_outvars = new_jaxpr.jaxpr.outvars
    new_jaxpr_outvars = (
        new_jaxpr_outvars[0:num_carry] + new_jaxpr_outvars[-2:] +
        new_jaxpr_outvars[num_carry:-2])
    new_jaxpr = new_jaxpr.replace(jaxpr=new_jaxpr.jaxpr.replace(outvars=new_jaxpr_outvars))
    eqns.append(
        eqn.replace(
            invars=new_invars,
            # Output token is at the end of carry result
            outvars=(eqn.outvars[0:num_carry] + [output_token_var, output_itoken_var] +
                     eqn.outvars[num_carry:]),
            params=dict(
                eqn.params,
                jaxpr=new_jaxpr,
                num_carry=num_carry + 2,
                linear=linear[0:nr_const_and_carry] + (False, False) + linear[nr_const_and_carry:])))
  elif eqn.primitive is pxla.xla_pmap_p:
    # We broadcast the input token into an array of tokens
    call_jaxpr = cast(core.Jaxpr, eqn.params["call_jaxpr"])
    eqns.append(
        eqn.replace(
            invars=eqn.invars + [input_token_var, input_itoken_var],
            outvars=eqn.outvars + [output_token_var, output_itoken_var],
            params=dict(
                eqn.params,
                call_jaxpr=_rewrite_jaxpr(call_jaxpr, True, True),
                donated_invars=eqn.params["donated_invars"] + (False, False),
                # Sharding/unsharding of tokens in pmap_translation are special
                # cased to just pass-through the token
                in_axes=eqn.params["in_axes"] + (None, None),
                out_axes=eqn.params["out_axes"] + (0, 0))))
  elif eqn.primitive is custom_derivatives.custom_jvp_call_p:
    fun_jaxpr = eqn.params["call_jaxpr"]

    def unreachable_thunk():
      assert False, "Should not be reached"
    unreachable_thunk.reset_stores = lambda: None

    eqns.append(
        eqn.replace(
            invars=eqn.invars + [input_token_var, input_itoken_var],
            outvars=eqn.outvars + [output_token_var, output_itoken_var],
            params=dict(
                eqn.params,
                call_jaxpr=_rewrite_closed_jaxpr(fun_jaxpr, True, True),
                jvp_jaxpr_thunk=unreachable_thunk
            )))
  elif eqn.primitive is custom_derivatives.custom_vjp_call_jaxpr_p:
    fun_jaxpr = eqn.params["fun_jaxpr"]
    new_invars = [*eqn.invars, input_token_var, input_itoken_var]

    def unreachable_thunk():
      assert False, "Should not be reached"

    eqns.append(
        eqn.replace(
            invars=new_invars,
            outvars=eqn.outvars + [output_token_var, output_itoken_var],
            params=dict(
                eqn.params,
                fun_jaxpr=_rewrite_closed_jaxpr(fun_jaxpr, True, True),
                fwd_jaxpr_thunk=unreachable_thunk,
                # The following are illegal values for the parameters, they
                # should not be needed because this rewrite is just before
                # compilation to XLA, which does not use those parameters.
                bwd="illegal param",
                out_trees="illegal param")))
  elif eqn.primitive is pjit.pjit_p:
    jaxpr = cast(core.ClosedJaxpr, eqn.params["jaxpr"])
    eqns.append(
        eqn.replace(
            invars=eqn.invars + [input_token_var, input_itoken_var],
            outvars=eqn.outvars + [output_token_var, output_itoken_var],
            params=dict(
                eqn.params,
                jaxpr=_rewrite_closed_jaxpr(jaxpr, True, True),
                donated_invars=eqn.params["donated_invars"] + (False, False),
                in_shardings=(
                    eqn.params["in_shardings"]
                    + (sharding_impls.UNSPECIFIED, sharding_impls.UNSPECIFIED)
                ),
                out_shardings=(
                    eqn.params["out_shardings"]
                    + (sharding_impls.UNSPECIFIED, sharding_impls.UNSPECIFIED)
                ),
                in_layouts=(eqn.params["in_layouts"] + (None, None)),
                out_layouts=(eqn.params["out_layouts"] + (None, None)),
            ),
        )
    )
  elif eqn.primitive is ad_checkpoint.remat_p:
    jaxpr_ = cast(core.Jaxpr, eqn.params["jaxpr"])
    eqns.append(
        eqn.replace(
            invars=eqn.invars + [input_token_var, input_itoken_var],
            outvars=eqn.outvars + [output_token_var, output_itoken_var],
            params=dict(
                eqn.params,
                jaxpr=_rewrite_jaxpr(jaxpr_, True, True),
            )))
  else:
    raise NotImplementedError(f"outfeed rewrite {eqn.primitive}")


def _rewrite_while_outfeed_cond(eqn: core.JaxprEqn, eqns: list[core.JaxprEqn],
                                input_token_var: core.Var,
                                output_token_var: core.Var,
                                input_itoken_var: core.Var,
                                output_itoken_var: core.Var,
                                mk_new_var: Callable):
  """Rewrite a while whose cond has outfeed"""
  cond_jaxpr, cond_nconsts, body_jaxpr, body_nconsts = util.split_dict(
      eqn.params, ["cond_jaxpr", "cond_nconsts", "body_jaxpr", "body_nconsts"])
  transformed_cond_jaxpr = _rewrite_closed_jaxpr(cond_jaxpr, True, True)
  carry_invars = eqn.invars[cond_nconsts + body_nconsts:]
  # pred1, token1, itoken1 = rewrite(COND)(cond_consts, carry_invars, input_token, input_itoken)
  pred1_and_token1 = [
      mk_new_var(ov.aval) for ov in transformed_cond_jaxpr.jaxpr.outvars
  ]
  eqns.append(
      core.new_jaxpr_eqn(
          eqn.invars[0:cond_nconsts] + carry_invars + [input_token_var, input_itoken_var],
          pred1_and_token1, core.call_p,
          dict(
              call_jaxpr=transformed_cond_jaxpr.jaxpr,
              name="cond_before"),
          transformed_cond_jaxpr.jaxpr.effects,
          eqn.source_info))
  # Make a new cond "lambda pred, carry, token, itoken: pred"
  new_cond_pred_invar = mk_new_var(cond_jaxpr.out_avals[0])
  new_cond_invars = (
      [new_cond_pred_invar] + [mk_new_var(cv.aval) for cv in carry_invars] +
      [mk_new_var(input_token_var.aval),
       mk_new_var(input_itoken_var.aval)])
  new_cond_jaxpr = core.ClosedJaxpr(
      core.Jaxpr([], new_cond_invars, [new_cond_pred_invar], [], set()), [])
  # Make a new body:
  #   "lambda cond_constvars, body_constvars, pred, carry, token, itoken:
  #        carry2, token2, itoken2 = rewrite(BODY)(body_constvars, carry, token, itoken)
  #        pred2, token3, itoken3 = rewrite(COND)(cond_constvars, carry2, token2, itoken2)
  #        (pred2, carry2, token3, itoken3)
  transformed_body_jaxpr = _rewrite_closed_jaxpr(body_jaxpr, True, True)
  new_body_invars_cond_constvars = [
      mk_new_var(v.aval) for v in eqn.invars[0:cond_nconsts]
  ]
  new_body_invars_body_constvars = [
      mk_new_var(v.aval)
      for v in eqn.invars[cond_nconsts:cond_nconsts + body_nconsts]
  ]
  new_body_invars_pred = mk_new_var(cond_jaxpr.out_avals[0])
  new_body_invars_carry = [mk_new_var(cv.aval) for cv in carry_invars]
  new_body_invars_token = mk_new_var(input_token_var.aval)
  new_body_invars_itoken = mk_new_var(input_itoken_var.aval)

  new_body_carry2 = [mk_new_var(cv.aval) for cv in carry_invars]
  new_body_token2 = mk_new_var(input_token_var.aval)
  new_body_itoken2 = mk_new_var(input_itoken_var.aval)
  new_body_pred2 = mk_new_var(cond_jaxpr.out_avals[0])
  new_body_token3 = mk_new_var(input_token_var.aval)
  new_body_itoken3 = mk_new_var(input_itoken_var.aval)

  new_body_eqns = [
      core.new_jaxpr_eqn(
          new_body_invars_body_constvars + new_body_invars_carry +
          [new_body_invars_token, new_body_invars_itoken],
          new_body_carry2 + [new_body_token2, new_body_itoken2],
          core.call_p,
          dict(
              call_jaxpr=transformed_body_jaxpr.jaxpr,
              name="body"),
          transformed_body_jaxpr.effects,
          eqn.source_info),
      core.new_jaxpr_eqn(
          new_body_invars_cond_constvars + new_body_carry2 + [new_body_token2, new_body_itoken2],
          [new_body_pred2, new_body_token3, new_body_itoken3], core.call_p,
          dict(
              call_jaxpr=transformed_cond_jaxpr.jaxpr,
              name="cond_body"),
          transformed_cond_jaxpr.effects,
          eqn.source_info)
  ]
  effects = core.join_effects(*(eqn.effects for eqn in new_body_eqns))
  new_body_jaxpr = core.ClosedJaxpr(
      core.Jaxpr([], (new_body_invars_cond_constvars +
                      new_body_invars_body_constvars + [new_body_invars_pred] +
                      new_body_invars_carry + [new_body_invars_token, new_body_invars_itoken]),
                 ([new_body_pred2] + new_body_carry2 + [new_body_token3, new_body_itoken3]),
                 new_body_eqns, effects), [])

  pred_out = mk_new_var(cond_jaxpr.out_avals[0])
  eqns.append(
      core.new_jaxpr_eqn(
          (eqn.invars[0:cond_nconsts + body_nconsts] + [pred1_and_token1[0]] +
           carry_invars + pred1_and_token1[1:]),
          ([pred_out] + eqn.outvars + [output_token_var, output_itoken_var]),
          lax.while_p,
          dict(
              cond_jaxpr=new_cond_jaxpr,
              cond_nconsts=0,
              body_jaxpr=new_body_jaxpr,
              body_nconsts=cond_nconsts + body_nconsts),
          new_body_jaxpr.effects,
          eqn.source_info))


# We need an identity primitive to simplify rewriting
id_p = core.Primitive("id")
id_p.multiple_results = True
id_p.def_impl(lambda *args: args)
id_p.def_abstract_eval(lambda *args: args)
xla.register_translation(id_p, lambda ctx, avals_in, avals_out, *args: args)

dispatch.outfeed_rewriter = lambda j: _rewrite_jaxpr(j, False, False)


class CallbackException(Exception):
  """Signals that some callback function had exceptions.

  Raised by :func:`barrier_wait`. See the :mod:`jax.experimental.host_callback`
  module documentation for details.
  """
  pass

TapFunctionException = CallbackException  # For backwards compatibility

class _CallbackHandlerData:
  """Keep track of the outfeed receiver data."""
  receiver: Any
  initialized: bool
  on_exit: bool
  lock: threading.Lock
  last_callback_exception: tuple[Exception, str] | None
  clients: tuple[XlaLocalClient, ...]
  devices: tuple[XlaDevice, ...]
  consumer_registry: dict[Callable, int]
  consumer_registry_by_id: dict[int, Callable]

  def __init__(self):
    self.receiver = None  # Initialize lazily, when first needed
    self.initialized = False
    self.on_exit = False
    self.lock = threading.Lock()
    self.last_callback_exception = None
    self.clients = ()
    self.devices = ()
    # The consumer registries must be live for the lifetime of the program,
    # because we may have cached compilations that embed consumer ids, and we
    # do not want the id reused for other shapes.
    # Used only for the outfeed mechanism.
    self.callback_registry = {}
    self.callback_registry_by_id = {}
    # For now we keep here the keep_alives for the emit_python_callback. This is
    # a leak. We ought to attach these to the executable.
    self.keep_alives = []

  def stop(self):
    """Wait for all pending outfeeds and stop the receiver."""
    self.receiver = None  # GC will trigger the destructor
    self.initialized = False
    self.clients = ()
    self.devices = ()
    # Do not clear the consumer registries.


_callback_handler_data = _CallbackHandlerData()


# This function is called from C++; it must not allow exceptions through.
def _callback_input_received(device, consumer_id, arrays: tuple):
  array_repr = ", ".join([f"({a.dtype}{a.shape})" for a in arrays])
  logger.debug("Callback input received on device %s for consumer %s arrays: %s",
    device, consumer_id, array_repr)
  callback = _callback_handler_data.callback_registry_by_id.get(consumer_id)
  assert callback is not None, "We should have crashed in the runtime"
  try:
    return callback(arrays, device)
  except Exception as e:
    formatted_e = traceback.format_exc()
    logger.error("Postponing exception raised in callback function: %s", formatted_e)
    _callback_handler_data.last_callback_exception = (e, formatted_e)


def _register_callback(callback: Callable) -> int:
  """Registers a callback function, cache by hash of callback.

  The callback is a function to be invoked as `callback(arrays, device)`.
  """
  callback_id = _callback_handler_data.callback_registry.get(callback)
  if callback_id is not None:
    return callback_id
  callback_id = hash(callback) & 0xFFFFFFFC  # pybind11 has trouble here with large ints
  callback_id += 1  # Reserve the consumer ID 0
  assert callback_id not in _callback_handler_data.callback_registry, (
      "callback id collision")
  _callback_handler_data.callback_registry[callback] = callback_id
  _callback_handler_data.callback_registry_by_id[callback_id] = callback
  return callback_id


def _initialize_outfeed_receiver(
    max_callback_queue_size_bytes: int = int(256 * 1e6)):
  """Creates and starts the outfeed_receiver.

  This function is called lazily only when we compile an id_tap.

  Args:
    * clients: the list of clients (backends) on whose devices to listen on.
    * max_callback_queue_size_bytes: an optional integer to bound the maximum
      size of arrays in the callback queue. When this limit is reached the
      device listener pauses.
  """
  outfeed_receiver_module = xla_extension.outfeed_receiver

  with _callback_handler_data.lock:
    if _callback_handler_data.initialized:
      return

    # By default, all devices on all supported backends.
    clients = [backend for name, backend in xb.backends().items()
               if name in ("cpu", "cuda", "rocm", "tpu")]
    devices = list(
        itertools.chain(*[backend.local_devices() for backend in clients]))
    _callback_handler_data.clients = clients  # type: ignore[assignment]
    _callback_handler_data.devices = devices  # type: ignore[assignment]
    clients_with_outfeed = [c for c in clients if _use_outfeed(c.platform)]
    for client in clients_with_outfeed:
      _raise_if_using_outfeed_with_pjrt_c_api(client)
    if clients_with_outfeed:
      devices_with_outfeed = list(
        itertools.chain(*[backend.local_devices() for backend in clients_with_outfeed]))
      if logger.isEnabledFor(logging.DEBUG):
        device_repr = ", ".join([str(d) for d in devices_with_outfeed])
        logger.debug("Starting outfeed_receiver for %s. max_callback_queue_size_bytes=%s",
                       device_repr, max_callback_queue_size_bytes)
      _callback_handler_data.receiver = outfeed_receiver_module.start(
          _callback_input_received, tuple(clients_with_outfeed),
          max_callback_queue_size_bytes,
          compiler.get_compile_options(1, 1).executable_build_options)

    def exit_handler():
      # Prevent logging usage during compilation, gives errors under pytest
      dispatch._on_exit = True
      if not _callback_handler_data.on_exit:
        _callback_handler_data.on_exit = True
        _deprecated_barrier_wait("at_exit")

    atexit.register(exit_handler)  # We wait as long as we have callbacks
    _callback_handler_data.initialized = True


def _deprecated_barrier_wait(logging_name: str | None = None):
  """Blocks the calling thread until all current outfeed is processed.

  Waits until all callbacks from computations already running on all devices
  have been received and processed by the Python callbacks. Raises
  CallbackException if there were exceptions while processing the callbacks.

  This works by enqueueing a special tap computation to all devices to which
  we are listening for outfeed. Once all those tap computations are done, we
  return from barrier_wait.

  Note: If any of the devices are busy and cannot accept new computations,
  this will deadlock.

  Args:
    logging_name: an optional string that will be used in the logging statements
      for this invocation. See `Debugging` in the module documentation.

  For more details see the :mod:`jax.experimental.host_callback` module documentation.
  """
  if not _HOST_CALLBACK_LEGACY.value:
    jax.effects_barrier()
    return

  logging_name = logging_name or ""
  logger.debug("barrier_wait[%s]: start", logging_name)

  lock = threading.Lock()
  cv = threading.Condition(lock=lock)
  devices_at_barrier = []  # Protected by lock
  def barrier_tap_received(dev_idx, _):
    device = _callback_handler_data.devices[dev_idx]
    logger.debug(
      "barrier_wait[%s]: at barrier_tap for device %s. Thread %s",
      logging_name, device, threading.current_thread()
    )
    with lock:
      devices_at_barrier.append(device)
      if logger.isEnabledFor(logging.DEBUG):
        waiting_for_devices = [d for d in _callback_handler_data.devices
                               if d not in devices_at_barrier]
        logger.debug(
          "barrier_wait[%s]: still waiting for %s devices at barrier (%s)",
          logging_name, len(waiting_for_devices), waiting_for_devices
        )
      cv.notify()

  for d_idx, d in enumerate(_callback_handler_data.devices):
    logger.debug("barrier_wait[%s]: enqueueing barrier on device %s", logging_name, d)
    x_on_dev = api.device_put(d_idx, device=d)
    api.jit(lambda x: _deprecated_id_tap(barrier_tap_received, x), device=d)(x_on_dev)

  logger.debug("barrier_wait[%s]: waiting for callbacks", logging_name)

  with lock:
    cv.wait_for(lambda: len(devices_at_barrier) == len(_callback_handler_data.devices))

  logger.debug("barrier_wait[%s]: done", logging_name)

  if _callback_handler_data.last_callback_exception is not None:
    last_exception, formatted_last_exception = _callback_handler_data.last_callback_exception
    _callback_handler_data.last_callback_exception = None
    raise CallbackException(
        "There were exceptions during callback processing. "
        f"Last one was: {formatted_last_exception}") from last_exception


def _deprecated_stop_outfeed_receiver():
  """Stops the outfeed receiver runtime.

  .. warning::
    The host_callback APIs are deprecated as of March 20, 2024.
    The functionality is subsumed by the
    `new JAX external callbacks <https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html>`_

  This waits for all outfeeds from computations already running on all devices,
  and then stops the outfeed receiver runtime. The runtime will be restarted
  next time you use a tap function.

  It should not be necessary to use this function, unless you want to start
  using lax.outfeed directly after having used host callbacks.
  """
  _callback_handler_data.stop()

_deprecation_msg = (
    "The host_callback APIs are deprecated as of March 20, 2024. The functionality "
    "is subsumed by the new JAX external callbacks. "
    "See https://github.com/google/jax/issues/20385.")

_deprecations = {
    # Added March 20, 2024
    "id_tap": (_deprecation_msg, _deprecated_id_tap),
    "id_print": (_deprecation_msg, _deprecated_id_print),
    "call": (_deprecation_msg, _deprecated_call),
    "barrier_wait": (_deprecation_msg, _deprecated_barrier_wait),
    "stop_outfeed_receiver": (_deprecation_msg, _deprecated_stop_outfeed_receiver),
}

import typing
if typing.TYPE_CHECKING:
  id_tap = _deprecated_id_tap
  id_print = _deprecated_id_print
  call = _deprecated_call
  barrier_wait = _deprecated_barrier_wait
  stop_outfeed_receiver = _deprecated_stop_outfeed_receiver
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
