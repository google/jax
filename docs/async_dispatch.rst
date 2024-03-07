.. _async-dispatch:

Asynchronous dispatch
=====================

JAX uses asynchronous dispatch to hide Python overheads. Consider the following
program:

>>> import numpy as np
>>> import jax.numpy as jnp
>>> from jax import random
>>> x = random.uniform(random.key(0), (1000, 1000))
>>> # Printing the result (i.e. evaluating `repr(result)` or `str(result)`)
>>> # will block until the value is ready.
>>> jnp.dot(x, x) + 3.  # doctest: +SKIP
Array([[258.01971436, 249.64862061, 257.13372803, ...,
        236.67948914, 250.68939209, 241.36853027],
        [265.65979004, 256.28912354, 262.18252563, ...,
        242.03181458, 256.16757202, 252.44122314],
        [262.38916016, 255.72747803, 261.23059082, ...,
        240.83563232, 255.41094971, 249.62471008],
        ...,
        [259.15814209, 253.09197998, 257.72174072, ...,
        242.23876953, 250.72680664, 247.16642761],
        [271.22662354, 261.91204834, 265.33398438, ...,
        248.26651001, 262.05389404, 261.33700562],
        [257.16134644, 254.7543335, 259.08300781, ..., 241.59848022,
        248.62597656, 243.22348022]], dtype=float32)

When an operation such as :code:`jnp.dot(x, x)` is executed, JAX does not wait
for the operation to complete before returning control to the Python program.
Instead, JAX returns a :class:`jax.Array` value, which is a future,
i.e., a value that will be produced in the future on an accelerator device but
isn't necessarily available immediately. We can inspect the shape or type of a
:class:`jax.Array` without waiting for the computation that produced it to
complete, and we can even pass it to another JAX computation, as we do with the
addition operation here. Only if we actually inspect the value of the array from
the host, for example by printing it or by converting it into a plain old
:class:`numpy.ndarray` will JAX force the Python code to wait for the
computation to complete.

Asynchronous dispatch is useful since it allows Python code to "run ahead" of
an accelerator device, keeping Python code out of the critical path.
Provided the Python code enqueues work on the device faster than it can be
executed, and provided that the Python code does not actually need to inspect
the output of a computation on the host, then a Python program can enqueue
arbitrary amounts of work and avoid having the accelerator wait.

Asynchronous dispatch has a slightly surprising consequence for microbenchmarks.

>>> %time jnp.dot(x, x)  # doctest: +SKIP
CPU times: user 267 µs, sys: 93 µs, total: 360 µs
Wall time: 269 µs 
Array([[255.01972961, 246.64862061, 254.13371277, ...,
        233.67948914, 247.68939209, 238.36853027],
        [262.65979004, 253.28910828, 259.18252563, ...,
        239.03181458, 253.16757202, 249.44122314],
        [259.38916016, 252.72747803, 258.23059082, ...,
        237.83563232, 252.41094971, 246.62471008],
        ...,
        [256.15814209, 250.09197998, 254.72172546, ...,
        239.23876953, 247.72680664, 244.16642761],
        [268.22662354, 258.91204834, 262.33398438, ...,
        245.26651001, 259.05389404, 258.33700562],
        [254.16134644, 251.7543335, 256.08300781, ..., 238.59848022,
        245.62597656, 240.22348022]], dtype=float32)

269µs is a surprisingly small time for a 1000x1000 matrix multiplication on CPU!
However it turns out that asynchronous dispatch is misleading us and we are not
timing the execution of the matrix multiplication, only the time to dispatch
the work. To measure the true cost of the operation we must either read the
value on the host (e.g., convert it to a plain old host-side numpy array), or
use the :meth:`~jax.Array.block_until_ready` method on a
:class:`jax.Array` value to wait for the computation that produced it to
complete.

>>> %time np.asarray(jnp.dot(x, x))  # doctest: +SKIP
CPU times: user 61.1 ms, sys: 0 ns, total: 61.1 ms
Wall time: 8.09 ms
Out[16]: 
array([[255.01973, 246.64862, 254.13371, ..., 233.67949, 247.68939,
        238.36853],
       [262.6598 , 253.28911, 259.18253, ..., 239.03181, 253.16757,
        249.44122],
       [259.38916, 252.72748, 258.2306 , ..., 237.83563, 252.41095,
        246.62471],
       ...,
       [256.15814, 250.09198, 254.72173, ..., 239.23877, 247.7268 ,
        244.16643],
       [268.22662, 258.91205, 262.33398, ..., 245.26651, 259.0539 ,
        258.337  ],
       [254.16135, 251.75433, 256.083  , ..., 238.59848, 245.62598,
        240.22348]], dtype=float32)
>>> %time jnp.dot(x, x).block_until_ready()  # doctest: +SKIP
CPU times: user 50.3 ms, sys: 928 µs, total: 51.2 ms
Wall time: 4.92 ms
Array([[255.01972961, 246.64862061, 254.13371277, ...,
        233.67948914, 247.68939209, 238.36853027],
        [262.65979004, 253.28910828, 259.18252563, ...,
        239.03181458, 253.16757202, 249.44122314],
        [259.38916016, 252.72747803, 258.23059082, ...,
        237.83563232, 252.41094971, 246.62471008],
        ...,
        [256.15814209, 250.09197998, 254.72172546, ...,
        239.23876953, 247.72680664, 244.16642761],
        [268.22662354, 258.91204834, 262.33398438, ...,
        245.26651001, 259.05389404, 258.33700562],
        [254.16134644, 251.7543335, 256.08300781, ..., 238.59848022,
        245.62597656, 240.22348022]], dtype=float32)

Blocking without transferring the result back to Python is usually faster, and
is often the best choice when writing microbenchmarks of computation times.
