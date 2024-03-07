Rank promotion warning
======================

`NumPy broadcasting rules
<https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules>`_
allow the automatic promotion of arguments from one rank (number of array axes)
to another. This behavior can be convenient when intended but can also lead to
surprising bugs where a silent rank promotion masks an underlying shape error.

Here's an example of rank promotion:

>>> import numpy as np
>>> x = np.arange(12).reshape(4, 3)
>>> y = np.array([0, 1, 0])
>>> x + y
array([[ 0,  2,  2],
       [ 3,  5,  5],
       [ 6,  8,  8],
       [ 9, 11, 11]])

To avoid potential surprises, :code:`jax.numpy` is configurable so that
expressions requiring rank promotion can lead to a warning, error, or can be
allowed just like regular NumPy. The configuration option is named
:code:`jax_numpy_rank_promotion` and it can take on string values
:code:`allow`, :code:`warn`, and :code:`raise`. The default setting is
:code:`allow`, which allows rank promotion without warning or error.
The :code:`raise` setting raises an error on rank promotion, and :code:`warn`
raises a warning on the first occurrence of rank promotion.

Rank promotion can be enabled or disabled locally with the :func:`jax.numpy_rank_promotion`
context manager:

.. code-block:: python

   with jax.numpy_rank_promotion("warn"):
     z = x + y

This configuration can also be set globally in several ways.
One is by using :code:`jax.config` in your code:

.. code-block:: python

  import jax
  jax.config.update("jax_numpy_rank_promotion", "warn")

You can also set the option using the environment variable
:code:`JAX_NUMPY_RANK_PROMOTION`, for example as
:code:`JAX_NUMPY_RANK_PROMOTION='warn'`. Finally, when using :code:`absl-py`
the option can be set with a command-line flag.
