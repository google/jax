``jax.stages`` module
=====================

.. automodule:: jax.stages

Classes
-------

.. currentmodule:: jax.stages

.. autoclass:: Wrapped
   :members: trace, lower
   :special-members: __call__

.. autoclass:: Lowered
   :members: in_tree, out_tree, compile, as_text, compiler_ir, cost_analysis

.. autoclass:: Compiled
   :members: in_tree, out_tree, as_text, cost_analysis, memory_analysis, runtime_executable
   :special-members: __call__
