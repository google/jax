JAX: High-Performance Array Computing
=====================================

JAX is a Python library for accelerator-oriented array computation and program transformation,
designed for high-performance numerical computing and large-scale machine learning.

If you're looking to train neural networks, use Flax_ and start with its documentation.
Some associated tools are Optax_ and Orbax_.
For an end-to-end transformer library built on JAX, see MaxText_.

.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Familiar API
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      JAX provides a familiar NumPy-style API for ease of adoption by researchers and engineers.

   .. grid-item-card:: Transformations
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      JAX includes composable function transformations for compilation, batching, automatic differentiation, and parallelization.

   .. grid-item-card:: Run Anywhere
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      The same code executes on multiple backends, including CPU, GPU, & TPU

.. grid:: 3

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Getting Started
      :columns: 12 6 6 4
      :link: beginner-guide
      :link-type: ref
      :class-card: getting-started

    .. grid-item-card:: :material-regular:`library_books;2em` User Guides
      :columns: 12 6 6 4
      :link: user-guides
      :link-type: ref
      :class-card: user-guides

    .. grid-item-card:: :material-regular:`laptop_chromebook;2em` Developer Docs
      :columns: 12 6 6 4
      :link: contributor-guide
      :link-type: ref
      :class-card: developer-docs


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart
   notebooks/Common_Gotchas_in_JAX
   faq

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Further Resources

   user_guides
   advanced_guide
   contributor_guide
   building_on_jax
   notes
   jax


.. toctree::
   :hidden:
   :maxdepth: 1

   changelog
   glossary


.. _Flax: https://flax.readthedocs.io/
.. _Orbax: https://orbax.readthedocs.io/
.. _Optax: https://optax.readthedocs.io/
.. _MaxText: https://github.com/google/maxtext/
