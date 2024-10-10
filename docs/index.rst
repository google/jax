JAX: High performance array computing
=====================================

JAX is a Python library for accelerator-oriented array computation and program transformation,
designed for high-performance numerical computing and large-scale machine learning.

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

   .. grid-item-card:: Run anywhere
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      The same code executes on multiple backends, including CPU, GPU, & TPU

.. grid:: 3

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Getting started
      :columns: 12 6 6 4
      :link: beginner-guide
      :link-type: ref
      :class-card: getting-started

    .. grid-item-card:: :material-regular:`library_books;2em` User guides
      :columns: 12 6 6 4
      :link: user-guides
      :link-type: ref
      :class-card: user-guides

    .. grid-item-card:: :material-regular:`laptop_chromebook;2em` Developer notes
      :columns: 12 6 6 4
      :link: contributor-guide
      :link-type: ref
      :class-card: developer-docs

If you're looking to train neural networks, use Flax_ and start with its tutorials.
For an end-to-end transformer library built on JAX, see MaxText_.

Ecosystem
---------
JAX itself is narrowly-scoped and focuses on efficient array operations & program
transformations. Built around JAX is an evolving ecosystem of machine learning and
numerical computing tools; the following is just a small sample of what is out there:

.. grid:: 4
    :class-container: ecosystem-grid

    .. grid-item:: :material-outlined:`hub;2em` **Neural networks, federated learning**

       - Flax_
       - Equinox_
       - Keras_
       - FedJAX_

    .. grid-item:: :material-regular:`show_chart;2em` **Optimizers, loss functions, & solvers**

       - Optax_
       - Optimistix_
       - Lineax_
       - Diffrax_

    .. grid-item:: :material-outlined:`storage;2em` **Data loading**

       - Grain_
       - `TensorFlow Datasets`_
       - `Hugging Face Datasets`_

    .. grid-item:: :material-regular:`construction;2em` **Checkpointing and other utilities**

       - Orbax_
       - Chex_
       - PIX_

    .. grid-item:: :material-regular:`lan;2em` **Probabilistic programming**

       - Oryx_
       - Blackjax_
       - Numpyro_
       - PyMC_

    .. grid-item:: :material-regular:`bar_chart;2em` **Probabilistic modeling**

       - `TensorFlow Probabilty`_
       - Distrax_

    .. grid-item:: :material-outlined:`animation;2em` **Differentiable physics & simulation**

       - `JAX MD`_
       - Brax_

    .. grid-item:: :material-regular:`language;2em` **LLMs & large scale deep learning**

       - MaxText_
       - AXLearn_
       - Levanter_
       - EasyLM_


Many more JAX-based libraries have been developed; the community-run `Awesome JAX`_ page
maintains an up-to-date list.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials

   notebooks/Common_Gotchas_in_JAX

   faq

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: More guides/resources

   user_guides
   advanced_guide
   contributor_guide
   extensions
   notes
   jax


.. toctree::
   :hidden:
   :maxdepth: 1

   changelog
   glossary


.. _Awesome JAX: https://github.com/n2cholas/awesome-jax
.. _AXLearn: https://github.com/apple/axlearn
.. _Blackjax: https://blackjax-devs.github.io/blackjax/
.. _Brax: https://github.com/google/brax/
.. _Chex: https://chex.readthedocs.io/
.. _Diffrax: https://docs.kidger.site/diffrax/
.. _Distrax: https://github.com/google-deepmind/distrax
.. _EasyLM: https://github.com/young-geng/EasyLM
.. _Equinox: https://docs.kidger.site/equinox/
.. _FedJAX: https://github.com/google/fedjax
.. _Flax: https://flax.readthedocs.io/
.. _Grain: https://github.com/google/grain
.. _Hugging Face Datasets: https://huggingface.co/docs/datasets/
.. _JAX MD: https://jax-md.readthedocs.io/
.. _Keras: https://keras.io/
.. _Levanter: https://github.com/stanford-crfm/levanter
.. _Lineax: https://github.com/patrick-kidger/lineax
.. _MaxText: https://github.com/google/maxtext/
.. _Numpyro: https://num.pyro.ai/en/latest/index.html
.. _Optax: https://optax.readthedocs.io/
.. _Optimistix: https://github.com/patrick-kidger/optimistix
.. _Orbax: https://orbax.readthedocs.io/
.. _Oryx: https://github.com/jax-ml/oryx
.. _PyMC: https://www.pymc.io/
.. _PIX: https://github.com/deepmind/dm_pix
.. _TensorFlow Datasets: https://www.tensorflow.org/datasets
.. _TensorFlow Probabilty: https://www.tensorflow.org/probability
