:orphan:

.. _beginner-guide:

Getting Started with JAX
========================
Welcome to JAX! The JAX documentation contains a number of useful resources for getting started.
:doc:`quickstart` is the easiest place to jump-in and get an overview of the JAX project.

If you're accustomed to writing NumPy code and are starting to explore JAX, you might find the following resources helpful:

- :doc:`notebooks/thinking_in_jax` is a conceptual walkthrough of JAX's execution model.
- :doc:`notebooks/Common_Gotchas_in_JAX` lists some of JAX's sharp corners.
- :doc:`faq` answers some frequent jax questions.

Tutorials
---------
If you're ready to explore JAX more deeply, the JAX tutorials go into much more detail:

.. toctree::
   :maxdepth: 2

   tutorials

If you prefer a video introduction here is one from JAX contributor Jake VanderPlas:

.. raw:: html

	<iframe width="640" height="360" src="https://www.youtube.com/embed/WdTeDXsOSj4"
	 title="Intro to JAX: Accelerating Machine Learning research"
	frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
	allowfullscreen></iframe>

Building on JAX
---------------
JAX provides the core numerical computing primitives for a number of tools developed by the
larger community. For example, if you're interested in using JAX for training neural networks,
two well-supported options are Flax_ and Haiku_.

For a community-curated list of JAX-related projects across a wide set of domains,
check out `Awesome JAX`_.

Finding Help
------------
If you have questions about JAX, we'd love to answer them! Two good places to get your
questions answered are:

- `JAX GitHub discussions`_
- `JAX on StackOverflow`_

.. _Awesome JAX: https://github.com/n2cholas/awesome-jax
.. _Flax: https://flax.readthedocs.io/
.. _Haiku: https://dm-haiku.readthedocs.io/
.. _JAX on StackOverflow: https://stackoverflow.com/questions/tagged/jax
.. _JAX GitHub discussions: https://github.com/google/jax/discussions