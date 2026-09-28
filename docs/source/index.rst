JAX-ESM documentation
=====================

JAX-ESM (imported as ``jem``) is a JAX-based, differentiable coupling framework for Earth
system components. It couples independent atmosphere, ocean, land, and sea-ice models —
such as JCM, Veros, and JEM's own slab models — into a single JIT-compilable simulation loop
built on ``jax.lax.scan``.

- New to JEM? Start with :doc:`getting_started` for install and your first
  coupled run.
- Building a coupled model in Python? See :doc:`python_api` for the complete
  construction.
- Integrating your own model? Follow :doc:`adding_a_component`.
- Looking for a specific class or function? See :doc:`api_superset`.

JEM is Alpha software: its API is subject to change without deprecation
until 1.0. The version shown throughout these docs is read from
``jem.__version__``, the single source of truth pyproject.toml's own
version metadata reads from in turn.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   python_api
   examples
   adding_a_component
   experimental

   issues
   design
   developers
   api_superset

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
