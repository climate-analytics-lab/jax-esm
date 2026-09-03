JAX-ESM documentation
=====================

JAX-ESM (imported as ``jem``) is a JAX-based, differentiable coupling framework for Earth
system components. It couples independent atmosphere, ocean, land, and sea-ice models —
such as JCM, Veros, and JEM's own slab models — into a single JIT-compilable simulation loop
built on ``jax.lax.scan``.

- New to JEM? Start with :doc:`quick_start` for a copy-and-paste aquaplanet example.
- Want to see it in action first? Browse :doc:`examples`.
- Integrating your own model? Follow :doc:`tutorial`.
- Looking for a specific class or function? See :doc:`api_superset`.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick_start
   examples
   tutorial
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
