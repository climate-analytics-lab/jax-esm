Experimental Setups
===================

The simulations presented here are not tested extensively and serve as
prototypes for future development.


JCM with Slabs
--------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   examples/02_experimental/01_earth


JCM with Veros
--------------

Two configurations couple the SPEEDY T31L8 atmosphere to the Veros ocean
GCM, each runnable as one command with no generated inputs and no
``PYTHONPATH``:

``python -m jem.main +configuration=veros-double-drake``
    An idealised two-continent ("double-drake") ocean on a uniform lat-lon
    grid, sharing the atmosphere's longitudes and its shape. Its latitude
    axis is uniform where the atmosphere's is Gaussian (up to ~0.9 degrees
    apart near the poles), and the exchange between them is un-regridded --
    see jax-esm#121.
``python -m jem.main +configuration=veros-earth``
    A realistic, rotated-pole global ocean read from a SCRIP grid, exchanged
    with the atmosphere through ESMF regridding weights.

Both compose ``ocean=veros`` with a case setup importable from
:mod:`jem.components.veros.setups`
(:mod:`~jem.components.veros.setups.double_drake` /
:mod:`~jem.components.veros.setups.earth`), and neither has a land or
sea-ice component -- see each configuration's own comment for what that
means for the atmosphere's land boundary conditions.

The two components are coupled through :class:`jem.fluxes.VerosExchange`,
named as ``coupling.exchanger``: it computes the surface wind stress from the
atmosphere's near-surface wind with a bulk drag law (rotated into the ocean
grid's local frame for ``veros-earth``, whose grid is not true east/north),
applies a "swamp" sea-ice mask to the heat and freshwater fluxes once the
surface reaches the freezing point, and carries the rest of
:data:`jem.exchangers.VEROS_OCEAN_EXCHANGES` (the fluxes onto the ocean, its
sea surface temperature back) -- so the ocean is mechanically as well as
thermodynamically forced.

A Veros configuration also runs the **atmosphere** in double precision:
importing Veros sets ``jax_enable_x64`` process-wide, and which of the
atmosphere's own fields stay float32 depends on build order (whatever
jax-gcm allocated before Veros was imported); :class:`~jem.fluxes.
VerosExchange` reads each destination field's dtype at trace time rather
than assuming one, so the coupling itself is robust to that.

Veros is an optional dependency -- the jittable fork this project is built
against, cloned and ``pip install -e``d as shown in the main
``README.md``'s install steps -- and is required for both configurations.
