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
    An idealised two-continent ("double-drake") planet. The ocean's cell
    centres are the atmosphere's own -- the uniform longitude axis and the
    Gaussian latitude axis of the T31 grid, read from its SCRIP file
    (``ocean.atmosphere_grid_file``) -- because the two are exchanged
    index-to-index with no regridding. The continents are the packaged
    ``terrain_double_drake_T31.nc`` land-sea mask, which must have the
    atmosphere grid's shape. The atmosphere's land boundary conditions are
    jax-gcm's defaults: bare-land albedo 0, no snow cover, no soil water and a
    flat 288.15 K skin-temperature climatology, so SPEEDY's slab land absorbs
    all the sunlight it receives. Earth's climatology is not used, because it
    describes Earth's continents, not these.
``python -m jem.main +configuration=veros-earth``
    A rotated-pole global ocean read from a SCRIP grid, exchanged with the
    atmosphere through ESMF regridding weights. The atmosphere runs on
    jax-gcm's packaged T30 observed climatology (the file ``earth-slab``
    uses): bare-land albedo, snow cover, soil water and land skin
    temperature. Its sea surface temperature comes from the ocean, not from
    that file, which also holds the observed sea-ice cover the atmosphere
    sees.

Both compose ``ocean=veros`` with a case setup importable from
:mod:`jem.components.veros.setups`
(:mod:`~jem.components.veros.setups.double_drake` /
:mod:`~jem.components.veros.setups.earth`), and neither has a land or
sea-ice component: the land is SPEEDY's own slab land inside the
atmosphere. Each configuration's own comment says why every setting is what
it is.

The ocean is idealised in both, and a short run should be read with that in
mind:

* **No bathymetry.** The land-sea mask is the only geometry read (for
  ``veros-earth`` the packaged ERA5 land fraction, thresholded at 0.5), so
  every ocean column has the same flat 5200 m bottom.
* **A uniform initial state.** Salinity is 35 and the sea surface temperature
  is 288.03 K in every ocean column, poles included. Temperature falls
  linearly with depth to zero at 4510 m, the top of the deepest layer, so the
  deepest layer starts at -1.15 degC. The meridional structure comes from the
  atmosphere's fluxes during the run.
* **No sea ice.** There is no sea-ice component (the Winton ice cannot be
  composed with Veros, jax-esm#160). The only ice effect on the ocean is the
  "swamp" mask on the heat and freshwater fluxes once the surface reaches the
  freezing point.

The two components are coupled through :class:`jem.fluxes.VerosExchange`,
named as ``coupling.exchanger``: it hands the ocean the surface wind stress
the atmosphere itself received -- jax-gcm's published, stability-corrected
stress, the same for every physics package, so momentum is conserved across
the interface -- regridded conservatively and rotated into the ocean grid's
local frame for ``veros-earth``, whose grid is not true east/north;
applies a "swamp" sea-ice mask to the heat and freshwater fluxes once the
surface reaches the freezing point, and carries the rest of
:data:`jem.exchangers.VEROS_OCEAN_EXCHANGES` (the fluxes onto the ocean, its
sea surface temperature back) -- so the ocean is mechanically as well as
thermodynamically forced.

``veros-earth``'s ocean output is written on index dimensions ``x``/``y``
with 2-D ``lon``/``lat`` coordinates giving each cell's true position, like
any curvilinear grid (see :doc:`design/output`): the grid's own axes are the
rotated ones, kept in the file as ``native_lon``/``native_lat``, and ``u``,
``v`` and the wind stress are components along those axes rather than east
and north.

Veros applies its ``device`` runtime setting to the whole process
(``jax_platform_name``), and its default is ``"cpu"``.
:func:`jem.components.veros_component.configure_veros_runtime` therefore sets
it to the platform JAX would use anyway (``VEROS_DEVICE`` still chooses
explicitly), so a Veros configuration runs on the GPU of a GPU host --
atmosphere included -- rather than silently on its CPU.

A Veros configuration also runs the **atmosphere** in double precision:
importing Veros sets ``jax_enable_x64`` process-wide, and which of the
atmosphere's own fields stay float32 depends on build order (whatever
jax-gcm allocated before Veros was imported); :class:`~jem.fluxes.VerosExchange`
reads each destination field's dtype at trace time rather than assuming one,
so the coupling itself is robust to that.

Veros is an optional dependency -- the jittable fork this project is built
against, cloned and ``pip install -e``d as shown in the main
``README.md``'s install steps -- and is required for both configurations.


A realistic Veros ocean for sensitivity studies
-----------------------------------------------

:mod:`jem.components.veros.setups.global_latlon` is Veros on its own
one-degree ``global_1deg`` grid with the Levitus bathymetry, started from the
Levitus temperature and salinity with an observed SST (an ERA5 analysis, say)
blended into the mixed layer -- the ocean to use when the initial ocean state
itself is under study. Coupled to a Gaussian atmosphere it exchanges through
the closed-form weights of :mod:`jem.utils.latlon_weights`, and its gradient
over a long, high-resolution run comes from
:func:`jem.adjoint.checkpointed_value_and_grad`; see
:doc:`design/differentiating_long_runs` for a worked T255 case.
