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
    one spatially uniform, time-constant land from the packaged
    ``forcing_idealised_T31.nc`` (bare-land albedo 0.2, no snow cover,
    soil-water availability 0.65 and a 290 K skin-temperature climatology: the
    rounded means of Earth's ice-free land), written by
    :func:`jem.tools.idealised_terrain.idealised_forcing`. Earth's climatology
    is not used, because it describes Earth's continents, not these; left at
    jax-gcm's defaults the land albedo would be 0 and the land would absorb all
    the sunlight it receives. Another geography on the same grid reuses the
    file; another resolution needs ``idealised_terrain`` and
    ``idealised_forcing`` calls for its own grid.
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
sea-ice component (``veros-earth-winton`` adds one): the land is SPEEDY's
own slab land inside the atmosphere. Each configuration's own comment says why every setting is what
it is.

The ocean is idealised in both, and a short run should be read with that in
mind:

* **Real bathymetry at the grid's resolution in** ``veros-earth``, **a flat
  bottom in** ``veros-double-drake``. The land-sea mask decides which
  columns are ocean (for ``veros-earth`` the packaged ERA5 land fraction,
  thresholded at 0.5), and the packaged
  ``jem/data/bathymetry_RotatedGaussianLatLon.nc`` decides how deep each of
  them is: NOAA NCEI's ETOPO 2022 relief (60 arc-second, ice-surface
  version; NOAA National Centers for Environmental Information, 2022,
  https://doi.org/10.25921/fd45-gt74, a public-domain product released under
  CC0) averaged onto the grid's cells, whose size is about 3.75 degrees, so
  shelves, ridges and straits are not resolved. Each column holds as many of
  the 15 layers as fit above its depth, at least two (a minimum depth of
  120 m) and at most all of them (5200 m), without changing which columns
  are ocean. The file's attributes record its source, terms and counts, and
  ``jem.tools.ocean_bathymetry`` says how to regenerate it.
  ``ocean.bathymetry_file=null`` restores the flat 5200 m bottom, and
  ``ocean.bathymetry_file=/path/to/depth.nc`` takes a file made by
  ``jem.tools.ocean_bathymetry.ocean_bathymetry`` from another source.
  ``veros-double-drake`` has the same flat 5200 m bottom in every ocean
  column.
* **An observed mean state in** ``veros-earth``, **a uniform one in**
  ``veros-double-drake``. ``veros-earth`` starts from the World Ocean Atlas
  2018 annual-mean temperature and salinity (NOAA National Centers for
  Environmental Information; Boyer et al., 2018, https://www.ncei.noaa.gov/archive/accession/NCEI-WOA18;
  Locarnini et al., 2019, NOAA Atlas NESDIS 81, https://doi.org/10.25923/e5rn-9711
  and Zweng et al., 2019, NOAA Atlas NESDIS 82, https://doi.org/10.25923/9pgv-1224;
  the data are "openly available to the public" with a request to
  acknowledge them, and the packaged file carries that citation and the
  disclaimer), in
  ``jem/data/ocean_initial_state_RotatedGaussianLatLon.nc``. WOA holds
  in-situ temperature and practical salinity; Veros' ``temp`` and ``salt``
  are Conservative Temperature and Absolute Salinity, so the tool converts
  every WOA point (UNESCO 1983 potential temperature, TEOS-10 Conservative
  Temperature, Reference-Composition Salinity; the salinity anomaly
  ``delta SA`` is taken as zero) before averaging onto the grid's cells, and
  each model layer takes the mean of the profile over its thickness. The
  annual mean is used because the monthly fields stop at 1500 m and a run can
  start in any month. The surface therefore starts at about 29 degC in the
  warm pool and at the freezing point near the poles, with the observed
  salinity contrasts, and the deep ocean cold. Wet cells with no WOA value
  (WOA's sea floor is shallower, or a lake) are filled from their neighbours
  or the layer above, and the counts are in the file's attributes. About
  1.4 % of vertically adjacent pairs (467 of 32304) are statically unstable
  in the setup's linear equation of state, nearly all south of 60 S with
  densities differing by a few hundredths of a kg m-3 (the linear equation
  overstates the thermal expansion of near-freezing water; one Baltic cell
  differs by 2 kg m-3); in a coupled run the count is 638, 329 and 390 after
  1, 30 and 90 days, and no pair is more than 0.03 kg m-3 inverted. The
  currents start at rest, so the first days are a geostrophic adjustment of
  the observed density field (the sea-surface-height range reaches about
  -2.2 m within days and stays there). Over a 90-day
  January run the band-mean sea surface temperature moves by 0.4 to 2.9
  degrees (the annual mean is not January's state), and the polar surface
  reaches and passes the freezing point with no sea ice to stop it: that is
  the state a sea-ice component would start from. The file holds values for
  exactly the wet cells of the packaged bathymetry, so it is valid only with it: the setup refuses a
  file made for another bathymetry, and
  ``jem.tools.ocean_initial_state`` says how to regenerate one.
  ``ocean.initial_state_file=null`` restores the uniform start (salinity 35
  and 288.03 K at the surface in every ocean column, falling linearly with depth to zero at
  4510 m, so the deepest layer starts at -1.15 degC). ``veros-double-drake``
  always starts uniformly.
* **No sea ice by default.** ``veros-earth`` and ``veros-double-drake`` have
  no sea-ice component; the only ice effect on the ocean is the "swamp" mask on
  the heat and freshwater fluxes once the surface reaches the freezing point,
  which stops the surface cooling but not the water under it, so the polar
  surface of ``veros-earth`` drifts below the freezing point.
  ``+configuration=veros-earth-winton`` composes the Winton ice on the ocean's
  grid: the ocean then clamps its top cell at the freezing point and hands the
  ice the deficit, and the ice hands the ocean its heat and freshwater (see
  :doc:`design/winton_seaice`).

The two components are coupled through :class:`jem.fluxes.VerosExchange`,
named as ``coupling.exchanger``: it hands the ocean the surface wind stress
the atmosphere itself received -- jax-gcm's published, stability-corrected
stress, the same for every physics package, so momentum is conserved across
the interface -- regridded conservatively and rotated into the ocean grid's
local frame for ``veros-earth``, whose grid is not true east/north;
applies a "swamp" sea-ice mask to the heat and freshwater fluxes once the
surface reaches the freezing point (unless a Winton ice is composed, which is
then the insulation and passes the ocean its own fluxes), and carries the rest of
:data:`jem.exchangers.VEROS_OCEAN_EXCHANGES` (the fluxes onto the ocean, its
sea surface temperature back) -- so the ocean is mechanically as well as
thermodynamically forced.

``veros-earth``'s ocean output is written on index dimensions ``x``/``y``
with 2-D ``longitude``/``latitude`` coordinates giving each cell's true
position, like any curvilinear grid (see :doc:`design/output`, which also says
why they are not called ``lon``/``lat``): the grid's own axes are the
rotated ones, kept in the file as ``native_lon``/``native_lat``, and ``u``,
``v`` and the wind stress are components along those axes rather than east
and north.

Veros applies its ``device`` runtime setting to the whole process
(``jax_platform_name``), and its default is ``"cpu"``.
:func:`jem.components.veros_component.configure_veros_runtime` therefore sets
it to the platform JAX would use anyway (``VEROS_DEVICE`` still chooses
explicitly), so a Veros configuration runs on the GPU of a GPU host --
atmosphere included -- rather than silently on its CPU.

A Veros configuration runs the **whole model**, atmosphere included, in double
precision, from the start. Veros runs float64 and switches JAX's process-wide
``jax_enable_x64`` on when its operators are first imported, which happens
while the ocean is built, after the atmosphere; so ``ocean=veros`` selects
``jax_enable_x64=true``, and the runner applies it before any component is
built (see :doc:`design/configuration`). Nothing is allocated in float32 and
then stepped in float64, and :class:`~jem.fluxes.VerosExchange` has no mixed
carry to reconcile. A configuration that asks for ``jax_enable_x64=false``
with Veros is refused at build time. The few atmosphere leaves that stay
float32 are the ones jax-gcm builds as float32 whatever the flag (the
solar-geometry scalars, an empty analytic-ozone array and, under
``forcing=from_file``, the arrays read from the forcing file): 6 of 246
floating-point leaves in ``veros-double-drake`` and 12 in ``veros-earth``.

The cost, measured for two coupled days on a CPU host that other jobs had
loaded heavily (wall times are good to perhaps 20 %), is a few percent to a
quarter in wall time and little memory: ``veros-earth`` took 68 s and 1.70 GB
peak resident memory, ``veros-double-drake`` 56 s and 1.65 GB. Against the same
runs whose atmosphere was built in float32 (54 s and
1.71 GB, 52 s and 1.67 GB) the memory is unchanged, because Veros' own float64
state dominates it, and the fields after two days differ at the level of
single-precision round-off grown by the flow: relative differences of
1e-6 to 1e-4 in temperature, wind and the ocean's temperature and velocity,
and up to about 5e-3 in the clear-sky emissivities, with no NaN. The slab
``aquaplanet-slab`` takes 38 s and 1.18 GB in float32 and 39 s and 1.29 GB with
``jax_enable_x64=true``. A Veros run cannot share a process with a float32 run
(:doc:`issues`).

Veros is an optional dependency -- the jittable fork this project is built
against, checked out at the one revision recorded as ``VEROS_SUPPORTED_REV``
in ``jem.components.veros`` and put on ``PYTHONPATH`` after
``pip install -r requirements.txt``, as shown in the main ``README.md``'s
install steps (and as CI does) -- and is required for both configurations.
The fork's ``main`` is a different commit from that revision.

The supported fork revision carries an AD fix its own ``PR.md`` marks as
needing physical review before merge. The fix is
``veros.core.utilities.sqrt_singularity_removed``, ``sqrt(maximum(x, 1e-12))``,
which the fork uses in place of ``sqrt(maximum(0, x))`` for the buoyancy
frequency ``Nsqr`` (in the TKE mixing length, and in the EKE and IDEMIX
terms), the TKE and the EKE. Where ``Nsqr`` is negative -- a statically
unstable water column -- or below ``1e-12``, the clamp is active and the
derivative through that term is zero. The fork's authors have not confirmed
that discarding the sensitivity of the convective-instability branch is
physically right, rather than a smoother relaxation of ``Nsqr`` that keeps
some of it. For a JAX-ESM user this means the gradient of a Veros-coupled run
with respect to anything that acts through convectively unstable columns
(``Nsqr`` below ``1e-12``) comes out of this clamp and is not a validated
sensitivity; the forward (primal) solution is, per ``PR.md``, unchanged by
that clamp. The same revision also floors sub-surface TKE at zero after the
implicit solve, which does change the primal, by the undershoot
(``PR.md`` quotes ~1e-7 against ~1e-4) it removes. Both ``veros-earth`` and
``veros-double-drake`` enable TKE and EKE.


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
