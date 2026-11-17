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
    two zonally uniform, time-constant surfaces from the packaged
    ``forcing_idealised_T31.nc``, written by
    :func:`jem.tools.idealised_terrain.idealised_forcing`: the ice-free land
    (bare-land albedo 0.2, no snow, soil-water availability 0.65, and a
    skin-temperature climatology falling from 302 K at the equator to 265 K at
    75 degrees as ``302 - 40 sin(latitude)**2``: the rounded fit to Earth's
    ice-free land) and the polar cap poleward of 75 degrees, an ice sheet with
    full snow cover, background albedo 0.47, soil-water availability 0.4 and a
    237 K climatology (Earth's ice-sheet values). The land is SPEEDY's own
    slab land inside the atmosphere, whose snow cover and soil water are
    prescribed (snow only changes the albedo to 0.6) and whose temperature is
    the climatology plus an anomaly driven by the land's heat budget and
    relaxed to it on 40 days; a cap that had the ice-free land's values would
    warm to 315 K in its summer (with the polar cap as an ice sheet, a two-year
    run keeps every land skin temperature below 302 K, the cap's mean below
    265 K in every month, and the slab land in a repeating seasonal cycle by
    its second year). Earth's climatology is not used, because it
    describes Earth's continents, not these; left at jax-gcm's defaults the
    land albedo would be 0 and the land would absorb all the sunlight it
    receives. Another geography on the same grid reuses the file; another
    resolution needs ``idealised_terrain`` and ``idealised_forcing`` calls for
    its own grid.
``python -m jem.main +configuration=veros-earth``
    A rotated-pole global ocean read from a SCRIP grid, exchanged with the
    atmosphere through ESMF regridding weights. The atmosphere runs on
    jax-gcm's packaged T30 observed climatology (the file ``earth-slab``
    uses): bare-land albedo, snow cover, soil water and land skin
    temperature. Its sea surface temperature comes from the ocean, not from
    that file, and its sea-ice cover is the Winton ice's, formed on the
    ocean's grid, not the observed cover that file holds.

Both compose ``ocean=veros`` with a case setup importable from
:mod:`jem.components.veros.setups`
(:mod:`~jem.components.veros.setups.double_drake` /
:mod:`~jem.components.veros.setups.earth`), and neither has a land component:
the land is SPEEDY's own slab land inside the atmosphere. ``veros-earth`` has
the Winton sea ice and ``veros-double-drake`` none. Each configuration's own
comment says why every setting is what it is.

Both run SPEEDY with the parameters jax-gcm calibrated for coupled use, at the
30-minute step the calibration was made at. They arrive by composing jax-gcm's
``speedy-t31`` configuration bundle under ``atmosphere`` and are not restated
in JEM's YAML, so a retune in jax-gcm reaches both configurations. The values
are fitted at T31 and 30 minutes against observations with prescribed SSTs
(the top-of-atmosphere budget and the surface fluxes handed to an ocean), so a
coupled run is not guaranteed to be drift-free. Each configuration reads its
own terrain and forcing files, not the bundle's.

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
  reaches the freezing point within days, where the Winton ice forms. The
  file holds values for
  exactly the wet cells of the packaged bathymetry, so it is valid only with it: the setup refuses a
  file made for another bathymetry, and
  ``jem.tools.ocean_initial_state`` says how to regenerate one.
  ``ocean.initial_state_file=null`` restores the uniform start (salinity 35
  and 288.03 K at the surface in every ocean column, falling linearly with depth to zero at
  4510 m, so the deepest layer starts at -1.15 degC). ``veros-double-drake``
  always starts uniformly.
* **A Winton sea ice in** ``veros-earth``, **none in**
  ``veros-double-drake``. ``veros-earth`` composes the Winton ice on the
  ocean's grid: the ocean clamps its top cell at the freezing point and hands
  the ice the deficit, and the ice hands the ocean its heat and freshwater and
  the atmosphere its cover and skin temperature (see
  :doc:`design/winton_seaice`). It starts ice-free over the observed ocean and
  grows from there: in a 90-day run from 1 January it covers 4.7 % of the
  northern ocean after 30 days, 7.7 % after 60 and 7.6 % after 90, 1.2 m thick
  on average poleward of 65N on day 90 (2.6 m at most), the sea surface holds
  at the -1.8 degC freezing point under it, and the ice's energy budget closes
  to round-off (5e-12 W m-2). That is a cover growing over an annual-mean ocean, not an
  equilibrated seasonal cycle, which needs years of integration.
  ``veros-double-drake`` has no sea-ice component; the only ice effect on its
  ocean is the "swamp" mask on the heat and freshwater fluxes once the surface
  reaches the freezing point, which stops the surface cooling but not the
  water under it.

The two components are coupled through :class:`jem.fluxes.VerosExchange`,
named as ``coupling.exchanger``: it hands the ocean the surface wind stress
the atmosphere itself received -- jax-gcm's published, stability-corrected
stress, the same for every physics package, so momentum is conserved across
the interface -- regridded conservatively and rotated into the ocean grid's
local frame for ``veros-earth``, whose grid is not true east/north;
applies a "swamp" sea-ice mask to the heat and freshwater fluxes once the
surface reaches the freezing point (unless a Winton ice is composed, as in
``veros-earth``, which is then the insulation and passes the ocean its own
fluxes), and carries the rest of
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
