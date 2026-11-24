# Output conventions

`Coupler.to_xarray(diagnostics, time=None, *, first_step=0)` returns one
`xarray.Dataset` per component that implements `SupportsXarray`; components
that do not are skipped, so an output-less component does not stop a run
producing output. A component that returns a *mapping* of datasets (a nested
`Coupler`, see {doc}`nesting`) contributes its entries under their own
names, and a name that collides with one already written is a `ValueError`.
`first_step` is the coupled step the first record covers — the `step` of the
carry the trajectory started from — and defaults to 0. **Pass it when
writing a chunked run**, or the second chunk is labelled with the first
chunk's dates. The positional `time` is the *component* form of the call: a
`TimeAxis` an outer coupler hands a nested `Coupler` (see {doc}`nesting`); a
run leaves it `None` and the coupler builds the axis from `first_step`.

Each component is handed a `TimeAxis` (start date, the record's step indices
and the record interval) so every dataset from one run shares one time
coordinate; `TimeAxis.datetimes()` and `TimeAxis.attrs` are the `(values,
attrs)` pair xarray wants, and every component's `to_xarray` calls them
directly.

A component the workflow runs *n > 1* times per coupled step wrote *n*
records per step, and its stacked diagnostics arrive as `(steps, n, ...)`.
The two leading axes are folded into one — they are already in time order,
record `s * n + k` being call *k* of step *s* — and the component is handed
a `TimeAxis` spaced at `coupling_timestep / n` and starting at sub-step
`first_step * n`. So `first_step` is always given in *coupled* steps,
whatever rate a component runs at, and an hourly component in a daily
coupler writes 24 records per coupled step, each stamped at the midpoint of
its hour (00:30, 01:30, ...).
Components with `n == 1` are unchanged, and the datasets of a fast and a
slow component are deliberately *not* on one time axis: they are different
sampling rates of one run, and `xr.merge` of the two is an outer join by
design.

The conventions, which are JCM's:

- **Dimensions and horizontal coordinates.** A separable lon/lat grid
  (every grid whose axes are meridians and parallels, the atmosphere's among
  them) is written on `("time", "lon", "lat")` with 1-D `lon`/`lat`
  coordinates. A curvilinear grid (the displaced-pole ocean, the rotated pole
  of `veros-earth`) is written on `("time", "x", "y")` — index dimensions, no
  coordinate values — with 2-D auxiliary coordinates named **`longitude`** and
  **`latitude`** (`standard_name` `longitude`/`latitude`, degrees, no `axis`
  attribute), and every variable spanning `x` and `y` carries the CF
  `coordinates = "longitude latitude"` attribute that points at them.
  `jem.components.coordinates` defines both layouts once, for the slab models
  and the Veros component alike.

  The 2-D coordinates are not called `lon`/`lat` for two reasons. CF and
  xarray forbid a 2-D variable named after one of its own dimensions, which
  is why the dimensions are `x`/`y`; and, more to the point, a dataset holds
  one variable per name, so in
  `xr.merge([atm, ocn], join="exact")` the ocean's 2-D `lon`/`lat` would meet
  the atmosphere's 1-D `lon`/`lat` dimension coordinates and **lose**. The
  merge does not fail: it keeps the atmosphere's axes and silently drops the
  ocean's positions, leaving `sea_surface_temperature` with no horizontal
  coordinate at all. No `compat` option helps (`equals`, `no_conflicts` and
  `override` all drop them), because the clash is between a dimension
  coordinate and a non-index coordinate of one name, not between values. CF
  does not care what an auxiliary coordinate is called — `standard_name`,
  `units` and the `coordinates` attribute identify it — so a name that cannot
  collide costs nothing, and the merge of an atmosphere, a curvilinear ocean
  and sea ice keeps `longitude`/`latitude` on every ocean and ice variable.
  `jem.plot.map_plot` and `area_mean` read `longitude`/`latitude` when a
  field has them and `lon`/`lat` otherwise.

  Two components on the *same* curvilinear grid (the slab ocean and the slab
  sea ice on the displaced-pole grid) share `x`/`y` and equal 2-D
  coordinates and merge exactly. **The limit:** `x`/`y` carry a size, not
  positions, so two components on curvilinear grids of *different shapes*
  cannot be merged — xarray raises an `AlignmentError` ("conflicting dimension
  sizes") rather than producing a union, and `join="outer"` would pad one
  grid against the other with no meaning. Nothing ships such a configuration;
  one that needs it has to give the second component its own dimension names
  or write its dataset separately.

  The Veros ocean declares what its output is labelled with, as one of:
  nothing (Veros' own T-grid axes `xt`/`yt` on `lon`/`lat`); the 1-D
  `geographic_longitude_axis`/`geographic_latitude_axis` (still `lon`/`lat`,
  but with exactly those values; they must be Veros' own axes to within 1e-6
  degrees, about 0.1 m, because a label must not move a cell); or the 2-D
  `geographic_longitude`/`geographic_latitude` of a grid whose lines are not
  meridians and parallels (written as above, with Veros' own axes kept as
  `native_lon`/`native_lat`: they are up to the pole's displacement — 12
  degrees on `veros-earth` — from where a cell is, so they are never written
  as positions). The 1-D form exists because Veros rebuilds `xt`/`yt` by
  cumulative sums, which cannot be made bit-identical to the atmosphere's
  `radians * 180 / pi`; `veros-double-drake` declares the atmosphere's own
  axes (rebuilt from its Gaussian grid, and checked against its SCRIP file),
  so its `atm` and `ocn` files merge with `join="exact"` and share `lon`/`lat`
  exactly.
- **Coordinate values** are degrees computed as `radians * 180 / pi`, in
  float64, which is character for character what `jcm.utils.data_to_xarray`
  does. A last-bit difference would be enough for `xr.merge` to treat two
  96-point longitude axes as different axes and produce a 119-point union.
- **The atmosphere's `level` coordinate** is the nominal sigma value
  (`a / p0 + b`) at each layer's mid-level, **ordered surface-first**: index 0
  is the lowest layer, `level[0] = 0.95` and `level[-1] = 0.025` for the
  8-layer SPEEDY grid. Its attributes say so: `units = "1"`,
  `positive = "down"`, `axis = "Z"`, and a `long_name` ("nominal sigma ... at
  layer mid-level, surface-first (index 0 is the surface)"). Pick the lowest
  model level with `sel(level=1.0, method="nearest")`, not `isel(level=-1)`,
  which is the top. Only the atmosphere has a `level`; the slab components
  and Veros (`depth`) have none.
- **The time label is the MIDPOINT of the interval a record covers**, as an
  absolute `datetime64[ms]`: record *k* holds the average over `[start_date
  + k dt, start_date + (k+1) dt)` and is stamped at that interval's
  midpoint. `TimeAxis.datetimes()` computes those midpoints on host with
  plain `datetime64[ms]`/`timedelta64[ms]` arithmetic — the same arithmetic
  as `jcm.predictions.ModelPredictions.time_labels` — so every component's
  dataset merges with the atmosphere's on one exact time axis
  (`xr.merge(join="exact")`). Each `to_xarray` hands those values, plus
  `TimeAxis.attrs`, straight to xarray. JCM's own output additionally
  carries a `time_bounds` variable with each interval's start and end; the
  other components do not, **by decision**: they merge with the atmosphere on
  the identical midpoint axis, so bounds would add nothing a merge needs, and
  the interval of record *k* is `[start_date + k dt, start_date + (k+1) dt)`
  for a reader that wants it. The dates are `jax_datetime`'s proleptic
  Gregorian — there is one clock and one calendar, so no component's labels
  can disagree with another's about what day it is.
- **Variable names**: state and derived quantities keep their plain names,
  and every variable that came from a component's *forcing* is written with
  a `forcing_` prefix — `jem.base.component.FORCING_VARIABLE_PREFIX`,
  applied by `forcing_variable(name)`, which a slab model's
  `_create_xarray_data_vars` and `VerosComponent.to_xarray` call. It is a
  convention of the packaged output, not a protocol requirement: the coupler
  never inspects a dataset, a wrapper around an external model may keep
  that model's own names, and the helper leaves a name that already carries
  the prefix unchanged. Two components legitimately hold the same physical
  field — one produced it, the other received it — and without the prefix
  the merge of their datasets collides on the shared name. So the slab
  atmosphere writes `forcing_total_heat_flux` while the ocean writes its own
  derived `total_heat_flux`; the sea ice writes
  `forcing_sea_surface_temperature` for the field the ocean published as
  `sea_surface_temperature`; the bucket land writes `forcing_precipitation`
  and `forcing_land_evaporation`, and the slab land those two and
  `forcing_land_total_heat_flux` (the land tile's fluxes, so that they do not
  collide with the `forcing_evaporation` and `forcing_total_heat_flux` a
  Winton ice, forced by the same fluxes over the sea part, writes); and Veros writes `forcing_heat_flux`,
  `forcing_freshwater_flux`, `forcing_surface_taux`, `forcing_surface_tauy`,
  `forcing_surface_air_temperature` and `forcing_runoff_mass_rate` for the
  six fields an exchanger hands it, keeping plain names for the `temp`,
  `salt`, `u`, `v`, `psi` and sea-surface fields it computes.

  The ocean's `psi` is the barotropic streamfunction, and which of two
  things it is depends on how the setup solves the external mode. With
  `settings.enable_streamfunction` (Veros' own default) it is Veros'
  prognostic `variables.psi`. With the linear free surface — what every
  Veros setup shipped with JEM selects — Veros reuses that same array for
  the surface pressure instead, a different quantity in different units, so
  the wrapper diagnoses the streamfunction by integrating the
  depth-integrated zonal transport northwards from a southern boundary where
  it vanishes (the discrete relation Veros itself inverts when it does solve
  for a streamfunction; a test pins the diagnosis against Veros' own `psi`
  where Veros has one, to machine precision up to the additive constant a
  streamfunction is defined up to). The variable's `comment` attribute says
  which of the two the file holds, with the masks a reader needs
  (`mask_surface_Z`, `mask_U`) published beside it. A free-surface run also
  publishes the sea surface height of its surface-pressure solve as `ssh`
  (m, `ssh = psi / grav`); a streamfunction run publishes no `ssh`. A run
  with GM skew diffusion (`settings.enable_neutral_diffusion` and
  `enable_skew_diffusion`, every shipped Veros setup) also publishes `B1_gm`,
  the zonal component of the GM streamfunction (m² s⁻¹, on `v`'s layout),
  which the eddy-induced overturning is the zonal integral of. Veros' own
  driver no longer evaluates it, so the wrapper runs Veros'
  `isoneutral_diag_streamfunction_kernel` on the evolved state each coupling
  step; it only reads the state.
- **A variable's role is metadata, not a name to parse.** Every packaged
  component tags each output variable with `jem_role`
  (`jem.base.component.role_attrs`), whose value is the section of the carry
  the variable came from: `state` (what the component integrates), `derived`
  (what it diagnosed for others to read) or `forcing` (what it was given).
  So `ds.filter_by_attrs(jem_role="forcing")` is the whole query. This does
  not replace the `forcing_` prefix and is not redundant with it: the prefix
  stops an `xr.merge` collision between a field one component computed and
  the lagged copy another received, while matching a prefix cannot tell a
  *received* `forcing_q_flux` from a model whose own field happens to start
  with the same word, and says nothing about the variables that are not
  forcing. A variable that is none of the three — a grid mask, a layer
  thickness, anything time-invariant from the component's configuration
  rather than its carry — is left untagged deliberately.
- **A configuration-dependent variable is decided by the run, not by the
  component object.** `SlabOceanModel` writes `forcing_q_flux` only when the
  trajectory actually applied a Q-flux, following the `forcing_method` in
  `carry["params"]` — which `initialize(params)` may set to something other
  than the method the model was constructed with. The step publishes the
  Q-flux snapshot it applied as a key of its own diagnostics, and
  `_create_xarray_data_vars` writes the variable when that key is present.
  This is safe under `lax.scan` because `forcing_method` is static
  (`pytree_node=False`): it cannot change during a run, so the diagnostics
  structure is constant even though it varies between runs.

Together these are what make `xr.merge([datasets["atm"], datasets["ocn"]])`
an N-long join rather than a 2N-long outer union.
