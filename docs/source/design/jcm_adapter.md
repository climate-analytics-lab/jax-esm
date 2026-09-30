# The jax-gcm dependency contract and the JCM adapter

## The jax-gcm dependency contract

JAX-ESM is built on jax-gcm but lives in its own repository and is installed
against a source *checkout* of it, not a PyPI release. Without a recorded
pin, "which jax-gcm does this work with?" has no answer, and a rename on the
jax-gcm side surfaces as an `AttributeError` or a `KeyError` deep inside
somebody's coupled run instead of at import time.

`jem/components/jcm/contract.py` records both halves of the contract:
`JCM_SUPPORTED_REV`, the revision every gate runs against (a full 40-character
sha, not an abbreviation, because that is what `actions/checkout` needs and
what `git rev-parse` in a jax-gcm checkout can be compared against directly),
and `JCM_INTEGRATION_POINTS`, every jax-gcm name JAX-ESM reaches for — the
functions it calls, the physics diagnostics fields the surface exchange is
read out of, the package data it resolves, and the constructors its
documented workflow asks a user to call — each entry saying what it is used
for, which is what makes it possible to decide whether an entry may be
deleted. `tests/unit/test_jcm_contract.py` walks that list against the
installed `jcm`, so a jax-gcm rename fails as "jax-gcm renamed or removed X,
which JAX-ESM used for Y, at revision Z" — at the cheapest possible moment,
rather than mid-run — and asserts that the revision the CI workflow checks
out (`JCM_REV` in `.github/workflows/tests.yml`) equals `JCM_SUPPORTED_REV`,
so the pin and the workflow cannot drift apart. A non-blocking
`canary-jcm-dev` job separately tracks jax-gcm's `dev` branch, so drift stays
visible without blocking a pull request.

The pin is a `dev` commit rather than a release (jax-gcm has no tagged 3.x
yet), chosen and bumped by the policy in `contract.py`'s own module
docstring: pin to a `dev` commit validated as a whole against this
repository's gates, never to the merge commit of one motivating pull request
or to whatever `dev` happens to be at bump time. The current pin is needed
for the package-independent surface-exchange contract's near-surface wind
vector (`wind_u`/`wind_v`, jax-gcm#911/#914), which is what lets JAX-ESM read
the wind the same way for every physics package (see *The JCM adapter*
below).

**One consequence a coupled configuration has to set explicitly.**
`jcm.forcing.resolve_align`'s `auto` resolves a boundary condition's
alignment (climatology vs. transient) only when the file is a known
mirror/packaged product; anything else needs `atmosphere.forcing.align` set
by hand in the configuration, rather than guessed from the file's own time
axis. See the `forcing.align` comments in
`jem/config/configuration/{veros-double-drake,veros-earth}.yaml`.

## The JCM adapter

`jem/components/jcm/component.py` wraps a `jcm.model.Model` (the spectral
atmosphere from jax-gcm) as `JCMComponent`. It is a wrapper object, not an
in-place adaptation: the atmosphere JEM drives is the same object the user
configured, and nothing in JCM has to know JEM exists. Its carry is:

```python
{
    "state":   <jcm modal (spectral) dycore state>,
    "physics": <jcm's cross-step physics carry, threaded, opaque>,
    "time":    <jax-gcm's own clock, a jax_datetime.Datetime>,
    "step":    <jax-gcm's own step counter>,
    "forcing": <jcm ForcingData; holds sea_surface_temperature, sice_am, ...>,
    "derived": JCMDerived(physics, total_heat_flux, total_freshwater_flux,
                          evaporation, precipitation,
                          eastward_wind_stress, northward_wind_stress,
                          u0, v0),
}
```

`initialize()` builds `state`/`physics` from the `(dycore_state,
physics_carry)` pair `Model.bootstrap_state()` returns and sets `time`/`step`
to the model's own `start_time` and zero; it does **not** integrate. Each
`step` calls `model.run_from_state_with_carry()` with the coupling interval
as both `save_interval` and `total_time`, and `carry["time"]`/`carry["step"]`
as `initial_time=`/`initial_step=`, so JCM sub-steps internally at its own
timestep, returns exactly one saved record per coupling step, and its own
clock keeps advancing continuously across coupling steps — carried
separately from the coupler's own `CoupledCarry.time` (see
{doc}`carry_and_clock`) because it is jax-gcm's `RunState`, not JEM's. If the
two ever disagree — which can only happen if the carry did not come from
this run, such as a checkpoint restored under the wrong start date —
`JCMComponent.step` logs an ERROR (through `jax.debug.callback`, since the
check runs inside the traced `lax.scan` and cannot raise) rather than
silently continuing.

**Reading the surface exchange.** `jem/components/jcm/exchange_fields.py`'s
`from_diagnostics()` is the single reader off jax-gcm's own
package-independent `diagnostics["surface_exchange"]` struct, published
identically by every physics package that resolves a surface (SPEEDY, ECHAM;
Held-Suarez opts out because it has no surface fluxes at all) — the heat and
water fluxes, the surface wind stress AND the near-surface wind vector. Translating that contract to
JEM's conventions is a sign flip, plus one reshape every field takes:

| jax-gcm field | jax-gcm convention | JEM field | JEM convention |
|---|---|---|---|
| `net_heat_flux` | W m⁻², positive **down** | `total_heat_flux` | W m⁻², positive **up** (negated here) |
| `evaporation` | kg m⁻² s⁻¹, positive up | `evaporation` | unchanged |
| `precipitation` | kg m⁻² s⁻¹, positive down | `precipitation` | unchanged |
| `stress_u` | N m⁻², positive down (stress **on** the surface) | `eastward_wind_stress` | unchanged |
| `stress_v` | N m⁻², positive down (stress **on** the surface) | `northward_wind_stress` | unchanged |
| `wind_u` | m s⁻¹, package's own reference | `u0` | unchanged |
| `wind_v` | m s⁻¹, package's own reference | `v0` | unchanged |

**Every field is also reshaped onto the atmosphere's `(ix, il)` nodal grid.**
ECHAM's `vectorize_columns=True` publishes every diagnostics field flat,
`(ix * il,)`, while SPEEDY publishes it already gridded; `from_diagnostics`
reshapes every field onto `nodal_shape` unconditionally, a no-op for SPEEDY
and the exact inverse of jax-gcm's own C-order flatten for ECHAM. See
`jem/components/jcm/exchange_fields.py`'s module docstring for the full
explanation.

`evaporation` and `precipitation` are already the convective+large-scale (or
convective+stratiform) total, computed once by the publisher, so no
package-specific arithmetic happens here at all beyond the reshape.

`stress_u`/`stress_v` keep jax-gcm's sign: "positive down" for a momentum
flux is the stress the atmosphere exerts *on* the surface (westerlies give a
positive eastward stress), which is JEM's convention for a stress and the one
Veros integrates. It is the stress the atmosphere column was actually given
that step, which is why `jem.fluxes.VerosExchange` hands it to the ocean
instead of applying a drag law of its own to the wind (jax-esm#132; see
{doc}`exchange`). Like every guaranteed field it is a grid-box mean over
land, sea and sea ice, so a coastal ocean cell receives some of the land's
drag (jax-esm#147).

`wind_u` and `wind_v` sit at whichever reference the *publishing* package's
own surface closure defines — the contract's static `wind_reference` field
names which (`"10m"`/`"lowest_level"`). JEM publishes the wind as `u0`/`v0`
for a consumer that needs the wind itself and derives no stress from it: a
drag law applied to winds at two different heights would give each physics
package a different ocean forcing for the same flow. An ECHAM-composed
coupled model completes a step like any other package, coupled to a slab
surface or to **Veros**. Importing `veros` sets `jax_enable_x64` process-wide,
so a Veros-coupled ECHAM atmosphere traces its Tiedtke-Nordeng convection and
RRTMGP radiation in 64-bit mode; jax-gcm supports that from the pinned
revision on (jax-gcm#927, fixed by jax-gcm#946).
`tests/unit/test_veros_setups.py::test_echam_veros_earth_configuration_steps`
steps `+configuration=veros-earth physics@atmosphere.physics=echam` twice and
checks every carry field is finite.

**Public surface.** Every JCM attribute the wrapper touches is public at the
pinned revision: the initial state and physics carry come from
`Model.bootstrap_state()`, and a
stacked `ModelPredictions` is repaired with `ModelPredictions.with_context(model)`
(which also stamps the atmosphere dataset's `jcm_prov_params` attribute,
recording that those parameter values were read from the live physics after
the trajectory was traced and scanned, rather than captured at trace time —
the truthful record for a coupled run). Each is a `JCM_INTEGRATION_POINTS`
entry, so a JCM refactor that moves one fails the contract test by name
instead of inside somebody's run.

`VerosComponent.step` compares Veros' own `variables.time` against the
coupler's clock, within `clock_tolerance_seconds`
(`jem.components.clock.clock_tolerance_seconds`) — one second, or eight
float32 ulps of the elapsed time, whichever is larger. Veros has no calendar,
so its counter is seconds since its setup's own start rather than since the
coupler's `start_date`: `bind` records the reading the setup holds when the
coupler adopts it, and the check compares `variables.time` minus that zero
point. A setup already integrated before it was wrapped therefore starts the
coupled run where it stands, while a *later* disagreement — a Veros restart
paired with a `CoupledCarry.time` from elsewhere in the run — is caught. Both
this check and the JCM clock-drift check above report through
`jax.debug.callback` rather than raising, since they run inside the coupled
`lax.scan`, where a Python exception cannot fire on a traced value.
