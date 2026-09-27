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

The pin is a `dev` commit — the merge commit of the jax-gcm pull request that
introduced the change JAX-ESM needs, rather than whatever `dev` happens to be
at bump time, since later unrelated `dev` commits are not revisions this
branch has been checked against. jax-gcm has no tagged 3.x release yet, so a
sha is the most precise thing there is to name; `pyproject.toml`'s
`jcm>=3.0.0rc1` is the loosest true statement of the same fact, since
jax-gcm's version string is bumped only at release and a release candidate
floor is satisfied by every later 3.x release too. The current pin carries
jax-gcm's unification onto one exact `jax_datetime.Datetime` clock
(`Model(start_time=)`, no separate `calendar`, output records labelled at
their interval midpoint) and the package-independent surface-exchange
contract every physics package that resolves a surface now publishes
identically (see *The JCM adapter* below); the module's own docstring is the
procedure for bumping the pin, and is kept current as later revisions are
adopted.

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
                          evaporation, precipitation, u0, v0),
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
Held-Suarez opts out because it has no surface fluxes at all). Translating
that contract to JEM's conventions is a sign flip and nothing else:

| jax-gcm field | jax-gcm convention | JEM field | JEM convention |
|---|---|---|---|
| `net_heat_flux` | W m⁻², positive **down** | `total_heat_flux` | W m⁻², positive **up** (negated here) |
| `evaporation` | kg m⁻² s⁻¹, positive up | `evaporation` | unchanged |
| `precipitation` | kg m⁻² s⁻¹, positive down | `precipitation` | unchanged |

`evaporation` and `precipitation` are already the convective+large-scale (or
convective+stratiform) total, computed once by the publisher, so no
package-specific arithmetic happens here at all.

**One package-specific read remains, and is not expected to go away with a
routine jax-gcm update.** jax-gcm's contract publishes only the *scalar*
`wind_speed`, never a near-surface wind *vector*, so
`jem.fluxes.bulk_wind_stress` (the bulk-drag law `jem.fluxes.VerosExchange`
applies for a Veros ocean) still reads SPEEDY's private
`_surface_flux.u0`/`.v0` directly; ECHAM has no wind vector anywhere in its
own diagnostics either, only a diagnosed speed. `from_diagnostics` reads the
wind *eagerly* to fill `JCMDerived.u0`/`.v0` on every coupled step, so **no
ECHAM-composed coupled model can complete a step**, whatever it is coupled
to — a slab ocean as much as Veros. No shipped JAX-ESM configuration composes
ECHAM, so nothing shipped is affected. Making the wind optional — through
`JCMDerived`, the coupled carry and the output — is tracked in jax-esm#129.

**Public surface.** Every JCM attribute the wrapper touches is public at the
pinned revision, apart from that one underscore-prefixed diagnostics key: the
initial state and physics carry come from `Model.bootstrap_state()`, and a
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
paired with a `CoupledCarry.step` from elsewhere in the run — is caught. Both
this check and the JCM clock-drift check above report through
`jax.debug.callback` rather than raising, since they run inside the coupled
`lax.scan`, where a Python exception cannot fire on a traced value.
