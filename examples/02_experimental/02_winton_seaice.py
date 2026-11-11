# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Aquaplanet
#
# SPEEDY T31L8 over a slab ocean with thermodynamic sea ice: JAX-ESM's own
# quick start, built directly in Python -- the same construction
# `docs/source/python_api.md` walks through, with the sea-ice component this
# run also couples added on top (see the note below the build cell). The same
# model, as one command:
#
# ```bash
# python -m jem.main +configuration=aquaplanet-slab
# ```
#
# Building it directly here, rather than composing that command's config and
# calling `jem.runners.run(cfg)`, is the point of this notebook: every object
# below is a plain Python class from `jem.components`, and nothing about it
# depends on Hydra. The build cell passes `time_step=12` explicitly for the
# same reason -- without it, `jcm.model.Model` would pick the physics' own
# stable step (30 minutes) rather than the 12 minutes
# `+configuration=aquaplanet-slab` actually runs at, which would make "the
# same model, as one command" above false at the physics level.
# `tests/unit/test_notebook_equivalence.py` checks the claim holds, `dt`
# included. `02_aquaplanet_customized_initial_condition.ipynb` and
# `03_aquaplanet_response_to_SST_perturbation_using_gradient.ipynb` build
# their models the same direct way before perturbing an initial condition and
# differentiating through a run.

# %% [markdown]
# ## Build it

# %%
from pathlib import Path

from jem import plot

output_dir = (Path("output") / "01-01_aquaplanet").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# This next cell builds the model: a plain `jcm.model.Model` wrapped as a component, a `SlabGrid` taken from the atmosphere's own horizontal grid, the slab ocean, and the Winton three-layer sea ice, wired by one hand-written exchanger.
#
# The Winton ice computes its own bulk surface fluxes, so it needs the atmosphere's near-surface state and radiation, not just a net flux. `JCMDerived` publishes only the net surface exchange by name, so the exchanger reads the rest from JCM's per-step diagnostics (`derived.physics`):
#
# | Winton forcing | taken from | note |
# |---|---|---|
# | `rsds` | `_shortwave_rad.rsds` | downward shortwave at the surface, W m-2 |
# | `rlds` | `_surface_flux.rlds` | downward longwave at the surface, W m-2 |
# | `air_temperature` | `_surface_flux.t0` | SPEEDY's near-surface air temperature, K |
# | `air_specific_humidity` | `_humidity.rh[-1] * qsat[-1]` | lowest-level humidity, which is what SPEEDY's sea flux uses (`fhum0 = 0`); g/kg converted to kg/kg |
# | `wind_speed` | `surface_exchange.wind_speed` | the fwind0-scaled near-surface wind; the ice adds the gust speed itself |
# | `normalized_surface_pressure` | inverted from `surface_exchange.air_density` and `air_potential_temperature` | jcm does not publish the surface pressure |
# | `frozen_precipitation` | `precipitation` where `t0 < 0 degC` | SPEEDY does not split rain and snow |
# | `liquid_precipitation` | the rest of `precipitation` | |
# | `atm_sea_heat_flux` | `-total_heat_flux` | downward; on an aquaplanet the grid mean is the sea flux |
# | `evaporation` | `evaporation` | upward; passed on to the ocean by the ice |
# | `sea_surface_temperature` | `ocn.state` | |
# | `ice_frazil_melt_energy` | `ocn.derived` | CESM `frzmlt` sign |
#
# Everything the atmosphere publishes is a mean over the coupling day, so the ice is forced with daily means.

# %%
import jax.numpy as jnp
import jax_datetime as jdt
import jcm
import jcm.constants as jcm_constants
from jcm.physics.speedy.speedy_coords import get_speedy_coords

from jem import Coupler, run_chunked
from jem.components import JCMComponent, SlabOceanModel
from jem.components.slab import SlabGrid
from jem.components.slab.winton_seaice_model import WintonSeaiceModel

start_date = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")

# `time_step=12` (minutes) is the step `+configuration=aquaplanet-slab` runs
# at; without it `Model` picks SPEEDY's own stable step (30 minutes).
atm_model = jcm.model.Model(
    coords=get_speedy_coords(), start_time=start_date, time_step=12
)
# The forcing fields the exchanger overwrites every step, so the atmosphere
# takes them from the carry instead of from its own climatology.
atm = JCMComponent(
    atm_model, exchanged_forcing=("sea_surface_temperature", "sice_am"),
)

# Aquaplanet: the slab grid is the atmosphere's own grid, all ocean.
grid = SlabGrid.from_coords(atm_model.coords.horizontal)

components = {
    "atm": atm,
    "ocn": SlabOceanModel(grid),
    "seaice": WintonSeaiceModel(grid),
}

# Lowest full sigma level, needed to recover the surface pressure below.
sigma_bottom = float(atm_model.coords.vertical.centers[-1])


def atmosphere_forcing_for_ice(atm_derived):
    """The near-surface state and fluxes the Winton ice reads, in its units."""
    physics = atm_derived.physics
    surface_flux = physics["_surface_flux"]
    humidity = physics["_humidity"]
    exchange = physics["surface_exchange"]

    # SPEEDY's sea fluxes use the lowest-level specific humidity (fhum0 = 0).
    # SPEEDY works in g/kg; the ice works in kg/kg.
    air_specific_humidity = 1e-3 * humidity.rh[-1] * humidity.qsat[-1]

    # jcm does not publish the surface pressure, but it does publish the
    # lowest-level moist density and potential temperature,
    #   rho   = sigma p0 psa / (Rd T (1 + vtmpc1 q))
    #   theta = T (sigma psa) ** -kappa,
    # and eliminating T between the two gives sigma * psa.
    sigma_psa = (
        exchange.air_density * jcm_constants.rd * exchange.air_potential_temperature
        * (1.0 + jcm_constants.vtmpc1 * air_specific_humidity) / jcm_constants.p0
    ) ** (1.0 / (1.0 - jcm_constants.akap))

    air_temperature = surface_flux.t0
    # SPEEDY has no rain/snow split: precipitation falling through
    # sub-freezing near-surface air is taken as snow.
    frozen_precipitation = jnp.where(
        air_temperature < jcm_constants.tmelt, atm_derived.precipitation, 0.0)
    return dict(
        rsds=physics["_shortwave_rad"].rsds,
        rlds=surface_flux.rlds,
        air_temperature=air_temperature,
        air_specific_humidity=air_specific_humidity,
        wind_speed=exchange.wind_speed,
        normalized_surface_pressure=sigma_psa / sigma_bottom,
        liquid_precipitation=atm_derived.precipitation - frozen_precipitation,
        frozen_precipitation=frozen_precipitation,
        # Downward positive. On an aquaplanet the grid mean is the sea flux.
        atm_sea_heat_flux=-atm_derived.total_heat_flux,
        # Upward positive; the ice passes E - P on to the ocean.
        evaporation=atm_derived.evaporation,
    )


def exchanger(cpl_carry, time):
    atm = cpl_carry["atm"]
    ocn = cpl_carry["ocn"]
    seaice = cpl_carry["seaice"]

    sea_surface_temperature = ocn["state"].sea_surface_temperature
    total_heat_flux = atm["derived"].total_heat_flux
    ice_fraction = seaice["state"].ice_fraction
    
    # The exchange runs before the atmosphere's first step, when everything
    # it publishes is still zero (air at 0 K, no pressure), which the ice's
    # bulk fluxes cannot take. Until the atmosphere has run once, the ice
    # keeps the seed forcing it was initialized with.
    atmosphere_has_run = time.step > 0
    ice_atmosphere_forcing = {
        name: jnp.where(atmosphere_has_run, value, getattr(seaice["forcing"], name))
        for name, value in atmosphere_forcing_for_ice(atm["derived"]).items()
    }

    atm = dict(atm, forcing=atm["forcing"].replace(
        sea_surface_temperature=sea_surface_temperature,
        sice_am=ice_fraction,
        sea_ice_temperature=seaice["derived"].ice_surface_temperature_K,
    ))
    ocn = dict(ocn, forcing=ocn["forcing"].replace(
        # The ice returns the frazil latent heat in ocean_heat_flux_up, for an
        # ocean that stays below freezing until that heat arrives. The slab
        # ocean clamps at freezing and has already removed the deficit, so the
        # returned heat is taken back out here, or it would be counted twice.
        total_heat_flux=(
            seaice["derived"].ocean_heat_flux_up
            + seaice["derived"].ocean_frazil_heating
        ),
        # The ice passes on the atmosphere's E - P, holds back the snow that
        # landed on it until it melts, and adds the water ice growth takes.
        total_freshwater_flux=seaice["derived"].ocean_freshwater_flux_up,
    ))
    seaice = dict(seaice, forcing=seaice["forcing"].replace(
        sea_surface_temperature=sea_surface_temperature,
        # thsice's fzMlOc times dt: positive is the mixed layer's freezing
        # potential, the only way ice forms in open water; negative is the
        # heat it holds above freezing, which caps melting at the ice base.
        ice_frazil_melt_energy=ocn["derived"].ice_frazil_melt_energy,
        **ice_atmosphere_forcing,
    ))

    return {**cpl_carry, "atm": atm, "ocn": ocn, "seaice": seaice}


coupler = Coupler(
    components,
    {"exchanger": exchanger},
    workflow=["exchanger", "atm", "ocn", "seaice"],
    coupling_timestep=coupling_timestep,
    start_date=start_date,
)
print(repr(coupler))

# %%
initial_coupled_carry = coupler.initialize()
sea_surface_temperature = initial_coupled_carry.components["ocn"]["state"].sea_surface_temperature
initial_coupled_carry.components["ocn"]["state"].sea_surface_temperature = jnp.zeros_like(sea_surface_temperature) + 273.15 + 1.0

# %% [markdown]
# ## Run it

# %%
result = run_chunked(
    coupler,
    initial_carry = initial_coupled_carry,
    total_time="180 days",
    chunk="30 days",
    output_dir=str(output_dir),
    subsample=3,       # 10 records out of 30 coupled days
    checkpoint_path="test_output",
)
result.steps_completed, [p.name for p in result.paths]

# %% [markdown]
# ## What it wrote
#
# One file per component per chunk, named after the coupled step its chunk starts on (`<component>-<first step>.nc`).

# %%
atm_ds = plot.open_output(output_dir, "atm")
ocn_ds = plot.open_output(output_dir, "ocn")
seaice_ds = plot.open_output(output_dir, "seaice")
list(ocn_ds.data_vars)

# %% [markdown]
# ## Plot
#
# All three coupled components show up below: surface specific humidity (animated -- the field this example's top-level README entry is named after) and sea surface temperature from the atmosphere and ocean, and ice thickness from the sea-ice component.

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# `level` is a sigma coordinate, surface-first, so selecting by its
# value (rather than `.isel(level=0)`) says so without relying on that
# ordering -- see "Output conventions" in
# docs/source/design/output.md.
humidity = atm_ds["specific_humidity"].sel(level=1.0, method="nearest")
plot.map_plot(humidity.isel(time=-1), ax=axes[0],
              title="Surface specific humidity [kg/kg]")

sst = ocn_ds["sea_surface_temperature"].isel(time=-1) - 273.15
plot.map_plot(sst, ax=axes[1], title="Sea surface temperature [°C]")

# `levels=10`: an integer band *count* rather than explicit boundaries, so
# the sea-ice panel gets 10 discrete colour bands over its own data range
# instead of matplotlib's default (~7-ish, chosen by its own locator).
plot.map_plot(seaice_ds["ice_fraction"].isel(time=-1), ax=axes[2],
              levels=20, title="Sea ice fracion")
plt.tight_layout()

fig, ax = plt.subplots()
plot.area_mean(ocn_ds["sea_surface_temperature"]).plot(ax=ax)
ax.set_ylabel("Area-mean SST [K]")

# Animate the same field the static panel above shows, and the field
# the top-level README's gif is named after.
ani = plot.animate_map(humidity, title="Surface specific humidity [kg/kg]")
ani.save(output_dir / "specific_humidity.gif", writer="pillow")

# %%
