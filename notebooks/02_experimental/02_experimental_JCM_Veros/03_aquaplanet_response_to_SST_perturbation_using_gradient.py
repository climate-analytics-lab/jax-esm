# Response of atmosphere to sea surface temperature bump using jax.jvp
import os
from pathlib import Path

import jax
import jax.numpy as jnp

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
import jax_datetime as jdt

from jem.components import JCM, SlabOceanModel
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools

use_ipython = 'get_ipython' in globals()

# %% [markdown]
# ## Configurations

# %%
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
simulation_name = "01-03_aquaplanet_response_to_SST_perturbation_using_gradient"
output_dir = (Path("output") / simulation_name).resolve()
output_dir.mkdir(exist_ok=True, parents=True)
output_figure = output_dir / "sensitivity.png"
one_second = jdt.to_timedelta(1, "second")

# %% [markdown]
# ## Creating Flux and Scalar Exchange between Components

# %%
mapper = BasicMapper()
mapper.add_mapping(
    source = ("atm", "derived.total_heat_flux"),
    target = ("ocn", "forcing.total_heat_flux"),
)
mapper.add_mapping(
    source = ("ocn", "state.sea_surface_temperature"),
    target = ("atm", "forcing.sea_surface_temperature"),
)

# %% [markdown]
# ## Create Coupled Model

# %%
atm_model = jcm.model.Model(
    coords = get_speedy_coords(),
    start_date=start_datetime,
)

JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
)

model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=SlabOceanModel(
            start_datetime=start_datetime,
            timestep=coupling_timestep / one_second,
        ),
    ),
    mappers=dict(mapper=mapper),
)

print("Model info: ") 
tree_tools.print_tree(model.get_info(), root="Model")

# %% [markdown]
# ## Taking Gradient of the Coupled Model

# %% [markdown]
# Mathematically speaking, we are trying to assess
# $$
# x_{\mathrm{final}} = F_{t_0, T}(x_0)
# $$
# where $x_0$ and $x_{\mathrm{final}}$ the initial and final states of the system, $t_0$ the initial time, $T$ the length of simulation time, and $F_{t_0, T}$ the trajectory function that maps the input state from time $t = t_0$ to $t = t_0 +  T$.
#
# The following code attempts to compute the sensitivity of the trajectory function to a pulse function $g(x)$. That is, let initial condition $x_0$ be perturbed as
# $$
#     x'_0(x; \epsilon) = x_0(x) + \epsilon g(x)
# $$
# the response of the final state to $\delta(x)$ will then be
# $$
# x'_{\mathrm{final}} = x_{\mathrm{final}} + \epsilon \, \delta x_{\mathrm{final}}
# $$
# through which the sensitivity of final statet to $g$, noted as $s_{x_{\mathrm{final}}}$, is formally defined as
# $$
# s_{x_{\mathrm{final}}} = \frac{\partial x'_{\mathrm{final}} }{\partial \epsilon}
# $$
#
# The entire computation is achieved using `jax.jvp`. 

# %%
simulation_interval = jdt.to_timedelta(5, "day")
initial_coupled_carry = model.initialize()
trajectory_function = model.generate_trajectory_function(
    workflow=["mapper", "atm", "ocn"],
    iterations = int(simulation_interval / coupling_timestep),
)

@jax.jit
def forecast(sst):
    ocn_state = initial_coupled_carry["ocn"]["state"]
    initial_coupled_carry["ocn"]["state"] = ocn_state.copy({
        "sea_surface_temperature": sst,
    })
    final_carry, _ = trajectory_function(initial_coupled_carry)
    return final_carry["ocn"]["state"]["sea_surface_temperature"], final_carry["atm"]["derived"]["physics"].surface_flux.v0

sst_initial = initial_coupled_carry["ocn"]["state"]["sea_surface_temperature"]

# Put a point SST perturbation in the middle of domain
shape2D = sst_initial.shape
tangent_sst_initial = jnp.zeros_like(sst_initial).at[shape2D[0]//2, shape2D[1]//2].set(1.0)

# Use jax.jvp to obtain the sensitivity of surface meridional wind and SST
(sst_final, v_final), (tangent_sst_final, tangent_v_final) = jax.jvp(forecast, (sst_initial,), (tangent_sst_initial,))
# %% [markdown]
# ## Visualization

# %%
import matplotlib as mplt
if not use_ipython:
    mplt.use("Agg")
import matplotlib.pyplot as plt

# %%
lat = atm_model.coords.horizontal.latitudes * 180/jnp.pi
lon = atm_model.coords.horizontal.longitudes * 180/jnp.pi

fig, ax = plt.subplots(3, 2, figsize=(12, 16))

sst_levels = jnp.linspace(-2, 35, 11)
tangent_sst_levels = jnp.linspace(-1, 1, 11) * 0.5
v_levels = jnp.linspace(-1, 1, 11) * 5
tangent_v_levels = jnp.linspace(-1, 1, 11) * 0.2

ax[0, 0].contourf(lon, lat, (sst_initial-273.15).transpose(), levels=sst_levels)
ax[0, 1].contourf(lon, lat, tangent_sst_initial.transpose(), levels=tangent_sst_levels, cmap="bwr")
ax[1, 0].contourf(lon, lat, (sst_final-273.15).transpose(), levels=sst_levels)
ax[1, 1].contourf(lon, lat, tangent_sst_final.transpose(), levels=tangent_sst_levels, cmap="bwr")
ax[2, 0].contourf(lon, lat, v_final.transpose(), levels=v_levels, cmap="bwr")
ax[2, 1].contourf(lon, lat, tangent_v_final.transpose(), levels=tangent_v_levels, cmap="bwr")

ax[0, 0].set_title("(a) $\\mathrm{SST}_\\mathrm{init}$")
ax[0, 1].set_title("(b) $\\partial \\mathrm{SST}_\\mathrm{init}$")
ax[1, 0].set_title("(c) $\\mathrm{SST}_\\mathrm{final}$")
ax[1, 1].set_title("(d) $\\partial \\mathrm{SST}_\\mathrm{final}$")
ax[2, 0].set_title("(e) $\\mathrm{v}_\\mathrm{final}$")
ax[2, 1].set_title("(f) $\\partial \\mathrm{v}_\\mathrm{final}$")

fig.suptitle(f"Response time: {simulation_interval / jdt.to_timedelta(1, 'day'):.1f} days")
for _ax in ax.flatten():
    _ax.set_xlabel("longitude [deg]")
    _ax.set_ylabel("latitude [deg]")
    
print(f"Saving result figures into: {output_figure}")
plt.savefig(output_figure, dpi=200)

if use_ipython:
    plt.show()


# %%
