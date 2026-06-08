# Response of atmosphere to sea surface temperature bump using jax.jvp
import os
from pathlib import Path

import jax
import jax.numpy as jnp

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
import jax_datetime as jdt

import jem
from jem.components import JCM, SlabOceanModel
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools

from model_setup import build_model, get_ocean_surface_temperature, set_ocean_surface_temperature

print(f"jcm library is located at: {jcm.__file__}")
print(f"jem library is located at: {jem.__file__}")

# Check available devices
print(f"Available devices: {jax.devices()}")
print(f"Number of devices: {len(jax.devices())}")


# Configurations

start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
simulation_name = "sensitivity"
output_dir = (Path("output") / simulation_name).resolve()
output_dir.mkdir(exist_ok=True, parents=True)
one_second = jdt.to_timedelta(1, "second")
truncation_number=31
calendar = "365_day"
# Build the coupled JCM + Veros + SlabOceanModel system. Packaged as a
# function in `model_setup.py` so that other scripts (e.g. a jax.grad
# sensitivity experiment) can build exactly the same model.
model, config = build_model(
    truncation_number=truncation_number,
    start_datetime=start_datetime,
    coupling_timestep=coupling_timestep,
    calendar=calendar,
)
ocn_model = model.components["ocn"].raw_component

print("Model info: ") 
tree_tools.print_tree(model.get_info(), root="Model")

simulation_interval = jdt.to_timedelta(5, "day")
initial_coupled_carry = model.initialize()
trajectory_function = model.generate_trajectory_function(
    workflow=config["workflow"],
    iterations = int(simulation_interval / coupling_timestep),
)

@jax.jit
def forecast(sst):
    # Work on a structural copy of the ocean state so that perturbing it
    # here cannot mutate (or leak tracers into) `initial_coupled_carry`,
    # which is captured by closure and must stay reusable across calls.
    ocn_state = jax.tree_util.tree_map(lambda x: x, initial_coupled_carry["ocn"]["state"])
    ocn_state = set_ocean_surface_temperature(ocn_state, sst)
    modified_carry = dict(
        initial_coupled_carry,
        ocn=dict(initial_coupled_carry["ocn"], state=ocn_state),
    )
    final_carry, _ = trajectory_function(modified_carry)
    return (
        get_ocean_surface_temperature(final_carry["ocn"]["state"]),
        final_carry["atm"]["derived"]["physics"]["_surface_flux"].v0,
    )

sst_initial = get_ocean_surface_temperature(initial_coupled_carry["ocn"]["state"])

# Put a point SST perturbation in the middle of domain
shape2D = sst_initial.shape
tangent_sst_initial = jnp.zeros_like(sst_initial).at[shape2D[0]//2, shape2D[1]//2].set(1.0)

# Use jax.jvp to obtain the sensitivity of surface meridional wind and SST
print("Compute sensitivity using jax.jvp...")
(sst_final, v_final), (tangent_sst_final, tangent_v_final) = jax.jvp(forecast, (sst_initial,), (tangent_sst_initial,))

print("Compute sensitivity using direct method")
epsilon = 0.01
sst_final1, v_final1 = forecast(sst_initial)
sst_final2, v_final2 = forecast(sst_initial + tangent_sst_initial * epsilon)
sensitivity_sst = (sst_final2 - sst_final1) / epsilon
sensitivity_v = (v_final2 - v_final1) / epsilon

print("Visualization...")
import matplotlib as mplt
mplt.use("Agg")
import matplotlib.pyplot as plt

atm_model = model.components["atm"].raw_component

lat = atm_model.coords.horizontal.latitudes * 180/jnp.pi
lon = atm_model.coords.horizontal.longitudes * 180/jnp.pi

sst_levels = jnp.linspace(-2, 35, 11)
tangent_sst_levels = jnp.linspace(-1, 1, 11) * 0.5
v_levels = jnp.linspace(-1, 1, 11) * 5
tangent_v_levels = jnp.linspace(-1, 1, 11) * 0.2

def plot_sensitivity(sst_final, tangent_sst_final, v_final, tangent_v_final, output_figure, method_name):

    fig, ax = plt.subplots(3, 2, figsize=(12, 16))

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

    fig.suptitle(f"[{method_name}] Response time: {simulation_interval / jdt.to_timedelta(1, 'day'):.1f} days")
    for _ax in ax.flatten():
        _ax.set_xlabel("longitude [deg]")
        _ax.set_ylabel("latitude [deg]")

    print(f"Saving result figure into: {output_figure}")
    plt.savefig(output_figure, dpi=200)
    plt.close(fig)


output_figure_jvp = output_dir / "sensitivity_jvp.png"
output_figure_direct = output_dir / "sensitivity_direct.png"

plot_sensitivity(sst_final, tangent_sst_final, v_final, tangent_v_final, output_figure_jvp, "jax.jvp")
plot_sensitivity(sst_final1, sensitivity_sst, v_final1, sensitivity_v, output_figure_direct, "direct finite-difference")
