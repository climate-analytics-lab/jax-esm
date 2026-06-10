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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--total-simulation-days", type=int, help="Total time of simulation in days", default=5)
parser.add_argument("--truncation-number", type=int, help="Truncation number", default=31)
args = parser.parse_args()
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

simulation_interval = jdt.to_timedelta(args.total_simulation_days, "day")
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


atm_model = model.components["atm"].raw_component
llon, llat = jnp.meshgrid(
    atm_model.coords.horizontal.longitudes * 180/jnp.pi,
    atm_model.coords.horizontal.latitudes  * 180/jnp.pi,
    indexing="ij",
)

def gaussian(x, y, xc, yc, sigma_x, sigma_y):
    return jnp.exp( -  (x-xc)**2 / (2*sigma_x**2) - (y-yc)**2 / (2*sigma_y**2) )
    
tangent_sst_initial = gaussian(llon, llat, 180.0, 0.0, 5, 8) * 1
tangent_sst_initial /= jnp.sum(tangent_sst_initial**2)**0.5
#tangent_sst_initial = jnp.zeros_like(sst_initial).at[shape2D[0]//2, shape2D[1]//2].set(1.0)

# Use jax.jvp to obtain the sensitivity of surface meridional wind and SST
print("Compute sensitivity using jax.jvp...")
(sst_final, v_final), (tangent_sst_final, tangent_v_final) = jax.jvp(forecast, (sst_initial,), (tangent_sst_initial,))

def report_tangent(name, tangent):
    tangent = jnp.asarray(tangent)
    print(
        f"[{name}] max|tangent| = {float(jnp.max(jnp.abs(tangent))):.6e}, "
        f"sum|tangent| = {float(jnp.sum(jnp.abs(tangent))):.6e}, "
        f"any nonzero = {bool(jnp.any(tangent != 0.0))}, "
        f"any nan = {bool(jnp.any(jnp.isnan(tangent)))}"
    )

report_tangent("tangent_sst_final (jvp)", tangent_sst_final)
report_tangent("tangent_v_final (jvp)", tangent_v_final)

print("Compute sensitivity using direct method")
epsilon = 0.01
sst_final1, v_final1 = forecast(sst_initial)

ensemble_members = 10
ensemble_collection = []
for i in range(ensemble_members):
    print(f"Running ensemble member ({i:d}/{ensemble_members:d})")
    _sst_final, _v_final = forecast(sst_initial + tangent_sst_initial * epsilon * (jax.random.normal(key=jax.random.PRNGKey(i), shape=sst_initial.shape) * 0.05 + 1) )
    _sensitivity_sst = (_sst_final - sst_final1) / epsilon
    _sensitivity_v = (_v_final - v_final1) / epsilon

    ensemble_collection.append(dict(
        sst_final = sst_final1,
        v_final = v_final1,
        sensitivity_sst = _sensitivity_sst,
        sensitivity_v   = _sensitivity_v,
    ))

test_ensemble_members = [1, ensemble_members // 2, ensemble_members]

ensemble_mean_of_sensitivity = {}
for number in test_ensemble_members:
     ensemble_mean_of_sensitivity[number] = jax.tree.map(lambda *x: jnp.mean(jnp.stack(x), axis=0), *ensemble_collection[:number])
    

print("Visualization...")
import matplotlib as mplt
mplt.use("Agg")
import matplotlib.pyplot as plt

atm_model = model.components["atm"].raw_component
lat = atm_model.coords.horizontal.latitudes * 180/jnp.pi
lon = atm_model.coords.horizontal.longitudes * 180/jnp.pi

sst_levels = jnp.linspace(-2, 35, 11)
tangent_sst_levels = jnp.linspace(-1, 1, 11) * 0.01
v_levels = jnp.linspace(-1, 1, 11) * 5
tangent_v_levels = jnp.linspace(-1, 1, 11) * 0.05

# Each column is a method; each row is a quantity (perturbation/initial
# SST, SST response, meridional wind response). The reference column shows
# the actual fields, while the jvp and ensemble columns show the
# perturbation/sensitivity fields for direct comparison.
reference_column = (
    "Reference run",
    [
        ((sst_initial - 273.15), sst_levels,         {},              "$\\mathrm{SST}_\\mathrm{init}$"),
        ((sst_final - 273.15),   sst_levels,         {},              "$\\mathrm{SST}_\\mathrm{final}$"),
        (v_final,                v_levels,           {"cmap": "bwr"}, "$\\mathrm{v}_\\mathrm{final}$"),
    ],
)

jvp_column = (
    "jax.jvp",
    [
        (tangent_sst_initial, tangent_sst_levels, {"cmap": "bwr"}, "$\\partial \\mathrm{SST}_\\mathrm{init}$"),
        (tangent_sst_final,   tangent_sst_levels, {"cmap": "bwr"}, "$\\partial \\mathrm{SST}_\\mathrm{final}$"),
        (tangent_v_final,     tangent_v_levels,   {"cmap": "bwr"}, "$\\partial \\mathrm{v}_\\mathrm{final}$"),
    ],
)

ensemble_columns = [
    (
        f"direct (ens={number:d})",
        [
            (tangent_sst_initial,        tangent_sst_levels, {"cmap": "bwr"}, "$\\partial \\mathrm{SST}_\\mathrm{init}$"),
            (data["sensitivity_sst"],    tangent_sst_levels, {"cmap": "bwr"}, "$\\partial \\mathrm{SST}_\\mathrm{final}$"),
            (data["sensitivity_v"],      tangent_v_levels,   {"cmap": "bwr"}, "$\\partial \\mathrm{v}_\\mathrm{final}$"),
        ],
    )
    for number, data in ensemble_mean_of_sensitivity.items()
]

columns = [reference_column, jvp_column] + ensemble_columns


def plot_sensitivity_comparison(columns, output_figure, suptitle):

    nrows = 3
    ncols = len(columns)
    fig, ax = plt.subplots(nrows, ncols, figsize=(6 * ncols, 14), squeeze=False)

    for col_idx, (col_title, rows) in enumerate(columns):
        for row_idx, (data, levels, kwargs, row_title) in enumerate(rows):
            _ax = ax[row_idx, col_idx]
            cf = _ax.contourf(lon, lat, jnp.asarray(data).transpose(), levels=levels, extend="both", **kwargs)
            fig.colorbar(cf, ax=_ax, orientation="vertical", shrink=0.85, pad=0.02)
            _ax.set_title(row_title if row_idx > 0 else f"{col_title}\n{row_title}")
            _ax.set_xlabel("longitude [deg]")
            _ax.set_ylabel("latitude [deg]")

    fig.suptitle(suptitle)
    print(f"Saving result figure into: {output_figure}")
    plt.savefig(output_figure, dpi=200, bbox_inches="tight")
    plt.close(fig)


output_figure = output_dir / "sensitivity_comparison.png"
plot_sensitivity_comparison(
    columns,
    output_figure,
    f"Response time: {simulation_interval / jdt.to_timedelta(1, 'day'):.1f} days",
)
