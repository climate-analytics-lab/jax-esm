import numpy as np
import xarray as xr
import pandas as pd
import jax
import jax.numpy as jnp

import jcm
from jcm.model import get_coords
from jcm.boundaries import default_boundaries, boundaries_from_file, BoundaryData
from jcm.physics_interface import PhysicsState, Physics, get_physical_tendencies, dynamics_state_to_physics_state
from jcm.date import DateData, Timedelta, Timestamp
from datetime import datetime

from dinosaur import primitive_equations_states, primitive_equations

from pathlib import Path

from dataclasses import dataclass
import tree_math
from typing import Any, List


def stack_objects(
    objs : List,
):
    """
    A tool function that stack dataclasses together.

    Args:

        objs : A list of objects that need to be stacked

    Returns:

        stacked : Stacked object.
        
    """
    # objs is a list of pytrees with same structure
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)
    return stacked



@tree_math.struct
@dataclass
class SpeedyState:
    prog    : PhysicsState
    phydata : Any
    metadata    : primitive_equations_states

# Prepare boundary file
boundary_file = (Path(jcm.__file__).parent / "data" / "bc" / "t30" / "clim" / "boundaries_daily.nc").resolve()
print("Check boundary file:", str(boundary_file))

if not boundary_file.exists():
    print("Boundary file %s does not exist. Need to produce it." % (str(boundary_file),))
    import subprocess, sys
    interpolation_file = boundary_file.parent / "interpolate.py"
    subprocess.run([sys.executable, str(interpolation_file)], check=True)

if boundary_file.exists():
    print("Boundary file %s exists!" % (str(boundary_file), ) )
else:
    raise Exception("Something went wrong. The daily file is not generated. Please check.")

boundaries = boundaries_from_file(
    boundary_file,
    get_coords().horizontal,
)


       
def gen_initialize(
    model: jcm.model.Model,
    substeps: int,
    initial_state: PhysicsState | primitive_equations.State = None,
    boundaries: BoundaryData = None,
    start_date: Timestamp = Timestamp.from_datetime(datetime(2000, 1, 1)),
):

    # Copy from jax-gcm jcm/model.py
    if isinstance(initial_state, primitive_equations.State):
        model.initial_state = dynamics_state_to_physics_state(initial_state, model.primitive)
        model._final_modal_state = initial_state
    else:
        model.initial_state = initial_state
        model._final_modal_state = model._prepare_initial_modal_state(initial_state)

        if initial_state is None:
            model.initial_state = dynamics_state_to_physics_state(model._final_modal_state, model.primitive)
        
    model.start_date = start_date
    model.boundaries = default_boundaries(model.coords.horizontal, model.orography)
    D3_nodal_shape = model.geometry.nodal_shape

    _, init_phydata = model.physics.compute_tendencies(
        state      = model.initial_state,
        boundaries = model.boundaries,
        geometry   = model.geometry,
        date       = model._date_from_sim_time(jnp.array(model._final_modal_state.sim_time)),
    )

        
    init_state = stack_objects([model.initial_state,]*substeps)
    init_phydata = stack_objects([init_phydata,]*substeps)


    return SpeedyState(
        prog = init_state,
        phydata = init_phydata,
        metadata = model._final_modal_state,
    )


def gen_step_fn(
    model,
    save_interval,
    timestep,
):
   
    @jax.jit
    def step_fn(speedy_state: SpeedyState, t):
        jax.debug.print("Step t={t:d}", t=t) 
        new_atm_modal_state, predictions = model.run_from_state(
            initial_state = speedy_state.metadata,
            save_interval = save_interval / 86400.0, # in days
            total_time = timestep / 86400.0, # in days
            boundaries = model.boundaries,
        )

        # phydata is a stacked object. What do I do?
        return SpeedyState(
            prog = predictions.dynamics,
            metadata = new_atm_modal_state,
            phydata = predictions.physics,
        ), predictions
        
    return step_fn



#===========================================================


total_simulation_time =  5 * 86400.0 # sec
timestep              =  86400.0  # sec
total_steps           = int(total_simulation_time / timestep)

substeps          =  12           # count
save_interval     =  timestep / substeps # coupler_time_step # sec

print("Creating JCM") 
model = jcm.model.Model(
    time_step = timestep / substeps / 60.0, # in minutes
)

init_state = gen_initialize(
    model,
    substeps=substeps,
)

step_fn = gen_step_fn(
    model = model,
    save_interval = save_interval,
    timestep = timestep,
)

#print("Ad-hoc generating init_physics...")
#semiinit_state, predictions = step_fn(init_state, 0)

#init_state.prog    = predictions.dynamics
#init_state.phydata = predictions.physics 
#print("semiinit_state.prog.u_wind.shape = ", semiinit_state.prog.u_wind.shape)
#print("init_state.phydata.shortwave_rad.fsol.shape = ", init_state.phydata.shortwave_rad.fsol.shape)

print("Jitting step_fn...")
step_fn = jax.jit(step_fn)

print("Scanning...")
final_state, predictions = jax.lax.scan(
    step_fn,
    init_state,
    xs=jnp.arange(total_steps),
)




