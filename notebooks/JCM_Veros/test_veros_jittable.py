import jax
from tqdm import tqdm

from veros import runtime_settings

setattr(runtime_settings, "backend", "jax")
setattr(runtime_settings, "force_overwrite", True)

from acc import ACCSetup

veros_object = ACCSetup()
veros_object.setup()

with veros_object.state.settings.unlock() :
    veros_object.state.settings.enable_eke = False

#with veros_object.state.variables.unlock() :
#     veros_object.state.variables.r_bot += 1e-5
#     veros_object.state.variables.K_gm_0 += 1000.0

@jax.jit
def step_function(state, step):
    """
        Convert the state function into a "pure step" copying the input state
    """
    n_state = state#.copy()
    veros_object.step(n_state)  # This is a function that modifies state object inplace
    return n_state

warmup_steps = 2
state = veros_object.state 

for step in tqdm(range(warmup_steps)) :
    state = step_function(state, step)

