from acc import ACCSetup
import jax

veros_model = ACCSetup()
veros_model.setup()

@jax.jit
def step_function(state, step):
    veros_model.step(state)
    return state, None


print("Calling step_function")
step_function(veros_model.state, 1)
