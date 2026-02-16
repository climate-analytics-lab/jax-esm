# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# %% [markdown]
# # JEM application: Spring System
#
# %%
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from dataclasses import dataclass
import tree_math

from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools

# %% [markdown]
# # Define Model
# %%
@tree_math.struct
@dataclass
class SpringCarry:
    t: ArrayLike  # time
    x: ArrayLike  # position
    v: ArrayLike  # velocity
    m: ArrayLike  # mass
    k: ArrayLike  # spring coefficient
    f: ArrayLike  # external force

class Spring:

    def __init__(self, init_x, init_v, k, m, dt):
        self.init_x = init_x
        self.init_v = init_v
        self.k = k
        self.m = m
        self.dt = dt

    def initialize(self):
        return SpringCarry(
            t = jnp.array(0),
            x = jnp.array(self.init_x),
            v = jnp.array(self.init_v),
            m = jnp.array(self.m),
            k = jnp.array(self.k),
            f = jnp.array(0),
        )

    def generate_step_function(self):
        dt = self.dt
        def step_function(carry, step):
            """Integrates one time step of a harmonic oscillator."""

            # Physics: a = -k/m * x + f
            acceleration = - (carry.k * carry.x + carry.f) / carry.m

            # Update state (Semi-implicit Euler for better stability)
            new_v = carry.v + acceleration * dt
            new_x = carry.x + new_v * dt

            carry.v = new_v
            carry.x = new_x
            carry.t += dt

            return carry, dict(t=carry.t, x=carry.x, v=carry.v)
        return step_function

# %% [markdown]
# # Define Interaction

# %%
interaction_strength = 1.0
def mapper(coupled_carry):
    f = (coupled_carry["spring2"].x - coupled_carry["spring1"].x) * interaction_strength
    coupled_carry["spring1"].f = f
    coupled_carry["spring2"].f = - f
    return coupled_carry

# %% [markdown]
# # Create Model

# %%
total_time = 50
dt = 0.01
model = Coupler(
    components=dict(
        spring1=Spring(init_x=0, init_v=2, k=5.0, m=1.0, dt=dt),
        spring2=Spring(init_x=2, init_v=5, k=5.0, m=5.0, dt=dt),
    ),
    mappers=dict(mapper=mapper),
)

# %% [markdown]
# # Run Model
# %%
initial_coupled_carry, final_coupled_carry, predictions = model.run(
    workflow=["mapper", "spring1", "spring2"],
    iterations = int(total_time / dt),
)


# %% [markdown]
# # Display
# %%
import matplotlib.pyplot as plt

x1 = predictions["spring1"]["x"]
v1 = predictions["spring1"]["v"]
x2 = predictions["spring2"]["x"]
v2 = predictions["spring2"]["v"]
t = predictions["spring1"]["t"]

fig, ax = plt.subplots(2,1)
ax[0].plot(t, x1, label="x1")
ax[0].plot(t, x2, label="x2")
ax[1].plot(t, v1, label="v1")
ax[1].plot(t, v2, label="v2")

ax[0].legend()
ax[1].legend()

fig, ax = plt.subplots(2,1)
ax[0].plot(x1, x2, label="x")
ax[1].plot(v1, v2, label="v")

ax[0].legend()
ax[1].legend()

plt.show()
