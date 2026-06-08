# Diagnostic: localize where the SST-perturbation tangent vanishes.
#
# Stage 0 perturbs SST and reads it back without stepping the ocean at all.
# Stage 1 runs the perturbed state through a single ocean coupling-step.
# Stage 2 chains several ocean-only coupling-steps (no atmosphere feedback).
#
# Comparing max|tangent| across these stages tells us whether the signal is
# lost immediately (stage 0), within one ocean step (stage 1), or only fades
# out gradually under repeated stepping (stage 2) -- e.g. due to diffusion.
import jax
import jax.numpy as jnp
import jax_datetime as jdt

from model_setup import build_model, get_ocean_surface_temperature, set_ocean_surface_temperature

start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
truncation_number = 31
calendar = "365_day"

model, config = build_model(
    truncation_number=truncation_number,
    start_datetime=start_datetime,
    coupling_timestep=coupling_timestep,
    calendar=calendar,
)
ocn_model = model.components["ocn"].raw_component
ocn_step_function = ocn_model.generate_step_function()

initial_coupled_carry = model.initialize()
ocn_carry0 = initial_coupled_carry["ocn"]

sst_initial = get_ocean_surface_temperature(ocn_carry0["state"])
shape2D = sst_initial.shape
tangent_sst_initial = jnp.zeros_like(sst_initial).at[shape2D[0]//2, shape2D[1]//2].set(1.0)


def report(name, tangent):
    tangent = jnp.asarray(tangent)
    print(
        f"[{name}] max|tangent| = {float(jnp.max(jnp.abs(tangent))):.6e}, "
        f"sum|tangent| = {float(jnp.sum(jnp.abs(tangent))):.6e}, "
        f"any nonzero = {bool(jnp.any(tangent != 0.0))}"
    )


def perturb(sst):
    ocn_state = jax.tree_util.tree_map(lambda x: x, ocn_carry0["state"])
    return set_ocean_surface_temperature(ocn_state, sst)


# Stage 0: write the perturbation and read it straight back (no stepping)
def stage0(sst):
    return get_ocean_surface_temperature(perturb(sst))


print("Computing stage 0 (perturb + read back, no stepping)...")
_, t0 = jax.jvp(stage0, (sst_initial,), (tangent_sst_initial,))
report("stage0: write-only", t0)


# Stage 1: a single ocean coupling-step
@jax.jit
def stage1(sst):
    perturbed_carry = dict(ocn_carry0, state=perturb(sst))
    new_carry, _ = ocn_step_function(perturbed_carry, 0)
    return get_ocean_surface_temperature(new_carry["state"])


print("Computing stage 1 (one ocean coupling-step)...")
_, t1 = jax.jvp(stage1, (sst_initial,), (tangent_sst_initial,))
report("stage1: one ocean coupling-step", t1)


# Stage 2: five ocean-only coupling-steps chained (no atmosphere)
@jax.jit
def stage2(sst):
    carry = dict(ocn_carry0, state=perturb(sst))
    for step in range(5):
        carry, _ = ocn_step_function(carry, step)
    return get_ocean_surface_temperature(carry["state"])


print("Computing stage 2 (five ocean-only coupling-steps)...")
_, t2 = jax.jvp(stage2, (sst_initial,), (tangent_sst_initial,))
report("stage2: five ocean-only coupling-steps", t2)

print("Done.")
