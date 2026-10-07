# Differentiating long, high-resolution coupled runs

`jax.grad` of a coupled trajectory is the project's reason for being, and for
a T31 atmosphere over a slab ocean it is one line:
`jax.grad(objective)(coupler.generate_trajectory_function(n, remat=True))`.
This page records what changes when the atmosphere is T255 and the ocean is a
one-degree Veros, and the gradient is wanted over a forecast of a week or
two. The worked case throughout is a T255 SPEEDY atmosphere started from
ERA5 on 24 December 2022, coupled hourly to the one-degree
`jem.components.veros.setups.global_latlon` ocean started from Levitus and
that day's ERA5 SST, and the gradient of the five-day rainfall over the ocean
off the US West Coast seven to twelve days later.

## Memory: what a checkpoint holds

`remat=True` stores one coupled carry per step on the device and recomputes
everything inside the step. At T255L8 coupled hourly to a 360 x 160 x 25 Veros
ocean in double precision the carry is 2.6 GB (the ocean's `VerosState` is
1.4 GB of it, the atmosphere's state, physics carry and derived fields the
rest), and a twelve-day sensitivity has 288 steps: about 750 GB.

Two things bring that within reach.

**The forcing window.** jax-gcm keeps a boundary condition read from a file
as a `TimeSeries` holding every record, and JEM keeps the atmosphere's forcing
in its carry because exchangers write parts of it. jax-gcm's daily T255
climatology is five `(365, 768, 384)` fields: 4.3 GB, stored at every
checkpoint of a run that reads a fortnight of it.
`jem.components.jcm.forcing_window.restrict_forcing_to_window` re-expresses a
daily or monthly climatology as a by-date series of one record per midnight
in the window, and cuts a dated series to the records that bracket it. A
daily climatology is selected piecewise-constant by nominal date, so the
by-date series selects exactly the same record at every step;
`tests/unit/test_forcing_window.py` checks this hour by hour against
`ForcingData.select`.

**Host checkpoints.** `jem.adjoint.checkpointed_value_and_grad` is two-level
checkpointing with the outer level in host memory: a forward run that copies
the carry to the host at the start of each block, then, block by block from
the last, a recomputation of the block on the device and a `jax.vjp` of one
coupled step at a time, chaining the cotangent backwards. The device holds
`block_size` carries plus one step's VJP (for the case above: 50 GB peak with
six-hour blocks of hourly steps, against 80 GB; four-hour blocks leave more
headroom on a shared GPU); the host holds one carry per block (130 GB for
six-hour blocks). Each step is one compiled program called repeatedly, so a
288-step run compiles a step and a step's VJP once each -- unrolling the steps
inside one `jit` instead takes tens of minutes to compile and, without
`remat`, asks for hundreds of gigabytes. The objective is a weighted sum over
steps, so a window mean is a weight vector, and one reverse sweep gives the
gradient with respect to the carry at every block start: the sensitivity at
every lead time, delivered through the `on_block_gradient` callback.

Two details of Veros' pytrees shaped the implementation.
`VerosVariables.tree_unflatten` casts every leaf back to its declared dtype
with `jnp.asarray`: unflattening a host copy puts it straight back on the
device (so the host copies are flat lists of numpy leaves, never a rebuilt
carry), and a `float0` cotangent for an integer leaf cannot be cast at all (so
the step is differentiated with respect to the floating-point leaves only,
with masks and step counters passed through). And `VerosVariables` is
mutable: perturbing a carry's ocean state in place perturbs every carry that
shares it, so a perturbation experiment copies the state first
(`jax.tree.map(lambda a: a, state)`).

## Switches: smoothing the derivative, not the model

A reference formulation is full of clips and switches, and a few of them have
a derivative that is enormous in a window far narrower than any perturbation
of interest and zero outside it. jax-gcm's convention for these
(`jcm.physics.surrogate_gradient.with_surrogate_gradient`, design page
`surrogate_gradients.md`) is to keep the value bit for bit and take the
derivative from a smooth surrogate, as operational 4D-Var does with its
linearised physics.

The one that matters in this coupled system is in Veros' TKE closure. The
Prandtl number is `clip(6.6 Ri, 1, 10)` with `Ri = N^2 / max(shear^2,
1e-12)`. In a column with no shear -- an ocean started from rest, or any
quiescent column -- that is a switch from convective (1) to stable (10) as
`N^2` crosses a window about `1e-12 s^-2` wide, and convecting columns sit at
`N^2` near zero. Differentiating it as written gave ocean sensitivities four
to five orders of magnitude above their neighbours at isolated cells (the
Yucatan channel, off Japan, near Mindanao); a finite-difference check in an
ocean-only run reproduced them only for perturbations of `1e-4` K, and a 0.01 K
perturbation responded five times less. Freezing the Prandtl number in the
reverse pass removed them, which located the switch. The fix, in the Veros
fork (`veros.core.tke.prandtl_number`), keeps the reference value and takes the
derivative from the same formula with the shear floor raised smoothly to
`settings.tke_prandtl_surrogate_shear_floor` (default `1e-7 s^-2`, a weak
current shear of a few cm/s per 100 m) and the clip rounded over
`tke_prandtl_surrogate_width` (0.5). With it the adjoint reproduces the
response to a 0.01 K perturbation at the former spike cells and is unchanged
at every ordinary cell; `1e-8` was too narrow to matter and `1e-6` smoothed
the real convective response away. The fork's earlier
`sqrt(max(x, eps))` regularisation of the closures' square roots turned out not
to be the source of the spikes: a surrogate there changed nothing.

The forward model is unchanged by the surrogate up to floating-point
reassociation (the compiled program fuses differently): in an ocean-only run
the objective agrees to `1e-12`. In a coupled run a week long, that last-bit
difference grows like any other perturbation, which is the next point.

## Chaos: what the gradient of a forecast a week out means

The atmosphere's adjoint grows by about `e` per day backwards in time at
T255 with a six-minute step: the growth rate of errors in a moist
high-resolution atmosphere. The gradient a week back is the correct
derivative of the model, but it describes the fastest-growing atmospheric
modes, and the linear regime around the forecast is far smaller than an
observation error. In the worked case (five-day rainfall `J` = 3.97
mm/day; a +/-0.5 K change of the upstream mixed layer):

| Lead | Perturbed runs | Gradient | 0.01 K random noise alone |
|---|---|---|---|
| 1 day | +0.075 mm/day | +0.052 mm/day | 0.017 mm/day |
| 7 days | +0.083 +/- 0.027 mm/day (3 members) | +651 mm/day | 0.099 mm/day rms |

A day ahead the gradient is a usable stand-in for perturbation runs; a week
ahead it is not, and the deterministic ocean effect is an ensemble statement
-- of the same size as the chaotic spread from a 0.01 K perturbation. The
useful horizon is resolution dependent: at T42 with a 30-minute step
jax-gcm's gradients stay informative for nearly two weeks. For a longer lead
at high resolution, the answer to "how much does the ocean matter" is the
mean response of perturbed runs against their spread, or the gradient of an
ensemble-mean objective, not the gradient of one forecast.

## Reproducibility: the last bit is a perturbation too

XLA's GPU scatter-add accumulates in a nondeterministic order by default, and
JEM's regridding is a scatter-add. Two otherwise identical coupled runs then
differ in the last bit after one step, and a week later in `J` by about 1 %
-- the same size as the effects being measured. A perturbation experiment
or a finite-difference check of a coupled gradient therefore needs
`XLA_FLAGS=--xla_gpu_deterministic_ops=true`, and its base and perturbed runs
must go through the same compiled program (a differently fused program
rounds differently): with both, the adjoint's own forward pass reproduces a
separately compiled forecast's `J` exactly.
