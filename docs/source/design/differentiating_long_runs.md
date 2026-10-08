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
fork (`veros.core.tke.prandtl_number`,
[veros-jittable#2](https://github.com/meteorologytoday/veros-jittable/pull/2)), keeps the reference value and takes the
derivative from the same formula with the shear floor raised smoothly to
`settings.tke_prandtl_surrogate_shear_floor` and the clip rounded over
`tke_prandtl_surrogate_width` (0.5).

The same switch also survives, less violently, where the shear is resolved
but weak. Between its clips the Prandtl number is `6.6 N^2 / shear^2`, a
ramp about `1.4 shear^2` wide in `N^2`; with `shear^2` near `1e-7 s^-2` (a
few mm/s across a 10 m layer) the whole switch fits inside the `~2e-6 s^-2`
change in `N^2` that a 0.01 K surface perturbation makes. The derivative
there is the true local slope -- a `1e-4` K finite difference reproduces it --
but it overstates the 0.01 K response five- to tenfold, and a global
one-degree ocean-only adjoint (24 hourly steps, a random-weighted global SST
objective) showed it as isolated spikes after the zero-shear switch was
smoothed. Freezing the Prandtl number again removed them. The floor therefore
has to cover these columns too, and a scan against 0.01 K finite differences
fixed it:

| Floor (s^-2) | max / 99th pct | spike cells: gradient vs 0.01 K FD | change in median, 99th pct |
|---|---|---|---|
| `1e-7` | 25 | 28 vs 3.2; -12 vs -2.0; 9.9 vs 2.4 | (reference) |
| **`3e-7`** (default) | 19 | 2.5 vs 3.2; -0.9 vs -2.0; 5.1 vs 2.4 | 0 %, -1 % |
| `1e-6` | 18 | 0.5 vs 3.2; +0.7 vs -2.0; 0.9 vs 2.4 | 0 %, -5 % |
| `1e-5` | 15 | 0.1 vs 3.2; +0.8 vs -2.0; 0.6 vs 2.4 | -1 %, -7 % |

`3e-7` brings the spike cells to within a factor of about two of the 0.01 K
response without touching ordinary cells; a larger floor discards real
sensitivity (and flips a sign) rather than just the switch. What remains
above the bulk at `3e-7` is not a switch: the largest cells agree in size with a
`1e-4` K finite difference, and the largest of all, in the Arctic, runs
through the mixing length rather than the Prandtl number, and the 0.01 K
response there is larger still. Freezing all the vertical diffusivities would
flatten the gradient further (max / 99th percentile 3.5), but only by deleting
that real sensitivity with the switch.

The fork's earlier
`sqrt(max(x, eps))` regularisation of the closures' square roots turned out not
to be the source of the spikes: a surrogate there changed nothing.

### The isoneutral tensor: a linearisation that is not dissipative

With the Prandtl number smoothed, the coupled adjoint still had isolated ocean
cells 10^3-10^4 times the 99th percentile of the SST sensitivity, at a
different place every day or two (the Andaman Sea, off Cape Hatteras, the
Philippine Sea, the Gulf of Alaska). They did not move with the Prandtl floor
or with jax-gcm's SPEEDY surrogates, and the ocean-only test above never
produced them: they need the realistic, time-varying ocean of the coupled run.
Their signature is not a switch's. Replaying a few hours of the coupled run
from a saved carry and recording the cotangent every step, the cotangent at
one cell grows a thousandfold within three or four hourly steps, *changing
sign every step*, and then collapses -- the behaviour of an unstable
tangent-linear operator, not of one steep function.

Stopping the derivative through one process at a time in those replays located
it. The vertical velocity, the flux limiter of the tracer advection, TKE and
EKE made no difference; holding the TKE diffusivities fixed shrank the spikes
but left them in place; holding the
isoneutral mixing tensor (slopes `Ai_*` and diffusivities `K_11`, `K_22`,
`K_33`) fixed removed it in both windows tested. The slopes go as `1 /
drho/dz`, and in Griffies' triad scheme the slope terms of the flux of a
density-carrying tracer nearly cancel; differentiating the slopes keeps both
halves of that cancellation in the tangent-linear model with nothing to keep
it stable. Smoothing does not rescue it: holding only `K_33` fixed, or
flooring the stratification in the slope derivative at `drho/dz = -1e-4 kg
m^-4` (`N^2 ~ 1e-6 s^-2`), broke the cancellation and gave growth of
10^6-10^8 elsewhere, and a floor of `1e-3 kg m^-4` -- ordinary thermocline
stratification -- was indistinguishable from holding the tensor fixed.

The fork therefore holds the tensor fixed in the derivative
(`veros.core.isoneutral.isoneutral_diffusion_pre`, in the same
[veros-jittable#2](https://github.com/meteorologytoday/veros-jittable/pull/2); the setting
`enable_isoneutral_tensor_derivative` restores the full linearisation). The
value is unchanged and the isoneutral and skew fluxes are still differentiated
with respect to the tracer they mix, so within each step the tangent-linear
model sees a fixed, symmetric positive semi-definite diffusion, as the forward
model does -- the same kind of simplification operational 4D-Var makes when it
neglects perturbations of mixing coefficients in its linearised physics. In
the worked case (seven-day gradient, every six hours):

| | worst SST max / 99th pct | daily SST max / 99th pct | 1-day lead, +/-0.5 K upstream mixed layer |
|---|---|---|---|
| tensor differentiated | 28 988 | 19 - 2 221 | +/-0.060 mm/day |
| **tensor held fixed** | **88** | **7 - 46** | **+/-0.062 mm/day** |
| perturbed runs | | | +0.089 / -0.047 mm/day |

Below the 99th percentile the two gradients correlate at 0.96-0.97 and their
medians agree to 3 %: what is lost is the unstable mode, not the sensitivity.

The forward model is unchanged by either change up to floating-point
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
