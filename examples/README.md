# Examples

Every example here (the table below lists all of them, the eight named
configurations among them) is either one `python -m jem.main` command, or a notebook
built entirely in Python -- there is no Hydra in any notebook (issue #131).
Python is JAX-ESM's primary interface; see `docs/source/python_api.md` for
the complete direct construction the notebooks below are built from, and for
`jem.configurations` -- the *recipe door* that loads one of the validated
`jem/config/configuration/*.yaml` (the same recipe `+configuration=<name>`
composes) as built Python objects, with no Hydra visible to the caller:

```python
from jem import configurations

exp = configurations.load("aquaplanet-slab")   # a jem.configurations.LoadedConfiguration
exp.coupler, exp.config, exp.run_kwargs        # built Coupler, plain-dict config, run_chunked() kwargs
```

A notebook that *teaches* how a coupled model is put together builds its
components directly (`01_aquaplanet.ipynb`,
`02_aquaplanet_customized_initial_condition.ipynb`,
`04_jcm_slabs_mixed_grid_aqua_planet.ipynb`). Of those, 01 and 04 take their
wiring from `default_exchangers`, the standard exchange table, and 02 writes
the same exchange out as a plain function, so what passes between the
components is on the page. A notebook that *runs* a validated configuration to
demonstrate something else -- differentiating through a trajectory, an
Earth-like climate -- loads it through the door instead of copying its
settings into Python, which is exactly the drift the door exists to prevent.

| Example | How to run it |
| --- | --- |
| Aquaplanet (`01_basic/01_aquaplanet.ipynb`) | notebook: direct construction (atmosphere, slab ocean, slab sea ice) with SPEEDY's reference parameters; the same coupled structure as `python -m jem.main +configuration=aquaplanet-slab`, which runs jax-gcm's calibrated SPEEDY |
| Custom initial SST, hand-written exchange (`01_basic/02_aquaplanet_customized_initial_condition.ipynb`) | notebook: direct construction of the same aquaplanet as 01 with the exchange written as a plain function in place of `default_exchangers`, then one field of the initial carry is replaced before the run |
| Response to an SST bump (`01_basic/03_aquaplanet_response_to_SST_perturbation_using_gradient.ipynb`) | notebook: `configurations.load("aquaplanet-slab", seaice="none")`, then `jax.jvp` through the coupled trajectory |
| Mixed grid (`01_basic/04_jcm_slabs_mixed_grid_aqua_planet.ipynb`) | notebook: direct construction of an all-ocean aquaplanet with the ocean and sea ice on a displaced-pole grid, with ESMF regridding, at SPEEDY's reference parameters; the same coupled structure as `python -m jem.main +configuration=aquaplanet-slab-mixed-grid`, which runs jax-gcm's calibrated SPEEDY |
| Earth-like (`02_experimental/01_earth.ipynb`) | notebook: `configurations.load("earth-slab")`; equivalent to `python -m jem.main +configuration=earth-slab` |
| Aquaplanet with ECHAM physics | `python -m jem.main +configuration=aquaplanet-slab-echam` (jax-gcm's `t63-echam-1m` atmosphere, T63 with 47 levels, over the free slab ocean; about half an hour per simulated day on 16 CPU threads, and the first run fetches jax-gcm's data mirror) |
| Earth-like with ECHAM physics | `python -m jem.main +configuration=earth-slab-echam` (jax-gcm's `t63-echam-1m` configuration unchanged, over a slab ocean relaxed to its forcing file's SST; append `configuration@atmosphere=t63-echam-2m` for another jax-gcm atmosphere) |
| Earth-like with Winton sea ice | `python -m jem.main +configuration=earth-slab-winton` (`earth-slab`, so calibrated SPEEDY at the 30-minute step, with the thermodynamic Winton ice and a freezing-clamped ocean) |
| Earth-like with JEM's slab land | `python -m jem.main +configuration=earth-slab land=slab` (`earth-slab` with the land's temperature, snow and soil water evolved by `SlabLandModel`; the one option also switches SPEEDY's own land off) |
| Long aquaplanet at T106 | `python -m jem.main +configuration=aquaplanet-slab coupled_run=long_run atmosphere.grid.spectral_truncation=106 atmosphere.run.time_step=10` |
| Double drake with a Veros ocean | `python -m jem.main +configuration=veros-double-drake` |
| Earth with a Veros ocean and the Winton sea ice | `python -m jem.main +configuration=veros-earth` (the thermodynamic Winton ice on the ocean's grid, a freezing-clamped ocean, and SPEEDY's coupled-calibrated parameters) |
| Coupled springs (`03_non_geoscience/01_SpringSystem.ipynb`) | notebook: the coupler with no climate in it at all |

The `python -m jem.main` commands above remain exactly valid; the "How to run
it" column says how each *notebook* gets to the same model, since none of
them compose Hydra any more. The three directly-built notebooks (01, 02, 04)
build the same coupled structure as their configuration, `dt` included: a plain
`jcm.model.Model(...)` takes SPEEDY T31L8's stable 30-minute step, which is the
step the configurations run at.
`tests/unit/test_notebook_equivalence.py` checks that structurally, and that
02's hand-written exchange gives the same two coupled steps as the standard
table. They do
not build the same atmosphere parameters. The SPEEDY configurations (all but the two `-echam`) run jax-gcm's SPEEDY parameters calibrated for coupled use
(its `speedy-t31` bundle; jax-gcm's `docs/source/design/speedy_retune.md`),
while a Python-built `Model(...)` runs SPEEDY's reference values. The
notebooks that load a configuration (03, `02_experimental/01_earth`) run
the calibrated set; 01, 02 and 04 are teaching builds of the wiring and make no
claim to a calibrated or realistic climate. The calibration was a fit at T31
against observations with prescribed SSTs, so a coupled run is not guaranteed
drift-free, and it does not carry to the T106 command above.

jax-gcm ships T106 grid options only for ECHAM physics
(`echam_t106_l47_hybrid`, `echam_t106_l95_hybrid`); it has no SPEEDY T106
option, so the SPEEDY resolution above is raised with the grid's own key,
`atmosphere.grid.spectral_truncation=106`, paired with a 10-minute atmosphere
timestep (`atmosphere.run.time_step=10`). A long run checkpoints into its Hydra
output directory by default, so it can be resumed by pointing a second launch
at the first's `coupled_run.output_dir`. The run length need not be a multiple
of `coupled_run.chunk` (30 days under `long_run`): a last short chunk finishes
it.

Every command above writes `outputs/<date>/<time>/<component>-<first coupled
step>.nc` plus a `checkpoint/` directory into that same run directory, unless
told otherwise with `coupled_run.output_dir=...`.

`coupled_run=short_run` is the two-day version of any of the commands above,
for checking that a configuration composes, builds, steps and writes output
on a given machine before committing to a real integration.

`python -m jem.main --help` explains the configuration groups, how the
atmosphere's groups are re-rooted under `atmosphere`, and worked overrides; the
options of a group are the files in `jem/config/<group>/`, and
`python -m jem.main +configuration=<name> --cfg job` prints a configuration
fully composed. `jem <overrides>` is the same command as `python -m jem.main
<overrides>`.

## Running the notebooks

`./examples/run_notebooks.sh [PATH ...]` executes the notebooks in place (all
of them if no path is given). Plotting needs the `plot` extra, and running a
notebook needs a Jupyter runtime (`ipykernel`, and `nbconvert` for the script),
which no extra provides:

```bash
pip install -e ".[plot]" jupyter
```

## Committing a notebook

Notebooks are committed with their outputs cleared:

```bash
./examples/clean_notebooks.sh      # clear outputs and execution counts in place
./examples/clean_notebooks.sh -c   # check only; exits 1 if a notebook is not clean
```
