# Examples

Every example here is either one `python -m jem.main` command, or a notebook
built entirely in Python -- there is no Hydra in any notebook (issue #131).
Python is JAX-ESM's primary interface; see `docs/source/python_api.md` for
the complete direct construction the notebooks below are built from, and for
`jem.configurations` -- the *recipe door* that loads one of the validated
`jem/config/configuration/*.yaml` (the same recipe `+configuration=<name>`
composes) as built Python objects, with no Hydra visible to the caller:

```python
from jem import configurations

exp = configurations.load("aquaplanet-slab")   # a jem.base.coupler.Coupler
exp.coupler, exp.config, exp.run_kwargs        # built, plain-dict, run_chunked() kwargs
```

A notebook that *teaches* how a coupled model is put together builds its
components directly (`01_aquaplanet.ipynb`,
`04_jcm_slabs_mixed_grid_aqua_planet.ipynb`); a notebook that *runs* a
validated configuration to demonstrate something else -- perturbing an
initial condition, differentiating through a trajectory -- loads it through
the door instead of copying its settings into Python, which is exactly the
drift the door exists to prevent.

| Example | How to run it |
| --- | --- |
| Aquaplanet (`01_basic/01_aquaplanet.ipynb`) | notebook: direct construction (atmosphere, slab ocean, slab sea ice); equivalent to `python -m jem.main +configuration=aquaplanet-slab` |
| Custom initial SST (`01_basic/02_aquaplanet_customized_initial_condition.ipynb`) | notebook: `configurations.load("aquaplanet-slab")`, then one field of the initial carry is replaced before the run |
| Response to an SST bump (`01_basic/03_aquaplanet_response_to_SST_perturbation_using_gradient.ipynb`) | notebook: `configurations.load("aquaplanet-slab", seaice="none")`, then `jax.jvp` through the coupled trajectory |
| Mixed grid (`01_basic/04_jcm_slabs_mixed_grid_aqua_planet.ipynb`) | notebook: direct construction on a displaced-pole ocean grid with ESMF regridding; equivalent to `python -m jem.main +configuration=aquaplanet-slab-mixed-grid` |
| Earth-like (`02_experimental/01_earth.ipynb`) | notebook: `configurations.load("earth-slab")`; equivalent to `python -m jem.main +configuration=earth-slab` |
| Long aquaplanet at T106 | `python -m jem.main +configuration=aquaplanet-slab coupled_run=long_run atmosphere.grid.spectral_truncation=106 atmosphere.run.time_step=10` |
| Double drake with a Veros ocean | `python -m jem.main +configuration=veros-double-drake` |
| Earth with a Veros ocean | `python -m jem.main +configuration=veros-earth` |
| Coupled springs (`03_non_geoscience/01_SpringSystem.ipynb`) | notebook: the coupler with no climate in it at all |

The `python -m jem.main` commands above remain exactly valid; the "How to run
it" column says how each *notebook* gets to the same model, since none of
them compose Hydra any more.

jax-gcm ships no T106 grid option, so the resolution above is raised with the
grid's own keys, `atmosphere.grid.spectral_truncation=106`, paired with a
10-minute atmosphere timestep (`atmosphere.run.time_step=10`). A long run
needs two things the short examples do not: `coupled_run.total_time=...` must
be a whole multiple of `coupled_run.chunk` (30 days under `long_run`), and the
run checkpoints into its Hydra output directory by default, so it can be
resumed by pointing a second launch at the first's `coupled_run.output_dir`.

Every command above writes `outputs/<date>/<time>/<component>-<first coupled
step>.nc` plus a `checkpoint/` directory into that same run directory, unless
told otherwise with `coupled_run.output_dir=...`.

`coupled_run=short_run` is the two-day version of any of the commands above,
for checking that a configuration composes, builds, steps and writes output
on a given machine before committing to a real integration.

`python -m jem.main --help` lists every configuration group and its options.

## Running the notebooks

`./examples/run_notebooks.sh [PATH ...]` executes the notebooks in place (all
of them if no path is given). Plotting needs the `plot` extra:

```bash
pip install -e ".[plot]"
```

## Committing a notebook

Notebooks are committed with their outputs cleared:

```bash
./examples/clean_notebooks.sh      # clear outputs and execution counts in place
./examples/clean_notebooks.sh -c   # check only; exits 1 if a notebook is not clean
```
