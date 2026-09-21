# Examples

Every example here is either one `python -m jem.main` command or a notebook
that does one thing the command line cannot -- customise an initial condition,
or differentiate through a coupled trajectory.

| Example | How to run it |
| --- | --- |
| Aquaplanet (`01_basic/01_aquaplanet.ipynb`) | `python -m jem.main +configuration=aquaplanet-slab` |
| Custom initial SST (`01_basic/02_aquaplanet_customized_initial_condition.ipynb`) | notebook: one field of the initial carry is replaced before the run |
| Response to an SST bump (`01_basic/03_aquaplanet_response_to_SST_perturbation_using_gradient.ipynb`) | notebook: `jax.jvp` through the coupled trajectory |
| Mixed grid (`01_basic/04_jcm_slabs_mixed_grid_aqua_planet.ipynb`) | `python -m jem.main +configuration=aquaplanet-slab-mixed-grid` |
| Earth-like (`02_experimental/01_earth.ipynb`) | `python -m jem.main +configuration=earth-slab` |
| Long aquaplanet at T106 | `python -m jem.main +configuration=aquaplanet-slab coupled_run=long_run atmosphere.grid.spectral_truncation=106 atmosphere.run.time_step=10` |
| Double drake with a Veros ocean | `python -m jem.main +configuration=veros-double-drake` |
| Earth with a Veros ocean | `python -m jem.main +configuration=veros-earth` |
| Coupled springs (`03_non_geoscience/01_SpringSystem.ipynb`) | notebook: the coupler with no climate in it at all |

The plan this repository followed for the long-aquaplanet row named a
`grid@atmosphere.grid=speedy_t106_l8` option that does not exist -- jax-gcm
ships no `speedy_t106_l8` grid (`/home/user/jax-gcm-dev/jcm/config/grid/` has
only `speedy_t31_l8`, `held_suarez_t31_l8` and the ECHAM grids). The
resolution is raised by overriding the grid's own keys instead,
`atmosphere.grid.spectral_truncation=106`, with `atmosphere.run.time_step=10`
(minutes) as the deleted `03_long_aquaplanet.py` script used. A long run needs
two things the short examples do not: `coupled_run.total_time=...` must be a
whole multiple of `coupled_run.chunk` (30 days under `long_run`), and the run
checkpoints into its Hydra output directory by default, so it can be resumed
by pointing a second launch at the first's `coupled_run.output_dir`.

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
