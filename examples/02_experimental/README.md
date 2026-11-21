# Experimental Setups

This directory holds one notebook and no scripts: every runnable setup is a
named `jem.main` configuration, runnable as one command. The setups here are
the least validated of them.

- **`01_earth.ipynb`** -- a notebook coupling JCM to JEM's own slab components
  on realistic Earth boundary conditions. It loads the `earth-slab`
  configuration (`configurations.load("earth-slab")`), so it is the notebook
  form of `python -m jem.main +configuration=earth-slab`.
- **Two configurations coupling JCM to the Veros ocean GCM**, which are not
  directories to run a script in:

  | Configuration | Command |
  | --- | --- |
  | Double drake with a Veros ocean | `python -m jem.main +configuration=veros-double-drake` |
  | Earth with a Veros ocean | `python -m jem.main +configuration=veros-earth` |

  Veros is an optional dependency; see the main `README.md`'s install steps.
  Both configurations couple the ocean through `jem.fluxes.VerosExchange`
  (see `docs/source/experimental.rst`), and neither has a land or sea-ice
  component (`docs/source/issues.rst`).

What is tested: `tests/examples/test_configurations.py` composes every named
configuration, `earth-slab` and both Veros ones included, and integrates two
coupled days of each, checking that every component writes output (the Veros
ones are skipped where Veros is not installed). That shows a configuration
builds, steps and writes; it is not a validation of the climate it produces,
and `tests/examples/test_examples.py` executes every example notebook,
`01_earth.ipynb` included, end to end (it skips where the `plot` extra is
missing).
