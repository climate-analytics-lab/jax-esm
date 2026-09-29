# Experimental Setups

Prototypes that are not tested as extensively as `examples/01_basic`. Two
kinds live here:

- **`01_earth.ipynb`** -- a notebook coupling JCM to JEM's own slab
  components on realistic Earth boundary conditions.
- **Two configurations coupling JCM to the Veros ocean GCM.** These are not
  directories to run a script in -- they are named `jem.main` configurations,
  runnable as one command with no setup:

  | Configuration | Command |
  | --- | --- |
  | Double drake with a Veros ocean | `python -m jem.main +configuration=veros-double-drake` |
  | Earth with a Veros ocean | `python -m jem.main +configuration=veros-earth` |

  Veros is an optional dependency; see the main `README.md`'s install steps.
  Both configurations couple the ocean through `jem.fluxes.VerosExchange`
  (see `docs/source/experimental.rst`), and neither has a land or sea-ice
  component.
