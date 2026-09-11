#!/usr/bin/env python
"""Hydra entry point: one coupled run from the command line.

Examples
--------
The shipped aquaplanet, two coupled days, to check a machine can run
anything at all::

    python -m jem.main +configuration=aquaplanet-slab coupled_run=smoke

An Earth-like year, chunked by month, with monthly-mean output::

    python -m jem.main +configuration=earth-slab coupled_run=longrun \
        coupled_run.total_time="1 year"

The atmosphere is configured by jax-gcm's own groups, re-rooted under
``atmosphere`` (so the group's package is spelled out), and everything else by
JAX-ESM's own::

    python -m jem.main physics@atmosphere.physics=held_suarez \
        grid@atmosphere.grid=held_suarez_t31_l8 atmosphere.run.time_step=15 \
        ocean=slab_relax ocean.sst_clim_file=${jcm_data:bc/t30/clim/forcing.nc}

``python -m jem.main --help`` lists the groups and the override spellings;
``--cfg job`` prints the fully composed config without running anything.

"""

import logging

import hydra
from omegaconf import DictConfig

# Importing the config package registers the ${jcm_data:} / ${jem_data:}
# resolvers its YAML uses. `@hydra.main(config_path="config")` reads this
# package's files off disk without importing it, so the import has to be
# here -- a config that names a packaged data file is otherwise composed
# before the resolver that can read it exists.
import jem.config  # noqa: F401
from jem import runners

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one coupled simulation, configured entirely by ``cfg``."""
    # The whole package's logger, not the root one: Hydra already configures
    # the root logger for the job, and `coupled_run.log_level` is about how
    # much JAX-ESM itself says.
    logging.getLogger("jem").setLevel(cfg.coupled_run.log_level)
    result = runners.run(cfg)
    logger.info(
        "Finished: %d coupled steps, completed=%s, %d file(s) written.",
        result.steps_completed, result.completed, len(result.paths),
    )
    if not result.completed:
        logger.error(
            "The run stopped early: the health gate rejected the state at "
            "coupled step %d. The last report was: %s",
            result.steps_completed,
            result.reports[-1] if result.reports else "(none)",
        )


if __name__ == "__main__":
    main()
