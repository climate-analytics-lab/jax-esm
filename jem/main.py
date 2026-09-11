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

Exit status
-----------
``0`` when the run reached the time it was asked for, ``1`` when the health
gate stopped it early, and whatever Hydra reports for a configuration or
build error. A run that stops early is a *failure* as far as a scheduler,
a shell ``&&`` or a CI job is concerned -- the output it wrote is kept and
the reason is logged, but the command must not look like it succeeded.

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
    """Run one coupled simulation, configured entirely by ``cfg``.

    Returns normally when the run completed, and raises ``SystemExit(1)``
    when the health gate stopped it early; see the module docstring.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The composed config, as Hydra hands it over.

    Raises
    ------
    SystemExit
        With code 1 if :attr:`jem.driver.RunResult.completed` is False.

    """
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
        # The only thing a scheduler, a `&&` or a CI job can see is the exit
        # status, so a run the gate stopped must not exit 0. The output and
        # the checkpoint written so far are kept; this only reports that the
        # run did not get to the end.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
