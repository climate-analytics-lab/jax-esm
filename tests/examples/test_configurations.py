"""Every named configuration composes, builds, steps and writes output.

This is the smoke test that closes the loop `docs/source/experimental.rst`
and every configuration's own WHY comment promise: ``python -m jem.main
+configuration=<name>`` is a runnable command, not just a file that
composes. It runs under `tests/examples` -- slow, and reaching real
packaged data and (for the `veros-*` configurations) an optional dependency
-- so CI runs it on pull requests only, not on every push.

``CONFIGURATIONS`` is discovered from ``jem/config/configuration/*.yaml``
rather than listed by hand, so a newly added configuration is covered
automatically the next time this runs.

A configuration whose atmosphere is finer than T31 is composed here but not
integrated: the ECHAM configurations run jax-gcm's T63 L47 atmosphere, which
takes minutes per simulated day on a CPU and reads jax-gcm's data mirror, so
their two days would cost an hour of a pull-request job and a network the job
may not have. Their plumbing is integrated once, on jax-gcm's smallest ECHAM
grid, by ``tests/unit/test_echam_configuration.py``.
"""

import importlib.util
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module

import jem.config
import jem.runners
from jem.output import output_file_step

CONFIG_MODULE = "jem.config"

#: The finest atmosphere a smoke run integrates (see the module docstring).
SMOKE_RUN_TRUNCATION = 31

CONFIGURATIONS = sorted(
    p.stem for p in (Path(jem.config.__file__).parent / "configuration").glob("*.yaml")
)


@pytest.mark.slow
@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_configuration_runs(configuration, tmp_path):
    """Compose, build and integrate two coupled days of ``configuration``."""
    if configuration.startswith("veros-") and importlib.util.find_spec("veros") is None:
        pytest.skip(f"{configuration!r} needs the optional `veros` dependency")

    with initialize_config_module(config_module=CONFIG_MODULE, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"+configuration={configuration}",
                "coupled_run=short_run",
                f"coupled_run.output_dir={tmp_path}",
            ],
        )

    if cfg.atmosphere.grid.spectral_truncation > SMOKE_RUN_TRUNCATION:
        pytest.skip(
            f"{configuration!r} runs a T{cfg.atmosphere.grid.spectral_truncation} "
            "atmosphere; see this module's docstring"
        )

    result = jem.runners.run(cfg)

    assert result.completed, (configuration, result.reports)
    assert result.steps_completed == 2

    written = {path.name for path in result.paths}
    component_names = list(result.final_carry.components)
    for name in component_names:
        matches = [path for path in result.paths if path.stem.startswith(f"{name}-")]
        assert matches, (
            f"{configuration!r}: no output file for component {name!r} "
            f"among {sorted(written)}"
        )
        for path in matches:
            # `output_file_step` parses the step back out of the name
            # `write_chunk` gave the file, so this also proves the file is
            # named the way the rest of the output layer expects to read it
            # back, not merely present.
            assert path.exists()
            assert output_file_step(path, component_names) is not None
