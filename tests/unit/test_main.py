"""Tests for :mod:`jem.main` -- the command line around one coupled run.

The composition and the building are :mod:`jem.runners`' (and are tested in
``test_runners.py``); what is left here is the thin shell around them, and the
one thing only the shell can get wrong: the **exit status**. A run the health
gate stopped has to leave a status a scheduler can see, so these call ``main``
with ``runners.run`` stubbed out and look at what comes back.

``@hydra.main`` passes a config straight through to the task function when it
is called with one, so no command line, no run directory and no model are
involved.
"""

import logging
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
from omegaconf import OmegaConf

from jem import runners
from jem.driver import RunResult
from jem.main import main


@pytest.fixture
def cfg():
    """Return the only part of a composed config that ``main`` itself reads."""
    return OmegaConf.create({"coupled_run": {"log_level": "INFO"}})


def stub_run(result: RunResult):
    """Return a ``runners.run`` replacement that returns ``result``."""
    def run(cfg):
        del cfg
        return result
    return run


def test_main_returns_when_the_run_completed(cfg, monkeypatch):
    """A run that reached `total_time` exits 0, i.e. returns without raising."""
    monkeypatch.setattr(
        runners, "run", stub_run(RunResult(None, 4, True, [{"chunk": 0}], []))
    )
    assert main(cfg) is None


def test_main_exits_non_zero_when_the_health_gate_stops_the_run(
    cfg, monkeypatch, caplog
):
    """A run stopped by the gate is a failure a scheduler can see.

    The output written so far is kept and the reason is logged, but the
    command must not exit 0: a `&&`, a queue system or a CI job has nothing
    else to go on.
    """
    report = {"chunk": 2, "elapsed_days": 6.0, "reason": "temperature"}
    monkeypatch.setattr(
        runners, "run", stub_run(RunResult(None, 6, False, [report], []))
    )

    with caplog.at_level(logging.ERROR, logger="jem.main"):
        with pytest.raises(SystemExit) as stopped:
            main(cfg)

    assert stopped.value.code == 1
    assert "health gate" in caplog.text
    # The last report is what says why, so it is in the message.
    assert "temperature" in caplog.text


def test_a_stopped_run_exits_one_as_a_process(tmp_path):
    """The status reaches the shell, not just the Python function.

    `SystemExit` has to survive `@hydra.main`'s own error handling to become
    the process's exit status, and that is the only part a scheduler sees. A
    subprocess is the only way to check it; `runners.run` is stubbed out so
    no model is built and the composition is all that really runs.
    """
    repository = pathlib.Path(__file__).resolve().parents[2]
    program = textwrap.dedent(
        """
        from jem import runners
        from jem.driver import RunResult

        # Stopped by the health gate after three coupled steps.
        runners.run = lambda cfg: RunResult(None, 3, False, [{"chunk": 1}], [])

        from jem.main import main
        main()
        """
    )
    environment = dict(os.environ, JAX_PLATFORMS="cpu")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    finished = subprocess.run(
        [sys.executable, "-c", program,
         "+configuration=aquaplanet-slab", "coupled_run=smoke"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=600,
    )
    assert finished.returncode == 1, finished.stdout[-2000:] + finished.stderr[-2000:]


def test_main_sets_the_package_log_level_from_the_config(cfg, monkeypatch):
    """`coupled_run.log_level` configures the `jem` logger, not the root one.

    The level is process-global state, so it is put back afterwards: a test
    that left the package logger at WARNING would silence every later test
    that reads its own log output.
    """
    monkeypatch.setattr(runners, "run", stub_run(RunResult(None, 0, True, [], [])))
    package_logger = logging.getLogger("jem")
    restore = package_logger.level
    cfg.coupled_run.log_level = "WARNING"
    try:
        main(cfg)
        assert package_logger.level == logging.WARNING
    finally:
        package_logger.setLevel(restore)
