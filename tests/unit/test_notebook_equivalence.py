"""Structural equivalence between the directly-built notebooks and the CLI.

``01_aquaplanet.ipynb`` and ``04_jcm_slabs_mixed_grid_aqua_planet.ipynb`` both
claim, in their own markdown, to build "the same model"
``+configuration=aquaplanet-slab`` / ``...-mixed-grid`` composes. This module
executes each notebook's OWN construction cell (not a copy of it, so it
cannot silently drift from what the notebook actually runs) and compares the
resulting :class:`~jem.base.coupler.Coupler` against
:func:`jem.runners.build_coupler` on the equivalent composed config:
component names and types, workflow, coupling timestep, calendar, start date,
the coupled carry's pytree structure, and -- the specific thing a local
review of issue #131 found missing here -- the atmosphere's own physics
timestep (``model.dt_si``). Building a plain ``jcm.model.Model`` with no
explicit ``time_step`` silently adopts the physics' *stable* step (30 minutes
for SPEEDY T31L8) rather than the 12 minutes
``+configuration=aquaplanet-slab`` actually runs at
(``atmosphere.run.time_step``, from jax-gcm's ``run/default.yaml``, which the
recipe never overrides) -- exactly the kind of divergence the other
structural checks (component types, carry shape) cannot see, since a wrong
timestep changes nothing about a carry's *shape*.

Slow: each test builds two full JCM atmospheres (the notebook's own and the
CLI's reference).
"""

import unittest
from pathlib import Path

import jax
import nbformat
import numpy as np
import pytest

from jem import runners
from jem.configurations import _compose

REPO_ROOT = Path(__file__).resolve().parents[2]


def _exec_construction_cell(notebook_path: Path):
    """Execute ``notebook_path``'s own ``coupler = Coupler(`` cell.

    Returns the ``coupler`` it built, by running the cell's actual source
    (via :mod:`nbformat`) in a fresh namespace -- not a hand-copied
    transcription of it, so this cannot drift from what the notebook itself
    does.
    """
    nb = nbformat.read(notebook_path, as_version=4)
    matches = [
        cell.source for cell in nb.cells
        if cell.cell_type == "code" and "coupler = Coupler(" in cell.source
    ]
    assert len(matches) == 1, (
        f"{notebook_path} has {len(matches)} code cells naming "
        "`coupler = Coupler(`; expected exactly one, its construction cell."
    )
    namespace: dict = {"__name__": "__main__"}
    exec(compile(matches[0], str(notebook_path), "exec"), namespace)
    return namespace["coupler"]


def _term_names(coupler) -> dict:
    return {name: type(component).__name__
            for name, component in coupler.components.items()}


def _assert_structurally_equivalent(coupler, ref_coupler) -> None:
    """Assert two couplers describe the same coupled model, ``dt`` included."""
    assert list(coupler.components) == list(ref_coupler.components)
    assert _term_names(coupler) == _term_names(ref_coupler)
    assert coupler.workflow == ref_coupler.workflow
    assert coupler.coupling_timestep == ref_coupler.coupling_timestep
    assert coupler.start_date == ref_coupler.start_date
    assert coupler.calendar == ref_coupler.calendar
    # The atmosphere's own physics timestep -- not implied by anything else
    # checked here, and the specific field this test exists to guard.
    dt = float(coupler.components["atm"].model.dt_si.m)
    ref_dt = float(ref_coupler.components["atm"].model.dt_si.m)
    assert dt == ref_dt, (
        f"atmosphere timestep differs: notebook builds dt={dt}s, "
        f"the configuration runs at dt={ref_dt}s"
    )
    assert (jax.tree_util.tree_structure(coupler.initialize())
            == jax.tree_util.tree_structure(ref_coupler.initialize()))


@pytest.mark.slow
class TestNotebookEquivalence(unittest.TestCase):
    def test_01_aquaplanet_matches_aquaplanet_slab(self):
        """01_aquaplanet.ipynb's direct build matches +configuration=aquaplanet-slab."""
        notebook = REPO_ROOT / "examples" / "01_basic" / "01_aquaplanet.ipynb"
        coupler = _exec_construction_cell(notebook)

        ref_cfg = _compose("aquaplanet-slab", [])
        ref_coupler = runners.build_coupler(ref_cfg)

        _assert_structurally_equivalent(coupler, ref_coupler)

    def test_04_mixed_grid_matches_aquaplanet_slab_mixed_grid(self):
        """04_...mixed_grid...ipynb's direct build matches the mixed-grid recipe."""
        notebook = (REPO_ROOT / "examples" / "01_basic"
                    / "04_jcm_slabs_mixed_grid_aqua_planet.ipynb")
        coupler = _exec_construction_cell(notebook)

        ref_cfg = _compose("aquaplanet-slab-mixed-grid", [])
        ref_coupler = runners.build_coupler(ref_cfg)

        _assert_structurally_equivalent(coupler, ref_coupler)

        # Mixed-grid-specific: the same ocean grid (shape and land/ocean
        # split) and the same set of exchange rows that cross grids.
        ocean_grid = coupler.components["ocn"].grid
        ref_grid = ref_coupler.components["ocn"].grid
        assert ocean_grid.shape == ref_grid.shape
        np.testing.assert_array_equal(
            ocean_grid.fractional_mask, ref_grid.fractional_mask)

        exchange = coupler.exchangers["exchange"]
        ref_exchange = ref_coupler.exchangers["exchange"]
        crossing = {spec.regrid for spec in exchange.specs if spec.regrid is not None}
        ref_crossing = {spec.regrid for spec in ref_exchange.specs
                        if spec.regrid is not None}
        assert crossing == ref_crossing


if __name__ == "__main__":
    unittest.main()
