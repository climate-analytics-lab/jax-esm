"""Structural equivalence between the directly-built notebooks and the CLI.

``01_aquaplanet.ipynb``, ``02_aquaplanet_customized_initial_condition.ipynb``
and ``04_jcm_slabs_mixed_grid_aqua_planet.ipynb`` build their coupled model
directly, and say in their own markdown what it is the same as: the coupled
structure ``+configuration=aquaplanet-slab`` / ``...-mixed-grid`` composes.
``02`` also writes its exchange as a plain function and says that it is the
standard exchange table, which is checked here by running both. This module
executes each notebook's OWN construction cell (not a copy of it, so it
cannot silently drift from what the notebook actually runs) and compares the
resulting :class:`~jem.base.coupler.Coupler` against
:func:`jem.runners.build_coupler` on the equivalent composed config:
component names and types, workflow, coupling timestep, start date, the
coupled carry's pytree structure, and -- the specific field this test exists
to guard -- the atmosphere's own physics timestep (``model.dt_si``). A
plain ``jcm.model.Model`` with no explicit ``time_step`` takes the physics'
stable step (30 minutes for SPEEDY T31L8), and the configurations run the
``speedy-t31`` bundle's own 30 minutes; a wrong timestep changes nothing
about a carry's *shape*, so the other structural checks cannot see it.

What is deliberately NOT compared is the SPEEDY parameters: the notebooks run
the Python defaults (SPEEDY's reference values) and the configurations run
jax-gcm's coupled-calibrated set, which is why the notebooks claim the same
structure and not the same climate.

Slow: each test builds one or two full JCM atmospheres (the notebook's own
and, for a structural comparison, the CLI's reference).
"""

import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import nbformat
import numpy as np
import pytest

from jem import Coupler, default_exchangers, replace_field, runners
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

    def test_02_customized_initial_condition_matches_aquaplanet_slab(self):
        """02's direct build, hand-written exchange included, is aquaplanet-slab's.

        The carry structure is the part a function could get wrong unseen:
        the atmosphere's forcing has to hold the same leaves whether a table
        or a function overwrites two of them.
        """
        notebook = (REPO_ROOT / "examples" / "01_basic"
                    / "02_aquaplanet_customized_initial_condition.ipynb")
        coupler = _exec_construction_cell(notebook)

        ref_cfg = _compose("aquaplanet-slab", [])
        ref_coupler = runners.build_coupler(ref_cfg)

        _assert_structurally_equivalent(coupler, ref_coupler)

    def test_02_hand_written_exchange_is_the_standard_table(self):
        """The function 02 writes out moves what ``default_exchangers`` moves.

        The notebook says its ``exchange`` is the standard table written by
        hand. Two coupled steps from one initial carry, once through the
        notebook's own coupler and once through the same components wired by
        ``default_exchangers``, have to end in the same carry, leaf for leaf.

        The carry starts with polar caps below the ice-free temperature,
        because the notebook's own warm ocean never forms ice: the ice
        fraction would be zero whether or not the function moved it, and the
        comparison would pass with that row missing. Each of the four
        destinations is also required to have changed from its initial
        value, so that no row is compared as zero against zero. What this
        cannot tell apart is the sea tile's flux from the grid-cell mean,
        which are the same field on a planet with no land.
        """
        notebook = (REPO_ROOT / "examples" / "01_basic"
                    / "02_aquaplanet_customized_initial_condition.ipynb")
        by_hand = _exec_construction_cell(notebook)
        assert not hasattr(by_hand.exchangers["exchange"], "specs"), (
            "02 is the notebook that writes its exchange as a plain function"
        )
        components = by_hand.components
        by_table = Coupler(
            components,
            default_exchangers(components),
            coupling_timestep=by_hand.coupling_timestep,
            start_date=by_hand.start_date,
        )
        assert by_table.workflow == by_hand.workflow

        latitude = components["ocn"].grid.latitude_radian
        carry = replace_field(
            by_hand.initialize(), "ocn.state.sea_surface_temperature",
            271.5 + 28.0 * jnp.cos(latitude) ** 2,
        )
        final_by_hand, _ = by_hand.generate_trajectory_function(2)(carry)
        final_by_table, _ = by_table.generate_trajectory_function(2)(carry)

        assert (jax.tree_util.tree_structure(final_by_hand)
                == jax.tree_util.tree_structure(final_by_table))
        by_hand_leaves = jax.tree_util.tree_leaves_with_path(final_by_hand)
        by_table_leaves = jax.tree_util.tree_leaves(final_by_table)
        for (leaf_path, leaf), ref_leaf in zip(by_hand_leaves, by_table_leaves):
            np.testing.assert_array_equal(
                np.asarray(leaf), np.asarray(ref_leaf),
                err_msg=jax.tree_util.keystr(leaf_path),
            )
        for component, field in (
            ("ocn", "total_heat_flux"),
            ("atm", "sea_surface_temperature"),
            ("atm", "sice_am"),
            ("seaice", "sea_surface_temperature"),
        ):
            written = getattr(final_by_hand.components[component]["forcing"], field)
            initial = getattr(carry.components[component]["forcing"], field)
            assert not np.array_equal(np.asarray(written), np.asarray(initial)), (
                f"{component}.forcing.{field} is still its initial value, so "
                "this run does not show that the exchange writes it"
            )

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
