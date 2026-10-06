"""Equivalence between the directly-built notebooks and the CLI.

``01_aquaplanet.ipynb`` and ``04_jcm_slabs_mixed_grid_aqua_planet.ipynb`` both
claim, in their own markdown, to build "the same model"
``+configuration=aquaplanet-slab`` / ``...-mixed-grid`` composes. This module
executes each notebook's OWN construction cell (not a copy of it, so it
cannot silently drift from what the notebook actually runs) and compares the
resulting :class:`~jem.base.coupler.Coupler` against
:func:`jem.runners.build_coupler` on the equivalent composed config.

The notebooks write their exchange out by hand, as one function, where the
configuration builds the declarative :class:`~jem.exchangers.Exchange` from
the standard table. Two couplers wired that differently cannot be compared
by their exchanger *objects* -- one has rows and the other has none -- so
they are compared by what the exchangers *do*:

- component names and types, the order the components step in, the coupling
  timestep, the start date and the coupled carry's pytree structure;
- the atmosphere's own physics timestep (``model.dt_si``). Building a plain
  ``jcm.model.Model`` with no explicit ``time_step`` silently adopts the
  physics' *stable* step (30 minutes for SPEEDY T31L8) rather than the 12
  minutes ``+configuration=aquaplanet-slab`` actually runs at
  (``atmosphere.run.time_step``, from jax-gcm's ``run/default.yaml``, which
  the recipe never overrides) -- a divergence nothing about a carry's
  *shape* can show;
- **the exchange itself**: each coupler's exchanger is applied to the same
  carry, with every field an exchange reads made distinct and non-trivial
  first, and the two results must agree leaf for leaf. A hand-written
  exchanger that drops a field, sends it to the wrong place or -- on the
  mixed grid -- regrids it with the wrong map fails here.

Slow: each test builds two full JCM atmospheres (the notebook's own and the
CLI's reference).
"""

import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
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


def _component_order(coupler) -> list:
    """Return the workflow with its exchangers dropped: who steps, in order.

    The exchanger's own name is the notebook's to choose, so it is not
    compared; that both workflows *start* with their one exchanger is.
    """
    assert len(coupler.exchangers) == 1, list(coupler.exchangers)
    assert coupler.workflow[0] in coupler.exchangers, coupler.workflow
    return [name for name in coupler.workflow if name in coupler.components]


def _distinct_carry(coupler):
    """Return ``coupler``'s initial components with every field made distinct.

    An initial carry is mostly zeros, and two exchangers that disagree about
    a field of zeros agree on the result. Every floating leaf of every
    ``state``/``derived``/``forcing`` section is therefore replaced by a
    smooth, strictly positive field that is different for every leaf, so a
    field read from the wrong place, or regridded with the wrong map, shows
    up as a different number.
    """
    components = dict(coupler.initialize().components)
    counter = iter(range(1, 10_000))

    def distinct(leaf):
        if not (hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
                and jnp.ndim(leaf) == 2):
            return leaf
        k = next(counter)
        i, j = jnp.meshgrid(
            jnp.linspace(0.0, 1.0, leaf.shape[0]),
            jnp.linspace(0.0, 1.0, leaf.shape[1]), indexing="ij")
        return (k + jnp.sin(k + 3.0 * i) * jnp.cos(2.0 * j)).astype(leaf.dtype)

    for name, carry in components.items():
        components[name] = dict(carry, **{
            section: jax.tree.map(distinct, carry[section])
            for section in ("state", "derived", "forcing") if section in carry
        })
    return components


def _assert_exchanges_agree(coupler, ref_coupler) -> None:
    """Assert the two couplers' exchangers move the same fields the same way."""
    (exchanger,) = coupler.exchangers.values()
    (ref_exchanger,) = ref_coupler.exchangers.values()
    # One carry for both: the couplers are structurally equivalent (checked
    # by the caller), so either one's carry is a valid input to both.
    components = _distinct_carry(ref_coupler)
    time = ref_coupler.coupling_time(0, ref_coupler.start_date)

    result = exchanger(dict(components), time)
    expected = ref_exchanger(dict(components), time)
    for name in ref_coupler.components:
        for section in ("state", "derived", "forcing"):
            if section not in expected[name]:
                continue
            got = jax.tree_util.tree_leaves_with_path(result[name][section])
            want = jax.tree_util.tree_leaves_with_path(expected[name][section])
            assert [path for path, _ in got] == [path for path, _ in want]
            for (path, got_leaf), (_, want_leaf) in zip(got, want):
                np.testing.assert_allclose(
                    np.asarray(got_leaf), np.asarray(want_leaf), rtol=1e-6,
                    err_msg=(f"{name}.{section}{jax.tree_util.keystr(path)}: "
                             "the notebook's exchanger and the standard "
                             "exchange disagree"),
                )
    # And the exchange did something: the atmosphere's SST is the ocean's,
    # not the value it started with.
    assert not np.allclose(
        np.asarray(result["atm"]["forcing"].sea_surface_temperature),
        np.asarray(components["atm"]["forcing"].sea_surface_temperature))


def _assert_structurally_equivalent(coupler, ref_coupler) -> None:
    """Assert two couplers describe the same coupled model, ``dt`` included."""
    assert list(coupler.components) == list(ref_coupler.components)
    assert _term_names(coupler) == _term_names(ref_coupler)
    assert _component_order(coupler) == _component_order(ref_coupler)
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
        _assert_exchanges_agree(coupler, ref_coupler)

    def test_04_mixed_grid_matches_aquaplanet_slab_mixed_grid(self):
        """04_...mixed_grid...ipynb's direct build matches the mixed-grid recipe."""
        notebook = (REPO_ROOT / "examples" / "01_basic"
                    / "04_jcm_slabs_mixed_grid_aqua_planet.ipynb")
        coupler = _exec_construction_cell(notebook)

        ref_cfg = _compose("aquaplanet-slab-mixed-grid", [])
        ref_coupler = runners.build_coupler(ref_cfg)

        _assert_structurally_equivalent(coupler, ref_coupler)

        # Mixed-grid-specific: the same ocean grid (shape and land/ocean
        # split), and -- because the two ocean grids are the same -- an
        # exchange that crosses to and from it with the same maps.
        ocean_grid = coupler.components["ocn"].grid
        ref_grid = ref_coupler.components["ocn"].grid
        assert ocean_grid.shape == ref_grid.shape
        np.testing.assert_array_equal(
            ocean_grid.fractional_mask, ref_grid.fractional_mask)

        _assert_exchanges_agree(coupler, ref_coupler)


if __name__ == "__main__":
    unittest.main()
