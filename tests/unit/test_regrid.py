"""Tests for `jem.regrid.ESMFRegridders`, on the packaged weight files."""

from importlib import resources

import numpy as np
import pytest

from jem.regrid import ESMFRegridders

DATA = resources.files("jem.data")

#: The displaced-pole mixed-grid configuration's four maps, as
#: `jem/config/regrid/esmf.yaml` names them (`a2o` = atmosphere to ocean).
WEIGHTS = {
    "a2o_conserve": "weight_algo-conserve_JCM_T31_to_DisplacedPoleGrid.nc",
    "a2o_bilinear": "weight_algo-bilinear_JCM_T31_to_DisplacedPoleGrid.nc",
    "o2a_conserve": "weight_algo-conserve_DisplacedPoleGrid_to_JCM_T31.nc",
    "o2a_bilinear": "weight_algo-bilinear_DisplacedPoleGrid_to_JCM_T31.nc",
}

T31_SHAPE = (96, 48)
DISPLACED_POLE_SHAPE = (120, 59)


@pytest.fixture(scope="module")
def regridders():
    """Return the four packaged displaced-pole regridders."""
    return ESMFRegridders(
        **{name: str(DATA / file) for name, file in WEIGHTS.items()}
    )


def test_names_and_directions(regridders):
    """Each name maps between the grids its direction says it does."""
    assert sorted(regridders) == sorted(WEIGHTS)
    assert len(regridders) == 4
    for name, regridder in regridders.items():
        source, destination = (
            (T31_SHAPE, DISPLACED_POLE_SHAPE) if name.startswith("a2o")
            else (DISPLACED_POLE_SHAPE, T31_SHAPE)
        )
        assert tuple(regridder.src_shape) == source
        assert tuple(regridder.dst_shape) == destination


def test_regridders_apply(regridders):
    """Every regridder maps a field onto the destination grid.

    A uniform field must come back uniform: that is what a wrong index order
    or a wrong grid shape would break, and it holds for both algorithms.
    """
    for name, regridder in regridders.items():
        result = np.asarray(regridder(np.ones(regridder.src_shape)))
        assert result.shape == tuple(regridder.dst_shape), name
        assert 0.0 <= result.min() and result.max() <= 1.0 + 1e-5, name
        # A handful of destination cells are only partly covered by the source
        # grid -- the two grids do not share a domain boundary -- and a
        # conservative map returns the covered fraction there.
        assert np.isclose(result, 1.0, rtol=1e-5).mean() > 0.99, name


def test_mapping_is_read_only(regridders):
    """It is a `Mapping`, so a run cannot acquire a regridder halfway through."""
    with pytest.raises(TypeError):
        regridders["a2o_conserve"] = None  # type: ignore[index]


def test_unknown_name_says_what_is_configured(regridders):
    """Asking for a map the configuration did not build names the ones it did."""
    with pytest.raises(KeyError, match="a2o_conserve"):
        regridders["a2o_nearest"]


def test_empty_collection_is_refused():
    """An empty collection would silently exchange nothing."""
    with pytest.raises(ValueError, match="same_grid"):
        ESMFRegridders()
