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


# ---------------------------------------------------------------------------
# ESMFRegridder: input shape, jit, precision
# ---------------------------------------------------------------------------


def test_a_transposed_source_is_refused(regridders):
    """(n_lat, n_lon) has the right size but the wrong layout: it must raise."""
    regridder = regridders["a2o_conserve"]
    with pytest.raises(ValueError, match=r"a2o_conserve|weight_algo-conserve_JCM_T31") as error:
        regridder(np.ones((48, 96)))
    assert "(96, 48)" in str(error.value)
    assert "(48, 96)" in str(error.value)


def test_a_too_small_source_is_refused(regridders):
    """A (10, 10) field would be gathered from with clamped indices."""
    with pytest.raises(ValueError, match=r"\(10, 10\)"):
        regridders["a2o_conserve"](np.ones((10, 10)))


def test_a_correct_call_works_under_jit(regridders):
    """The shape check is static, so it does not stop the call being traced."""
    import jax

    regridder = regridders["a2o_conserve"]
    field = np.random.default_rng(0).uniform(size=T31_SHAPE).astype(np.float32)
    jitted = jax.jit(regridder)(field)
    np.testing.assert_allclose(jitted, regridder(field), rtol=1e-6)
    with pytest.raises(ValueError):
        jax.jit(regridder)(np.ones((48, 96), np.float32))


_PRECISION_SCRIPT = """
import jax
jax.config.update("jax_enable_x64", {x64})
import numpy as np
from importlib import resources
from jem.utils.esmf_regrid import ESMFRegridder
file = resources.files("jem.data") / "weight_algo-conserve_JCM_T31_to_DisplacedPoleGrid.nc"
r = ESMFRegridder(str(file))
print(r.weights.weights.dtype)
out = np.asarray(r(np.ones(r.src_shape)))
covered = out > 0.5
print(float(np.abs(out[covered] - 1.0).max()))
"""


@pytest.mark.parametrize("x64", [False, True])
def test_weights_follow_the_processs_float_precision(x64):
    """float64 weights under x64 (R(1) is 1 to 1e-12), float32 otherwise."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _PRECISION_SCRIPT.format(x64=x64)],
        capture_output=True, text=True, check=True,
    )
    dtype, error = result.stdout.split()
    assert dtype == ("float64" if x64 else "float32")
    assert float(error) < (1e-10 if x64 else 1e-5)
