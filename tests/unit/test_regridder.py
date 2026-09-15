"""Tests for jem.mapping.regridder.BilinearRegridder.

Before these existed, `BilinearRegridder.__call__` imported
`scipy.interpolate.RegularRegridder`, which is not a SciPy symbol under any
version, so every call raised ImportError.  The class was untested.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import coordax as cx

from jem.base.exceptions import ValidationError
from jem.mapping.grid import Grid
from jem.mapping.regridder import BilinearRegridder


def _grid(ny, nx, bmask=None, lat0=-89.0, lat1=89.0):
    coord = cx.compose_coordinates(
        cx.LabeledAxis("lat", np.linspace(lat0, lat1, ny)),
        cx.LabeledAxis("lon", np.linspace(0.0, 360.0, nx, endpoint=False)),
    )
    return Grid(coordinate=coord, bmask=bmask)


class TestBilinearRegridder:
    def test_call_does_not_raise_import_error(self):
        src, dst = _grid(8, 16), _grid(5, 10)
        out = BilinearRegridder(src, dst)(jnp.ones((8, 16)))
        assert out.shape == (5, 10)

    def test_identity_grids_roundtrip_exactly(self):
        src = _grid(8, 16)
        data = jnp.asarray(np.random.RandomState(0).rand(8, 16))
        out = BilinearRegridder(src, _grid(8, 16))(data)
        np.testing.assert_allclose(np.asarray(out), np.asarray(data), atol=1e-5)

    def test_linear_field_is_reproduced(self):
        """Bilinear interpolation must be exact on a field linear in lat/lon."""
        src, dst = _grid(9, 18), _grid(6, 12)
        slat = np.linspace(-89, 89, 9)[:, None]
        slon = np.linspace(0, 360, 18, endpoint=False)[None, :]
        data = jnp.asarray(2.0 * slat + 0.5 * slon)
        dlat = np.linspace(-89, 89, 6)[:, None]
        dlon = np.linspace(0, 360, 12, endpoint=False)[None, :]
        expect = 2.0 * dlat + 0.5 * dlon
        out = BilinearRegridder(src, dst)(data)
        np.testing.assert_allclose(np.asarray(out), expect, rtol=1e-4)

    def test_is_differentiable_and_jittable(self):
        """The point of a pure-JAX regridder: SciPy broke the gradient path."""
        src, dst = _grid(8, 16), _grid(5, 10)
        rg = BilinearRegridder(src, dst)
        g = jax.grad(lambda d: jnp.sum(rg(d) ** 2))(jnp.ones((8, 16)))
        assert g.shape == (8, 16)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.sum(jnp.abs(g)) > 0
        out = jax.jit(rg)(jnp.ones((8, 16)))
        assert jnp.all(jnp.isfinite(out))

    def test_values_are_bounded_by_the_source(self):
        """Clamping, not extrapolating, past the edge of the source grid."""
        src = _grid(8, 16, lat0=-60.0, lat1=60.0)
        dst = _grid(6, 12, lat0=-89.0, lat1=89.0)  # extends beyond the source
        data = jnp.asarray(np.random.RandomState(1).rand(8, 16))
        out = BilinearRegridder(src, dst)(data)
        assert float(jnp.min(out)) >= float(jnp.min(data)) - 1e-5
        assert float(jnp.max(out)) <= float(jnp.max(data)) + 1e-5

    def test_missing_ticks_raises_rather_than_guessing(self):
        src, dst = _grid(8, 16), _grid(5, 10)
        rg = BilinearRegridder(src, dst)
        rg.source_grid = None
        with pytest.raises(ValidationError):
            rg(jnp.ones((8, 16)))


class TestMaskedRegridding:
    def test_land_never_bleeds_into_ocean(self):
        """Land at 100, ocean at 10: a masked regrid must return only ~10."""
        ny, nx = 8, 16
        bmask = np.zeros((ny, nx), dtype=np.float32)
        bmask[:3, :] = 1.0  # a land band
        data = jnp.asarray(np.where(bmask == 1, 100.0, 10.0))
        src = _grid(ny, nx, bmask=jnp.asarray(bmask))
        dst = _grid(5, 10, bmask=jnp.zeros((5, 10)))
        out = BilinearRegridder(src, dst, mask_land=True)(data)
        assert float(jnp.max(out)) <= 10.0 + 1e-4, f"land bled in: max {jnp.max(out)}"

    def test_unmasked_call_does_bleed(self):
        """Control: without the mask the same field does contaminate."""
        ny, nx = 8, 16
        bmask = np.zeros((ny, nx), dtype=np.float32)
        bmask[:3, :] = 1.0
        data = jnp.asarray(np.where(bmask == 1, 100.0, 10.0))
        src = _grid(ny, nx, bmask=jnp.asarray(bmask))
        out = BilinearRegridder(src, _grid(5, 10), mask_land=False)(data)
        assert float(jnp.max(out)) > 10.0 + 1e-4

    def test_all_land_stencil_at_ocean_destination_is_never_zero(self):
        """An SST field must not come back as 0 K where the stencil is all land."""
        ny, nx = 8, 16
        bmask = np.ones((ny, nx), dtype=np.float32)
        bmask[6:, :] = 0.0  # only a narrow ocean band exists
        data = jnp.asarray(np.where(bmask == 1, 0.0, 290.0))
        src = _grid(ny, nx, bmask=jnp.asarray(bmask))
        dst = _grid(5, 10, bmask=jnp.zeros((5, 10)))  # all-ocean destination
        out = BilinearRegridder(src, dst, mask_land=True)(data)
        assert float(jnp.min(out)) > 250.0, f"got {jnp.min(out)} K at an ocean cell"

    def test_mask_land_without_bmask_raises(self):
        src, dst = _grid(8, 16), _grid(5, 10)
        with pytest.raises(ValidationError):
            BilinearRegridder(src, dst, mask_land=True)(jnp.ones((8, 16)))
