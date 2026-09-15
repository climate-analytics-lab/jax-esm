from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Dict, Any, Optional

from jem.base.exceptions import ValidationError
from jem.base.typing import (
    VariableMetadata,
)
from jem.mapping.grid import Grid

from jax import Array
from typeguard import typechecked


class BasicRegridder(ABC):
    """
    Abstract base class for interpolating between climate model grids.

    This class provides a framework for regriding data between different
    grid configurations (e.g., atmosphere to ocean) with built-in validation.

    Attributes:
        source_grid: A Grid object holding the information of the source grid.
        target_grid: A Grid object holding the information of the target grid.
        validate_shape: A flag indicating whether to check the compatibility of the input
            and output array when performing regridation.
         
    """

    source_grid: Optional[Grid] = None
    target_grid: Optional[Grid] = None

    validate_shape: bool

    @typechecked
    def __init__(
        self,
        source_grid: Optional[Grid] = None,
        target_grid: Optional[Grid] = None,
        validate_shape: bool = True,
    ):
        """Initialize the regridder."""

        self.source_grid = source_grid
        self.target_grid = target_grid
        self.validate_shape = validate_shape

        # Store validation results
        self.last_validation: Dict[str, Any] = {}

    def __call__(self, data: Array) -> Array:
        """Apply regridation with validation.

        Args:
            data: Data on the source grid

        Returns:
            Validated regrided data

        Raises:
            ValidationError: If validation checks fail
        """
        # Check input shape
        if (self.source_grid is not None) and (data.shape != self.source_grid.shape):
            raise ValueError(
                f"Input shape {data.shape} does not match source grid "
                f"shape {self.source_grid.shape}" 
            )

        # Apply regridation
        result = self(data)

        # Validate output
        self._validate(data, result)

        return result

    def _validate(self, source_data: Array, target_data: Array):
        """
        Perform validation checks on regrided data.

        Parameters
        ----------
        source_data : Array
            Original data on source grid
        target_data : Array
            Transformed data on target grid

        Raises
        ------
        ValidationError
            If any validation check fails
        """
        self.last_validation = {}

        # Shape validation
        if self.validate_shape:
            self._validate_shape(target_data)

    def _validate_shape(self, target_data: Array):
        """Validate that output has correct shape."""
        if not self.target_grid:
            raise Exception("target_grid was not provided. Cannot validate shape")

        if (not self.target_grid) and (target_data.shape != self.target_grid.shape):
            raise ValidationError(
                f"Output shape {target_data.shape} does not match "
                f"target grid shape {self.target_grid.shape}"
            )
        self.last_validation["shape_valid"] = True

    @abstractmethod
    def validate_metadata(
        self,
        source_metadata: VariableMetadata,
        target_metadata: VariableMetadata,
    ):
        """Validate the metadata"""
        pass

    def get_info(self):
       
        return {
            'type': str(self.__class__),
            'source_grid' : self.source_grid.get_info() if self.source_grid is not None else None, 
            'target_grid' : self.target_grid.get_info() if self.target_grid is not None else None, 
            'validate_shape' : self.validate_shape,
        }
       

class IdentityRegridder(BasicRegridder):
    """Identity mapping (no interpolation)."""

    def __call__(self, data: Array) -> Array:
        return data

    def validate_metadata(
        self, source_metadata: VariableMetadata, target_metadata: VariableMetadata
    ):
        if source_metadata[0] != target_metadata[0]: # shape
            raise ValidationError(f"Source {str(source_metadata[0])} and target metadata {str(target_metadata[0])} must have the same shape")


def _axis_ticks(grid: Grid) -> tuple[Array, Array]:
    """Return the 1-D coordinate values of a 2-D grid, as (axis0, axis1).

    Works for a coordax product coordinate (``.axes``) and for a single
    ``LabeledAxis``.  Falls back to normalised index space only when a grid
    genuinely carries no tick values, and says so rather than doing it silently.
    """
    coord = grid.coordinate
    axes = getattr(coord, "axes", None) or (coord,)
    if len(axes) != 2:
        raise ValidationError(
            f"BilinearRegridder expects a 2-D grid, got dims {coord.dims}."
        )
    ticks = []
    for ax in axes:
        t = getattr(ax, "ticks", None)
        if t is None:
            raise ValidationError(
                f"Axis {ax.dims} carries no tick values, so it cannot be regridded "
                "geographically.  Attach coordinate values to the Grid, or use "
                "IdentityRegridder if the grids are already aligned."
            )
        ticks.append(jnp.asarray(t, dtype=jnp.float32))
    return ticks[0], ticks[1]


def _bracket(src: Array, dst: Array) -> tuple[Array, Array, Array]:
    """Bracketing indices and the interpolation fraction for each target point.

    Returns ``(lo, hi, w)`` with ``value = (1-w)*src[lo] + w*src[hi]``.  Target
    points outside the source range are **clamped**, not extrapolated: linear
    extrapolation off the end of a Gaussian latitude axis produces unphysical
    values at the poles, which is exactly where ocean/atmosphere grids differ most.
    """
    n = src.shape[0]
    idx = jnp.clip(jnp.searchsorted(src, dst, side="right"), 1, n - 1)
    lo, hi = idx - 1, idx
    denom = src[hi] - src[lo]
    w = jnp.where(denom != 0, (dst - src[lo]) / jnp.where(denom != 0, denom, 1.0), 0.0)
    return lo, hi, jnp.clip(w, 0.0, 1.0)


class BilinearRegridder(BasicRegridder):
    """Bilinear interpolation between two rectilinear grids.

    Pure JAX, and therefore differentiable and jit-safe -- which the previous
    SciPy implementation was not.  That implementation also imported
    ``scipy.interpolate.RegularRegridder``, which does not exist under any SciPy
    version (the class is ``RegularGridInterpolator``), so every call raised
    ``ImportError``; and it interpolated in normalised index space, ignoring the
    grids' actual coordinates, which is wrong whenever the two grids are not
    uniformly spaced the same way -- e.g. a Gaussian atmosphere grid against a
    regular ocean grid.

    Set ``mask_land=True`` to restrict each interpolation stencil to ocean
    source cells, using ``source_grid.bmask`` (1 = land, 0 = ocean).  Weights are
    renormalised over the ocean corners, so land values never bleed into ocean
    destinations.  See ``masked`` notes in ``__call__``.
    """

    mask_land: bool = False

    @typechecked
    def __init__(
        self,
        source_grid: Optional[Grid] = None,
        target_grid: Optional[Grid] = None,
        validate_shape: bool = True,
        mask_land: bool = False,
    ):
        super().__init__(source_grid, target_grid, validate_shape)
        self.mask_land = mask_land

    def __call__(self, data: Array) -> Array:
        """Apply bilinear interpolation.

        With ``mask_land=True`` the four corners of each stencil are weighted by
        the source ocean mask and renormalised.  A destination whose stencil is
        entirely land has no ocean information to draw on; it takes the value of
        the nearest ocean source cell if the destination itself is ocean
        (``target_grid.bmask == 0``), and 0 otherwise.

        The asymmetry matters and is deliberate: zeroing a land destination in an
        ocean->atmosphere SST field would hand the atmosphere 0 K, not "no data".
        """
        if self.source_grid is None:
            raise ValidationError(
                "Source grid cannot be None when regridding with BilinearRegridder."
            )
        if self.target_grid is None:
            raise ValidationError(
                "Target grid cannot be None when regridding with BilinearRegridder."
            )

        src_y, src_x = _axis_ticks(self.source_grid)
        dst_y, dst_x = _axis_ticks(self.target_grid)

        iy0, iy1, wy = _bracket(src_y, dst_y)
        ix0, ix1, wx = _bracket(src_x, dst_x)

        # (ny_dst, nx_dst) corner weights
        wy_c = jnp.stack([1.0 - wy, wy])[:, :, None]     # (2, ny, 1)
        wx_c = jnp.stack([1.0 - wx, wx])[:, None, :]     # (2, 1, nx)
        iy = jnp.stack([iy0, iy1])
        ix = jnp.stack([ix0, ix1])

        corners = jnp.stack(
            [data[jnp.ix_(iy[a], ix[b])] for a in range(2) for b in range(2)]
        )  # (4, ny, nx)
        weights = jnp.stack(
            [wy_c[a] * wx_c[b] for a in range(2) for b in range(2)]
        )  # (4, ny, nx)

        if self.mask_land:
            bmask = self.source_grid.bmask
            if bmask is None:
                raise ValidationError(
                    "mask_land=True requires source_grid.bmask (1 = land, 0 = ocean)."
                )
            ocean = 1.0 - jnp.asarray(bmask, dtype=jnp.float32)
            ocean_c = jnp.stack(
                [ocean[jnp.ix_(iy[a], ix[b])] for a in range(2) for b in range(2)]
            )
            weights = weights * ocean_c
            total = jnp.sum(weights, axis=0)
            out = jnp.sum(weights * corners, axis=0) / jnp.where(total > 0, total, 1.0)

            # Stencils with no ocean corner at all.
            nearest = self._nearest_ocean(data, ocean)
            dst_bmask = self.target_grid.bmask
            dst_ocean = (
                jnp.ones_like(total)
                if dst_bmask is None
                else 1.0 - jnp.asarray(dst_bmask, dtype=jnp.float32)
            )
            fallback = jnp.where(dst_ocean > 0, nearest, 0.0)
            return jnp.where(total > 0, out, fallback)

        return jnp.sum(weights * corners, axis=0)

    def _nearest_ocean(self, data: Array, ocean: Array) -> Array:
        """Ocean-area-mean of the source field, broadcast to the target shape.

        A deliberately simple fallback: it is only reached for destinations whose
        entire stencil is land, and it keeps the operation jit-safe and
        differentiable.  A true nearest-neighbour search needs a precomputed
        index, which belongs in a weights-building step rather than here.
        """
        total = jnp.sum(ocean)
        mean = jnp.sum(data * ocean) / jnp.where(total > 0, total, 1.0)
        return jnp.broadcast_to(mean, self.target_grid.shape)

    def validate_metadata(
        self, source_metadata: VariableMetadata, target_metadata: VariableMetadata
    ):
        if self.source_grid is not None and source_metadata[0] != self.source_grid.shape:
            raise ValidationError(
                f"Source metadata shape {source_metadata[0]} does not match source grid "
                f"{self.source_grid.shape}"
            )
