"""Minimal grid concept owned by the slab models.

Slab models do no advection, so unlike a full coupling-layer grid they only
need a fractional (land) mask and the coordinates that go with it -- shape and
the binary mask are both derived from ``fractional_mask``, so nothing here can
drift out of sync.

Two constructors cover every grid JEM runs on:

- :meth:`SlabGrid.from_coords` -- a separable lon/lat grid taken straight from
  the dinosaur horizontal grid the atmosphere is discretized on
  (``jcm.utils.get_coords(...).horizontal`` or
  ``jcm.physics.speedy.speedy_coords.get_speedy_coords(...).horizontal``). This
  replaced the ``"JCM::T31"`` specification-string DSL: the atmosphere's grid
  object is the single source of truth for the grid a coupled run uses, and
  re-deriving it from a string was one more place for the two to disagree.
- :meth:`SlabGrid.from_scrip` -- a 2-D structured grid from a SCRIP file, which
  may be curvilinear (a displaced-pole ocean grid).

The land-sea mask comes from the atmosphere too:
``jcm.terrain.TerrainData.from_coords(coords).fmask`` (or ``.from_file(...)`` /
``.aquaplanet(coords)``) is already shaped ``(n_lon, n_lat)`` with latitude
ascending south to north -- exactly ``SlabGrid``'s layout -- so it is passed
through as ``fractional_mask`` with no transpose or flip.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import Array

#: Largest departure from separability, in radians, still treated as a
#: separable lon/lat grid. 1e-6 rad is ~6 m at the Earth's surface: far below
#: any grid spacing, and above the float32 noise of a grid whose 2-D
#: coordinate fields were written out cell by cell from 1-D axes.
SEPARABILITY_TOLERANCE_RADIAN = 1e-6


def to_degrees(radians) -> np.ndarray:
    """Convert radians to degrees the way JCM does, in float64.

    ``x * 180 / pi`` and ``numpy.rad2deg(x)`` (a single multiply by the
    precomputed ``180/pi``) do not round identically. JCM writes its ``lon``
    and ``lat`` coordinates with the two-operation form
    (``jcm.utils.data_to_xarray``), so a slab dataset must too: a last-bit
    difference in a coordinate value is enough for ``xr.merge`` to treat the
    two grids as different axes and produce a 119-point longitude union out of
    two 96-point grids, which is exactly what happened before this existed.
    """
    return np.asarray(radians, dtype=np.float64) * 180.0 / np.pi


def _separable_axes(
    longitude_radian: Array,
    latitude_radian: Array,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the 1-D ``(longitude, latitude)`` axes, or None if curvilinear.

    A grid is separable when longitude varies only along axis 0 and latitude
    only along axis 1. That is what lets the model write CF 1-D ``lon``/``lat``
    coordinates instead of 2-D auxiliary ones, and what lets a climatology file
    written on 1-D axes be checked against the grid at all.
    """
    longitude = np.asarray(longitude_radian, dtype=np.float64)
    latitude = np.asarray(latitude_radian, dtype=np.float64)

    longitude_axis = longitude[:, 0]
    latitude_axis = latitude[0, :]
    separable = np.allclose(
        longitude, longitude_axis[:, None], atol=SEPARABILITY_TOLERANCE_RADIAN, rtol=0.0
    ) and np.allclose(
        latitude, latitude_axis[None, :], atol=SEPARABILITY_TOLERANCE_RADIAN, rtol=0.0
    )
    return (longitude_axis, latitude_axis) if separable else None


@dataclass
class SlabGrid:
    """Grid information needed by slab models.

    ``fractional_mask``, ``latitude_radian`` and ``longitude_radian`` fully
    specify the grid: shape, the derived binary mask, whether the grid is
    separable and the dimension names of its output all come from those three
    alone, so there is a single source of truth instead of several attributes
    that could drift out of sync.

    Attributes
    ----------
    fractional_mask : jax.Array
        Fraction of each grid cell occupied by land, in [0, 1], shaped
        ``(n_lon, n_lat)``.
    latitude_radian : jax.Array
        2-D latitudes in radians, matching ``fractional_mask.shape``.
    longitude_radian : jax.Array
        2-D longitudes in radians, matching ``fractional_mask.shape``.
    threshold : float
        Fractional-mask value at or above which a cell counts as land in
        :attr:`binary_mask`.
    longitude_axis_radian, latitude_axis_radian : numpy.ndarray or None
        The 1-D axes of a separable grid, in radians, in float64. Derived in
        ``__post_init__`` when not given, and ``None`` for a curvilinear grid.
        They are kept in float64 numpy (not jax) deliberately: they are grid
        *metadata*, and the degrees written into the output coordinates have to
        match JCM's bit for bit or ``xr.merge`` of two components' datasets
        would align on nothing.

    """

    fractional_mask: Array
    latitude_radian: Array
    longitude_radian: Array
    threshold: float = 1.0
    longitude_axis_radian: np.ndarray | None = field(default=None)
    latitude_axis_radian: np.ndarray | None = field(default=None)

    def __post_init__(self):
        """Check the shapes agree and work out whether the grid is separable."""
        if self.latitude_radian.shape != self.fractional_mask.shape:
            raise ValueError(
                f"latitude_radian has shape {tuple(self.latitude_radian.shape)} but "
                f"fractional_mask has shape {tuple(self.fractional_mask.shape)}."
            )
        if self.longitude_radian.shape != self.fractional_mask.shape:
            raise ValueError(
                f"longitude_radian has shape {tuple(self.longitude_radian.shape)} but "
                f"fractional_mask has shape {tuple(self.fractional_mask.shape)}."
            )
        if len(self.fractional_mask.shape) != 2:
            raise ValueError(
                "A SlabGrid is 2-D (n_lon, n_lat); got fractional_mask with shape "
                f"{tuple(self.fractional_mask.shape)}."
            )

        if self.longitude_axis_radian is None or self.latitude_axis_radian is None:
            axes = _separable_axes(self.longitude_radian, self.latitude_radian)
            if axes is not None:
                self.longitude_axis_radian, self.latitude_axis_radian = axes

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape, ``(n_lon, n_lat)``."""
        return tuple(self.fractional_mask.shape)

    @property
    def is_separable(self) -> bool:
        """True when the grid has genuine 1-D longitude and latitude axes."""
        return (
            self.longitude_axis_radian is not None
            and self.latitude_axis_radian is not None
        )

    @property
    def dims(self) -> tuple[str, str]:
        """Dimension names of a 2-D field on this grid, in shape order.

        A separable grid uses JCM's own axis names, so a slab dataset and a JCM
        dataset share dimensions and merge. A curvilinear grid must NOT: its
        latitude and longitude are 2-D auxiliary coordinates, and CF (and
        xarray) forbid a 2-D variable named after one of its own dimensions --
        hence the index-space names.
        """
        return ("lon", "lat") if self.is_separable else ("x", "y")

    @property
    def binary_mask(self) -> Array:
        """Binary land(1)/ocean(0) mask, derived from ``fractional_mask``."""
        return jnp.where(self.fractional_mask >= self.threshold, 1.0, 0.0)

    @classmethod
    def from_coords(
        cls,
        horizontal,
        fractional_mask: Array | None = None,
        threshold: float = 0.5,
    ) -> "SlabGrid":
        """Build a SlabGrid on an atmosphere's horizontal grid.

        Parameters
        ----------
        horizontal : dinosaur.spherical_harmonic.Grid
            The horizontal grid of the coordinate system the atmosphere runs
            on -- ``coords.horizontal`` for ``coords`` from
            ``jcm.utils.get_coords`` or
            ``jcm.physics.speedy.speedy_coords.get_speedy_coords``. Its
            ``nodal_axes`` give longitude in radians and the sine of latitude,
            in the same ``(n_lon, n_lat)`` layout SlabGrid uses.
        fractional_mask : jax.Array, optional
            Land fraction in [0, 1], shaped ``(n_lon, n_lat)``. Take it from
            ``jcm.terrain.TerrainData`` -- ``TerrainData.from_coords(coords).fmask``,
            ``TerrainData.from_file(path, coords).fmask`` or
            ``TerrainData.aquaplanet(coords).fmask`` -- which is already on
            this layout and orientation. Defaults to all-ocean.
        threshold : float
            Land fraction at or above which a cell counts as land.

        Returns
        -------
        SlabGrid

        Raises
        ------
        ValueError
            If ``fractional_mask`` is not on the grid's shape or leaves [0, 1].

        """
        # arcsin of the nodal sine-latitude axis, rather than the grid's own
        # ``latitudes`` attribute, because this is character for character what
        # ``jcm.utils.data_to_xarray`` writes into its ``lat`` coordinate; any
        # other route risks a last-bit difference that stops xr.merge aligning.
        longitude_axis, sin_latitude = horizontal.nodal_axes
        longitude_axis = np.asarray(longitude_axis, dtype=np.float64)
        latitude_axis = np.arcsin(np.asarray(sin_latitude, dtype=np.float64))
        shape = (longitude_axis.size, latitude_axis.size)

        if fractional_mask is None:
            fractional_mask = jnp.zeros(shape)
        else:
            fractional_mask = jnp.asarray(fractional_mask)
            if tuple(fractional_mask.shape) != shape:
                raise ValueError(
                    f"fractional_mask has shape {tuple(fractional_mask.shape)}, but this "
                    f"grid is {shape} (n_lon, n_lat). TerrainData.fmask is already on "
                    "this layout; a transposed mask is a sign it came from elsewhere."
                )
            if not bool(jnp.all((fractional_mask >= 0.0) & (fractional_mask <= 1.0))):
                raise ValueError("fractional_mask must lie in [0, 1].")

        longitude_2d = jnp.asarray(np.broadcast_to(longitude_axis[:, None], shape))
        latitude_2d = jnp.asarray(np.broadcast_to(latitude_axis[None, :], shape))

        return cls(
            fractional_mask=fractional_mask,
            latitude_radian=latitude_2d,
            longitude_radian=longitude_2d,
            threshold=threshold,
            longitude_axis_radian=longitude_axis,
            latitude_axis_radian=latitude_axis,
        )

    @classmethod
    def from_scrip(
        cls,
        scrip_file: str,
        fractional_mask: Array | None = None,
        threshold: float = 0.5,
    ) -> "SlabGrid":
        """Build a SlabGrid from a SCRIP-convention grid file.

        The file describes a 2-D structured (possibly curvilinear) grid -- e.g.
        a displaced-pole ocean grid. This is not a general unstructured-mesh
        reader: only SCRIP files with ``grid_rank == 2`` are supported.
        ``grid_dims = [n_lon, n_lat]`` already matches SlabGrid's convention;
        per-cell fields are stored flat in Fortran order (longitude varies
        fastest) within that shape. Only ``grid_center_lat``/``grid_center_lon``
        and ``grid_imask`` are read; ``grid_corner_*`` and ``grid_area`` exist
        for computing regridding weights, which the slab models do not do.

        Parameters
        ----------
        scrip_file : str
            Path to a SCRIP-convention netCDF grid file.
        fractional_mask : jax.Array, optional
            Land fraction in [0, 1] matching the file's shape. If None, it is
            derived from the file's own ``grid_imask`` (see
            :func:`_load_scrip_fractional_mask` for the assumed polarity).
        threshold : float
            Land fraction at or above which a cell counts as land.

        Returns
        -------
        SlabGrid

        """
        ds = xr.open_dataset(scrip_file)

        if "grid_dims" not in ds:
            raise ValueError(
                f"'{scrip_file}' has no grid_dims variable -- not a SCRIP grid file."
            )
        grid_dims = ds["grid_dims"].values
        if len(grid_dims) != 2:
            raise ValueError(
                "This reader only supports 2D SCRIP grids (grid_rank == 2); "
                f"got grid_dims={grid_dims!r}."
            )
        ni, nj = (int(n) for n in grid_dims)

        for name in ("grid_center_lat", "grid_center_lon"):
            if name not in ds:
                raise ValueError(f"'{scrip_file}' is missing required variable '{name}'.")

        grid_size = ds["grid_center_lat"].shape[0]
        if ni * nj != grid_size:
            raise ValueError(
                f"grid_dims {(ni, nj)} implies {ni * nj} cells, but "
                f"'{scrip_file}' has grid_size={grid_size}."
            )

        longitude_2d = _scrip_latlon_to_radians(ds["grid_center_lon"], ni, nj, scrip_file)
        latitude_2d = _scrip_latlon_to_radians(ds["grid_center_lat"], ni, nj, scrip_file)

        if jnp.any((latitude_2d < -jnp.pi / 2) | (latitude_2d > jnp.pi / 2)):
            raise ValueError(f"'{scrip_file}' has latitude values outside [-90, 90] degrees.")
        if jnp.any((longitude_2d < -2 * jnp.pi) | (longitude_2d > 2 * jnp.pi)):
            raise ValueError(f"'{scrip_file}' has longitude values outside [-360, 360] degrees.")

        if fractional_mask is None:
            fractional_mask = _load_scrip_fractional_mask(ds, ni, nj)

        return cls(
            fractional_mask=fractional_mask,
            latitude_radian=latitude_2d,
            longitude_radian=longitude_2d,
            threshold=threshold,
        )


def _reshape_scrip_field(values, ni: int, nj: int) -> Array:
    """Reshape a flat (grid_size,) SCRIP field into JEM's (n_lon, n_lat).

    SCRIP stores per-cell fields flat, in Fortran order (the first
    grid_dims axis -- longitude -- varies fastest). grid_dims = [ni, nj]
    already gives JEM's (n_lon, n_lat) shape directly, so an order='F'
    reshape recovers it with no transpose needed.
    """
    return jnp.asarray(values).reshape((ni, nj), order="F")


def _scrip_latlon_to_radians(da: xr.DataArray, ni: int, nj: int, scrip_file: str) -> Array:
    """Reshape and unit-convert a SCRIP grid_center_lat/lon field to radians."""
    values_2d = _reshape_scrip_field(da.values, ni, nj)
    units = da.attrs.get("units")
    if units == "degrees":
        return jnp.deg2rad(values_2d)
    elif units == "radians":
        return values_2d
    else:
        raise ValueError(
            f"'{scrip_file}' variable '{da.name}' has unrecognized units "
            f"{units!r}; expected 'degrees' or 'radians'."
        )


def _load_scrip_fractional_mask(ds: xr.Dataset, ni: int, nj: int) -> Array:
    """Derive a fractional_mask (1 = land) from a SCRIP grid_imask.

    SCRIP's grid_imask carries no CF-style flag_meanings metadata, so the
    land/ocean polarity can't be verified from the file itself. This follows
    the standard SCRIP/ESMF convention used by ocean-model grid files (e.g.
    POP, CESM): imask=1 marks a valid/active (ocean) cell, imask=0 marks a
    masked-out (land) cell. Pass `fractional_mask` explicitly to
    `SlabGrid.from_scrip` if a file uses the opposite convention.
    """
    if "grid_imask" not in ds:
        # A SCRIP file without a mask describes the geometry only; an all-ocean
        # mask is the neutral choice and is what the caller can override.
        return jnp.zeros((ni, nj))
    imask_2d = _reshape_scrip_field(ds["grid_imask"].values, ni, nj)
    return jnp.where(imask_2d == 0, 1.0, 0.0)
