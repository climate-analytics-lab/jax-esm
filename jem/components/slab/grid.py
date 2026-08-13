"""Minimal grid concept owned by the slab models.

Slab models do no advection, so unlike a full coupling-layer grid they only
need a fractional (land) mask and the coordinates that go with it -- shape
and the binary mask are both derived from fractional_mask, so nothing here
can drift out of sync.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import dinosaur
import jax.numpy as jnp
import xarray as xr
from jax import Array

import jem


@dataclass
class SlabGrid:
    """Grid information needed by slab models.

    fractional_mask, latitude_radian, and longitude_radian fully specify the
    grid: shape and the derived binary mask both come from fractional_mask
    alone, so there's a single source of truth instead of several attributes
    that could drift out of sync.

    Attributes:
        fractional_mask: Fraction of each grid cell occupied by land, in [0, 1].
        latitude_radian: 2D array of latitudes in radians, matching
            fractional_mask.shape.
        longitude_radian: 2D array of longitudes in radians, matching
            fractional_mask.shape.
        threshold: Fractional-mask value at or above which a cell counts as
            land in `binary_mask`.
        dims: Axis names for fractional_mask/latitude_radian/longitude_radian,
            in shape order (used for xarray output labeling).
    """

    fractional_mask: Array
    latitude_radian: Array
    longitude_radian: Array
    threshold: float = 1.0
    dims: tuple[str, str] = ("longitude", "latitude")

    def __post_init__(self):
        assert self.latitude_radian.shape == self.fractional_mask.shape, (
            "latitude_radian must match fractional_mask shape"
        )
        assert self.longitude_radian.shape == self.fractional_mask.shape, (
            "longitude_radian must match fractional_mask shape"
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.fractional_mask.shape

    @property
    def binary_mask(self) -> Array:
        """Binary land(1)/ocean(0) mask, derived from fractional_mask."""
        return jnp.where(self.fractional_mask >= self.threshold, 1.0, 0.0)


def _parse_grid_specification(grid_specification_string: str) -> tuple[str, str]:
    """Parse a grid specification string of format "<grid_universe>::<grid_family>".

    Returns:
        (grid_universe, grid_family)
    """
    match = re.match(r"^([^:]+)::(.+)$", grid_specification_string)

    if not match:
        raise ValueError(
            f"Invalid grid specification format: '{grid_specification_string}'. "
            f"Expected format: '<grid_universe>::<grid_family>'"
        )

    return match.group(1), match.group(2)


def _broadcast_separable_grid_to_2d(
    latitude_1d: Array,
    longitude_1d: Array,
    shape: tuple[int, int],
    dims: tuple[str, str],
) -> tuple[Array, Array]:
    """Broadcast independent 1D latitude/longitude axes into 2D fields.

    Only valid for grids where latitude and longitude are separable (each
    varies along a single axis, as for JCM/Veros regular lat-lon grids). A
    curvilinear/displaced-pole grid builder would instead load its 2D
    latitude/longitude fields directly and skip this helper.
    """
    lat_dim_idx = dims.index("latitude")
    lon_dim_idx = dims.index("longitude")

    latitude_2d = jnp.repeat(
        jnp.expand_dims(latitude_1d, axis=lon_dim_idx),
        repeats=shape[lon_dim_idx],
        axis=lon_dim_idx,
    )
    longitude_2d = jnp.repeat(
        jnp.expand_dims(longitude_1d, axis=lat_dim_idx),
        repeats=shape[lat_dim_idx],
        axis=lat_dim_idx,
    )
    return latitude_2d, longitude_2d


def generate_slab_grid(
    grid_specification: str,
    fractional_mask: Array | None = None,
) -> SlabGrid:
    """Convenience helper: build a SlabGrid from one of JEM's canonical grid
    specification strings, instead of assembling one by hand.

    Loading a fractional mask from a file is the caller's responsibility --
    see `load_jcm_fractional_mask` for the conventional per-universe loaders.

    Args:
        grid_specification: String in format "<grid_universe>::<grid_family>",
            e.g. "JCM::T31"
        fractional_mask: Fraction of each grid cell occupied by land, in
            [0, 1], matching the grid's shape. If None, defaults to all-zero
            (no land).

    Returns:
        A SlabGrid, with the binary-mask threshold set to the conventional
        value for the grid_universe (0.95 for JCM).
    """
    grid_universe, grid_family = _parse_grid_specification(grid_specification)

    if grid_universe == "JCM":
        return _generate_jcm_slab_grid(
            horizontal_resolution=int(grid_family[1:]),
            fractional_mask=fractional_mask,
        )
    raise ValueError(f"Error: unrecognized grid_universe '{grid_universe}'.")


def load_jcm_fractional_mask(mask_file: str) -> Array:
    """Load a JCM land-sea mask file into a fractional_mask array."""
    ds = xr.open_dataset(mask_file, engine="netcdf4")
    fmask = jnp.asarray(ds["lsm"])

    assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), (
        "Land-sea mask must be between 0 and 1"
    )

    return fmask


def _generate_jcm_slab_grid(
    horizontal_resolution: int,
    fractional_mask: Array | None = None,
) -> SlabGrid:
    """
    Builds a SlabGrid for the given horizontal resolution (21, 31, 42, 85,
    106, 119, 170, 213, 340, or 425).
    """
    try:
        horizontal_grid = getattr(
            dinosaur.spherical_harmonic.Grid, f"T{horizontal_resolution:d}"
        )
    except AttributeError:
        raise ValueError(
            f"Invalid horizontal grid name: T{horizontal_resolution:d}. Must be one "
            "of: T21, T31, T42, T85, T106, T119, T170, T213, T340, or T425."
        )

    one_layer_coords = dinosaur.coordinate_systems.CoordinateSystem(
        horizontal=horizontal_grid(radius=1.0),
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates([0.0, 1.0]),
    )
    hgrid = one_layer_coords.horizontal

    latitude = jnp.asarray(hgrid.latitudes)
    longitude = jnp.asarray(hgrid.longitudes)
    dims = ("longitude", "latitude")
    shape = (longitude.shape[0], latitude.shape[0])

    if fractional_mask is None:
        print("Notice: No fractional_mask given. Set fmask = 0.")
        fmask = jnp.zeros(shape)
    else:
        fmask = fractional_mask

    latitude_2d, longitude_2d = _broadcast_separable_grid_to_2d(
        latitude, longitude, shape, dims
    )

    return SlabGrid(
        fractional_mask=fmask,
        latitude_radian=latitude_2d,
        longitude_radian=longitude_2d,
        threshold=0.95,
        dims=dims,
    )


def _reshape_ugrid_face_field(values, ny: int, nx: int) -> Array:
    """Reshape a flat (nface,) UGRID face array into JEM's (n_lon, n_lat).

    UGRID stores per-face fields flat; a `shape` attribute (project-specific,
    not part of the UGRID spec) gives the (ny, nx) raster shape the flat
    array reshapes into, in row-major standard CF (lat, lon) axis order.
    """
    return jnp.asarray(values).reshape(ny, nx).T


def _load_ugrid_fractional_mask(ds: xr.Dataset, ny: int, nx: int) -> Array:
    """Derive a fractional_mask (1 = land) from a UGRID CF flag mask.

    Reads the flag_values/flag_meanings that identify which value means
    "land" rather than assuming a fixed convention, since UGRID/CF leaves
    the flag ordering up to the file.
    """
    mask_var = ds["mask"]
    flag_values = mask_var.attrs.get("flag_values")
    flag_meanings = mask_var.attrs.get("flag_meanings")
    if flag_values is None or flag_meanings is None:
        raise ValueError(
            "UGRID 'mask' variable is missing CF flag_values/flag_meanings "
            "attributes -- cannot determine which value means land."
        )

    meanings = flag_meanings.split()
    if "land" not in meanings:
        raise ValueError(
            f"mask flag_meanings {meanings!r} does not include 'land'."
        )
    land_value = flag_values[meanings.index("land")]

    mask_2d = _reshape_ugrid_face_field(mask_var.values, ny, nx)
    return jnp.where(mask_2d == land_value, 1.0, 0.0)


def generate_slab_grid_from_ugrid(
    ugrid_file: str,
    fractional_mask: Array | None = None,
    threshold: float = 1.0,
) -> SlabGrid:
    """Build a SlabGrid from a UGRID netCDF file describing a 2D structured
    (curvilinear) grid -- e.g. a displaced-pole ocean grid.

    This is not a general unstructured-mesh reader: it only supports UGRID
    files that declare a `shape` global attribute giving the (ny, nx) raster
    shape the flat per-face arrays reshape into (row-major, standard CF
    (lat, lon) axis order) -- i.e. grids that are topologically 2D despite
    being stored in UGRID's face-based layout. Node/edge connectivity
    (face_nodes, face_edges, edge_nodes, node_lon/lat) is not used; this
    reader only needs face-center coordinates (the mesh's face_coordinates).

    Args:
        ugrid_file: Path to a UGRID-convention netCDF file.
        fractional_mask: Fraction of each grid cell occupied by land, in
            [0, 1], matching the grid's shape. If None, read from the
            file's own `mask` variable (a CF flag variable).
        threshold: Passed through to SlabGrid -- the mask in the example
            file is already strictly binary, so the default of 1.0 is
            exactly right there.

    Returns:
        A SlabGrid built from the file's face-center coordinates and mask.
    """
    ds = xr.open_dataset(ugrid_file)

    mesh_var_name = next(
        (
            name
            for name, da in ds.variables.items()
            if da.attrs.get("cf_role") == "mesh_topology"
        ),
        None,
    )
    if mesh_var_name is None:
        raise ValueError(
            f"No UGRID mesh_topology variable found in '{ugrid_file}'."
        )
    mesh_attrs = ds[mesh_var_name].attrs

    if mesh_attrs.get("topology_dimension") != 2:
        raise ValueError(
            "This reader only supports 2D UGRID meshes; "
            f"got topology_dimension={mesh_attrs.get('topology_dimension')!r}."
        )

    if "face_coordinates" not in mesh_attrs:
        raise ValueError(
            f"UGRID mesh '{mesh_var_name}' has no face_coordinates -- this "
            "reader needs face-center lon/lat, not just node coordinates."
        )
    lon_name, lat_name = mesh_attrs["face_coordinates"].split()
    if lon_name not in ds or lat_name not in ds:
        raise ValueError(
            f"face_coordinates names '{lon_name}', '{lat_name}' not found "
            f"in '{ugrid_file}'."
        )

    if "shape" not in ds.attrs:
        raise ValueError(
            f"'{ugrid_file}' has no 'shape' attribute -- this reader only "
            "supports UGRID files that declare their implied 2D (ny, nx) "
            "raster shape; general unstructured meshes are not supported."
        )
    ny, nx = (int(n) for n in ds.attrs["shape"])

    nface = ds[lon_name].shape[0]
    if ny * nx != nface:
        raise ValueError(
            f"shape attribute {(ny, nx)} implies {ny * nx} faces, but "
            f"'{ugrid_file}' has {nface} faces."
        )

    latitude_2d = jnp.deg2rad(_reshape_ugrid_face_field(ds[lat_name].values, ny, nx))
    longitude_2d = jnp.deg2rad(_reshape_ugrid_face_field(ds[lon_name].values, ny, nx))

    if jnp.any((latitude_2d < -jnp.pi / 2) | (latitude_2d > jnp.pi / 2)):
        raise ValueError(f"'{ugrid_file}' has latitude values outside [-90, 90] degrees.")
    if jnp.any((longitude_2d < -2 * jnp.pi) | (longitude_2d > 2 * jnp.pi)):
        raise ValueError(f"'{ugrid_file}' has longitude values outside [-360, 360] degrees.")

    if fractional_mask is None:
        fractional_mask = _load_ugrid_fractional_mask(ds, ny, nx)

    return SlabGrid(
        fractional_mask=fractional_mask,
        latitude_radian=latitude_2d,
        longitude_radian=longitude_2d,
        threshold=threshold,
        dims=("longitude", "latitude"),
    )


def _reshape_scrip_field(values, ni: int, nj: int) -> Array:
    """Reshape a flat (grid_size,) SCRIP field into JEM's (n_lon, n_lat).

    SCRIP stores per-cell fields flat, in Fortran order (the first
    grid_dims axis -- longitude -- varies fastest). grid_dims = [ni, nj]
    already gives JEM's (n_lon, n_lat) shape directly, so an order='F'
    reshape recovers it with no transpose needed (unlike the UGRID reader,
    whose `shape` attribute is (nlat, nlon) and needs one).
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

    Unlike UGRID, SCRIP's grid_imask carries no CF-style flag_meanings
    metadata, so the land/ocean polarity can't be verified from the file
    itself. This follows the standard SCRIP/ESMF convention used by
    ocean-model grid files (e.g. POP, CESM): imask=1 marks a valid/active
    (ocean) cell, imask=0 marks a masked-out (land) cell -- the same
    polarity as this codebase's UGRID reader ended up with. Pass
    `fractional_mask` explicitly to `generate_slab_grid_from_scrip` if a
    file uses the opposite convention.
    """
    if "grid_imask" not in ds:
        print("Notice: No grid_imask in SCRIP file. Set fmask = 0.")
        return jnp.zeros((ni, nj))
    imask_2d = _reshape_scrip_field(ds["grid_imask"].values, ni, nj)
    return jnp.where(imask_2d == 0, 1.0, 0.0)


def generate_slab_grid_from_scrip(
    scrip_file: str,
    fractional_mask: Array | None = None,
    threshold: float = 1.0,
) -> SlabGrid:
    """Build a SlabGrid from a SCRIP-convention grid file describing a 2D
    structured (curvilinear) grid -- e.g. a displaced-pole ocean grid.

    This is not a general unstructured-mesh reader: it only supports SCRIP
    files with grid_rank == 2 (grid_dims has two entries), i.e. grids that
    are topologically 2D. grid_dims = [n_lon, n_lat] already matches JEM's
    SlabGrid convention directly; per-cell fields are stored flat in
    Fortran order (longitude varies fastest) within that shape. Only
    grid_center_lat/lon and grid_imask are used; grid_corner_lat/lon,
    grid_area (present for regridding weight computation, which JEM's slab
    models don't do) are not read.

    Args:
        scrip_file: Path to a SCRIP-convention netCDF grid file.
        fractional_mask: Fraction of each grid cell occupied by land, in
            [0, 1], matching the grid's shape. If None, read from the
            file's own `grid_imask` variable (see `_load_scrip_fractional_mask`
            for the assumed land/ocean convention).
        threshold: Passed through to SlabGrid.

    Returns:
        A SlabGrid built from the file's cell-center coordinates and mask.
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

    return SlabGrid(
        fractional_mask=fractional_mask,
        latitude_radian=latitude_2d,
        longitude_radian=longitude_2d,
        threshold=threshold,
        dims=("longitude", "latitude"),
    )


