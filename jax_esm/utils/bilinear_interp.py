from typing import Any, Tuple, Optional, cast
import math
import jax
import jax.numpy as jnp
from jax import Array, lax
from jax import config as _jax_config
import matplotlib.pyplot as plt


_jax_config.update("jax_enable_x64", True)


def _wrap_longitudes(lon_deg: Array, base0_deg: Array) -> Array:
    r"""Maps longitudes (deg) into the [base0, base0+360) interval.

    Mathematics:
        When longitude is treated as periodic, wrap every target longitude
        :math:`\lambda^{*}_{deg}` into the same half-open interval
        :math:`[ \lambda^{*}_{0}, \lambda^{*}_{0} + 360 )`
        of the (internally ascending) source grid:
        .. math::
            \tilde{\lambda}^{*}_{deg} = \lambda^{deg}_{0} + \mathrm{mod}(\lambda^{*}_{deg} - \lambda^{deg}_{0}, 360)
            \text{where } \lambda^{deg}_{0} = \text{base0\_deg}

        This guarantees consistent bracketing even across the dateline.

    Arguments:
        lon_deg: 1-D array of longitudes (degrees)
        base0_deg: base longitude (degrees) for wrapping

    Returns:
        An array of same shape as lon_deg with wrapped longitudes
    """

    return base0_deg + jnp.mod(lon_deg - base0_deg, 360.0)


def _unit_vectors_east_north(lon_rad: Array, lat_rad: Array) -> Tuple[Array, Array]:
    r"""Computes unit vectors (east, north) in 3-D for given lon/lat (radians).

    Mathematics:
        At any geographic point (λ,φ) on the unit sphere, define:
        - Radial unit vector (position on the unit sphere):
            .. math::
                \mathbf{r(\lambda, \phi)} = \begin{pmatrix}
                    \cos\phi \cos\lambda \\
                    \cos\phi \sin\lambda \\
                    \sin\phi
                \end{pmatrix}

        - Orthonormal tangent basis:
            .. math::
                \mathbf{e}_{\text{east}} = \frac{\partial \mathbf{r}}{\partial \lambda} =
                \begin{pmatrix}
                    -\sin\lambda \\
                    \cos\lambda \\
                    0
                \end{pmatrix}, \quad
                \mathbf{e}_{\text{north}} = \frac{\partial \mathbf{r}}{\partial \phi} =
                \begin{pmatrix}
                    -\sin\phi \cos\lambda \\
                    -\sin\phi \sin\lambda \\
                    \cos\phi
                \end{pmatrix}

        These satisfy:
            .. math::
                \mathbf{e}_{\text{east}} \cdot \mathbf{e}_{\text{north}} = 0 \quad \text{and} \quad
                \|\mathbf{e}_{\text{east}}\| = \|\mathbf{e}_{\text{north}}\| = 1.

    Arguments:
        lon_rad: array of longitudes (radians)
        lat_rad: array of latitudes (radians)

    Returns:
        (tuple):
            - e_east: array of shape (...,3) with east unit vectors
            - e_north: array of shape (...,3) with north unit vectors
    """

    sin_lon, cos_lon = jnp.sin(lon_rad), jnp.cos(lon_rad)
    sin_lat, cos_lat = jnp.sin(lat_rad), jnp.cos(lat_rad)
    e_east = jnp.stack((-sin_lon, cos_lon, jnp.zeros_like(lon_rad)), axis=-1)
    e_north = jnp.stack((-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat), axis=-1)
    return (e_east, e_north)


def _great_circle_distance_rad(
    lon1: Array, lat1: Array, lon2: Array, lat2: Array
) -> Array:
    r"""Haversine great-circle distance (radians) between points on the unit sphere.

    Mathematics:
        The haversine formula gives the central angle (great-circle distance in
        radians) between two points with geographic coordinates
        (longitude, latitude) = (λ, φ):

        .. math::
            \Delta\lambda &= \lambda_2 - \lambda_1,\\
            \Delta\varphi &= \varphi_2 - \varphi_1,\\[6pt]
            a &= \sin^2\!\left(\frac{\Delta\varphi}{2}\right)
                + \cos\varphi_1\,\cos\varphi_2\,
                \sin^2\!\left(\frac{\Delta\lambda}{2}\right),\\[6pt]
            c &= 2\,\operatorname{atan2}\!\left(\sqrt{a},\sqrt{1-a}\right).

        The returned value is the central angle :math:`c` in radians. For an
        earth-radius-scaled distance multiply :math:`c` by the desired radius.

    Arguments:
        lon1:
            Longitudes of the first point(s) in radians. May be scalar or array.
        lat1:
            Latitudes of the first point(s) in radians. Must be broadcastable
            with ``lon1``.
        lon2:
            Longitudes of the second point(s) in radians. May be scalar or array.
            Must be broadcastable with ``lat2`` and the other inputs.
        lat2:
            Latitudes of the second point(s) in radians. Must be broadcastable
            with ``lon2`` and the other inputs.

    Returns:
        An array of great-circle distances (central angles) in radians.
        The shape is the result of NumPy broadcasting of the inputs.

    Notes:
        - The implementation uses the haversine formulation for numerical
        robustness for small distances.
        - Intermediate value ``a`` is clamped to ``[0, 1]`` to avoid NaNs from
        floating point round-off when computing ``atan2``.
    """

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    sin_delta_lat = jnp.sin(0.5 * delta_lat)
    sin_delta_lon = jnp.sin(0.5 * delta_lon)
    a = (
        sin_delta_lat * sin_delta_lat
        + jnp.cos(lat1) * jnp.cos(lat2) * sin_delta_lon * sin_delta_lon
    )
    a = jnp.clip(a, 0.0, 1.0)
    return 2.0 * jnp.arctan2(jnp.sqrt(a), jnp.sqrt(1.0 - a))


def _geo_to_cart(lon_rad: Array, lat_rad: Array) -> Array:
    """Convert geographic coordinates (longitude, latitude) in radians to 3-D
    Cartesian unit vectors on the unit sphere.

    Mathematics:
        For longitude λ and latitude φ (radians) the corresponding point on the
        unit sphere is

            x = cos(φ) * cos(λ)
            y = cos(φ) * sin(λ)
            z = sin(φ)

        These follow from the spherical-to-Cartesian coordinate transform with the
        convention that latitude is the angle north of the equator and longitude
        is the angle east of the prime meridian.

    Arguments:
        lon_rad (ndarray):
            Longitudes in radians. Can be scalar or an array;
            must be broadcastable with lat_rad.
        lat_rad (ndarray):
            Latitudes in radians. Can be scalar or an array;
            must be broadcastable with lon_rad.

    Returns:
        (ndarray): array of shape (..., 3) (broadcasted shape of the inputs plus a trailing
        length-3 axis) containing the Cartesian unit vectors [x, y, z] for each
        input coordinate pair.
    """

    return jnp.stack(
        [
            jnp.cos(lat_rad) * jnp.cos(lon_rad),
            jnp.cos(lat_rad) * jnp.sin(lon_rad),
            jnp.sin(lat_rad),
        ],
        axis=-1,
    )


def _check_longitude_latitude(lon_deg: Array, lat_deg: Array, grid_type: str) -> None:
    """Checks validity of longitude and latitude arrays.

    Arguments:
        lon_deg: 1-D array of longitudes (degrees)
        lat_deg: 1-D array of latitudes (degrees)
        grid_type: 'source' or 'target' for error messages

    Returns:
        exception.ValueError: If the input arrays are not valid.
    """

    assert lon_deg.ndim == 1 and lat_deg.ndim == 1

    ascending_longitude = bool(jnp.all(jnp.diff(lon_deg) > 0.0))
    descending_longitude = bool(jnp.all(jnp.diff(lon_deg) < 0.0))
    if not (ascending_longitude or descending_longitude):
        raise ValueError(f"lon_{grid_type} must be strictly monotonic")
    if descending_longitude:
        raise ValueError(f"lon_{grid_type} must be ascending")

    ascending_latitude = bool(jnp.all(jnp.diff(lat_deg) > 0.0))
    descending_latitude = bool(jnp.all(jnp.diff(lat_deg) < 0.0))
    if not (ascending_latitude or descending_latitude):
        raise ValueError(f"lat_{grid_type} must be strictly monotonic")


class BilinearInterpolator:
    r"""
    JIT-friendly bilinear interpolator for rectilinear lat/lon grids
    with fixed-size (k, extrapolation_block_size) extrapolation.

    - periodic longitude handling
    - ascending or descending latitude
    - non-uniform spacing
    - optional NaN-aware renormalization
    - support for scalar and vector (u, v) fields
    - source/target masks
    - nearest / IDW extrapolation for masked-out areas

    Coordinates:
        - longitude_source: shape (nx,), degrees, strictly monotonic ascending (wrap allowed across dateline)
        - latitude_source: shape (ny,), degrees, strictly monotonic (ascending or descending)
        - longitude_target, latitude_target: target coordinates, any shape

    Masks:
        - source_mask: boolean array (ny, nx) = K_all where True means valid; if None, all valid
        - target_mask: boolean array like longitude_target/latitude_target of :math:`\Tau` shape;
                    True means compute/keep output, False -> fill_value

    Vector interpolation:
        - (u,v) are eastward and northward components on the sphere (m/s etc.)
        - We rotate vectors properly by projecting each corner vector to 3-D (using local EN basis),
          bilinear-blending in 3-D, then projecting onto the target EN basis.

    Extrapolation:
        - If all four bilinear corners are invalid, we extrapolate from sources:
            - mode='nearest': nearest valid source point
            - mode='idw': inverse-distance weighted blend of k nearest valid sources

        Config notes:
        - idw_k: number of neighbors for IDW (k=1 acts like 'nearest')
        - extrapolation_mode: 'idw' or 'nearest' or 'none'
        - extrapolation_block_size: block size for extrapolation; larger = faster but more memory
    """

    def __init__(
        self,
        longitude_source_deg: Array | list | tuple,
        latitude_source_deg: Array | list | tuple,
        longitude_target_deg: Array | list | tuple,
        latitude_target_deg: Array | list | tuple,
        periodic_longitude: bool = True,
        nan_renorm: bool = True,
        target_mask: Optional[Array] = None,
        source_mask: Optional[Array] = None,
        extrapolation_mode: str = "idw",
        idw_k: int = 6,
        idw_eps: float = 1e-12,
        extrapolation_block_size: int = 4096,
        fill_value: float = jnp.nan,
    ):
        self.periodic = bool(periodic_longitude)
        self.nan_renorm = bool(nan_renorm)
        if extrapolation_mode in ["idw", "nearest", "none"]:
            self.extrapolation_mode = extrapolation_mode
        else:
            raise ValueError(f"Invalid extrapolation_mode: {extrapolation_mode}")
        self.idw_k = idw_k
        self.idw_eps = float(idw_eps)
        self.extrapolation_block_size = int(extrapolation_block_size)
        self.fill_value = fill_value

        self.longitude_source_deg = jnp.asarray(longitude_source_deg, float)
        self.latitude_source_deg = jnp.asarray(latitude_source_deg, float)
        self.nx_source = int(self.longitude_source_deg.size)
        self.ny_source = int(self.latitude_source_deg.size)

        _check_longitude_latitude(
            self.longitude_source_deg, self.latitude_source_deg, "source"
        )

        self.latitude_source_ascending = bool(
            jnp.all(jnp.diff(self.latitude_source_deg) > 0.0)
        )
        longitude_source_rad, latitude_source_rad = jnp.meshgrid(
            jnp.deg2rad(self.longitude_source_deg),
            jnp.deg2rad(self.latitude_source_deg),
            indexing="xy",
        )

        longitude_target_deg = jnp.asarray(longitude_target_deg, float)
        latitude_target_deg = jnp.asarray(latitude_target_deg, float)

        _check_longitude_latitude(longitude_target_deg, latitude_target_deg, "target")

        self.longitude_target_deg, self.latitude_target_deg = jnp.meshgrid(
            longitude_target_deg, latitude_target_deg, indexing="xy"
        )
        self.target_grid_shape = tuple(self.longitude_target_deg.shape)
        self.target_grid_size = int(self.longitude_target_deg.size)
        self.longitude_target_rad = jnp.deg2rad(self.longitude_target_deg)
        self.latitude_target_rad = jnp.deg2rad(self.latitude_target_deg)

        # Source mask
        if source_mask is None:
            self.source_mask = jnp.ones((self.ny_source, self.nx_source), dtype=bool)
        else:
            self.source_mask = jnp.broadcast_to(
                jnp.asarray(source_mask, bool), (self.ny_source, self.nx_source)
            )

        # Target mask
        if target_mask is None:
            self.target_mask = jnp.ones(self.target_grid_shape, dtype=bool)
        else:
            self.target_mask = jnp.broadcast_to(
                jnp.asarray(target_mask, bool), self.target_grid_shape
            )

        # Precompute bilinear indices/weights
        self._precompute_cells_and_weights()

        # EN bases
        self._e_east_t, self._e_north_t = _unit_vectors_east_north(
            self.longitude_target_rad, self.latitude_target_rad
        )
        self._e_east_source, self._e_north_source = _unit_vectors_east_north(
            longitude_source_rad, latitude_source_rad
        )

        # ---- Candidate window of static size ----
        window_i = jnp.arange(-self.idw_k, self.idw_k + 1, dtype=jnp.int32)
        window_j = jnp.arange(-self.idw_k, self.idw_k + 1, dtype=jnp.int32)
        # C = (2K+1)^2
        window_i_2d, window_j_2d = jnp.meshgrid(window_i, window_j, indexing="xy")
        self._candidate_window_i = window_i_2d.reshape(-1)  # (C,)
        self._candidate_window_j = window_j_2d.reshape(-1)  # (C,)

        # precompute per-target candidate indices (saves doing it inside extrapolation loop)
        i0_flat = self.i0.reshape(-1).astype(jnp.int32)
        j0_flat = self.j0.reshape(-1).astype(jnp.int32)
        index_i = (
            i0_flat[:, None] + self._candidate_window_i[None, :]
        ) % self.nx_source
        index_j = jnp.clip(
            j0_flat[:, None] + self._candidate_window_j[None, :], 0, self.ny_source - 1
        )
        # Using linear (row-major) indexing: k = j * nx + i
        self._candidate_index = (index_j * self.nx_source + index_i).astype(
            jnp.int32
        )  # (target_grid_size, C)

        # Unit vectors for all sources & targets
        self._r_source_all = _geo_to_cart(
            longitude_source_rad, latitude_source_rad
        ).reshape(-1, 3)  # (K,3)
        self._r_target_all = _geo_to_cart(
            self.longitude_target_rad, self.latitude_target_rad
        ).reshape(-1, 3)  # (target_grid_size,3)
        # -----------------

        # Blocking schedule (static)
        self._num_blocks = int(
            math.ceil(self.target_grid_size / self.extrapolation_block_size)
        )

    # ----------------- indices/weights ----------------- #
    def _precompute_cells_and_weights(self) -> None:
        r"""For each target point, compute (i0,i1,j0,j1) and (fx,fy) and corner weights.

        Mathematics:
            For each target :math:`(\tilde{\lambda}^{*}, \phi^{*})` we find bracketing indices
            .. math::
                (i_0, i_1) \in \{0, \ldots, N_x - 1\}^2, \quad
                (j_0, j_1) \in \{0, \ldots, N_y - 1\}^2,

            such that :math:`(i_0, i_1)` are consecutive longitudes around
            :math:`(\tilde{\lambda}^*)`,
            and :math:`(j_0, j_1)` are consecutive latitudes around
            :math:`(\varphi^*)`.

            If the target lies beyond the non-periodic ends, indices are clamped;
            for periodic longitude, indices wrap modulo :math:`(N_x)`.

            Let
            .. math::
                \lambda_0 = \lambda_{i_0}, \quad
                \lambda_1 = \lambda_{i_1}, \quad
                \varphi_0 = \varphi_{j_0}, \quad
                \varphi_1 = \varphi_{j_1}.

            **Forward (wrapped) longitudinal difference**

            Across the dateline, we must measure the **forward** difference from
            :math:`(i_0)` to :math:`(i_1)`.
            Because longitude wraps, we define the **forward** cell width

            .. math::
                \Delta \lambda_{\text{cell}} =
                \begin{cases}
                    (\lambda_1 + 2\pi) - \lambda_0, & \text{if } i_1 \le i_0 \text{ (wrapped cell)}, \\
                    \lambda_1 - \lambda_0, & \text{otherwise.}
                \end{cases}

            and the **forward displacement**

            .. math::
                \Delta \tilde{\lambda}^* = \tilde{\lambda}^* - \lambda_0, \quad
                \Delta \tilde{\lambda}^* \leftarrow
                \begin{cases}
                    \Delta \tilde{\lambda}^* + 2\pi, & \text{if } \Delta \tilde{\lambda}^* < 0, \\
                    \Delta \tilde{\lambda}^*, & \text{otherwise.}
                \end{cases}

            Then the fractional longitudinal coordinate is

            .. math::
                f_x = \frac{\Delta \tilde{\lambda}^*}{\Delta \lambda_{\text{cell}}} \in [0, 1],

            (after clipping if needed).

            **Latitudinal fraction**

            Regardless of ascending/descending latitude ordering,

            .. math::
                \Delta \varphi_{\text{cell}} = \varphi_1 - \varphi_0,
                \quad
                f_y = \frac{\varphi^* - \varphi_0}{\Delta \varphi_{\text{cell}}}.

            and then clip :math:`f_y` to :math:`[0, 1]`. If latitudes are descending,
            :math:`(\Delta \varphi_{\text{cell}} < 0)`,
            and the fraction remains consistent after clipping.

            **Bilinear shape functions (weights)**

            On the rectangle :math:`(i_0, i_1) \times (j_0, j_1)`,
            the four standard bilinear basis functions are

            .. math::
                w_{00} = (1 - f_x)(1 - f_y), \quad
                w_{10} = f_x(1 - f_y),
            .. math::
                w_{01} = (1 - f_x)f_y, \quad
                w_{11} = f_x f_y.

            They satisfy :math:`w_{ab} \ge 0` and :math:`\sum w_{ab} = 1`.

            **Corner mapping:**

            .. math::
                (0,0) \mapsto (j_0, i_0), \quad
                (1,0) \mapsto (j_0, i_1), \quad
                (0,1) \mapsto (j_1, i_0), \quad
                (1,1) \mapsto (j_1, i_1).
        """

        nx, ny = self.nx_source, self.ny_source
        longitude_source = self.longitude_source_deg
        latitude_source = self.latitude_source_deg

        branch_periodic_index = jnp.array(0 if self.periodic else 1, dtype=jnp.int32)

        def _longitude_target_periodic_case(operand: Tuple[Array, Array]) -> Array:
            longitude_source, longitude_target = operand
            base0 = longitude_source[0].astype(float)
            longitude_target_mapped = _wrap_longitudes(longitude_target, base0)
            return longitude_target_mapped

        def _longitude_target_nonperiodic_case(operand: Tuple[Array, Array]) -> Array:
            longitude_source, longitude_target = operand
            longitude_target_mapped = jnp.clip(
                longitude_target, longitude_source.min(), longitude_source.max()
            )
            return longitude_target_mapped

        longitude_target_mapped = lax.switch(
            branch_periodic_index,
            [_longitude_target_periodic_case, _longitude_target_nonperiodic_case],
            (longitude_source, self.longitude_target_deg),
        )

        i1 = jnp.searchsorted(longitude_source, longitude_target_mapped, side="right")
        i0 = i1 - 1

        def _wrap_periodic_indices(
            operand: Tuple[Array, Array, Array],
        ) -> Tuple[Array, Array]:
            i0, i1, nx = operand
            i0 = jnp.mod(i0, nx)
            i1 = jnp.mod(i1, nx)
            return (i0, i1)

        def _clip_nonperiodic_indices(
            operand: Tuple[Array, Array, Array],
        ) -> Tuple[Array, Array]:
            i0, _, nx = operand
            i0 = jnp.clip(i0, 0, nx - 2)
            i1 = i0 + 1
            return (i0, i1)

        i0, i1 = lax.switch(
            branch_periodic_index,
            [_wrap_periodic_indices, _clip_nonperiodic_indices],
            (i0, i1, nx),
        )

        branch_asc_desc = jnp.array(
            0 if self.latitude_source_ascending else 1, dtype=jnp.int32
        )

        def _ascending_latitude_case(
            operand: Tuple[Array, Array],
        ) -> Tuple[Array, Array]:
            latitude_source, latitude_target = operand
            j1 = jnp.searchsorted(latitude_source, latitude_target, side="right")
            j0 = j1 - 1
            return (j0, j1)

        def _descending_latitude_case(
            operand: Tuple[Array, Array],
        ) -> Tuple[Array, Array]:
            latitude_source, latitude_target = operand
            lat_inv = latitude_source[::-1]
            j1_inv = jnp.searchsorted(lat_inv, latitude_target, side="right")
            j0_inv = j1_inv - 1
            j0 = (ny - 1) - jnp.clip(j0_inv, 0, ny - 2) - 1
            j1 = j0 + 1
            return (j0, j1)

        j0, j1 = lax.switch(
            branch_asc_desc,
            [_ascending_latitude_case, _descending_latitude_case],
            (latitude_source, self.latitude_target_deg),
        )

        j0 = jnp.clip(j0, 0, ny - 2)
        j1 = j0 + 1

        lon0 = jnp.deg2rad(longitude_source)[i0]
        lon1 = jnp.deg2rad(longitude_source)[i1]
        lam_tgt = self.longitude_target_rad
        dlon = lon1 - lon0
        wrap = i1 <= i0
        dlon = jnp.where(wrap, (lon1 + 2.0 * jnp.pi) - lon0, dlon)

        dlam = lam_tgt - lon0
        dlam = jnp.where(dlam < 0.0, dlam + 2.0 * jnp.pi, dlam)

        fx = jnp.where(dlon != 0.0, dlam / dlon, 0.0)
        fx = jnp.clip(fx, 0.0, 1.0)

        lat0 = jnp.deg2rad(latitude_source)[j0]
        lat1 = jnp.deg2rad(latitude_source)[j1]
        dphi = lat1 - lat0
        fy = jnp.where(dphi != 0.0, (self.latitude_target_rad - lat0) / dphi, 0.0)
        fy = jnp.clip(fy, 0.0, 1.0)

        self.i0 = i0.astype(jnp.int32)
        self.i1 = i1.astype(jnp.int32)
        self.j0 = j0.astype(jnp.int32)
        self.j1 = j1.astype(jnp.int32)

        self.w00 = (1.0 - fx) * (1.0 - fy)
        self.w10 = fx * (1.0 - fy)
        self.w01 = (1.0 - fx) * fy
        self.w11 = fx * fy

    # ----------------- utilities ----------------- #
    def _ensure_source_mask(self, source: Array) -> Array:
        """Ensures boolean validity mask for source data.

        Arguments:
            source: scalar source field (ny, nx)

        Returns:
            Boolean mask
        """

        return self.source_mask & jnp.isfinite(source)

    def _apply_bilinear_scalar(self, source: Array) -> Any:
        r"""Core bilinear (with optional NaN/mask renormalization).

        Mathematics:
            **Mask/NaN-aware renormalization**

            Let the corner validity be

            .. math::
                \mu_{00} = m^{\text{source}}_{j_0, i_0}, \quad
                \mu_{10} = m^{\text{source}}_{j_0, i_1}, \quad
                \mu_{01} = m^{\text{source}}_{j_1, i_0}, \quad
                \mu_{11} = m^{\text{source}}_{j_1, i_1}
                \in \{0, 1\}.

            (If values are NaN, take the corresponding :math:`\mu = 0`.)

            We down-weight invalid corners:

            .. math::
                \tilde{w}_{ab} = w_{ab} \mu_{ab}, \quad
                W = \sum_{a,b \in \{0,1\}} \tilde{w}_{ab}.

            * If :math:`W > 0` (some valid corners): **renormalize**

            .. math::
                \hat{w}_{ab} = \frac{\tilde{w}_{ab}}{W}, \quad
                \sum \hat{w}_{ab} = 1,

            and the scalar interpolation is

            .. math::
                s^* = \sum_{a,b} \hat{w}_{ab} \, s_{j_b, i_a},

            where :math:`i_{0/1} = i_0/i_1` and :math:`j_{0/1} = j_0/j_1`.

            * If :math:`W = 0`: all four corners invalid ⇒ **extrapolate** (Section 7).

            (If renormalization is **disabled**,
            then :math:`s^* = \sum w_{ab} s_{j_b, i_a}`
            only if all four corners are valid; otherwise
            :math:`s^* = \text{NaN}` and we fall back to extrapolation.)

        Arguments:
            source: (ny, nx) source scalar field

        Returns:
            tuple (array, array): Interpolated values and valid weight sum.
        """

        source = jnp.asarray(source, float)
        if tuple(source.shape) != (self.ny_source, self.nx_source):
            raise ValueError(
                f"source must have shape ({self.ny_source},{self.nx_source})"
            )
        valid = self._ensure_source_mask(source)

        v00 = source[self.j0, self.i0]
        v10 = source[self.j0, self.i1]
        v01 = source[self.j1, self.i0]
        v11 = source[self.j1, self.i1]

        m00 = valid[self.j0, self.i0]
        m10 = valid[self.j0, self.i1]
        m01 = valid[self.j1, self.i0]
        m11 = valid[self.j1, self.i1]

        branch_renorm_index = jnp.array(0 if self.nan_renorm else 1, dtype=jnp.int32)
        operand = (m00, m10, m01, m11, v00, v10, v01, v11)

        def _renorm_case(operand: Tuple[Array, ...]) -> Tuple[Array, Array]:
            m00, m10, m01, m11, v00, v10, v01, v11 = operand
            zeros = jnp.zeros_like(m00, dtype=float)
            w00 = self.w00 * m00
            w10 = self.w10 * m10
            w01 = self.w01 * m01
            w11 = self.w11 * m11
            wsum = w00 + w10 + w01 + w11

            num = (
                lax.select(m00, w00 * v00, zeros)
                + lax.select(m10, w10 * v10, zeros)
                + lax.select(m01, w01 * v01, zeros)
                + lax.select(m11, w11 * v11, zeros)
            )

            out = lax.select(wsum > 0.0, num / wsum, jnp.nan * zeros)
            return (out, wsum)

        def _no_renorm_case(operand: Tuple[Array, ...]) -> Tuple[Array, Array]:
            m00, m10, m01, m11, v00, v10, v01, v11 = operand
            all_ok = m00 & m10 & m01 & m11
            num = self.w00 * v00 + self.w10 * v10 + self.w01 * v01 + self.w11 * v11
            out = jnp.where(all_ok, num, jnp.nan)
            wsum = jnp.where(all_ok, 1.0, 0.0)
            return (out, wsum)

        return lax.switch(branch_renorm_index, [_renorm_case, _no_renorm_case], operand)

    # ----------------- fixed-size extrapolation core ----------------- #
    def _extrapolate_blocks_scalar(
        self, need_mask: Array, source_flat: Array, valid_flat: Array
    ) -> Any:
        r"""
        Fixed-size (k, extrapolation_block_size) extrapolation over ALL targets using lax.fori_loop.
        Returns a flat array of length ntarget with fill values for the 'need' entries.

        Mathematics:
            For each target :math:`t \in \{0, \ldots, N_T - 1\}` that needs extrapolation:

            1. **Distances to all source points** :math:`k \in \{1, \ldots, K\}`:

            .. math::
                d_{t,k} = gc\_dist \big( (\lambda_t^*, \phi_t^*), (\lambda_k, \phi_k) \big),
                \quad
                d_{t,k} = +\infty \text{ if source } k \text{ invalid.}

            2. **Weights:**

            .. math::
                \omega_{t,k} = \frac{1}{d_{t,k} + \varepsilon}.

            3. **Selection:**

            - If *nearest*: choose :math:`k^* = \arg \max_k \omega_{t,k}` and
              set :math:`s_t^{\text{ext}} = s_{k^*}`.
            - If *idw*: choose the top :math:`k` weights
              :math:`\{\omega_{t,k_r}\}_{r=1}^k` and compute

                .. math::
                    s_t^{\text{ext}} =
                    \begin{cases}
                        \dfrac{\sum_{r=1}^k \omega_{t,k_r} s_{k_r}}{\sum_{r=1}^k \omega_{t,k_r}}, & \sum_{r=1}^k \omega_{t,k_r} > 0, \\
                        \text{fill}, & \text{otherwise.}
                    \end{cases}

            ---

            The block machinery simply partitions the row set
            :math:`\{0, \ldots, N_T - 1\}` into disjoint chunks of size :math:`M`
            (except the last), but the **math is identical** to computing on all targets at once.

        Arguments:
            need_mask: (ntarget,) bool
                Flattened mask over targets: True where bilinear failed (needs extrapolation).
            source_flat: (K,) float
                Source scalar values flattened over the 2-D source grid; here
                K = NX × NY.
            valid_flat: (K,) bool
                Validity of each flattened source point (mask ∧ finite).

        Returns:
            An array of extrapolated scalar values of shape (ntarget,).
        """

        # NT
        ntarget = self.target_grid_size
        # M
        ext_blk_size = self.extrapolation_block_size
        idw_k = self.idw_k

        # Static-length block indices [0..ext_blk_size-1] to keep shapes static under JIT
        index0 = jnp.arange(ext_blk_size, dtype=jnp.int32)

        # Pre-broadcasted / flattened helpers
        i0_flat = self.i0.reshape(ntarget).astype(jnp.int32)
        j0_flat = self.j0.reshape(ntarget).astype(jnp.int32)
        window_i = self._candidate_window_i  # (C,) int32 offsets in i (x / longitude)
        window_j = self._candidate_window_j  # (C,) int32 offsets in j (y / latitude)

        # Unit vectors for all sources/targets (precomputed)
        r_source_all = self._r_source_all  # (K,3) float32
        r_target_all = self._r_target_all  # (ntarget,3) float32

        init = jnp.full((ntarget,), self.fill_value, dtype=source_flat.dtype)

        def body(b: int, out_flat: Array) -> Array:
            base = b * ext_blk_size
            index = base + index0  # (ext_blk_size,)
            index_clip = jnp.minimum(index, ntarget - 1)  # (ext_blk_size,)
            active = (index < ntarget) & need_mask[index_clip]  # (ext_blk_size,)

            # Candidate indices per target in the block: (ext_blk_size, C)
            i0_blk = i0_flat[index_clip][:, None]  # (ext_blk_size,1)
            j0_blk = j0_flat[index_clip][:, None]  # (ext_blk_size,1)
            index_i = (i0_blk + window_i[None, :]) % self.nx_source  # (ext_blk_size,C)
            index_j = jnp.clip(
                j0_blk + window_j[None, :], 0, self.ny_source - 1
            )  # (ext_blk_size,C)
            cand = (index_j * self.nx_source + index_i).astype(
                jnp.int32
            )  # (ext_blk_size,C)

            valid_c = valid_flat[cand]  # (ext_blk_size,C) bool

            # Dot-product distances: cos d = r_t · r_s
            rt_blk = r_target_all[index_clip]  # (ext_blk_size,3)  f32
            rs_blk = r_source_all[cand]  # (ext_blk_size,C,3) f32
            cosd = jnp.sum(rt_blk[:, None, :] * rs_blk, axis=-1)  # (ext_blk_size,C)
            cosd = jnp.clip(cosd, -1.0, 1.0)
            d = jnp.arccos(cosd)  # (ext_blk_size,C)  f64

            w = 1.0 / (d + self.idw_eps)
            w = jnp.where(valid_c, w, 0.0)
            # Invalid sources receive zero weight.
            # Inactive rows are zeroed so they don’t affect updates.
            w = jnp.where(active[:, None], w, 0.0)

            branch_index = jnp.array(
                0 if self.extrapolation_mode == "nearest" else 1, dtype=jnp.int32
            )
            row_ix_1d = jnp.arange(ext_blk_size, dtype=jnp.int32)
            operand = (w, cand, source_flat, active, row_ix_1d)

            def _nearest_case(args: Tuple[Array, ...]) -> Array:
                w_blk, cand_blk, source_flat, active_blk, row_ix = args
                imax = jnp.argmax(w_blk, axis=1)  # (ext_blk_size,)
                source_ix = cand_blk[row_ix, imax]  # (ext_blk_size,)
                picked = source_flat[source_ix]  # (ext_blk_size,)
                val_blk = jnp.where(active_blk, picked, self.fill_value)
                return val_blk

            def _idw_case(args: Tuple[Array, ...]) -> Array:
                w_blk, cand_blk, source_flat, active_blk, row_ix = args
                w_top, index_top = lax.top_k(w_blk, idw_k)  # (ext_blk_size,k)
                cand_top = cand_blk[row_ix[:, None], index_top]  # (ext_blk_size,k)
                v_top = source_flat[cand_top]  # (ext_blk_size,k)
                wsum = jnp.sum(w_top, axis=1)  # (ext_blk_size,)
                num = jnp.sum(w_top * v_top, axis=1)  # (ext_blk_size,)
                val_blk = jnp.where(wsum > 0.0, num / wsum, self.fill_value)
                val_blk = jnp.where(active_blk, val_blk, self.fill_value)
                return val_blk

            val_blk = lax.switch(branch_index, [_nearest_case, _idw_case], operand)

            # Inactive-write guard for the last (padded) block
            current = out_flat[index_clip]
            new_vals = jnp.where(active, val_blk, current)
            out_flat = out_flat.at[index_clip].set(new_vals)
            return out_flat

        return lax.fori_loop(0, self._num_blocks, body, init)

    def _extrapolate_blocks_vector(
        self, need_mask: Array, u_flat: Array, v_flat: Array, valid_flat: Array
    ) -> Any:
        """
        Same as scalar but computes (u,v) using the same neighbor sets/weights.

        Arguments:
            need_mask: (ntarget,) bool
                Flattened mask over targets: True where bilinear failed (needs extrapolation).
            u_flat, v_flat: (K,) float
                Source vector values flattened over the 2-D source grid; here
                K = NX × NY.
            valid_flat: (K,) bool
                Validity of each flattened source point (mask ∧ finite).

        Returns:
            A tuple of extrapolated vector arrays of shape (ntarget,).
        """

        # NT
        ntarget = self.target_grid_size
        # M
        ext_blk_size = self.extrapolation_block_size
        idw_k = self.idw_k

        index0 = jnp.arange(ext_blk_size, dtype=jnp.int32)

        i0_flat = self.i0.reshape(ntarget).astype(jnp.int32)
        j0_flat = self.j0.reshape(ntarget).astype(jnp.int32)
        window_i = self._candidate_window_i
        window_j = self._candidate_window_j

        r_source_all = self._r_source_all  # (K,3) f32
        r_target_all = self._r_target_all  # (ntarget,3) f32

        init_u = jnp.full((ntarget,), self.fill_value, dtype=u_flat.dtype)
        init_v = jnp.full((ntarget,), self.fill_value, dtype=v_flat.dtype)

        def body(b: int, carry: Tuple[Array, Array]) -> Tuple[Array, Array]:
            u_out, v_out = carry
            base = b * ext_blk_size
            index = base + index0
            index_clip = jnp.minimum(index, ntarget - 1)
            active = (index < ntarget) & need_mask[index_clip]

            # Candidate indices per target (ext_blk_size,C)
            i0_blk = i0_flat[index_clip][:, None]
            j0_blk = j0_flat[index_clip][:, None]
            index_i = (i0_blk + window_i[None, :]) % self.nx_source
            index_j = jnp.clip(j0_blk + window_j[None, :], 0, self.ny_source - 1)
            cand = (index_j * self.nx_source + index_i).astype(
                jnp.int32
            )  # (ext_blk_size,C)

            valid_c = valid_flat[cand]  # (ext_blk_size,C)

            # Dot-product distances
            rt_blk = r_target_all[index_clip]  # (ext_blk_size,3)
            rs_blk = r_source_all[cand]  # (ext_blk_size,C,3)
            cosd = jnp.sum(rt_blk[:, None, :] * rs_blk, axis=-1)  # (ext_blk_size,C)
            cosd = jnp.clip(cosd, -1.0, 1.0)
            d = jnp.arccos(cosd)  # (ext_blk_size,C)

            w = 1.0 / (d + self.idw_eps)
            w = jnp.where(valid_c, w, 0.0)
            w = jnp.where(active[:, None], w, 0.0)

            branch_index = jnp.array(
                0 if self.extrapolation_mode == "nearest" else 1, dtype=jnp.int32
            )
            row_ix_1d = jnp.arange(ext_blk_size, dtype=jnp.int32)
            operand = (w, cand, u_flat, v_flat, active, row_ix_1d)

            def _nearest_case(args: Tuple[Array, ...]) -> Tuple[Array, Array]:
                w_blk, cand_blk, u_source_flat, v_source_flat, active_blk, row_ix = args
                imax = jnp.argmax(w_blk, axis=1)
                source_ix = cand_blk[row_ix, imax]
                u_blk_local = u_source_flat[source_ix]
                v_blk_local = v_source_flat[source_ix]
                u_blk_local = jnp.where(active_blk, u_blk_local, self.fill_value)
                v_blk_local = jnp.where(active_blk, v_blk_local, self.fill_value)
                return u_blk_local, v_blk_local

            def _idw_case(args: Tuple[Array, ...]) -> Tuple[Array, Array]:
                w_blk, cand_blk, u_source_flat, v_source_flat, active_blk, row_ix = args
                w_top, index_top = lax.top_k(w_blk, idw_k)
                cand_top = cand_blk[row_ix[:, None], index_top]
                u_top = u_source_flat[cand_top]
                v_top = v_source_flat[cand_top]
                wsum = jnp.sum(w_top, axis=1)
                u_blk_local = jnp.where(
                    wsum > 0.0, jnp.sum(w_top * u_top, axis=1) / wsum, self.fill_value
                )
                v_blk_local = jnp.where(
                    wsum > 0.0, jnp.sum(w_top * v_top, axis=1) / wsum, self.fill_value
                )
                u_blk_local = jnp.where(active_blk, u_blk_local, self.fill_value)
                v_blk_local = jnp.where(active_blk, v_blk_local, self.fill_value)
                return u_blk_local, v_blk_local

            u_blk, v_blk = lax.switch(branch_index, [_nearest_case, _idw_case], operand)

            # Inactive-write guard for padding in the last block
            current_u = u_out[index_clip]
            current_v = v_out[index_clip]
            new_u = jnp.where(active, u_blk, current_u)
            new_v = jnp.where(active, v_blk, current_v)
            u_out = u_out.at[index_clip].set(new_u)
            v_out = v_out.at[index_clip].set(new_v)
            return (u_out, v_out)

        return lax.fori_loop(0, self._num_blocks, body, (init_u, init_v))

    # ----------------- public API ----------------- #
    def apply_scalar(self, source) -> Array:
        """Interpolate a scalar field defined on the source grid to the target grid.

        Arguments:
            source: (ny, nx,) Source scalar field.

        Returns:
            Target-shaped interpolated float array
        """

        out_bilin, _ = self._apply_bilinear_scalar(source)
        need = ~jnp.isfinite(out_bilin)

        yes_no_extrapolation_index = jnp.array(
            0 if self.extrapolation_mode == "none" else 1, dtype=jnp.int32
        )

        def _no_extrapolation_case(operand: Tuple[Array, ...]) -> Array:
            out_bilin, need, _ = operand
            out = jnp.where(need, self.fill_value, out_bilin)
            return out

        def _extrapolation_case(operand: Tuple[Array, ...]) -> Any:
            out_bilin, need, source = operand
            source = jnp.asarray(source, float)
            valid = self._ensure_source_mask(source).reshape(-1)
            source_flat = source.reshape(-1)  # (K_all,)
            fill_flat = self._extrapolate_blocks_scalar(
                need.reshape(-1), source_flat, valid
            )
            out = jnp.where(need, fill_flat.reshape(self.target_grid_shape), out_bilin)
            return out

        out = lax.switch(
            yes_no_extrapolation_index,
            [_no_extrapolation_case, _extrapolation_case],
            (out_bilin, need, source),
        )

        out = lax.select(
            self.target_mask, out, lax.full_like(out, fill_value=self.fill_value)
        )
        return out.reshape(self.target_grid_shape)

    def apply_vector(self, u_source: Array, v_source: Array) -> Tuple[Array, Array]:
        r"""Interpolate a vector field (u,v) in east/north components.

        Steps:
            1) Project each source corner (u,v) to 3-D using that corner's local EN basis.
            2) Bilinear blend those 3-D vectors with NaN/mask renormalization.
            3) Project blended 3-D vector to the target EN basis to get (u_target, v_target).
            4) Extrapolate where needed using scalar fallback on |V| and direction from nearest.

        Mathematics:
            At each source corner :math:`(\lambda_{i_a}, \varphi_{j_b})`,
            convert :math:`(u, v)` to a 3-D vector:

            .. math::
                \mathbf{V}_{ab}
                = u_{j_b, i_a} \, \mathbf{e}_{\text{east}}(\lambda_{i_a}, \varphi_{j_b})
                + v_{j_b, i_a} \, \mathbf{e}_{\text{north}}(\lambda_{i_a}, \varphi_{j_b}).

            Then apply the **same mask-aware bilinear blend** to the 3-D vectors:

            .. math::
                \mathbf{V}^* = \sum_{a,b} \hat{w}_{ab} \, \mathbf{V}_{ab}
                \quad (\text{if } W > 0; \text{ else extrapolate}).

            Finally, **project** the blended 3-D vector onto the target tangent basis at
            :math:`(\lambda^*, \varphi^*)`:

            .. math::
                u^* = \mathbf{V}^* \cdot \mathbf{e}_{\text{east}}(\lambda^*, \varphi^*),
                \quad
                v^* = \mathbf{V}^* \cdot \mathbf{e}_{\text{north}}(\lambda^*, \varphi^*).

            This procedure automatically rotates vectors correctly across the dateline
            and anywhere on the sphere (because the local bases vary with
            :math:`\lambda, \varphi`), while keeping the interpolation linear.

        Arguments:
            u_source: (ny, nx) eastward components on source grid.
            v_source: (ny, nx) northward components on source grid.
            source_mask (optional):
                (ny, nx) boolean array indicating valid vector elements.
                If None, validity = isfinite(u) & isfinite(v).

        Returns:
            Tuple of target-shaped interpolated float arrays
        """

        u_source = jnp.asarray(u_source, float)
        v_source = jnp.asarray(v_source, float)
        if tuple(u_source.shape) != (self.ny_source, self.nx_source) or tuple(
            v_source.shape
        ) != (
            self.ny_source,
            self.nx_source,
        ):
            raise ValueError(
                f"(u_source,v_source) must have shape ({self.ny_source},{self.nx_source})"
            )

        # Build 3-D vector field at sources (for bilinear in 3-D)
        eE, eN = self._e_east_source, self._e_north_source
        vector3d_source = (
            u_source[..., None] * eE + v_source[..., None] * eN
        )  # (NY,NX,3)

        valid_bilin = self.source_mask & jnp.isfinite(u_source) & jnp.isfinite(v_source)

        # Gather corners for bilinear in 3-D
        V00 = vector3d_source[self.j0, self.i0, :]
        V10 = vector3d_source[self.j0, self.i1, :]
        V01 = vector3d_source[self.j1, self.i0, :]
        V11 = vector3d_source[self.j1, self.i1, :]

        m00 = valid_bilin[self.j0, self.i0]
        m10 = valid_bilin[self.j0, self.i1]
        m01 = valid_bilin[self.j1, self.i0]
        m11 = valid_bilin[self.j1, self.i1]

        branch_renorm_index = jnp.array(0 if self.nan_renorm else 1, dtype=jnp.int32)

        def _renorm_case(operand: Tuple[Array, ...]) -> Tuple[Array, Array]:
            m00, m10, m01, m11, V00, V10, V01, V11 = operand
            zeros = jnp.zeros_like(V00)

            w00 = self.w00 * m00
            w10 = self.w10 * m10
            w01 = self.w01 * m01
            w11 = self.w11 * m11
            wsum = (w00 + w10 + w01 + w11)[..., None]

            def _select(mask, weight, vec):
                mask = jnp.broadcast_to(mask[..., None], vec.shape)
                return lax.select(mask, weight[..., None] * vec, zeros)

            num = (
                _select(m00, w00, V00)
                + _select(m10, w10, V10)
                + _select(m01, w01, V01)
                + _select(m11, w11, V11)
            )

            mask3d = jnp.broadcast_to(wsum > 0.0, num.shape)
            wsum3d = jnp.broadcast_to(wsum, num.shape)
            vector3d_target = lax.select(mask3d, num / wsum3d, jnp.nan * zeros)
            need = ~jnp.isfinite(vector3d_target[..., 0])

            return (vector3d_target, need)

        def _no_renorm_case(operand: Tuple[Array, ...]) -> Tuple[Array, Array]:
            m00, m10, m01, m11, V00, V10, V01, V11 = operand
            all_ok = m00 & m10 & m01 & m11

            num = (
                self.w00[..., None] * V00
                + self.w10[..., None] * V10
                + self.w01[..., None] * V01
                + self.w11[..., None] * V11
            )

            vector3d_target = jnp.where(
                all_ok[..., None], num, jnp.nan * jnp.zeros_like(V00)
            )
            need = ~all_ok

            return (vector3d_target, need)

        vector3d_target, need = lax.switch(
            branch_renorm_index,
            [_renorm_case, _no_renorm_case],
            (m00, m10, m01, m11, V00, V10, V01, V11),
        )

        # Project to target EN basis
        u_target = jnp.sum(vector3d_target * self._e_east_t, axis=-1)
        v_target = jnp.sum(vector3d_target * self._e_north_t, axis=-1)

        yes_no_extrapolation_index = jnp.array(
            0 if self.extrapolation_mode != "none" else 1, dtype=jnp.int32
        )
        operand = (need, u_source, v_source, u_target, v_target, valid_bilin)

        # Extrapolate components (fixed-size, shared neighbor sets)
        def _extrapolation_case(
            operand: Tuple[Array, ...],
        ) -> Tuple[Array, Array]:
            need, u_source, v_source, u_target, v_target, valid_ext = operand

            u_source_flat = u_source.reshape(-1)
            v_source_flat = v_source.reshape(-1)

            u_fill, v_fill = self._extrapolate_blocks_vector(
                need.reshape(-1), u_source_flat, v_source_flat, valid_ext.reshape(-1)
            )

            u_target_filled = jnp.where(
                need, u_fill.reshape(self.target_grid_shape), u_target
            )
            v_target_filled = jnp.where(
                need, v_fill.reshape(self.target_grid_shape), v_target
            )

            return (u_target_filled, v_target_filled)

        u_target, v_target = cast(
            Tuple[Array, Array],
            lax.switch(
                yes_no_extrapolation_index,
                [_extrapolation_case, lambda x: (u_target, v_target)],
                operand,
            ),
        )

        u_target = jnp.where(self.target_mask, u_target, self.fill_value)
        v_target = jnp.where(self.target_mask, v_target, self.fill_value)

        return (
            u_target.reshape(self.target_grid_shape),
            v_target.reshape(self.target_grid_shape),
        )


if __name__ == "__main__":
    NXs, NYs = 180, 91
    longitude_source = jnp.linspace(0.0, 360.0 - 360.0 / NXs, NXs)
    latitude_source = jnp.linspace(-90.0, 90.0, NYs)

    longitude_source_2d, latitude_source_2d = jnp.meshgrid(
        longitude_source, latitude_source
    )
    scalar_source = jnp.cos(jnp.deg2rad(latitude_source_2d)) * jnp.cos(
        jnp.deg2rad(longitude_source_2d - 180.0)
    )
    u_source = jnp.cos(jnp.deg2rad(latitude_source_2d))
    v_source = 0.5 * jnp.sin(jnp.deg2rad(longitude_source_2d - 180.0))

    source_mask = jnp.ones_like(scalar_source, dtype=bool)
    source_mask = source_mask.at[
        (latitude_source_2d > 20.0) & (latitude_source_2d < 30.0)
    ].set(False)
    source_mask = source_mask.at[
        (latitude_source_2d > 0.0)
        & (latitude_source_2d < 10.0)
        & (longitude_source_2d > 50)
        & (longitude_source_2d < 70)
    ].set(False)
    source_mask = source_mask.at[
        (latitude_source_2d > -30.0)
        & (latitude_source_2d < -10.0)
        & (longitude_source_2d > 20)
        & (longitude_source_2d < 50)
    ].set(False)

    scalar_source = jnp.where(source_mask, scalar_source, jnp.nan)
    u_source = jnp.where(source_mask, u_source, jnp.nan)
    v_source = jnp.where(source_mask, v_source, jnp.nan)

    longitude_target = jnp.linspace(-180.0, 180.0, 361)
    latitude_target = jnp.linspace(-90.0, 90.0, 181)
    longitude_target_2d, latitude_target_2d = jnp.meshgrid(
        longitude_target, latitude_target
    )

    target_mask = jnp.ones_like(longitude_target_2d, dtype=bool)
    target_mask = target_mask.at[
        (latitude_target_2d > 20.0) & (latitude_target_2d < 30.0)
    ].set(False)
    target_mask = target_mask.at[
        (latitude_target_2d > 0.0)
        & (latitude_target_2d < 10.0)
        & (longitude_target_2d > 50)
        & (longitude_target_2d < 70)
    ].set(False)
    # target_mask = target_mask.at[(latitude_target_2d > -30.0) & (latitude_target_2d < -10.0)
    #                      & (longitude_target_2d > 20) & (longitude_target_2d < 50)].set(False)
    target_mask = target_mask.at[
        (latitude_target_2d > -25.0)
        & (latitude_target_2d < -15.0)
        & (longitude_target_2d > 25)
        & (longitude_target_2d < 35)
    ].set(False)

    itp = BilinearInterpolator(
        longitude_source,
        latitude_source,
        longitude_target,
        latitude_target,
        periodic_longitude=True,
        nan_renorm=True,
        source_mask=source_mask,
        target_mask=target_mask,
        extrapolation_mode="idw",
        idw_k=8,
        extrapolation_block_size=4096,  # tune for memory/speed 4096 | 16384 | 64800
        fill_value=jnp.nan,
    )

    jit_scalar = jax.jit(lambda s: itp.apply_scalar(s))
    jit_vector = jax.jit(lambda u, v: itp.apply_vector(u, v))

    scalar_target = jit_scalar(scalar_source)
    u_target, v_target = jit_vector(u_source, v_source)

    print("Scalar target shape:", scalar_target.shape)
    print("Vector target shape:", u_target.shape, v_target.shape)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")

    im = axs[0, 0].pcolormesh(
        longitude_source_2d,
        latitude_source_2d,
        scalar_source,
        shading="auto",
        cmap="coolwarm",
    )
    axs[0, 0].set_title("Initial Scalar Field")
    axs[0, 0].set_xlabel("Longitude")
    axs[0, 0].set_ylabel("Latitude")

    axs[0, 1].quiver(
        longitude_source_2d[::4, ::4],
        latitude_source_2d[::4, ::4],
        u_source[::4, ::4],
        v_source[::4, ::4],
        scale=40,
    )
    axs[0, 1].set_title("Initial Vector Field")
    axs[0, 1].set_xlabel("Longitude")
    axs[0, 1].set_ylabel("Latitude")

    axs[1, 0].pcolormesh(
        longitude_target_2d,
        latitude_target_2d,
        scalar_target,
        shading="auto",
        cmap="coolwarm",
    )
    axs[1, 0].set_title("Interpolated Scalar Field")
    axs[1, 0].set_xlabel("Longitude")
    axs[1, 0].set_ylabel("Latitude")

    axs[1, 1].quiver(
        longitude_target_2d[::4, ::8],
        latitude_target_2d[::4, ::8],
        u_target[::4, ::8],
        v_target[::4, ::8],
        scale=40,
    )
    axs[1, 1].set_title("Interpolated Vector Field")
    axs[1, 1].set_xlabel("Longitude")
    axs[1, 1].set_ylabel("Latitude")

    fig.colorbar(im, ax=axs, shrink=0.6)

    plt.show()
