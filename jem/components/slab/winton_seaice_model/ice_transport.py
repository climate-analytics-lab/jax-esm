"""Cheap, conservative sea-ice transport for the thermodynamic Winton model.

Ice moves with a prescribed velocity (ocean surface current plus a wind free-drift term, supplied by the
coupler as forcing) by first-order upwind advection, and the conserved ice quantities diffuse laterally
(thickness diffusion "as a proxy for ice dynamics", as in the MIT coupled aquaplanet of Ferreira, Marshall
and Rose 2011; advection by surface currents as in the GENIE and Bern3D EMICs). Flux form on a regular,
possibly rotated, lat-lon grid: exactly conservative with a cyclic x axis and no-flux land faces.

References: Ferreira, Marshall and Rose (2011), J. Climate 24, 992-1012 (thickness diffusion as a proxy for ice
dynamics); Edwards and Marsh (2005), Clim. Dyn. 24, 415-433 (GENIE: ice advected by the surface current);
Flato and Hibler (1992), J. Phys. Oceanogr. 22, 626-651 (cavitating fluid: no compressive convergence once the
cover is compact, which the `compact` mask here reduces to a flux limiter). Original code; see the package NOTICE.
"""
import jax
import jax.numpy as jnp


def _shift(x, axis, n, cyclic):
    """x shifted so that result[i] = x[i - n]; beyond a non-cyclic edge the value is 0."""
    y = jnp.roll(x, n, axis=axis)
    if cyclic:
        return y
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(0, n) if n > 0 else slice(x.shape[axis] + n, None)
    return y.at[tuple(idx)].set(0.0)


def transport_fields(fields, u, v, dx, dy, ocean, dt, diffusivity, n_substeps=12, cyclic_x=True, compact=None):
    """Advect and diffuse per-cell-area quantities.

    compact: optional bool mask of fully ice-covered cells; no advective flux may enter them (the cavitating-fluid
    idea of Flato & Hibler 1992: no tensile strength, no convergence once the cover is compact), which stops
    free drift from piling ice up against coasts. Diffusion still acts.

    fields: tuple of (nx, ny) arrays (amount per unit cell area); u, v: cell-centred velocities (m/s) along the
    grid's own x and y; dx, dy: cell sizes (m) at centres; ocean: bool mask. All fields see the same linear,
    positive operator, so ratios of transported fields are weighted averages of the originals.
    """
    oc = ocean.astype(u.dtype)
    dts = dt / n_substeps
    area = dx * dy
    # faces: east face of cell i sits between i and i+1, north face of cell j between j and j+1
    east = lambda x: _shift(x, 0, -1, cyclic_x)
    north = lambda x: _shift(x, 1, -1, False)
    m_e, m_n = oc * east(oc), oc * north(oc)                      # no flux through land faces / the y edges
    dx_e, dy_n = 0.5 * (dx + east(dx)), 0.5 * (dy + north(dy))
    len_e, len_n = 0.5 * (dy + east(dy)), 0.5 * (dx + north(dx))   # face lengths
    u_e, v_n = 0.5 * (u + east(u)) * m_e, 0.5 * (v + north(v)) * m_n
    if compact is not None:
        c = compact.astype(u.dtype)
        u_e = u_e * (1.0 - jnp.where(u_e > 0, east(c), c))
        v_n = v_n * (1.0 - jnp.where(v_n > 0, north(c), c))
    # explicit stability near the rotated poles where dx is small: limit the Courant number and the
    # local diffusivity (ponytail: a limiter rather than implicit diffusion; only bites within ~8 deg of a pole)
    u_e = jnp.clip(u_e, -0.4 * dx_e / dts, 0.4 * dx_e / dts)
    v_n = jnp.clip(v_n, -0.4 * dy_n / dts, 0.4 * dy_n / dts)
    D_e = jnp.minimum(diffusivity, 0.2 * dx_e ** 2 / dts) * m_e
    D_n = jnp.minimum(diffusivity, 0.2 * dy_n ** 2 / dts) * m_n

    def substep(fs, _):
        out = []
        for X in fs:
            X_e, X_n = east(X), north(X)
            F_e = (u_e * jnp.where(u_e > 0, X, X_e) - D_e * (X_e - X) / dx_e) * len_e
            F_n = (v_n * jnp.where(v_n > 0, X, X_n) - D_n * (X_n - X) / dy_n) * len_n
            div = F_e - _shift(F_e, 0, 1, cyclic_x) + F_n - _shift(F_n, 1, 1, False)
            out.append(jnp.maximum(X - dts / area * div, 0.0) * oc)
        return tuple(out), None

    fields, _ = jax.lax.scan(substep, tuple(fields), None, length=n_substeps)
    return fields
