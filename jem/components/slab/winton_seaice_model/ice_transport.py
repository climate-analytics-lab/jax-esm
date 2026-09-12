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


def transport_fields(fields, u, v, dx, dy, ocean, dt, diffusivity, n_substeps=12, cyclic_x=True, compact_threshold=None):
    """Advect and diffuse per-cell-area quantities.

    compact_threshold: if given, fields[0] is the ice fraction and no advective flux may enter a cell whose fraction
    is at or above the threshold, re-evaluated every substep (the cavitating-fluid idea of Flato & Hibler 1992: no
    tensile strength, no convergence once the cover is compact), which stops free drift from piling ice up against
    coasts. Diffusion still acts.

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
    west = lambda x: _shift(x, 0, 1, cyclic_x)
    south = lambda x: _shift(x, 1, 1, False)
    m_e, m_n = oc * east(oc), oc * north(oc)                      # no flux through land faces / the y edges
    dx_e, dy_n = 0.5 * (dx + east(dx)), 0.5 * (dy + north(dy))
    len_e, len_n = 0.5 * (dy + east(dy)), 0.5 * (dx + north(dx))   # face lengths
    u_e0, v_n0 = 0.5 * (u + east(u)) * m_e, 0.5 * (v + north(v)) * m_n
    # Every face flux is split into an outgoing part from each of its two cells, each proportional to that cell's
    # content: F_e = k_e_out(i) X_i - k_e_in(i) X_{i+1}, with k (m^2/s) = upwind advection + diffusion. The scheme
    # is then a positive linear operator, and non-negativity holds exactly when the total outgoing fraction of a
    # cell per substep, dts/area * sum_faces k_out, is <= 1. Cells that would exceed `cap` have all their
    # outgoing coefficients scaled down; the same scaled flux leaves one cell and enters its neighbour, so mass
    # and enthalpy are conserved to roundoff and nothing is clipped (near the rotated poles, where dx is small,
    # this is what limits the transport).
    cap = 0.9
    dif_e, dif_n = diffusivity * len_e / dx_e * m_e, diffusivity * len_n / dy_n * m_n

    def substep(fs, _):
        u_e, v_n = u_e0, v_n0
        if compact_threshold is not None:
            c = (fs[0] >= compact_threshold).astype(u.dtype)
            u_e = u_e * (1.0 - jnp.where(u_e > 0, east(c), c))
            v_n = v_n * (1.0 - jnp.where(v_n > 0, north(c), c))
        k_e_out = jnp.maximum(u_e, 0.0) * len_e + dif_e            # out of cell i through its east face
        k_e_in = jnp.maximum(-u_e, 0.0) * len_e + dif_e            # out of cell i+1 through the same face
        k_n_out = jnp.maximum(v_n, 0.0) * len_n + dif_n
        k_n_in = jnp.maximum(-v_n, 0.0) * len_n + dif_n
        total_out = k_e_out + west(k_e_in) + k_n_out + south(k_n_in)  # west face of i is the east face of i-1
        positive = total_out > 0
        denom = jnp.where(positive, total_out, 1.0)                # safe denominator: no inf in the unselected branch
        s = jnp.where(positive, jnp.minimum(1.0, cap * area / (dts * denom)), 1.0)
        out = []
        for X in fs:
            F_e = s * k_e_out * X - east(s * X) * k_e_in
            F_n = s * k_n_out * X - north(s * X) * k_n_in
            div = F_e - west(F_e) + F_n - south(F_n)
            out.append((X - dts / area * div) * oc)
        return tuple(out), None

    fields, _ = jax.lax.scan(substep, tuple(fields), None, length=n_substeps)
    return fields
