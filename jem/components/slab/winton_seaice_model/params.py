"""Tunable parameters of the Winton sea-ice model."""

import jax.numpy as jnp
from flax import struct
from jcm.physics.speedy.params import ModRadConParameters, SurfaceFluxParameters


@struct.dataclass
class WintonSeaiceParameters:
    """Parameters of :class:`~jem.components.slab.winton_seaice_model.WintonSeaiceModel`.

    Every numeric field is a pytree leaf, so ``jax.grad`` of a coupled run with
    respect to any of them works: the parameters travel in the component's
    carry (``carry["params"]``), not in a closure over the model object. The
    fields that select a code path or fix an array shape at trace time (the
    substep counts, the mask convention) are static.

    ``initial_ice_thickness`` is an **initial-condition** parameter: it is read
    once, by :meth:`WintonSeaiceModel.initialize`, and never by ``step``. Vary
    it by passing parameters to ``initialize`` (or to
    ``Coupler.initialize({"seaice": params})``); replacing that leaf in a carry
    that already exists changes nothing, because its value has already been
    copied into the state. Every other numeric field is a process parameter,
    read from the carry every step.

    Attributes
    ----------
    ice_albedo : float or jnp.ndarray
        Shortwave albedo of bare ice.
    snow_albedo : float or jnp.ndarray or None
        Albedo of dry snow on ice. ``None`` (the default) makes snow take
        ``ice_albedo``, so a gradient with respect to ``ice_albedo`` also
        moves the snow albedo; give a value to decouple the two.
    snow_melt_albedo : float or jnp.ndarray or None
        Albedo of melting snow (surface at 0 degC). ``None`` (the default)
        makes it take the dry-snow albedo, resolved as above.
    i0_fraction : float or jnp.ndarray
        Fraction of the absorbed shortwave that penetrates the surface of
        snow-free ice (thsice ``i0swFrac``).
    ksolar : float or jnp.ndarray
        Bulk extinction coefficient (1/m) of the penetrating shortwave in ice.
    lead_closing_thickness : float or jnp.ndarray
        Hibler (1979) lead-closing thickness (m): frazil ice closes the open
        fraction of a cell as if spread over this thickness. Must be positive.
    min_ice_thickness : float or jnp.ndarray
        Thickness (m) below which the ice-covered part of a cell is not
        integrated thermodynamically and a column that melts down to it
        vanishes.
    min_ice_fraction : float or jnp.ndarray
        Fraction below which a cell is not integrated thermodynamically.
    initial_ice_thickness : float or jnp.ndarray
        Uniform thickness (m) of the ice-covered part over ocean points at the
        start of a run; a cell with any ice starts fully covered. Initial
        condition; read only by ``initialize``.
    emissivity : float or jnp.ndarray
        Longwave emissivity of the ice surface. The default is JCM's surface
        emissivity, so ice and atmosphere agree unless one is overridden.
    surface_flux : jcm.physics.speedy.params.SurfaceFluxParameters
        The atmosphere's bulk surface-flux parameters (exchange coefficient
        ``chs``, gust speed ``vgust`` and the stability correction ``dtheta``,
        ``fstab`` and ``lscasym``) used for the fluxes over ice. The default is
        JCM's; pass the same object the coupled atmosphere runs with. Its
        leaves are differentiable like every other parameter.
    transport_diffusivity : float or jnp.ndarray
        Lateral diffusivity (m2/s) of the ice transport step. Read only when
        the model was built with transport.
    n_substeps : int
        Thermodynamic substeps per coupling step. Static.
    n_flux_iterations : int
        Newton iterations of the implicit surface-temperature solve per
        substep. Static. The oracle comparison validates a single iteration;
        coupled runs use three.
    transport_n_substeps : int
        Substeps of the transport step per coupling step. Static.
    ocean_mask_value : float
        Value of the grid's binary mask that marks an ocean cell (0 = ocean,
        1 = land). Static: it selects which cells the model integrates at
        trace time, and is a mask convention rather than a physical tunable.

    """

    ice_albedo: float | jnp.ndarray = 0.60
    snow_albedo: float | jnp.ndarray | None = None
    snow_melt_albedo: float | jnp.ndarray | None = None
    i0_fraction: float | jnp.ndarray = 0.3
    ksolar: float | jnp.ndarray = 1.5
    lead_closing_thickness: float | jnp.ndarray = 0.5
    min_ice_thickness: float | jnp.ndarray = 0.01
    min_ice_fraction: float | jnp.ndarray = 0.01
    initial_ice_thickness: float | jnp.ndarray = 0.0
    emissivity: float | jnp.ndarray = struct.field(
        default_factory=lambda: ModRadConParameters.default().emisfc
    )
    surface_flux: SurfaceFluxParameters = struct.field(
        default_factory=SurfaceFluxParameters.default
    )
    transport_diffusivity: float | jnp.ndarray = 2e4
    n_substeps: int = struct.field(pytree_node=False, default=4)
    n_flux_iterations: int = struct.field(pytree_node=False, default=3)
    transport_n_substeps: int = struct.field(pytree_node=False, default=12)
    ocean_mask_value: float = struct.field(pytree_node=False, default=0.0)

    @classmethod
    def default(cls) -> "WintonSeaiceParameters":
        """Return the default parameters."""
        return cls()
