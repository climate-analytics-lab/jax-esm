"""Tunable parameters of the Winton sea-ice model."""

import jax.numpy as jnp
from flax import struct
from jcm.physics.speedy.params import ModRadConParameters, SurfaceFluxParameters


@struct.dataclass
class IceSurfaceFluxParameters:
    """The bulk surface-flux parameters the ice uses, in the form ``jax.grad`` can take.

    These are the four numeric leaves and the one flag of JCM's
    ``SurfaceFluxParameters`` that the fluxes over ice read. JCM's own struct
    is not nested in :class:`WintonSeaiceParameters` because it also holds
    boolean fields (``lscasym``, ``lskineb``): a boolean leaf in
    ``carry["params"]`` makes ``jax.grad`` over the whole parameter pytree
    fail unless the caller passes ``allow_int``. Here the one flag the ice
    reads, ``lscasym``, is static, so every leaf is a float and ``jax.grad``
    over the whole ``WintonSeaiceParameters`` works. ``lskineb`` (skin
    temperature from an energy balance) belongs to the atmosphere's land and
    sea surface and plays no part in the ice, so it is not carried.

    Use :meth:`from_jcm` to take the values a coupled atmosphere runs with.

    Attributes
    ----------
    chs : float or jnp.ndarray
        Heat exchange coefficient over sea (JCM ``chs``).
    vgust : float or jnp.ndarray
        Wind speed (m/s) of sub-grid-scale gusts, added in quadrature to the wind.
    dtheta : float or jnp.ndarray
        Potential temperature gradient of the stability correction.
    fstab : float or jnp.ndarray
        Amplitude of the stability correction (fraction).
    lscasym : bool
        Whether the stability correction is asymmetric. Static: it selects a
        code path and is not a tunable.

    """

    chs: float | jnp.ndarray = 0.9e-3
    vgust: float | jnp.ndarray = 5.0
    dtheta: float | jnp.ndarray = 3.0
    fstab: float | jnp.ndarray = 0.67
    lscasym: bool = struct.field(pytree_node=False, default=True)

    @classmethod
    def from_jcm(cls, surface_flux: SurfaceFluxParameters) -> "IceSurfaceFluxParameters":
        """Take the parameters the ice reads from JCM's ``SurfaceFluxParameters``.

        ``lscasym`` must be concrete (a Python or NumPy bool) because it is
        static here, so build the ice parameters outside ``jit``.
        """
        return cls(
            chs=surface_flux.chs,
            vgust=surface_flux.vgust,
            dtheta=surface_flux.dtheta,
            fstab=surface_flux.fstab,
            lscasym=bool(surface_flux.lscasym),
        )

    @classmethod
    def default(cls) -> "IceSurfaceFluxParameters":
        """Return JCM's default surface-flux parameters, restricted to those the ice reads."""
        return cls.from_jcm(SurfaceFluxParameters.default())


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
    surface_flux : IceSurfaceFluxParameters
        The bulk surface-flux parameters (exchange coefficient ``chs``, gust
        speed ``vgust`` and the stability correction ``dtheta``, ``fstab`` and
        ``lscasym``) used for the fluxes over ice. The default is JCM's. To use
        the values the coupled atmosphere runs with, pass its
        ``SurfaceFluxParameters`` (converted with
        :meth:`IceSurfaceFluxParameters.from_jcm` on construction). Its numeric
        leaves are differentiable like every other parameter; the one flag
        (``lscasym``) is static, so every leaf of this dataclass is a float and
        ``jax.grad`` over the whole parameter pytree needs no ``allow_int``.
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
    surface_flux: IceSurfaceFluxParameters = struct.field(
        default_factory=IceSurfaceFluxParameters.default
    )
    transport_diffusivity: float | jnp.ndarray = 2e4
    n_substeps: int = struct.field(pytree_node=False, default=4)
    n_flux_iterations: int = struct.field(pytree_node=False, default=3)
    transport_n_substeps: int = struct.field(pytree_node=False, default=12)
    ocean_mask_value: float = struct.field(pytree_node=False, default=0.0)

    def __post_init__(self) -> None:
        """Accept JCM's ``SurfaceFluxParameters`` for ``surface_flux``.

        The atmosphere's own object is the natural thing to pass, so it is
        converted here rather than making every caller do it.
        """
        if isinstance(self.surface_flux, SurfaceFluxParameters):
            object.__setattr__(
                self, "surface_flux", IceSurfaceFluxParameters.from_jcm(self.surface_flux)
            )

    @classmethod
    def default(cls) -> "WintonSeaiceParameters":
        """Return the default parameters."""
        return cls()
