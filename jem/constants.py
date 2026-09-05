"""Surface constants JEM owns, and the rule for everything else.

Every constant the atmosphere also knows about is read from **jcm.constants**,
so the coupler and the atmosphere cannot disagree about ``grav``, ``cpd``,
``tmelt`` and friends::

    import jcm.constants as c
    ... c.grav, c.cpd, c.sbc, c.tmelt, c.rhoi, c.alhf, c.solc

Read them by *attribute access on the module*, never ``from jcm.constants
import grav``: the latter binds the value at import time and does not track a
process-global ``jcm.constants.set_constants(...)`` override.

What remains here is what ``jcm.constants`` genuinely does not define: the
properties of a seawater / land / ice *surface slab* and the bulk-formula
coefficients the idealized slab atmosphere uses. They follow the same pattern
as JCM's own module — a frozen dataclass, a live singleton, a
:func:`set_constants` override and a module ``__getattr__`` that forwards bare
name access to the singleton — so ``jem.constants.ocean_density`` honours an
override made after import.

Constants removed in v1.0 because ``jcm.constants`` already owns them, with
the JCM name to use instead (values differ where noted, and JCM's value now
applies):

===================================================  ==================  =========================
removed name                                         use instead         value change
===================================================  ==================  =========================
``g0``                                               ``c.grav``          none (9.81)
``stephan_boltzmann_const``                          ``c.sbc``           none (5.67e-8)
``freezing_point_K``, ``ice_melting_point_K``        ``c.tmelt``         none (273.15)
``ice_density``                                      ``c.rhoi``          none (917)
``ice_latent_heat_fusion``                           ``c.alhf``          3.34e5 -> 3.33e5 J/kg
``atmosphere_specific_heat_capacity_...``            ``c.cpd``           1004.0 -> 1004.64 J/K/kg
``solar_const``                                      ``c.solc``          1367 -> 1361 W/m2
===================================================  ==================  =========================

``ice_latent_heat_fusion`` and ``cpd`` change value because JCM's are the
internally consistent ones: JCM *derives* the heat of fusion from the latent
heats of sublimation and condensation (``alhf = alhs - alhc``), and its ``cpd``
is the higher-precision ECHAM-6.3 value that its own ``rd = akap*cpd`` is built
on. A coupler that kept the old numbers would put ~0.3 % of every ice-formation
enthalpy, and 0.06 % of every sensible-heat flux, on the floor between the two
models.

``default_mld_min``/``default_mld_max`` and ``default_land_depth_min``/
``default_land_depth_max`` were removed with no replacement here: a default for
a component parameter belongs to that component's ``Parameters`` dataclass
(``SlabOceanParameters.mixed_layer_depth_min``, ...), which is now the single
place it is written down.
"""

from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class SurfaceConstants:
    """Physical constants of the surface slabs, which ``jcm.constants`` lacks.

    Attributes
    ----------
    ocean_density : float
        Seawater density (kg/m3).
    ocean_specific_heat_capacity : float
        Specific heat capacity of seawater (J/kg/K).
    land_density : float
        Effective density of the land surface slab (kg/m3).
    land_specific_heat_capacity : float
        Effective specific heat capacity of the land surface slab (J/kg/K).
    surface_air_density : float
        Air density at the surface, for bulk flux formulas (kg/m3).
    bulk_drag_coefficient : float
        Bulk aerodynamic drag coefficient (dimensionless). The slab atmosphere
        carries its own value in its forcing so an exchanger can overwrite it
        per grid cell; this is the value that forcing is initialized to.
    atmosphere_column_mass : float
        Mass of the atmospheric column the slab atmosphere heats (kg/m2).
    ice_thermal_conductivity : float
        Thermal conductivity of sea ice (W/m/K).
    seawater_freezing_point_K : float
        Freezing point of seawater at salinity ~34 psu (K); -1.8 degC. Distinct
        from ``jcm.constants.tmelt``, which is the *fresh*-water melting point
        used for the ice/snow surface.

    Notes
    -----
    ``land_density``, ``land_specific_heat_capacity`` and
    ``ice_thermal_conductivity`` are not read by any component today:
    :class:`~jem.components.slab.slab_land_model.SlabLandModel` uses SPEEDY's
    volumetric heat capacities directly (they are per-layer tunables, so they
    live in ``SlabLandParameters``), and the sea-ice model has no conductive
    temperature profile. They are kept because they are the material properties
    any richer surface slab needs, and because deleting and re-deriving them is
    more error-prone than keeping them documented in one place.

    """

    ocean_density: float = 1025.0
    ocean_specific_heat_capacity: float = 3985.0
    land_density: float = 3000.0
    land_specific_heat_capacity: float = 830.0
    surface_air_density: float = 1.22
    bulk_drag_coefficient: float = 1e-3
    atmosphere_column_mass: float = 1e4
    ice_thermal_conductivity: float = 2.03
    seawater_freezing_point_K: float = 271.35


#: The live singleton every ``jem.constants.<name>`` access resolves against.
surface_constants = SurfaceConstants()


def set_constants(constants: SurfaceConstants | None = None, **overrides: float) -> None:
    """Rebind the process-global surface constants.

    Call it *before* constructing the components: a component reads the
    constants while building its initial carry and while tracing its step, so a
    later override does not reach an already-traced step function.

    Parameters
    ----------
    constants : SurfaceConstants, optional
        Replace the whole set at once.
    **overrides : float
        Override individual fields of the current set, e.g.
        ``set_constants(ocean_density=1030.0)``.

    Raises
    ------
    ValueError
        If both forms are combined, or a keyword does not name a field.

    """
    global surface_constants

    if constants is not None:
        if overrides:
            raise ValueError(
                "set_constants() takes either a SurfaceConstants instance or "
                "field overrides, not both."
            )
        surface_constants = constants
        return

    known = {field.name for field in fields(SurfaceConstants)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(
            f"Unknown surface constant(s) {unknown!r}; "
            f"SurfaceConstants has {sorted(known)!r}."
        )
    surface_constants = replace(surface_constants, **overrides)


def __getattr__(name: str) -> Any:
    """Forward bare-name access to the live singleton (PEP 562).

    This is what makes ``import jem.constants as constants; ...
    constants.ocean_density`` honour :func:`set_constants`, in the same way
    ``jcm.constants`` does for the constants it owns.
    """
    try:
        return getattr(surface_constants, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
