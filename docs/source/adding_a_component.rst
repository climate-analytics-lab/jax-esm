Adding a component
===================

A component is any object satisfying the :class:`jem.base.component.Component`
protocol: a ``name``, an ``initialize()`` and a ``step(carry, time)``. The
protocol is *runtime-checkable*, which for a :code:`typing.Protocol` means "has
these attributes" -- there is **no base class to inherit from**, so an external
model is adapted by a thin wrapper *object* rather than by attaching methods to
it or monkey-patching it. This page works through JEM's own wrapper for JCM,
:class:`~jem.components.jcm.component.JCMComponent` (Figure 1), as the worked
example; the same shape applies to any model.

.. figure:: _static/jcm_som.svg
   :scale: 25 %
   :alt: Schematic diagram showing the relationshipe between JCM and slab
         ocean model
   :align: center

   Figure 1: Schematic diagram showing the relationship between JCM and
   the slab ocean model. The atmosphere model needs the sea surface
   temperature, and the ocean model needs the heat flux.


The three required members
---------------------------

1. :code:`name: str` -- the component's name in the coupler's workflow, in the
   coupled carry and in its output.
2. :code:`initialize() -> Carry` -- returns the initial carry value. It must
   **not** integrate the model; building the initial pytrees is all it may do.
3. :code:`step(carry, time) -> tuple[Carry, Diagnostics]` -- advances the
   component by one coupling timestep.

   - :code:`time` is a :class:`jem.base.component.CouplingTime`: the coupler's
     clock (step index, simulation time in seconds, and the static calendar
     facts behind :code:`time.year_fraction`). The component keeps no clock of
     its own.
   - The returned carry must have exactly the pytree structure, shapes and
     dtypes of the one :code:`initialize` produced -- that is what
     :code:`jax.lax.scan` requires. Two pytrees can be compared with
     :code:`jax.tree_util.tree_structure`.
   - :code:`Diagnostics` is this step's output pytree. The coupler stacks it
     over the run, giving every leaf a leading time axis.

The coupler raises :code:`TypeError` naming the missing member if an object
does not satisfy the protocol.


The three optional capabilities
---------------------------------

Three further capabilities are optional, and each is detected with
:code:`isinstance` at the point it is used -- a component that omits one is
simply skipped there, never broken:

.. list-table::
   :header-rows: 1

   * - Capability
     - Method
     - What it is for
   * - :class:`~jem.base.component.SupportsXarray`
     - ``to_xarray(diagnostics, time)``
     - Label a run's **stacked** diagnostics (every step's output, already
       given a leading time axis by the coupler) as an ``xarray.Dataset`` on
       the coupler's :class:`~jem.base.component.TimeAxis` -- not the
       :class:`~jem.base.component.CouplingTime` the other two rows take --
       so a component's output can be written and merged.
   * - :class:`~jem.base.component.SupportsCheckpoint`
     - ``save_carry(carry, directory)`` / ``load_carry(directory)``
     - Write and restore a carry that is not a plain pytree -- Veros' restart
       is an HDF5 file, not an array JAX can flatten.
   * - :class:`~jem.base.component.SupportsBind`
     - ``bind(*, coupling_timestep, start_date, calendar)``
     - Called once, when the component is registered. The place to check the
       coupler's clock against the model's own and **refuse**, with
       ``ValueError``, a configuration that cannot work (a coupling timestep
       that does not divide the model's own).


Designing the carry
---------------------

The :code:`Carry` in JEM refers to the state object that is passed
from one iteration of a loop to the next, which is the same concept as
elaborated in `jax.lax.scan <https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>`__
. In the science field, this corresponds to the "state" of the system, carrying
necessary information for the system to evolve in time. In JEM, the carry value
means more than that. Carry values also include (1) any variables that will
participate in differentiability of the resulting model, such as forcing or
physical parameters, and (2) convenient variables that can be but hard to
diagnose from the state, such as turbulent heat fluxes.

Therefore, the desired coupling feature determines the structure of carry
values, which decide how much adaptation one would need to integrate the
chosen model into JEM. Every packaged component follows the same convention --
a plain :code:`dict` with :code:`state`/:code:`forcing`/:code:`derived` keys
(plus :code:`params` for a component with tunable parameters) -- not because
the coupler enforces it (it never looks inside a carry), but because it is
what keeps an exchanger readable: an exchanger only ever moves a
:code:`derived` (or :code:`state`) field of one component into a
:code:`forcing` field of another.

For JCM, we want to (1) allow JCM to export the total heat flux such that the
slab ocean model can use it -- heat fluxes are not part of the model's own
state variables -- and (2) force JCM with the sea surface temperature the
slab ocean simulates, which is the :code:`ForcingData` of the native JCM
objects. So the carry structure the adapter adopts is

.. code-block:: python

    # carry of the JCM component
    {
        "state":   jcm_native_modal_state,
        "physics": jcm_cross_step_physics_carry,  # threaded back in every step
        "forcing": jcm_native_forcing_data,       # holds sea_surface_temperature
        "derived": JCMDerived(
            physics,                # JCM's own per-step diagnostics, opaque
            total_heat_flux,        # W/m^2, positive upward
            total_freshwater_flux,  # kg/m^2/s, positive upward (evap - precip)
            evaporation, precipitation, u0, v0,
        ),
    }

For the slab ocean model, the carry is

.. code-block:: python

    # carry of the slab ocean model
    {
        "params":  SlabOceanParameters(...),   # differentiable tunables
        "state":   OceanState(sea_surface_temperature),
        "forcing": OceanForcing(total_heat_flux, q_flux),
        "derived": OceanDerived(...),
    }

Neither carries a simulation time: the coupler owns the one clock and hands it
to :code:`step`.

The carried parameters are differentiable, but only the ones :code:`step`
actually reads from the carry can be varied *there*: replacing
:code:`mixed_layer_depth_max` in :code:`carry["params"]` changes the run,
while replacing :code:`initial_sst` changes nothing, because it was read
once by :code:`initialize` and has already been copied into the state. An
initial-condition parameter is varied by passing parameters to
:code:`initialize` -- :code:`ocn.initialize(params)`, or
:code:`coupler.initialize({"ocn": params})` for the coupled model -- which
builds the initial state from them and carries them. See the *Parameters*
section of :doc:`design/architecture` for the pattern in full.


A worked wrapper: JCMComponent
---------------------------------

JEM ships this wrapper for JCM, so you do not have to write one yourself:
:class:`jem.components.jcm.component.JCMComponent` holds a
:code:`jcm.model.Model` and satisfies the protocol on its behalf. It is a
wrapper *object*, not an in-place adaptation -- nothing is attached to the
model, so the atmosphere you configured is the atmosphere JEM drives.

:code:`bind` is how the component learns the coupler's clock. The coupler
calls it once, when the component is registered, and this is the place to
refuse a configuration that cannot work:

.. literalinclude:: ../../jem/components/jcm/component.py
   :language: python
   :pyobject: JCMComponent.bind
   :linenos:

:code:`step` then advances the atmosphere by exactly one coupling interval:

.. literalinclude:: ../../jem/components/jcm/component.py
   :language: python
   :pyobject: JCMComponent.step
   :linenos:

Four things to note:

- :code:`initialize` does not integrate: it builds the initial pytrees from
  :code:`Model.bootstrap_state()` plus a structural template of the
  diagnostics dict. An earlier adapter ran a whole throwaway coupling step
  just to learn that structure, which cost a step per run *and* started the
  atmosphere one interval ahead of the coupler's clock.
- The **physics carry is threaded**: JCM keeps cross-step physics state
  (sub-cycled radiation, prior-step TKE, term-to-term tendencies) in a carry
  that :code:`run_from_state_with_carry` takes in and hands back. It lives
  under :code:`carry["physics"]` and goes straight back in; dropping it would
  reset that memory once per coupling interval.
- :code:`step` calls :code:`model.run_from_state_with_carry()` with the
  coupling interval as both :code:`save_interval` and :code:`total_time`, so
  JCM sub-steps internally at its own timestep and returns one saved record
  per coupling step.
- The surface fluxes are converted on the way out, in
  :mod:`jem.components.jcm.exchange_fields`: JCM publishes :code:`hfluxn`
  downward positive and its water fluxes in :code:`g m-2 s-1`, and JEM's
  convention is upward positive in :code:`kg m-2 s-1`. Doing this once, at
  the component boundary, is what keeps every exchanger downstream sign- and
  unit-consistent.


Exchanging with other components
------------------------------------

An exchanger is a plain function
:code:`(dict[str, Carry], CouplingTime) -> dict[str, Carry]`. It is traced with
the rest of the coupled step, so it must build new structs rather than assign
into the carries it was handed, and must not change their pytree structure:

.. code-block:: python

    def interaction_between_atm_and_ocn(components, time):
        del time  # this exchange does not depend on the date
        atm, ocn = components["atm"], components["ocn"]

        atm = dict(atm, forcing=atm["forcing"].replace(
            sea_surface_temperature=ocn["state"].sea_surface_temperature,
        ))
        ocn = dict(ocn, forcing=ocn["forcing"].replace(
            total_heat_flux=atm["derived"].total_heat_flux,
        ))

        return dict(components, atm=atm, ocn=ocn)

The clock is passed in so a time-dependent coupling (a ramped forcing, a
lagged exchange) needs no state of its own.

This exchange only *moves* fields, which is what most of a coupled model does,
so it can be written as a table instead --
:func:`~jem.exchangers.default_exchangers` builds exactly these two rows (and
the land and sea-ice ones, when those components are present) for you:

.. code-block:: python

    from jem import default_exchangers

    exchangers = default_exchangers({"atm": atm, "ocn": ocn})

or, spelled out as an explicit :class:`~jem.exchangers.Exchange` table (see
:doc:`python_api` for the full example, side by side with the standard
wiring):

.. code-block:: python

    from jem import Exchange, ExchangeSpec

    exchangers = {"exchange": Exchange([
        ExchangeSpec("atm.derived.total_heat_flux", "ocn.forcing.total_heat_flux"),
        ExchangeSpec("ocn.state.sea_surface_temperature",
                     "atm.forcing.sea_surface_temperature"),
    ])}

Write the function instead of the table when the exchange is something a
table cannot express: a flux computed from two components' states, a unit
conversion, a coupling that depends on the date. See :doc:`design/architecture`
for the standard table's rows, the regridding keys a mixed-grid run uses, and
the one-step lag the default workflow implies.


Registering it
----------------

.. code-block:: python

    aquaplanet_grid = SlabGrid.from_coords(atm_model.coords.horizontal)

    coupler = Coupler(
        {
            "atm": JCMComponent(atm_model),
            "ocn": SlabOceanModel(aquaplanet_grid),
        },
        {"interaction_between_atm_and_ocn": interaction_between_atm_and_ocn},
        coupling_timestep=coupling_timestep,
        start_date=start_date,
    )

The workflow defaults to every exchanger followed by every component, which
for this model is :code:`("interaction_between_atm_and_ocn", "atm", "ocn")` --
pass :code:`workflow=[...]` to the constructor to choose a different coupling
scheme. The sequence may be nested, and a name may appear more than once: an
element listed *n* times runs *n* times per coupled step, on a clock *n*
times faster. :code:`workflow=[["interaction_between_atm_and_ocn", "atm"] *
24, "ocn"]` runs the atmosphere hourly inside a daily ocean coupling.

Once the component is built and wired in, the checklist to make it a full
citizen of the codebase (:doc:`design/architecture`'s *Adding a new
component* has the complete version):

- Export it from ``jem/components/__init__.py`` (lazily, via the module's
  ``__getattr__``, if it pulls in an optional dependency, the way Veros does).
- Give it a ``jem/config/<group>/<option>.yaml`` naming the class as
  ``_target_`` and nothing else -- physics defaults live on the Python class,
  never repeated in YAML.
- Test it through ``Coupler.generate_trajectory_function(2)``, not just its
  own ``step`` in isolation: a component-only test cannot catch a
  carry-structure mismatch, which only ``lax.scan`` sees.

:doc:`python_api` puts the whole thing -- construction, coupling and the run
loop -- together into one script; :doc:`getting_started` shows the same model
composed from the command line.
