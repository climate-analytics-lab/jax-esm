Tutorial: Integrate an Existing Model into JEM 
==============================================

In this tutorial, we are going to couple a differentiable global atmosphere
model `JCM <https://github.com/climate-analytics-lab/jax-gcm>`__ to a slab
ocean model provided by JEM (Figure 1).

.. figure:: _static/jcm_som.svg
   :scale: 25 %
   :alt: Schematic diagram showing the relationshipe between JCM and slab 
         ocean model
   :align: center

   Figure 1: Schematic diagram showing the relationshipe between JCM and
   the slab ocean model. The atmosphere model needs the sea surface
   temperature, and the ocean model needs the heat flux. 


Step 1: Identifying the Gap to be JEM-compatible
------------------------------------------------

A model object will be JEM-compatible if it has two particular member functions:

1. :code:`initialize: Callable[[], ComponentCarry]`:
   Returns the initial carry value.
2. :code:`generate_step_function: Callable[[], StepFunction]`, where
   :code:`StepFunction` is
   :code:`Callable[[ComponentCarry, SimulationTime], tuple[ComponentCarry, Predictions]]`:
   returns the function that advances the component by one coupling timestep.

   - The returned carry value must be consistent with the ones returned
     from :code:`initialize`. User can test if two Pytrees are equivalent by
     using `jax.tree_utils.tree_structure`.
   - The first dimension of leaf nodes in :code:`Predictions` is expected to be
     time. The coupler will concat multiple predictions along this dimension.

The :code:`ComponentCarry` in JEM refers to the the state object that is passed
from one iteration of a loop to the next, which is the same  concept as 
elaborated in `jax.lax.scan <https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>`__
. In the science field, this corresponds to the "state" of the system, carrying
necessary information for the system to evolve in time. In JEM, the carry value
means more than that. Carry values also include (1) any variables that will 
participate in differentiability of the resulting model, such as forcing or 
physical parameters, and (2) convenient variables that can be but hard to
diagnosed from the state, such as turbulenet heat fluxes.

Therefore, the desired coupling feature determines the structure of carry values
, which decide how much adaptation one would need to integrate the chosen model
into JEM. In our case, we want to (1) allow JCM to export the total heat flux
such that the slab ocean model use it. However, because heat fluxes are not part
of the model state variables. Also (2) we want to force JCM with the sea surface
temperature simulated in the slab ocean model, which is the :code:`ForcingData`
of the native JCM objects. Therefore, we need to encapsulate the heat flux and
sea surface temperature forcing variable into the :code:`ComponentCarry` JCM. 
A carry value structure we can adopt is

.. code-block:: python
   
    # `ComponentCarry` of JCM
    {
        "state": jcm_native_modal_state,
        "derived": JCMDerived(
            physics,           # JCM's own physics carry, passed straight through
            ...,               # surface heat fluxes, W/m^2, positive upward
            total_freshwater_flux,
        ),
        "forcing": jcm_native_forcing_data,   # holds sea_surface_temperature
    }

For the slab ocean model, the component carry value we adopt is

.. code-block:: python
   
    # `ComponentCarry` of the slab ocean model
    {
        "state": OceanState(sim_time, sea_surface_temperature, mixed_layer_depth),
        "forcing": OceanForcing(total_heat_flux, q_flux),
        "derived": OceanDerived(...),
    }


Step 2: Adapt Component Functions
---------------------------------

JEM ships this adapter for JCM, so you do not have to write one yourself:
:code:`make_jem_compatible` attaches :code:`initialize`,
:code:`generate_step_function` and the two optional methods onto a stock
:code:`jcm.model.Model` instance. It is a useful worked example of the whole
contract, so it is reproduced here in full, directly from the source:

.. literalinclude:: ../../jem/components/jcm_component.py
   :language: python
   :pyobject: make_jem_compatible
   :linenos:

Three things to note:

- :code:`initialize` runs one throwaway coupling step, purely to learn the
  structure of JCM's physics carry. That structure is still changing upstream,
  and deriving it rather than hard-coding it keeps the adapter working across
  JCM releases.
- :code:`step_function` calls :code:`model.run_from_state_with_carry()` with the
  coupling interval as both :code:`save_interval` and :code:`total_time`, so JCM
  sub-steps internally at its own timestep and returns one saved record per
  coupling step.
- The surface fluxes are negated on the way out: JCM publishes them downward
  positive, and JEM's convention is upward positive. Doing this once, at the
  adapter boundary, is what keeps every mapper downstream sign-consistent.


Step 3: Exchange Variables Between Components
---------------------------------------------

.. code-block:: python

    def interaction_between_atm_and_ocn(coupled_carry):
        atm = coupled_carry["atm"]
        ocn = coupled_carry["ocn"]

        atm["forcing"].sea_surface_temperature = ocn["state"].sea_surface_temperature
        ocn["forcing"].total_heat_flux = atm["derived"].total_heat_flux

        return coupled_carry

Step 4: Couple JCM to the Slab Ocean Model
------------------------------------------

.. code-block:: python

    aquaplanet_grid = generate_slab_grid("JCM::T31")

    model = Coupler(
        components=dict(
            atm=atm_model,
            ocn=SlabOceanModel(
                grid=aquaplanet_grid,
                start_datetime=start_datetime,
                timestep=coupling_timestep / one_second,
            ),
        ),
        mappers=dict(interaction_between_atm_and_ocn=interaction_between_atm_and_ocn),
    )

    simulation_interval = jdt.to_timedelta(10, "day")
    initial_carry, final_carry, predictions = model.run(
        workflow=["interaction_between_atm_and_ocn", "atm", "ocn"],
        iterations = int(simulation_interval / coupling_timestep),
    )

    output_dict = model.predictions_to_xarray(predictions)

Full Code
---------

:doc:`quick_start` puts exactly these four steps together into one
self-contained script you can copy and run.
