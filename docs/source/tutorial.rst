Tutorial: Integrate an Existing Model into JEM 
==============================================

In this tutorial, we are going to couple a differentiable global atmosphere
model `JCM <https://github.com/climate-analytics-lab/jax-gcm>`__ to a slab
ocean model provided by JEM.

Step 1: Identifying the Gap to be JEM-compatible
------------------------------------------------

It is easy to be JEM compatible because JEM only cares if the given component
has two particular member functions:

1. :code:`initialize: Callable[[], ComponentCarry]`: Returns the initial carry
    value.
2. :code:`generate_step_function: Callable[[ComponentCarry], tuple[ComponentCarry, Predictions]]`: 
    Returns the new carry value and predictions. The returned carry value must
    be consistent with the ones returned from :code:`initialize`. User can test
    if two Pytrees are equivalent by using `jax.tree_utils.tree_structure`. The
    first dimension of leaf nodes in :code:`Predictions` is expected to be time
    . The coupler will concat multiple predictions along this dimension.

The :code:`ComponentCarry` in JEM refers to the the state or accumulated value
that is passed from one iteration of a loop to the next, which is the same 
concept as elaborated in `jax.lax.scan <https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>`__
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
temperature simulated in the slab ocean model, which is also not part of the 
native JCM state. Therefore, we need to encapsulate the heat flux and sea
surface temperature forcing variable into the :code:`ComponentCarry` JCM.

Step 2: Write the Adapted Functions
-----------------------------------

Step 3: Put it Together
-----------------------

Optional: Auxilary Functions
----------------------------

