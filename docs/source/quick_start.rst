Quick Start
=============


Install jem
-----------


.. code-block::
   
   pip install jem

Run the first coupled run
-------------------------

.. code-block::

   jem create_shell --directory exp --shell JCM_SlabOceanModel_SlabLandModel
   python3 exp/main.py


To integrate your model with jem, you need to create an adapter function similar to `make_jem_compatible` as in `jem/components/JCM.py`. In this adapter function, you need to make sure Samudra model object provide the following functions:

1. `initialize`: returns `(initial_state, initial_derived, initial_forcing)` which are all pytrees.
2. `generate_step_function`: Create a step_function that returns `(new_state, new_derived, predictions)`. This `new_derived` must match the structure of `initial_derived` returned by `initialize`
3. (Optional) `predictions_to_xarray`: Convert the received prediction object into an xarray, which will be used when couple models `predictions_to_xarray` is called.
4. (Optional) `get_info`: Returns a dict describing the model, which will be printed when coupled model's `get_info` is called.

Happy coupling!

