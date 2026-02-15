Tutorial: Integrate an Existing Model into JEM 
==============================================

JEM philosopy
-------------
Each component has its own :code:`ComponentCarry`, a Pytree. JEM collects them
together to form a :code:`CoupledCarry`. Then, JEM use `MapperFunction` to
exchange the information within to achieve coupling.

JEM compatibility
-----------------
JEM expects each of the component to provide

- :code:`initialize: Callable[[], ComponentCarry]`: Returns the initial carry
    value.
- :code:`generate_step_function: Callable[[ComponentCarry], tuple[
    ComponentCarry, Predictions]]`: Returns the new carry value and predictions.
    The returned carry value must be consistent with the ones returned from
    :code:`initialize`. User can test if two Pytrees are equivalent by using
    `jax.tree_utils.tree_structure`. The first dimension of leaf nodes in 
    :code:`Predictions` is expected to be time. The coupler will concat 
    multiple predictions along this dimension.

What is a Mapper?
-----------------
Mapper is a callable that modifies a :code:`CoupledCarray` and returns the
result. While JEM provides a default mapper `jem.mapping.mapper.BasicMapper`.
As long as the signature matches, JEM will any user-defined mapper. Therefore,
it allows users to do more sophisticated checks.

Adapting a Model to be JEM Compatible
-------------------------------------

A Coupled Example
-----------------

