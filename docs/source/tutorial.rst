Tutorial: Integrate an Existing Model into 
==========================================

JEM philosopy
-------------

Each component has three pytrees:

1. :code:`State`: Holding the state of the component. This state will be passed 
    to each component's evolution function.
2. :code:`Derived`: Holding the derived information from :code:`State`. It is
    an auxiliary object to help communicating information between coupled
    components. For example, while it may be to recompute surface heat flux
    from atmosphere models' state, the recomputing is difficult. Therefore, it
    is convenient to put the heat flux in atmosphere model's :code:`Derived`.
3. :code:`Forcing`: Holding the variables that can be passed from other compon-
    ents to influence the evolution of :code:`State` of the component. For ex-
    ample, the sea surface temperature is a forcing to drive atmosphere's evo-
    lution.

The components receive information of other models from their :code:`Forcing`.
To send the forcing, user can provide their :code:`ForcingMapper`, which takes
in the :code:`State` and :code:`Derived` from source models and map them to the
:code:`Forcing` of the target components.

JEM compatibility
-----------------

JEM requires each of the component to provide

- :code:`initialize: Callable[[], tuple[State, Derived, Forcing]]`: Returns
    three pytrees representing initial state, derived and forcing.


Forcing Mapping
---------------


Coupling Models
---------------



