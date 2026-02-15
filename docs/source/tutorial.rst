Tutorial: Integrate an Existing Model into JEM 
==============================================

JEM philosopy
-------------
Each component has its own :code:`ComponentCarry`, a Pytree. JEM collects them
together to form a :code:`CoupledCarry: Dict[ComponentName, ComponentCarry]`.
Then, JEM use `MapperFunction` to exchange the information within to achieve 
coupling.

JEM compatibility
-----------------
JEM expects each of the component to provide

- :code:`initialize: Callable[[], ComponentCarry]`: Returns the initial carry
    value.
- :code:`generate_step_function: Callable[[ComponentCarry], tuple[ComponentCarry, Predictions]]`: 
    Returns the new carry value and predictions. The returned carry value must
    be consistent with the ones returned from :code:`initialize`. User can test
    if two Pytrees are equivalent by using `jax.tree_utils.tree_structure`. The
    first dimension of leaf nodes in :code:`Predictions` is expected to be time
    . The coupler will concat multiple predictions along this dimension.

What is a Mapper?
-----------------
Mapper is a callable that modifies a :code:`CoupledCarray` and returns the
result, i.e., :code:`Callabel[[CoupledCarry], CoupledCarry]`. While JEM 
provides a default mapper `jem.mapping.mapper.BasicMapper`. As long as the
signature matches, JEM will any user-defined mapper. Therefore, it allows users
to do more sophisticated checks.

Adapting a Model to be JEM Compatible
-------------------------------------

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax.typing import ArrayLike
    from dataclasses import dataclass
    import tree_math

    @tree_math.struct
    @dataclass
    class SpringCarry:
        x: ArrayLike  # position
        v: ArrayLike  # velocity
        m: ArrayLike  # mass
        k: ArrayLike  # spring coefficient
        f: ArrayLike  # external force

    class Spring:

        def __init__(self, init_x, init_v, k, m, dt):
            self.init_x = init_x
            self.init_v = init_v
            self.k = k
            self.m = m
            self.dt = dt
         
        def initialize(self):
            return SpringCarry(
                x = jnp.array(self.init_x),
                v = jnp.array(self.init_v),
                m = jnp.array(self.m),
                k = jnp.array(self.k),
                f = jnp.array(0),
            )

        def generate_step_function(self):
            dt = self.dt
            def step_function(carry, step):
                """Integrates one time step of a harmonic oscillator."""
                
                # Physics: a = -k/m * x + f
                acceleration = - (carry.k * carry.x + carry.f) / carry.m
                
                # Update state (Semi-implicit Euler for better stability)
                new_v = carry.v + acceleration * dt
                new_x = carry.x + new_v * dt
               
                carry.v = new_v
                carry.x = new_x
                
                return carry, dict(x=new_x, v=new_v)
            return step_function

    #Simulation 
    total_time = 10
    spring = Spring(init_x=0, init_v=2, k=5.0, m=1.0, dt=0.01)
    final_carry, predictions = jax.lax.scan(
        spring.generate_step_function(),
        spring.initialize(),
        jnp.arange(int(total_time / spring.dt)),
    )

    # Display
    import matplotlib.pyplot as plt
    x = predictions["x"]
    v = predictions["v"]
    fig, ax = plt.subplots(1,1)
    ax.plot(jnp.arange(len(x)), x, label="x")
    ax.plot(jnp.arange(len(v)), v, label="v")
    ax.legend()
    plt.show()

A Coupled Example
-----------------

