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

The following code is a spring system.

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax.typing import ArrayLike
    from dataclasses import dataclass
    import tree_math

    @tree_math.struct
    @dataclass
    class SpringCarry:
        t: ArrayLike  # time
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
                t = jnp.array(0),
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
                carry.t += dt

                return carry, dict(t=carry.t, x=carry.x, v=carry.v)
            return step_function


And this is how you run it

.. code-block:: python

    total_time = 10
    spring = Spring(init_x=0, init_v=2, k=5.0, m=1.0, dt=0.01)
    final_carry, predictions = jax.lax.scan(
        spring.generate_step_function(),
        spring.initialize(),
        jnp.arange(int(total_time / spring.dt)),
    )


With JEM, you can put two springs together (no interactions yet),

.. code-block:: python

    total_time = 50
    dt = 0.01
    model = Coupler(
        components=dict(
            spring1=Spring(init_x=0, init_v=2, k=5.0, m=1.0, dt=dt),
            spring2=Spring(init_x=2, init_v=5, k=5.0, m=5.0, dt=dt),
        ),
    )

    initial_coupled_carry, final_coupled_carry, predictions = model.run(
        workflow=["mapper", "spring1", "spring2"],
        iterations = int(total_time / dt),
    )


Add in interaction

.. code-block:: python


    interaction_strength = 1.0
    def mapper(coupled_carry):
        f = (coupled_carry["spring2"].x - coupled_carry["spring1"].x) * interaction_strength
        coupled_carry["spring1"].f = f
        coupled_carry["spring2"].f = - f
        return coupled_carry
    
    model.add_mapper("mapper", mapper)
    
    initial_coupled_carry, final_coupled_carry, predictions = model.run(
        workflow=["mapper", "spring1", "spring2"],
        iterations = int(total_time / dt),
    )

To display the result

.. code-block:: python

    import matplotlib.pyplot as plt

    x1 = predictions["spring1"]["x"]
    v1 = predictions["spring1"]["v"]
    x2 = predictions["spring2"]["x"]
    v2 = predictions["spring2"]["v"]
    t = predictions["spring1"]["t"]

    fig, ax = plt.subplots(2,1)
    ax[0].plot(t, x1, label="x1")
    ax[0].plot(t, x2, label="x2")
    ax[1].plot(t, v1, label="v1")
    ax[1].plot(t, v2, label="v2")

    ax[0].legend()
    ax[1].legend()

    fig, ax = plt.subplots(2,1)
    ax[0].plot(x1, x2, label="x")
    ax[1].plot(v1, v2, label="v")

    ax[0].legend()
    ax[1].legend()

    plt.show()
