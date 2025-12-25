"""Example of coupling of simple slab atmosphere and ocean models."""

def test_integration():

    import matplotlib.pyplot as plt

    from jax_esm.components import SlabOceanModel, SlabAtmosphereModel
    import jax_datetime as jdt
    from jax_esm.coupling.factory.simple_coupling import couple_atm_ocn as couple
    from pathlib import Path

    resolution = 31
    grid_specification = f"JCM::T{resolution:d}"

    coupling_timestep = 86400.0
    start_datetime = jdt.to_datetime("2000-01-01")
    simulation_interval = jdt.to_timedelta(5, "day")
    output_dir = Path("output/SAM_SOM").resolve()

    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True)

    # Creating components
    components = dict(
        atm=SlabAtmosphereModel(
            grid_specification=grid_specification,
            timestep=3600.0 * 12,
            start_datetime=start_datetime,
            save_interval=coupling_timestep,
        ),
        ocn=SlabOceanModel(
            grid_specification=grid_specification,
            timestep=coupling_timestep,
            start_datetime=start_datetime,
            save_interval=coupling_timestep,
            relaxation_time=60 * 86400.0,
        ),
    )

    # Creating model
    model = couple(**components)

    # Obtain initial condition

    
    trajectory_function = model.generate_trajectory_function(
        start_time=0,
        end_time=simulation_interval / jdt.to_timedelta(1, "second"),
        jitted=True,
        show_progress=True,
        tqdm_kwargs=dict(desc="Simulation"),
    )

    def measure(param):
        initial_coupled_state_forcing = model.initialize()
        initial_coupled_state_forcing["atm"][1].scalar.bulk_drag_coefficient = param
        state_holder, predictions = trajectory_function(initial_coupled_state_forcing)
        result = state_holder["state"]["ocn"].prog.sea_surface_temperature.sum() 
        return result
    
    import jax 
    import jax.numpy as jnp 
    grad_measure = jax.grad(measure)
    
    measure_results = []
    grad_measure_results = []
    params = jnp.linspace(0, 1e-2, 5)
    for i, param in enumerate(params):
        print(f"[{i:d}] Evaluating param {param:f}...")
        measure_results.append(measure(param))
        grad_measure_result = grad_measure(param)
        print("Grad_measure: ", grad_measure_result)
        grad_measure_results.append(grad_measure_result) 

    fig, ax = plt.subplots(2,1)
    ax[0].plot(params, measure_results)    
    ax[1].plot(params, grad_measure_results)
    
    ax[0].set_title("Measure") 
    ax[1].set_title("Grad Measure") 
    plt.show()
    
if __name__ == "__main__":
    test_integration()


