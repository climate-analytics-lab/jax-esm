
from itertools import chain
import networkx as nx

def flatten(target: List):
    return list(chain.from_iterable(target))


regridders = dict(
    atm_to_exchangegrid = dict(
        bilinear = Regridder("atm_exchangegrid_regrid_weight_bilinear.nc"),
    ),
    exchangegrid_to_atm = dict(
        conserve = Regridder("exchangegrid_atm_regrid_weight_conserve.nc"),
    ),
    ocn_to_exchangegrid = dict(
        bilinear = Regridder("ocn_exchangegrid_regrid_weight_bilinear.nc"),
    ),
    exchangegrid_to_ocn = dict(
        conserve = Regridder("exchangegrid_ocn_regrid_weight_conserve.nc"),
    ),
    lnd_to_exchangegrid = dict(
        bilinear = Regridder("lnd_exchangegrid_regrid_weight_bilinear.nc"),
    ),
    exchangegrid_to_lnd = dict(
        conserve = Regridder("exchangegrid_lnd_regrid_weight_conserve.nc"),
    ),
)


atm = ATMOSPHERE(...)
ocn = OCEAN(...)
lnd = LAND(...)
flx = FLX(...)

all_components = [atm, ocn, lnd, flx]

FM_al_forward = ForcingMapper([atm, lnd, flx]) # from atm and lnd to flx
FM_al_backward = ForcingMapper([atm, lnd, flx]) # from flx_al to atm and lnd

FM_ao_forward = ForcingMapper([atm, ocn, flx]) # from atm and ocn to flx
FM_ao_backward = ForcingMapper([atm, ocn, flx]) # from flx_ao to atm and ocn

# Add mappings for atm--flx--lnd
FM_al_forward.add_mapping("atm.u10", "flx_al.u10", regridders["atm_to_exchangegrid"]["bilinear"])
FM_al_forward.add_mapping("atm.v10", "flx_al.v10", regridders["atm_to_exchangegrid"]["bilinear"])
FM_al_forward.add_mapping("atm.t2m", "flx_al.t2m", regridders["atm_to_exchangegrid"]["bilinear"])
FM_al_forward.add_mapping("lnd.lst", "flx_al.lst", regridders["lnd_to_exchangegrid"]["bilinear"])
FM_al_backward.add_mapping("flx.lhf", "atm.lhf" regridders["exchangegrid_to_atm"]["conserve"])
FM_al_backward.add_mapping("flx.shf", "atm.shf", regridders["exchangegrid_to_atm"]["conserve"])
FM_al_backward.add_mapping("flx.lhf", "lnd.lhf", regridders["exchangegrid_to_lnd"]["conserve"])
FM_al_backward.add_mapping("flx.shf", "lnd.shf", regridders["exchangegrid_to_lnd"]["conserve"])

# Add mappings for atm--flx--ocn
FM_ao_forward.add_mapping("atm.u10", "flx_ao.u10", regridders["atm_to_exchangegrid"]["bilinear"])
FM_ao_forward.add_mapping("atm.v10", "flx_ao.v10", regridders["atm_to_exchangegrid"]["bilinear"])
FM_ao_forward.add_mapping("atm.t2m", "flx_ao.t2m", regridders["atm_to_exchangegrid"]["bilinear"])
FM_ao_forward.add_mapping("ocn.sst", "flx_ao.sst", regridders["ocn_to_exchangegrid"]["bilinear"])
FM_ao_backward.add_mapping("flx.lhf", "atm.lhf" regridders["exchangegrid_to_atm"]["conserve"])
FM_ao_backward.add_mapping("flx.shf", "atm.shf", regridders["exchangegrid_to_atm"]["conserve"])
FM_ao_backward.add_mapping("flx.lhf", "ocn.lhf" regridders["exchangegrid_to_ocn"]["conserve"])
FM_ao_backward.add_mapping("flx.shf", "ocn.shf", regridders["exchangegrid_to_ocn"]["conserve"])

# Running sequcne 
sequence_atm_lnd = [atm, lnd, FM_al_forward, flx, FM_al_backward]
sequence_atm_ocn = [ocn, FM_ao_forward, flx, FM_ao_backward]

run_sequence = [
  sequence_atm_lnd * 3,
  sequence_atm_ocn,
]

state, forcing = coupled.initialize()



# What looks like internally in coupler RUN


def generate_step_function(run_sequence, jitted: bool=True):

    # Validating run_sequence
    for obj in flatten(run_sequence):
        if not hasattr(obj, "what"):
            raise UndefinedActionElementError(f"Cannot find `what` in the action element `{str(obj):s}`.")
        if obj.what not in ["component", "forcing_mapping"]:
            raise UndefinedActionElementError(f"Unrecognized action element: {str(obj.what):s}.")

    component_step_functions = { component.name : component.generate_step_function() for component in all_components }
    
    def step_function(carry, t):
        
        state = carry["state"]
        forcing = carry["forcing"]
    
        predictions = { component.name : [] for component in all_components }

        for obj in flatten(run_sequence):

            if obj.what == "component":
                component_name = obj.name
                state[component_name], _predictions = component_step_functions[component_name](state[component_name], forcing[component_name])
                predictions[obj.name].append(_predictions) 
    
            elif obj.what == "forcing_mapping":

                sub_state = { name : state[component_name] for component_name in obj.involved_component_names }
                sub_forcing = { name : forcing[component_name] for component_name in obj.involved_component_names }
                sub_state, sub_forcing = obj.map_forcing( sub_state, sub_forcing )

                for name in sub_state.keys():
                    state[name] = sub_state[name]
                    forcing[name] = sub_state[name]
        

        return carry, predictions
