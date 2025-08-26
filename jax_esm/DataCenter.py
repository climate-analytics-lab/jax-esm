from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

@dataclass
class DataPermission:
    r: List[str]  # A list of component strings
    w: List[str]  # A list of component strings

    @classmethod
    def getTemplatePermission(
        cls,
        components: List[str],
    ):

        permissions = dict()
        for component in components:
            r = [component for component in components]
            w = [component for component in components]
    
            permissions[component] = DataPermission(
                r = r,
                w = w,
            )

            
        return permissions


class DataCenter:

    def __init__(
        self,
        components: Dict[str, any],
        permissions: Dict[str, DataPermission],
    ):
        self.components = components
        self.permissions = permissions
        self.varname_mapping = dict()



    def registerAlias(
        self,
        varname_universal: str,
        component: str,
        varname_component: str,
    ):

        if varname_universal not in self.varname_mapping:
            self.varname_mapping[varname_universal] = dict()

        self.varname_mapping[varname_universal][component] = varname_component
            

    
    def setAccessPermission(
        self,
        component: str,
        action: str,
        readwrite: str,
        by_component: str,
    ):

        if readwrite not in ["r", "w"]:
            raise Exception("Error: `readwrite` should be either `r` or `w`.")

        print(self.permissions)
        p = getattr(self.permissions[component], readwrite)
        print(p)
        
        if action == "+":
            if by_component not in p:
                p.append(by_component)
            
        elif action == "-":
            if by_component in p:
                p.remove(by_component)
            
    def getVariable(
        self,
        component:    str,
        by_component: str,
        varname: str = None,
        is_universal_name: bool = False,
        idx = None,
    ):
        result = None
        p = self.permissions[component].r
        if by_component in p:
            varname_component = self.varname_mapping[varname][component] if is_universal_name else varname
            variable = getattr(self.components[component].state, varname_component)
            result = variable if idx is None else variable.at[idx]

        else:
            raise Exception("Permission denied.")

        return result
            
    def setVariable(
        self, 
        component:    str,
        by_component: str,
        varname:      str,
        values: any,
        is_universal_name: bool = False,
        idx = None,
    ):
        p = self.permissions[component].w
        if by_component in p:
            varname_component = self.varname_mapping[varname][component] if is_universal_name else varname                            
            old_data = getattr(self.components[component].state, varname_component)
            if idx is None:
                old_data = old_data.at[idx]
            
            new_data = old_data.set(values)
            result = setattr(self.components[component].state, varname_component, new_data)
        else:
            raise Exception("Permission denied.")
        