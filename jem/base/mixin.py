from typing import get_args, Any, Callable, Dict, Optional
import inspect

class MethodNotFoundError(Exception):
    pass

class MethodSignatureNotMatchError(Exception):
    pass



def verify_functions(
    target : Any,
    function_metadata : Dict[str, type],
    verbose: bool = False,
):

    for function_name, function_signature in function_metadata.items():
 
        if verbose:
            print(f"Checking `{function_name:s}` => `{str(function_signature)}`")

      
        if not isinstance(function_signature, Callable):
            raise ValueError(f"The function signature of `{function_name:s}` should be a Callable. ")
        
        if not hasattr(target, function_name):
            raise MethodNotFoundError(f"The method `{function_name:s}` is not found")


        target_function_sigature = inspect.signature(getattr(target, function_name))
        goal_argument_types, output_type = get_args(function_signature)

        if len(target_function_sigature.parameters.keys()) != len(goal_argument_types):
            raise MethodSignatureNotMatchError("The number of parameters in the method `function_name:s` is not the same as given `{str(function_signature)}`")

