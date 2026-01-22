from typing import get_args, Any, Callable, Dict, Optional
import inspect

class MethodNotFoundError(Exception):
    pass

class MemberNotFoundError(Exception):
    pass

class MemberTypeNotMatchError(Exception):
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
            raise MethodSignatureNotMatchError(f"The number of parameters in the method `{function_name:s}` ({len(target_function_sigature.parameters.keys()):d}) is not the same as given `{str(function_signature)}` ({len(goal_argument_types):d})")

def verify_members(
    target : Any,
    members_metadata : Dict[str, type],
    verbose: bool = False,
):

    for member_name, member_type in members_metadata.items():
 
        if verbose:
            print(f"Checking `{member_name:s}` => `{str(member_type)}`")

        if not hasattr(target, member_name):
            raise MemberNotFoundError(f"The member `{member_name:s}` is not found")

        member = getattr(target, member_name)
        if member_type != Any and not isinstance(member, member_type):
            raise MemberTypeNotMatchError(f"The member `{member_name:s}` is expected to be of type `{str(member_type)}`")

