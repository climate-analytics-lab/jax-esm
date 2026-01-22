from typing import get_args, get_type_hints, get_origin
from typing import Any, Callable, Dict, Optional
import inspect


class MethodNotFoundError(Exception):
    pass

class MemberNotFoundError(Exception):
    pass

class MemberTypeNotMatchError(Exception):
    pass

class MethodSignatureNotMatchError(Exception):
    pass

def verify(
    target : Any,
    reference_class: Optional[type] = None,
    metadata: Optional[Dict[str, type]] = None,
    verbose: bool = False,
) -> None:
    if reference_class is None and metadata is None:
        raise ValueError("Either reference_class or metadata must be given.")
    if reference_class is not None:
        metadata = get_type_hints(reference_class)
    for member_name, member_type in metadata.items():
        if verbose:
            print(f"Checking `{member_name:s}` => `{str(member_type)}`")
        if get_origin(member_type) is get_origin(Callable):
            _verify_function(target, member_name, member_type)
        else:
            _verify_member(target, member_name, member_type)


def _verify_function(
    target : Any,
    function_name: str,
    function_signature: Callable,
    verbose: bool = False,
) -> None:

    if not isinstance(function_signature, Callable):
        raise ValueError(f"The function signature of `{function_name:s}` should be a Callable. ")
    
    if not hasattr(target, function_name):
        raise MethodNotFoundError(f"The method `{function_name:s}` is not found")

    target_function_sigature = inspect.signature(getattr(target, function_name))
    goal_argument_types, output_type = get_args(function_signature)

    if len(target_function_sigature.parameters.keys()) != len(goal_argument_types):
        raise MethodSignatureNotMatchError(f"The number of parameters in the method `{function_name:s}` ({len(target_function_sigature.parameters.keys()):d}) is not the same as given `{str(function_signature)}` ({len(goal_argument_types):d})")

def _verify_member(
    target : Any,
    member_name: str,
    member_type: type,
) -> None:

    if not hasattr(target, member_name):
        raise MemberNotFoundError(f"The member `{member_name:s}` is not found")

    member = getattr(target, member_name)
    if member_type != Any and not isinstance(member, member_type):
        raise MemberTypeNotMatchError(f"The member `{member_name:s}` ({str(type(member))}) is expected to be of type `{str(member_type)}`.")

