import inspect

# `Callable` must come from `typing`, not `collections.abc`: this module compares
# `get_origin(...)` against `get_origin(Callable)`, and only `typing.Callable`
# (unsubscripted) has a `get_origin` matching a subscripted Callable's origin.
from typing import Any, Callable, get_args, get_origin, get_type_hints  # noqa: UP035


class MethodNotFoundError(Exception):
    pass

class MemberNotFoundError(Exception):
    pass

class MemberTypeNotMatchError(Exception):
    pass

class NumberOfMethodParametersNotMatchError(Exception):
    pass

def resolve_interface(
    target : Any,
    reference_class: type,
    verbose: bool = False,
    customized_mapping_name:str = "__JEM_CUSTOMIZED_MAPPING__",
    optional: str | list[str] | None = None,
    skip: str | list[str] | None = None,
) -> dict[str, Any]:
    
    if skip is None:
        skip = []
    if optional is None:
        optional = []
    metadata = get_type_hints(reference_class)
    customized_mapping = getattr(target, customized_mapping_name, {})
    if isinstance(skip, str):
        skip = [skip,]
    result: dict[str, str | None] = {}
    for member_name, member_type in metadata.items():
        resolved_name = None
        if member_name in skip:
            pass
        else:
            if verbose:
                print(f"Checking `{member_name:s}` => `{member_type!s}`")

            try:
                if get_origin(member_type) is get_origin(Callable):
                    resolved_name = _resolve_function(target, member_name, member_type, customized_mapping)
                else:
                    resolved_name = _resolve_member(target, member_name, member_type, customized_mapping)
            except Exception as e:
                print(f"Exception happens when resolving `{member_name}`: {e!s}.")
                if member_name in optional:
                    print(f"Member name `{member_name}` is optional. Now set to None.")
                    resolved_name = None
                else:
                    raise

            if resolved_name is None:
                result[member_name] = None
            else:
                result[member_name] = getattr(target, resolved_name)

    return result

def _resolve_function(
    target : Any,
    function_name: str,
    function_signature: Callable,
    customized_mapping: dict[str, Any],
    verbose: bool = True,
) -> str:

    if get_origin(function_signature) is not get_origin(Callable):
        raise ValueError(f"The function signature of `{function_name:s}` should be a Callable. ")
    
    if (function_name not in customized_mapping) and (not hasattr(target, function_name)):
        raise MethodNotFoundError(f"The method `{function_name:s}` is not found")

    if function_name in customized_mapping:
        remapped_function_name = customized_mapping[function_name]
        if verbose:
            print(f"Customized mapping found: `{function_name}` is mapped to `{remapped_function_name}`")
        function_name = remapped_function_name
    
    target_function_sigature = inspect.signature(getattr(target, function_name))
    goal_argument_types, _output_type = get_args(function_signature)

    if len(target_function_sigature.parameters.keys()) != len(goal_argument_types):
        raise NumberOfMethodParametersNotMatchError(f"The number of parameters in the method `{function_name:s}` ({len(target_function_sigature.parameters.keys()):d}) is not the same as given `{function_signature!s}` ({len(goal_argument_types):d})")

    return function_name

def _resolve_member(
    target : Any,
    member_name: str,
    member_type: type,
    customized_mapping: dict[str, Any],
    verbose: bool = True,
) -> str:

    if (member_name not in customized_mapping) and (not hasattr(target, member_name)):
        raise MemberNotFoundError(f"The member `{member_name:s}` is not found")

    if member_name in customized_mapping:
        remapped_member_name = customized_mapping[member_name]
        if verbose:
            print(f"Customized mapping found: `{member_name}` is mapped to `{remapped_member_name}`")
        member_name = remapped_member_name
     
    member = getattr(target, member_name)
    if member_type != Any and not isinstance(member, get_origin(member_type) or member_type):
        raise MemberTypeNotMatchError(f"The member `{member_name:s}` ({type(member)!s}) is expected to be of type `{member_type!s}`.")
    return member_name
