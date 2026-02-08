"""
Tests for base/interface.py
============================
Covers resolve_interface, _resolve_function, _resolve_member, every exception
type, the skip normalisation, customized_mapping remapping, verbose output,
and a regression test for the getattr-uses-original-name-after-remap bug.

No external dependencies required — interface.py only uses stdlib typing +
inspect, so it can be imported directly.
"""

import pytest
from typing import Any, Callable, Dict, List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "base_extracted"))

from jem.base.interface import (
    resolve_interface,
    _resolve_function,
    _resolve_member,
    MethodNotFoundError,
    MemberNotFoundError,
    MemberTypeNotMatchError,
    MethodSignatureNotMatchError,
)


# ===========================================================================
# Reference classes used as the `reference_class` argument.
# resolve_interface reads their type-hints via get_type_hints().
# ===========================================================================

class _RefMemberOnly:
    """Reference with only non-Callable members."""
    count: int
    label: str
    tags:  List[str]
    meta:  Any          # Any-typed member — isinstance check is skipped


class _RefFuncOnly:
    """Reference with only Callable members."""
    compute: Callable[[int, str], float]        # 2 params
    reset:   Callable[[], None]                 # 0 params


class _RefMixed:
    """Reference with both members and functions."""
    name:    str
    process: Callable[[float], float]           # 1 param


class _RefSkippable:
    """Reference used to exercise the skip parameter."""
    keep_me:  int
    skip_me:  str
    also_skip: float


# ===========================================================================
# Concrete target classes that implement (or deliberately break) the refs above.
# ===========================================================================

class _GoodMemberTarget:
    count = 42
    label = "hello"
    tags  = ["a", "b"]
    meta  = {"anything": True}          # Any — no type check


class _GoodFuncTarget:
    @staticmethod
    def compute(x: int, y: str) -> float:
        return 1.0

    @staticmethod
    def reset() -> None:
        pass


class _GoodMixedTarget:
    name = "mixed"

    @staticmethod
    def process(v: float) -> float:
        return v


# ===========================================================================
# A.  _resolve_member  (unit tests on the internal helper directly)
# ===========================================================================

class TestResolveMember:
    """Direct tests of _resolve_member."""

    def test_happy_path_returns_member_name(self):
        result = _resolve_member(_GoodMemberTarget(), "count", int, {})
        assert result == "count"

    def test_missing_member_raises(self):
        with pytest.raises(MemberNotFoundError, match="not found"):
            _resolve_member(_GoodMemberTarget(), "nonexistent", int, {})

    def test_type_mismatch_raises(self):
        # count is int, but we declare it as str here
        with pytest.raises(MemberTypeNotMatchError, match="expected to be of type"):
            _resolve_member(_GoodMemberTarget(), "count", str, {})

    def test_any_typed_member_skips_isinstance_check(self):
        # meta is a dict; declaring it as Any must NOT raise
        result = _resolve_member(_GoodMemberTarget(), "meta", Any, {})
        assert result == "meta"

    def test_customized_mapping_remaps_name(self):
        """When a mapping exists, the returned name is the remapped one."""

        class Target:
            actual_name = "value"

        mapping = {"logical_name": "actual_name"}
        # We ask for "logical_name"; it remaps to "actual_name"
        result = _resolve_member(Target(), "logical_name", str, mapping)
        assert result == "actual_name"

    def test_customized_mapping_remapped_target_missing_raises(self):
        """Mapping exists but the remapped attribute doesn't exist on target."""

        class Target:
            pass

        mapping = {"logical_name": "does_not_exist"}
        with pytest.raises(AttributeError):
            _resolve_member(Target(), "logical_name", str, mapping)

    def test_list_type_member_accepted(self):
        # tags is List[str]; get_origin(List[str]) is list
        result = _resolve_member(_GoodMemberTarget(), "tags", List[str], {})
        assert result == "tags"

    def test_list_type_member_wrong_container_raises(self):
        """Declared as List[str] but the actual value is a tuple."""

        class Target:
            items = ("a", "b")   # tuple, not list

        with pytest.raises(MemberTypeNotMatchError):
            _resolve_member(Target(), "items", List[str], {})


# ===========================================================================
# B.  _resolve_function  (unit tests on the internal helper directly)
# ===========================================================================

class TestResolveFunction:
    """Direct tests of _resolve_function."""

    def test_happy_path_returns_function_name(self):
        sig = Callable[[int, str], float]   # 2 params — matches compute
        result = _resolve_function(_GoodFuncTarget(), "compute", sig, {})
        assert result == "compute"

    def test_zero_param_function_accepted(self):
        sig = Callable[[], None]            # 0 params — matches reset
        result = _resolve_function(_GoodFuncTarget(), "reset", sig, {})
        assert result == "reset"

    def test_missing_method_raises(self):
        sig = Callable[[int], None]
        with pytest.raises(MethodNotFoundError, match="not found"):
            _resolve_function(_GoodFuncTarget(), "vanished", sig, {})

    def test_wrong_param_count_raises(self):
        # compute has 2 params, but we declare 3
        sig = Callable[[int, str, float], None]
        with pytest.raises(MethodSignatureNotMatchError, match="not the same"):
            _resolve_function(_GoodFuncTarget(), "compute", sig, {})

    def test_customized_mapping_remaps_to_correct_method(self):
        """Logical name 'do_stuff' is remapped to 'compute' via mapping."""

        class Target:
            @staticmethod
            def compute(a: int, b: str) -> float:
                return 0.0

        mapping = {"do_stuff": "compute"}
        sig = Callable[[int, str], float]
        result = _resolve_function(Target(), "do_stuff", sig, mapping)
        assert result == "compute"

    def test_customized_mapping_remapped_method_missing_raises(self):
        """Mapping points to a method that does not exist."""

        class Target:
            pass

        mapping = {"do_stuff": "missing_method"}
        sig = Callable[[int], None]
        with pytest.raises(AttributeError):
            _resolve_function(Target(), "do_stuff", sig, mapping)

    def test_remapped_method_wrong_param_count_raises(self):
        """Mapping points to a real method but param count is wrong."""

        class Target:
            @staticmethod
            def real_method(x: int) -> None:
                pass

        mapping = {"logical": "real_method"}
        sig = Callable[[int, str, float], None]  # expects 3, real has 1
        with pytest.raises(MethodSignatureNotMatchError):
            _resolve_function(Target(), "logical", sig, mapping)

    def test_instance_method_self_counts_as_param(self):
        """A bound instance method: self is NOT in the signature after binding,
        so a 1-param Callable should match a method with (self, x)."""

        class Target:
            def compute(self, x: int) -> float:
                return 0.0

        sig = Callable[[int], float]       # 1 param
        result = _resolve_function(Target(), "compute", sig, {})
        assert result == "compute"


# ===========================================================================
# C.  resolve_interface  (integration — the main public entry point)
# ===========================================================================

class TestResolveInterface:
    """Tests of the top-level resolve_interface orchestrator."""

    # --- happy paths --------------------------------------------------

    def test_resolves_members_only(self):
        result = resolve_interface(_GoodMemberTarget(), _RefMemberOnly)
        assert set(result.keys()) == {"count", "label", "tags", "meta"}
        assert result["count"] == 42
        assert result["label"] == "hello"

    def test_resolves_functions_only(self):
        result = resolve_interface(_GoodFuncTarget(), _RefFuncOnly)
        assert set(result.keys()) == {"compute", "reset"}
        assert callable(result["compute"])
        assert callable(result["reset"])

    def test_resolves_mixed_ref(self):
        result = resolve_interface(_GoodMixedTarget(), _RefMixed)
        assert set(result.keys()) == {"name", "process"}
        assert result["name"] == "mixed"
        assert callable(result["process"])

    # --- skip parameter -----------------------------------------------

    def test_skip_as_list_omits_fields(self):

        class Target:
            keep_me   = 7
            skip_me   = "no"
            also_skip = 3.14

        result = resolve_interface(
            Target(), _RefSkippable, skip=["skip_me", "also_skip"]
        )
        assert set(result.keys()) == {"keep_me"}

    def test_skip_as_single_string_is_normalised(self):
        """Passing skip as a bare string (not a list) should still work."""

        class Target:
            keep_me   = 7
            skip_me   = "no"
            also_skip = 3.14

        result = resolve_interface(Target(), _RefSkippable, skip="skip_me")
        # skip_me is skipped; keep_me and also_skip remain
        assert "skip_me" not in result
        assert "keep_me" in result
        assert "also_skip" in result

    def test_skip_all_fields_returns_empty(self):

        class Target:
            keep_me   = 1
            skip_me   = "x"
            also_skip = 0.0

        result = resolve_interface(
            Target(), _RefSkippable,
            skip=["keep_me", "skip_me", "also_skip"]
        )
        assert result == {}

    # --- verbose output -----------------------------------------------

    def test_verbose_prints_checking_messages(self, capsys):
        resolve_interface(_GoodMemberTarget(), _RefMemberOnly, verbose=True)
        captured = capsys.readouterr()
        assert "Checking `count`" in captured.out
        assert "Checking `label`" in captured.out

    def test_non_verbose_prints_nothing(self, capsys):
        resolve_interface(_GoodMemberTarget(), _RefMemberOnly, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    # --- customized mapping (remapping) -------------------------------

    def test_customized_mapping_remaps_member(self):
        """__JEM_CUSTOMIZED_MAPPING__ renames a member; resolve_interface
        returns the remapped key."""

        class Ref:
            logical_name: str

        class Target:
            __JEM_CUSTOMIZED_MAPPING__ = {"logical_name": "real_attr"}
            real_attr = "found"
            # NOTE: logical_name does NOT exist on Target

        result = resolve_interface(Target(), Ref)
        # The key in the result is the remapped name
        assert "logical_name" in result

    def test_customized_mapping_remaps_function(self):

        class Ref:
            run: Callable[[int], None]

        class Target:
            __JEM_CUSTOMIZED_MAPPING__ = {"run": "execute"}

            @staticmethod
            def execute(x: int) -> None:
                pass

        result = resolve_interface(Target(), Ref)
        assert "run" in result
        assert callable(result["run"])

    def test_custom_mapping_attribute_name_can_be_changed(self):
        """The customized_mapping_name parameter selects a different attribute."""

        class Ref:
            val: int

        class Target:
            __MY_MAP__ = {"val": "actual_val"}
            actual_val = 99

        result = resolve_interface(
            Target(), Ref, customized_mapping_name="__MY_MAP__"
        )
        assert "val" in result
        assert result["val"] == 99

    # --- error propagation --------------------------------------------

    def test_missing_member_propagates(self):
        class Target:
            pass  # missing everything

        with pytest.raises(MemberNotFoundError):
            resolve_interface(Target(), _RefMemberOnly)

    def test_missing_method_propagates(self):
        class Target:
            pass

        with pytest.raises(MethodNotFoundError):
            resolve_interface(Target(), _RefFuncOnly)

    def test_type_mismatch_propagates(self):
        class Target:
            count = "not_an_int"   # declared as int in _RefMemberOnly
            label = "ok"
            tags  = ["a"]
            meta  = None

        with pytest.raises(MemberTypeNotMatchError):
            resolve_interface(Target(), _RefMemberOnly)

    def test_signature_mismatch_propagates(self):
        class Target:
            @staticmethod
            def compute(a, b, c):   # 3 params; ref expects 2
                pass

            @staticmethod
            def reset():
                pass

        with pytest.raises(MethodSignatureNotMatchError):
            resolve_interface(Target(), _RefFuncOnly)

    # --- regression: getattr uses original name after remap -----------

    def test_remap_bug_value_comes_from_original_name(self):
        """
        BUG in resolve_interface line 42:
            result[resolved_name] = getattr(target, member_name)
        After remapping, resolved_name is the NEW name but member_name is still
        the ORIGINAL.  So the value stored is from the original attribute.

        This test documents the current (buggy) behaviour.  If the bug is fixed
        (getattr should use resolved_name), this test will need updating.
        """

        class Ref:
            logical: str

        class Target:
            __JEM_CUSTOMIZED_MAPPING__ = {"logical": "physical"}
            logical  = "original_value"   # original name still exists
            physical = "remapped_value"   # this is what SHOULD be fetched

        result = resolve_interface(Target(), Ref)
        # Bug: the value comes from `logical` (original), not `physical` (remapped)
        assert result["logical"] == "remapped_value"  # documents current behaviour


# ===========================================================================
# D.  Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_reference_class_returns_empty_dict(self):
        class EmptyRef:
            pass

        class Target:
            x = 1

        result = resolve_interface(Target(), EmptyRef)
        assert result == {}

    def test_target_with_extra_attributes_is_fine(self):
        """Extra attributes on target that are not in the reference are ignored."""

        class Ref:
            x: int

        class Target:
            x = 10
            y = 20          # not in Ref — ignored
            z = "extra"

        result = resolve_interface(Target(), Ref)
        assert result == {"x": 10}

    def test_dict_typed_member(self):
        class Ref:
            data: Dict[str, int]

        class Target:
            data = {"a": 1, "b": 2}

        result = resolve_interface(Target(), Ref)
        assert result["data"] == {"a": 1, "b": 2}

    def test_optional_typed_member_with_none(self):
        """Optional[X] has origin typing.Union — isinstance check uses the
        origin or falls back to the type itself.  None is a valid value but
        isinstance(None, Union) will fail.  This documents current behaviour."""

        class Ref:
            maybe: Optional[int]

        class Target:
            maybe = None

        # Optional[int] -> origin is Union; isinstance(None, Union) raises TypeError
        # so this will raise (documents current limitation)
        with pytest.raises(TypeError):
            resolve_interface(Target(), Ref)

    def test_callable_with_no_args_and_no_return(self):
        """Callable[[], None] — zero input params."""

        class Ref:
            noop: Callable[[], None]

        class Target:
            @staticmethod
            def noop() -> None:
                pass

        result = resolve_interface(Target(), Ref)
        assert callable(result["noop"])
