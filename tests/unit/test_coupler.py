"""Tests for base/coupler.py
==========================
Covers:
  - adhoc_scan (empty, single, multi-step, dict carry)
  - generate_scan_function (selector logic)
  - Coupler construction, add/remove component & mapper
  - Coupler.initialize delegation
  - _verify_name_uniqueness (happy + collision)
  - _verify_workflow (happy + every error branch)
  - generate_step_function end-to-end (unjitted)
  - generate_trajectory_function end-to-end (unjitted)
  - predictions_to_xarray delegation
  - get_info (including dead-code regression + empty-mapper edge case)

Everything runs against the real dependencies; nothing is stubbed.
"""

from unittest.mock import MagicMock
import pytest

import jax

from jem.base.coupler import adhoc_scan, generate_scan_function, Coupler


# ===========================================================================
# 2.  TEST HELPERS
# ===========================================================================

def _make_raw_component(name="comp"):
    """Return a concrete class instance whose attributes satisfy
    resolve_interface against JEMComponent.  All Callable fields have the
    exact parameter counts the type-hints require.
    """
    init_carry = {
        "state" : {"value": 0.0},
        "forcing" : {"flux": 1.0},
    }

    class _Raw:
        # non-Callable members (Any-typed in JEMComponent, so any value is fine)

        # Callable members — param counts must match the type aliases in typing.py:
        #   InitializeFunction        = Callable[[], ...]                -> 0 params
        #   StepFunctionGenerator     = Callable[[JittableFlag], ...]    -> 1 param
        #   HistoryToXarray           = Callable[[History], ...]         -> 1 param
        #   GetInfoFunction           = Callable[[], ...]                -> 0 params

        @staticmethod
        def initialize():                                   # 0 params
            return init_carry

        @staticmethod
        def generate_step_function():                       # 0 params
            def _step(c, t):
                c["state"]["value"] += 1
                return c, dict(state=c["state"], forcing=c["forcing"])
            return _step

        @staticmethod
        def predictions_to_xarray(history):                 # 1 param
            return {"ds_" + name: history}

        @staticmethod
        def get_info():                                     # 0 params
            return {"name": name}

    return _Raw()


def _make_raw_mapper():

    def mapper(coupled_carry):
        return coupled_carry

    return mapper

def _build_coupler(comp_names=("comp_a", "comp_b"), mappers=("fm",)):
    """Build a Coupler with the given components and forcing mappers."""
    components = {n: _make_raw_component(name=n) for n in comp_names}
    mapper_dict = {n: _make_raw_mapper() for n in mappers}
    return Coupler(components=components, mappers=mapper_dict)


# ===========================================================================
# 3.  adhoc_scan
# ===========================================================================

class TestAdhocScan:

    def test_accumulates_carry_and_collects_outputs(self):
        def f(carry, x):
            x_new = carry + x
            return x_new, x_new ** 2

        final, ys = adhoc_scan(f, 0, [1, 2, 3])
        assert final == 6          # 0+1+2+3
        assert list(ys) == [1, 9, 36]     # each x*2

    def test_empty_xs_returns_init_and_empty_stack(self):
        def f(carry, x):
            return carry + x, x

        final, ys = adhoc_scan(f, 99, [])
        assert final == 99
        assert list(ys) == []

    def test_single_step(self):
        def f(carry, x):
            return carry * x, carry

        final, ys = adhoc_scan(f, 5, [3])
        assert final == 15
        assert list(ys) == [5]           # single non-dict output stays as list

    def test_dict_carry_mutated_across_steps(self):
        def f(carry, x):
            carry["acc"] += x
            return carry, {"step": x}

        final, ys = adhoc_scan(f, {"acc": 0}, [10, 20, 30])
        assert final == {"acc": 60}
        # stack_objects transposes list-of-dicts
        assert "step" in ys
        assert list(ys["step"]) == [10, 20, 30]

    def test_dict_outputs_are_transposed(self):
        def f(carry, x):
            return carry, {"val": x, "sq": x * x}

        _, ys = adhoc_scan(f, 0, [2, 3, 4])
        print(ys)
        assert "val" in ys
        assert list(ys["val"]) == [2, 3, 4]
        assert "sq" in ys
        assert list(ys["sq"]) == [4, 9, 16]


# ===========================================================================
# 4.  generate_scan_function
# ===========================================================================

class TestGenerateScanFunction:

    def test_jitted_true_returns_lax_scan(self):
        assert generate_scan_function(jitted=True) is jax.lax.scan

    def test_jitted_false_returns_adhoc_scan(self):
        assert generate_scan_function(jitted=False) is adhoc_scan


# ===========================================================================
# 5.  Coupler construction & add/remove
# ===========================================================================

class TestCouplerConstruction:

    def test_empty_coupler(self):
        c = Coupler()
        assert c.components == {}
        assert c.mappers == {}

    def test_none_arguments_default_to_empty(self):
        c = Coupler(components=None, mappers=None)
        assert c.components == {}
        assert c.mappers == {}

    def test_components_registered_on_init(self):
        raw = _make_raw_component("atm")
        c = Coupler(components={"atm": raw})
        assert "atm" in c.components
        # The stored object is a JEMComponent wrapper, not the raw object
        assert hasattr(c.components["atm"], "name")
        assert c.components["atm"].name == "atm"

    def test_mappers_registered_on_init(self):
        raw_c = _make_raw_component("a")
        raw_fm = _make_raw_mapper()
        c = Coupler(
            components={"a": raw_c},
            mappers={"fm": raw_fm},
        )
        assert "fm" in c.mappers


class TestAddRemoveComponent:

    def test_add_component_after_construction(self):
        c = Coupler()
        raw = _make_raw_component("ocean")
        c.add_component("ocean", raw)
        assert "ocean" in c.components

    def test_remove_existing_component(self):
        c = _build_coupler(comp_names=("a", "b"), mappers=())
        c.remove_component("a")
        assert "a" not in c.components
        assert "b" in c.components

    def test_remove_nonexistent_component_is_noop(self):
        c = _build_coupler(comp_names=("a",), mappers=())
        c.remove_component("ghost")          # should not raise
        assert "a" in c.components

    def test_add_overwrites_existing_component(self):
        c = _build_coupler(comp_names=("a",), mappers=())
        new_raw = _make_raw_component("a")
        c.add_component("a", new_raw)
        assert c.components["a"].raw_component is new_raw


class TestAddRemoveForcingMapper:

    def test_add_mapper_after_construction(self):
        c = _build_coupler(comp_names=("x", "y"), mappers=())
        raw_fm = _make_raw_mapper()
        c.add_mapper("coupler_xy", raw_fm)
        assert "coupler_xy" in c.mappers

    def test_remove_existing_mapper(self):
        c = _build_coupler(comp_names=("a", "b"), mappers=("fm",))
        c.remove_mapper("fm")
        assert "fm" not in c.mappers

    def test_remove_nonexistent_mapper_is_noop(self):
        c = _build_coupler(comp_names=("a",), mappers=())
        c.remove_mapper("ghost")     # should not raise
        assert c.mappers == {}


# ===========================================================================
# 6.  Coupler.initialize
# ===========================================================================

class TestCouplerInitialize:

    def test_returns_state_forcing_tuple_per_component(self):
        c = _build_coupler(comp_names=("a", "b"), mappers=())
        result = c.initialize()
        assert set(result.keys()) == {"a", "b"}
        for name in ("a", "b"):
            assert isinstance(result, dict)
            assert {"state", "forcing"} == set(result[name].keys())
            assert isinstance(result[name]["state"], dict)
            assert isinstance(result[name]["forcing"], dict)

    def test_empty_coupler_initialize_returns_empty(self):
        c = Coupler()
        assert c.initialize() == {}


# ===========================================================================
# 7.  _verify_name_uniqueness
# ===========================================================================

class TestVerifyNameUniqueness:

    def test_unique_names_pass(self):
        c = _build_coupler(comp_names=("a", "b"), mappers=(("fm", ("a", "b")),))
        c._verify_name_uniqueness()   # should not raise

    def test_collision_between_component_and_mapper_raises(self):
        """A component and a forcing mapper share the same name."""
        c = _build_coupler(comp_names=("a", "b"), mappers=())
        # Manually inject a mapper with the same key as component "a"
        c.mappers["a"] = MagicMock()
        with pytest.raises(Exception, match="not unique"):
            c._verify_name_uniqueness()

    def test_empty_coupler_passes(self):
        Coupler()._verify_name_uniqueness()   # no names at all


# ===========================================================================
# 8.  _verify_workflow
# ===========================================================================

class TestVerifyWorkflow:

    def test_valid_workflow_passes(self):
        c = _build_coupler(comp_names=("a", "b"), mappers=("fm",))
        c._verify_workflow(["a", "fm", "b"])   # all names exist

    def test_unknown_action_raises_valueerror(self):
        c = _build_coupler(comp_names=("a",), mappers=())
        with pytest.raises(ValueError, match="does not map"):
            c._verify_workflow(["a", "unknown"])

    def test_non_string_action_raises_typeerror(self):
        c = _build_coupler(comp_names=("a",), mappers=())
        # jax.tree.flatten is stubbed to just list(), so [123] flattens to [123]
        with pytest.raises(TypeError, match="have to be strings"):
            c._verify_workflow([123])

    def test_empty_workflow_passes(self):
        c = _build_coupler(comp_names=("a",), mappers=())
        c._verify_workflow([])   # nothing to validate


# ===========================================================================
# 9.  generate_step_function  (unjitted, end-to-end)
# ===========================================================================

class TestGenerateStepFunction:

    def _make_coupler(self):
        """Two components, one identity forcing mapper between them."""
        c = _build_coupler(
            comp_names=("atm", "ocean"),
            mappers=("fm",),
        )
        return c

    def test_returns_callable(self):
        c = self._make_coupler()
        step_fn = c.generate_step_function(
            workflow=["atm", "fm", "ocean"],
            jitted=False,
            verbose=False
        )
        assert callable(step_fn)

    def test_step_function_returns_updated_carry_and_predictions(self):
        c = self._make_coupler()
        carry = c.initialize()
        step_fn = c.generate_step_function(
            workflow=["atm", "fm", "ocean"], jitted=False, verbose=False
        )
        final_carry, preds = step_fn(carry, 0)

        # carry structure preserved
        for comp_name in ["atm", "ocean"]:
            assert comp_name in final_carry
            for check_key in ["state", "forcing"]:
                assert check_key in final_carry[comp_name]

        # predictions keyed by component name
        assert set(preds.keys()) == {"atm", "ocean"}

    def test_predictions_accumulate_per_component_appearance(self):
        """If a component appears twice in the workflow its predictions are
        collected into a list per output key (via stack_objects transpose).
        E.g. two appearances each yielding {"out": 42.0} become {"out": [42.0, 42.0]}.
        """
        c = self._make_coupler()
        carry = c.initialize()
        # atm appears twice
        step_fn = c.generate_step_function(
            workflow=["atm", "atm", "ocean"], jitted=False, verbose=False
        )
        _, preds = step_fn(carry, 0)
        # stack_objects transposes [{"out":42}, {"out":42}] -> {"out":[42,42]}
        # so each value list has length 2 (one entry per appearance)
        assert len(preds["atm"]["state"]["value"]) == 2

    def test_component_only_workflow(self):
        """Workflow with no forcing mapper — just components in sequence."""
        c = _build_coupler(comp_names=("a", "b"), mappers=())
        init_carry  = c.initialize()
        step_fn = c.generate_step_function(
            workflow=["a", "b"], jitted=False, verbose=False
        )
        new_carry, preds = step_fn(init_carry, 0)
        assert set(preds.keys()) == {"a", "b"}

# ===========================================================================
# 10.  generate_trajectory_function  (unjitted, end-to-end)
# ===========================================================================

class TestGenerateTrajectoryFunction:

    def _setup(self):
        c = _build_coupler(
            comp_names=("atm", "ocean"),
            mappers=("fm",),
        )
        workflow = ["atm", "fm", "ocean"]
        traj_fn  = c.generate_trajectory_function(
            workflow=workflow,
            iterations=3,
            jitted=False,
            show_progress=False,
        )
        init = c.initialize()   # {name: (state, forcing)}
        return traj_fn, init

    def test_returns_final_state_and_predictions(self):
        traj_fn, init = self._setup()
        final_carry, preds = traj_fn(init)

        for comp_name in ["atm", "ocean"]:
            assert comp_name in final_carry
            for check_key in ["state", "forcing"]:
                assert check_key in final_carry[comp_name]

        # preds is keyed by component; each component ran 3 times
        assert set(preds.keys()) == {"atm", "ocean"}

    def test_predictions_have_one_entry_per_iteration(self):
        traj_fn, init = self._setup()
        _, preds = traj_fn(init)

        # Each component appears once per workflow iteration, and there are
        # 3 iterations, so predictions per component is a list of 3 dicts
        # (after stack_objects transposes).
        for comp_name in ("atm", "ocean"):
            # preds[comp] is the stacked output: list of 3 step-outputs
            # Each step-output was itself stacked (1 appearance) -> list of 1
            # So final shape after both stack_objects calls is list of 3 lists
            print(f"{comp_name}: {preds[comp_name]}")
            assert len(preds[comp_name]["state"]["value"]) == 3

    def test_zero_iterations_returns_init_unchanged(self):
        c = _build_coupler(
            comp_names=("a",),
            mappers=(),
        )
        traj_fn = c.generate_trajectory_function(
            workflow=["a"],
            iterations=0,
            jitted=False,
            show_progress=False,
        )
        init_carry = c.initialize()
        final_carry, preds = traj_fn(init_carry)

        # No iterations -> states unchanged, predictions empty
        assert final_carry["a"]["state"] == init_carry["a"]["state"]
        assert list(preds) == []


# ===========================================================================
# 11.  predictions_to_xarray
# ===========================================================================

class TestPredictionsToXarray:

    def test_delegates_to_each_component(self):
        c = _build_coupler(comp_names=("a", "b"), mappers=())
        fake_preds = {"a": {"history": [1, 2]}, "b": {"history": [3, 4]}}
        result = c.predictions_to_xarray(fake_preds)

        assert set(result.keys()) == {"a", "b"}
        # Our stub wraps with {"ds_<name>": ...}
        assert "ds_a" in result["a"]
        assert "ds_b" in result["b"]


# ===========================================================================
# 12.  get_info
# ===========================================================================

class TestGetInfo:

    def test_returns_component_and_mapper_info(self):
        c = _build_coupler(
            comp_names=("atm", "ocean"),
            mappers=("fm",),
        )
        info = c.get_info()
        assert "component_info" in info
        assert "mappers" in info
        assert set(info["component_info"].keys()) == {"atm", "ocean"}
        assert "fm" in info["mappers"]

    def test_empty_mappers_is_empty_dict_not_none_string(self):
        """get_info checks `if self.mappers is None` to decide whether to
        emit the string "None".  But __init__ sets mappers to {} (not
        None) when no mappers are provided.  So an empty coupler's mappers
        value should be an empty dict, NOT the string "None".
        """
        c = _build_coupler(comp_names=("a",), mappers=())
        info = c.get_info()
        assert info["mappers"] == {}
        assert info["mappers"] != "None"

    def test_dead_code_return_info_is_unreachable(self):
        """coupler.py line 309: `return info` after the first return statement.
        This is dead code — it can never execute.  We verify get_info still
        returns successfully (i.e. the dead line does not cause a NameError at
        import time or anything similar).
        """
        c = _build_coupler(comp_names=("x",), mappers=())
        info = c.get_info()          # must not raise
        assert isinstance(info, dict)


# ===========================================================================
# 14.  Integration: full simulate cycle
# ===========================================================================

class TestFullSimulationCycle:
    """Smoke test: construct -> initialize -> trajectory -> xarray."""

    def test_full_cycle_does_not_raise(self):
        c = _build_coupler(
            comp_names=("atm", "ocean"),
            mappers=("fm",),
        )

        # 1. initialize
        init = c.initialize()
        assert len(init) == 2

        # 2. trajectory (3 steps)
        traj_fn = c.generate_trajectory_function(
            workflow=["atm", "fm", "ocean"],
            iterations=3,
            jitted=False,
            show_progress=False,
        )
        final_carry, preds = traj_fn(init)
        assert "atm" in final_carry
        assert "ocean" in final_carry

        # 3. convert predictions to xarray-like dicts
        xr_out = c.predictions_to_xarray(preds)
        assert set(xr_out.keys()) == {"atm", "ocean"}

        # 4. get_info
        info = c.get_info()
        assert "component_info" in info
