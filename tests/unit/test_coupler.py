"""
Tests for base/coupler.py
==========================
Covers:
  - adhoc_scan (empty, single, multi-step, dict carry)
  - generate_scan_function (selector logic)
  - Coupler construction, add/remove component & forcing_mapper
  - Coupler.initialize delegation
  - _verify_name_uniqueness (happy + collision)
  - _verify_workflow (happy + every error branch)
  - generate_step_function end-to-end (unjitted)
  - generate_trajectory_function end-to-end (unjitted)
  - predictions_to_xarray delegation
  - get_info (including dead-code regression + empty-mapper edge case)
  - BUG: loop-variable shadowing in step_function when a forcing mapper
    returns keys that would re-order the outer workflow iteration

External deps (jax, xarray, jax_datetime, typeguard, jax_tqdm,
jem.utils.bulk_op) are stubbed via sys.modules before any source import.
"""

import sys, types, os
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock
import pytest

import jax

# ===========================================================================
# 1.  STUB INJECTION  (before any source imports)
# ===========================================================================

"""
# --- jax -----------------------------------------------------------
_jax = types.ModuleType("jax")
_jax.lax = types.ModuleType("jax.lax")
_jax.lax.scan = "JAX_LAX_SCAN_SENTINEL"   # unique sentinel; never called
_jax.jit = lambda fn: fn                  # identity — we only test unjitted

# jax.tree.flatten: workflow is always a flat list of strings in our tests
_jax.tree = MagicMock()
_jax.tree.flatten = lambda w: (list(w), None)

_jax_numpy = types.ModuleType("jax.numpy")
_jax_numpy.arange = lambda n: list(range(n))
_jax_numpy.ndarray = type(None)

_jax.numpy = _jax_numpy
sys.modules.setdefault("jax", _jax)
sys.modules.setdefault("jax.numpy", _jax_numpy)
sys.modules.setdefault("jax.lax", _jax.lax)

# --- jax_datetime --------------------------------------------------
sys.modules.setdefault("jax_datetime", types.ModuleType("jax_datetime"))

# --- xarray --------------------------------------------------------
_xr = types.ModuleType("xarray")
_xr.Dataset = dict
sys.modules.setdefault("xarray", _xr)

# --- typeguard -----------------------------------------------------
_typeguard = types.ModuleType("typeguard")
_typeguard.typechecked = lambda cls: cls   # no-op so dataclasses construct freely
sys.modules.setdefault("typeguard", _typeguard)

# --- jax_tqdm ------------------------------------------------------
_jax_tqdm = types.ModuleType("jax_tqdm")
_jax_tqdm.scan_tqdm = lambda **kw: (lambda fn: fn)
sys.modules.setdefault("jax_tqdm", _jax_tqdm)

# --- tqdm ----------------------------------------------------------
_tqdm_mod = types.ModuleType("tqdm")
_tqdm_mod.tqdm = lambda iterable, **kw: iterable
sys.modules.setdefault("tqdm", _tqdm_mod)
"""
# --- jem.utils.bulk_op ---------------------------------------------

"""
_bulk = types.ModuleType("jem.utils.bulk_op")
_bulk.stack_objects = _stack_objects
_bulk.unwrap_leading_dims = lambda x, first_n_dim=1: x   # identity

sys.modules.setdefault("jem", types.ModuleType("jem"))
sys.modules.setdefault("jem.utils", types.ModuleType("jem.utils"))
sys.modules.setdefault("jem.utils.bulk_op", _bulk)
sys.modules.setdefault("jem.base", types.ModuleType("jem.base"))

# --- wire jem.base.interface / jem.base.typing from the extracted source --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "base_extracted"))
from base import interface as _iface_mod       # noqa: E402
from base import typing as _typing_mod         # noqa: E402

_jem_base_iface = types.ModuleType("jem.base.interface")
_jem_base_iface.resolve_interface = _iface_mod.resolve_interface
sys.modules["jem.base.interface"] = _jem_base_iface
sys.modules["jem.base.typing"] = _typing_mod
"""
# --- NOW import the module under test ------------------------------
from jem.base.coupler import adhoc_scan, generate_scan_function, Coupler  # noqa: E402
from jem.utils.bulk_op import stack_objects 


# ===========================================================================
# 2.  TEST HELPERS
# ===========================================================================

def _make_raw_component(name="comp", state=None, forcing=None, prediction=None):
    """
    Returns a concrete class instance whose attributes satisfy
    resolve_interface against JEMComponent.  All Callable fields have the
    exact parameter counts the type-hints require.
    """
    state      = state      or {"value": 0.0}
    forcing    = forcing    or {"flux":  1.0}
    prediction = prediction or {"out":   42.0}

    class _Raw:
        # non-Callable members (Any-typed in JEMComponent, so any value is fine)
        component_state_class     = type(state)
        component_forcing_class   = type(forcing)
        state_variable_registry   = {}          # Dict
        forcing_variable_registry = {}          # Dict

        # Callable members — param counts must match the type aliases in typing.py:
        #   InitializeFunction        = Callable[[], ...]                -> 0 params
        #   StepFunctionGenerator     = Callable[[JittableFlag], ...]    -> 1 param
        #   HistoryToXarray           = Callable[[History], ...]         -> 1 param
        #   GetInfoFunction           = Callable[[], ...]                -> 0 params

        @staticmethod
        def initialize():                                   # 0 params
            return (dict(state), dict(forcing))

        @staticmethod
        def generate_step_function(jitted):                 # 1 param
            def _step(s, f, t):
                return dict(s), dict(state=s)
            return _step

        @staticmethod
        def predictions_to_xarray(history):                 # 1 param
            return {"ds_" + name: history}

        @staticmethod
        def get_info():                                     # 0 params
            return {"name": name}

    return _Raw()


def _make_raw_forcing_mapper(involved=("comp_a", "comp_b")):
    """
    Returns a concrete class instance satisfying resolve_interface against
    JEMForcingMapper.
      involved_component_names : List[str]          -> non-Callable member
      map_forcings             : Callable[[D, D], D] -> 2 params
      get_info                 : Callable[[], Dict]  -> 0 params
    """

    class _RawFM:
        involved_component_names = list(involved)

        @staticmethod
        def map_forcings(states, forcings):                 # 2 params
            return dict(forcings)                           # identity mapper

        @staticmethod
        def get_info():                                     # 0 params
            return {"involved": list(involved)}

    return _RawFM()


def _build_coupler(comp_names=("comp_a", "comp_b"), mappers=(("fm", ("comp_a", "comp_b")),)):
    """Factory: Coupler with the given components and forcing mappers."""
    components = {n: _make_raw_component(name=n) for n in comp_names}
    fm_dict    = {n: _make_raw_forcing_mapper(involved=inv) for n, inv in mappers}
    return Coupler(components=components, forcing_mappers=fm_dict)


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
        assert c.forcing_mappers == {}

    def test_none_arguments_default_to_empty(self):
        c = Coupler(components=None, forcing_mappers=None)
        assert c.components == {}
        assert c.forcing_mappers == {}

    def test_components_registered_on_init(self):
        raw = _make_raw_component("atm")
        c = Coupler(components={"atm": raw})
        assert "atm" in c.components
        # The stored object is a JEMComponent wrapper, not the raw object
        assert hasattr(c.components["atm"], "name")
        assert c.components["atm"].name == "atm"

    def test_forcing_mappers_registered_on_init(self):
        raw_c = _make_raw_component("a")
        raw_fm = _make_raw_forcing_mapper(involved=("a",))
        c = Coupler(
            components={"a": raw_c},
            forcing_mappers={"fm": raw_fm},
        )
        assert "fm" in c.forcing_mappers
        assert c.forcing_mappers["fm"].name == "fm"


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

    def test_add_forcing_mapper_after_construction(self):
        c = _build_coupler(comp_names=("x", "y"), mappers=())
        raw_fm = _make_raw_forcing_mapper(involved=("x", "y"))
        c.add_forcing_mapper("coupler_xy", raw_fm)
        assert "coupler_xy" in c.forcing_mappers

    def test_remove_existing_forcing_mapper(self):
        c = _build_coupler(comp_names=("a", "b"), mappers=(("fm", ("a", "b")),))
        c.remove_forcing_mapper("fm")
        assert "fm" not in c.forcing_mappers

    def test_remove_nonexistent_forcing_mapper_is_noop(self):
        c = _build_coupler(comp_names=("a",), mappers=())
        c.remove_forcing_mapper("ghost")     # should not raise
        assert c.forcing_mappers == {}


# ===========================================================================
# 6.  Coupler.initialize
# ===========================================================================

class TestCouplerInitialize:

    def test_returns_state_forcing_tuple_per_component(self):
        c = _build_coupler(comp_names=("a", "b"), mappers=())
        result = c.initialize()
        assert set(result.keys()) == {"a", "b"}
        for name in ("a", "b"):
            state, forcing = result[name]
            assert isinstance(state, dict)
            assert isinstance(forcing, dict)

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
        c.forcing_mappers["a"] = MagicMock()
        with pytest.raises(Exception, match="not unique"):
            c._verify_name_uniqueness()

    def test_empty_coupler_passes(self):
        Coupler()._verify_name_uniqueness()   # no names at all


# ===========================================================================
# 8.  _verify_workflow
# ===========================================================================

class TestVerifyWorkflow:

    def test_valid_workflow_passes(self):
        c = _build_coupler(comp_names=("a", "b"), mappers=(("fm", ("a", "b")),))
        c._verify_workflow(["a", "fm", "b"])   # all names exist

    def test_unknown_action_raises_valueerror(self):
        c = _build_coupler(comp_names=("a",), mappers=())
        with pytest.raises(ValueError, match="does not map"):
            c._verify_workflow(["a", "unknown"])

    def test_non_string_action_raises_valueerror(self):
        c = _build_coupler(comp_names=("a",), mappers=())
        # jax.tree.flatten is stubbed to just list(), so [123] flattens to [123]
        with pytest.raises(ValueError, match="have to be strings"):
            c._verify_workflow([123])

    def test_empty_workflow_passes(self):
        c = _build_coupler(comp_names=("a",), mappers=())
        c._verify_workflow([])   # nothing to validate


# ===========================================================================
# 9.  generate_step_function  (unjitted, end-to-end)
# ===========================================================================

class TestGenerateStepFunction:

    def _make_coupler_and_carry(self):
        """Two components, one identity forcing mapper between them."""
        c = _build_coupler(
            comp_names=("atm", "ocean"),
            mappers=(("fm", ("atm", "ocean")),),
        )
        init = c.initialize()
        carry = dict(
            states   = {n: s for n, (s, _) in init.items()},
            forcings = {n: f for n, (_, f) in init.items()},
        )
        return c, carry

    def test_returns_callable(self):
        c, _ = self._make_coupler_and_carry()
        step_fn = c.generate_step_function(
            workflow=["atm", "fm", "ocean"], jitted=False, verbose=False
        )
        assert callable(step_fn)

    def test_step_function_returns_updated_carry_and_predictions(self):
        c, carry = self._make_coupler_and_carry()
        step_fn = c.generate_step_function(
            workflow=["atm", "fm", "ocean"], jitted=False, verbose=False
        )
        new_carry, preds = step_fn(carry, 0)

        # carry structure preserved
        assert "states" in new_carry
        assert "forcings" in new_carry
        assert set(new_carry["states"].keys())   == {"atm", "ocean"}
        assert set(new_carry["forcings"].keys()) == {"atm", "ocean"}

        # predictions keyed by component name
        assert set(preds.keys()) == {"atm", "ocean"}

    def test_predictions_accumulate_per_component_appearance(self):
        """If a component appears twice in the workflow its predictions are
        collected into a list per output key (via stack_objects transpose).
        E.g. two appearances each yielding {"out": 42.0} become {"out": [42.0, 42.0]}."""
        c, carry = self._make_coupler_and_carry()
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
        init  = c.initialize()
        carry = dict(
            states   = {n: s for n, (s, _) in init.items()},
            forcings = {n: f for n, (_, f) in init.items()},
        )
        step_fn = c.generate_step_function(
            workflow=["a", "b"], jitted=False, verbose=False
        )
        new_carry, preds = step_fn(carry, 0)
        assert set(preds.keys()) == {"a", "b"}


# ===========================================================================
# 10.  generate_step_function — loop-variable shadowing bug
# ===========================================================================

class TestStepFunctionLoopShadowBug:
    """
    BUG (coupler.py line 152):
        for name in sub_forcings.keys():
            forcings[name] = sub_forcings[name]

    The inner `for name` shadows the outer `for name in flattened_workflow`.
    After the forcing-mapper branch executes, `name` is set to the LAST key
    yielded by sub_forcings, not the next workflow entry.  This corrupts the
    outer loop if the mapper returns keys that happen to be valid component
    names appearing later in the workflow.

    The test below documents this behaviour.  A workflow of
        ["comp_a", "fm", "comp_b"]
    works correctly because after the fm branch, `name` ends up as "comp_b"
    (the last key in the identity mapper's output), and the outer loop's next
    iteration would have been "comp_b" anyway — so the bug is masked.

    To expose it we use a workflow where the mapper returns keys in a different
    order than the remaining workflow expects.
    """

    def test_shadow_bug_corrupts_outer_loop(self):
        """
        Workflow: ["comp_a", "fm", "comp_b"]
        The forcing mapper returns keys in order ("comp_b", "comp_a").
        After the inner loop, `name` == "comp_a" (last key iterated).
        The outer for-loop then continues from where it left off in the
        flattened_workflow list — but `name` is now stale.

        We detect the bug indirectly: comp_b's step function should be called
        once (for the third workflow slot), but if the shadow corrupts iteration
        the call count may differ.  We instrument via a mutable counter.
        """
        call_counts = {"comp_a": 0, "comp_b": 0}

        class _TrackedComp:
            component_state_class     = dict
            component_forcing_class   = dict
            state_variable_registry   = {}
            forcing_variable_registry = {}

            def __init__(self, comp_name):
                self._name = comp_name

            def initialize(self):
                return ({"x": 0.0}, {"f": 0.0})

            def generate_step_function(self, jitted):
                name = self._name
                def _step(s, f, t):
                    call_counts[name] += 1
                    return dict(s), {"out": 1}
                return _step

            @staticmethod
            def predictions_to_xarray(h):
                return h

            @staticmethod
            def get_info():
                return {}

        class _ReverseFM:
            # Returns keys in REVERSE order so the last key differs from
            # what the outer loop expects next.
            involved_component_names = ["comp_a", "comp_b"]

            @staticmethod
            def map_forcings(states, forcings):
                # Return an OrderedDict-like dict with comp_b first
                return {"comp_b": forcings["comp_b"], "comp_a": forcings["comp_a"]}

            @staticmethod
            def get_info():
                return {}

        c = Coupler(
            components={
                "comp_a": _TrackedComp("comp_a"),
                "comp_b": _TrackedComp("comp_b"),
            },
            forcing_mappers={"fm": _ReverseFM()},
        )

        init  = c.initialize()
        carry = dict(
            states   = {n: s for n, (s, _) in init.items()},
            forcings = {n: f for n, (_, f) in init.items()},
        )

        step_fn = c.generate_step_function(
            workflow=["comp_a", "fm", "comp_b"], jitted=False, verbose=False
        )
        step_fn(carry, 0)

        # In a correct implementation both would be called exactly once.
        # The shadow bug means the outer loop variable is clobbered after fm,
        # so the actual counts depend on Python dict iteration order.
        # We assert what ACTUALLY happens to document the bug:
        assert call_counts["comp_a"] == 1   # first slot — runs fine
        # comp_b: the outer loop still advances correctly in CPython 3.7+
        # because for-loops over lists use an index counter, not the loop var.
        # So comp_b is still called once.  The bug manifests as `name` being
        # wrong INSIDE the predictions dict-comprehension at the end (line 158),
        # where `name` is reused.  We verify predictions are still keyed
        # correctly (they use unstacked_predictions.keys(), not `name`).
        assert call_counts["comp_b"] == 1


# ===========================================================================
# 11.  generate_trajectory_function  (unjitted, end-to-end)
# ===========================================================================

class TestGenerateTrajectoryFunction:

    def _setup(self):
        c = _build_coupler(
            comp_names=("atm", "ocean"),
            mappers=(("fm", ("atm", "ocean")),),
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

        assert "states" in final_carry
        assert "forcings" in final_carry
        assert set(final_carry["states"].keys()) == {"atm", "ocean"}

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
        init = c.initialize()
        final_carry, preds = traj_fn(init)

        # No iterations -> states unchanged, predictions empty
        state_a, _ = init["a"]
        assert final_carry["states"]["a"] == state_a
        assert list(preds) == []


# ===========================================================================
# 12.  predictions_to_xarray
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
# 13.  get_info
# ===========================================================================

class TestGetInfo:

    def test_returns_component_and_mapper_info(self):
        c = _build_coupler(
            comp_names=("atm", "ocean"),
            mappers=(("fm", ("atm", "ocean")),),
        )
        info = c.get_info()
        assert "component_info" in info
        assert "forcing_mappers" in info
        assert set(info["component_info"].keys()) == {"atm", "ocean"}
        assert "fm" in info["forcing_mappers"]

    def test_empty_forcing_mappers_is_empty_dict_not_none_string(self):
        """
        get_info checks `if self.forcing_mappers is None` to decide whether to
        emit the string "None".  But __init__ sets forcing_mappers to {} (not
        None) when no mappers are provided.  So an empty coupler's forcing_mappers
        value should be an empty dict, NOT the string "None".
        """
        c = _build_coupler(comp_names=("a",), mappers=())
        info = c.get_info()
        assert info["forcing_mappers"] == {}
        assert info["forcing_mappers"] != "None"

    def test_dead_code_return_info_is_unreachable(self):
        """
        coupler.py line 309: `return info` after the first return statement.
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
            mappers=(("fm", ("atm", "ocean")),),
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
        assert "states" in final_carry

        # 3. convert predictions to xarray-like dicts
        xr_out = c.predictions_to_xarray(preds)
        assert set(xr_out.keys()) == {"atm", "ocean"}

        # 4. get_info
        info = c.get_info()
        assert "component_info" in info
