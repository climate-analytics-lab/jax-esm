"""Tests for the recipe door ``jem.configurations`` (issue #131).

Deliberately imports neither ``hydra`` nor ``omegaconf`` at module top level:
the door hides them, so a caller (and this test) needs only
``jem.configurations``. A meta-test below enforces that on this file's own
AST -- mirroring jax-gcm's own ``jcm/configurations_test.py``, this module's
reference formulation.

The "CLI path" a few tests compare against is built with
``configurations._compose`` + ``jem.runners.build_coupler`` /
``jem.runners.build_run_kwargs`` directly, rather than by importing
``hydra.compose``/``initialize_config_module`` in this file the way
``tests/unit/test_runners.py`` does -- so this file keeps needing only the
door's own module, which is the property the AST meta-test checks.
"""

import ast
import shutil
import unittest
from pathlib import Path

import jax

from jem import configurations, runners


def _term_names(coupler):
    return {name: type(component).__name__
            for name, component in coupler.components.items()}


class TestAvailable(unittest.TestCase):
    def test_lists_all_shipped_configurations_with_summaries(self):
        av = configurations.available()
        self.assertEqual(
            set(av),
            {p.stem for p in configurations.CONFIGURATION_DIR.glob("*.yaml")},
        )
        # Five configurations are shipped today; a new one only has to land
        # as a yaml file, this assertion included to catch the moment one
        # goes missing from the recipe store rather than staying silent.
        self.assertEqual(len(av), 5)
        self.assertTrue(all(isinstance(v, str) and v for v in av.values()))
        # The one-line summary is the yaml's first human comment.
        self.assertIn("SPEEDY", av["aquaplanet-slab"])
        self.assertIn("Earth", av["earth-slab"])


class TestLoad(unittest.TestCase):
    def test_unknown_name_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown configuration"):
            configurations.load("does-not-exist")

    def test_builds_and_hides_hydra(self):
        exp = configurations.load("aquaplanet-slab")
        from jem.base.coupler import Coupler

        self.assertIsInstance(exp.coupler, Coupler)
        self.assertIsInstance(exp.config, dict)
        self.assertEqual(exp.name, "aquaplanet-slab")
        # aquaplanet-slab builds atm/ocn/seaice; no land.
        self.assertEqual(set(exp.coupler.components), {"atm", "ocn", "seaice"})
        # No omegaconf container survives on the returned surface.
        self.assertNotIn("DictConfig", type(exp.config).__name__)
        for value in exp.run_kwargs.values():
            self.assertNotIn("DictConfig", type(value).__name__)
        # run_kwargs matches the default coupled_run recipe (jem/config/
        # coupled_run/default.yaml), unmodified.
        self.assertEqual(exp.run_kwargs["total_time"], "30 days")
        self.assertEqual(exp.run_kwargs["chunk"], "30 days")
        self.assertNotIn("log_level", exp.run_kwargs)

    def test_load_builds_through_the_cli_builders(self):
        """``load()`` calls the SAME ``jem.runners`` builders the CLI does.

        This only shows that ``load()`` does not skip, reorder or duplicate
        the ``build_coupler``/``build_run_kwargs`` assembly -- it composes
        through ``configurations._compose`` itself, so it cannot by itself
        prove the door agrees with the REAL command line
        (``python -m jem.main``); ``test_load_matches_the_cli_entry_point``
        below exercises that entry point directly, in a subprocess.
        ``tests/unit/test_runners.py`` tests the builders themselves in
        depth.
        """
        exp = configurations.load("aquaplanet-slab")

        cfg = configurations._compose("aquaplanet-slab", [])
        ref_coupler = runners.build_coupler(cfg)
        ref_run_kwargs = runners.build_run_kwargs(cfg)

        self.assertEqual(list(exp.coupler.components), list(ref_coupler.components))
        self.assertEqual(_term_names(exp.coupler), _term_names(ref_coupler))
        self.assertEqual(exp.coupler.workflow, ref_coupler.workflow)
        self.assertEqual(exp.coupler.coupling_timestep, ref_coupler.coupling_timestep)
        self.assertEqual(exp.coupler.start_date, ref_coupler.start_date)
        self.assertEqual(exp.coupler.calendar, ref_coupler.calendar)
        # The carry each coupler scans over has the identical pytree shape.
        self.assertEqual(jax.tree_util.tree_structure(exp.coupler.initialize()),
                         jax.tree_util.tree_structure(ref_coupler.initialize()))
        # run_kwargs matches `build_run_kwargs` on every key except
        # `output_dir`, which the door deliberately diverges on: it gets its
        # own fresh `outputs/<date>/<time>` (see `load`'s docstring),
        # `build_run_kwargs` called directly gets the plain `"outputs"`
        # fallback. Checked explicitly rather than silently dropped from the
        # comparison.
        exp_kwargs = dict(exp.run_kwargs)
        exp_output_dir = exp_kwargs.pop("output_dir")
        ref_kwargs = dict(ref_run_kwargs)
        ref_output_dir = ref_kwargs.pop("output_dir")
        self.assertEqual(exp_kwargs, ref_kwargs)
        self.assertEqual(ref_output_dir, "outputs")
        self.assertTrue(exp_output_dir.startswith("outputs/"), exp_output_dir)
        shutil.rmtree(exp_output_dir, ignore_errors=True)

    def test_load_matches_the_cli_entry_point(self):
        """The door's ``.config`` matches ``python -m jem.main``'s own composition.

        Runs the REAL entry point in a subprocess (``--cfg job --resolve``,
        which composes and prints without building or running anything) and
        compares its YAML against ``load(...).config`` -- unlike
        ``test_load_builds_through_the_cli_builders``, this cannot pass
        because both sides happen to call the same private helper; it only
        passes if ``python -m jem.main`` and ``jem.configurations.load``
        agree on what a recipe composes to.

        Today the two are byte-for-byte identical once parsed: ``compose()``
        (what the door uses) and ``--cfg job`` (what the CLI prints) both
        exclude Hydra's own ``hydra:`` config node already, and
        ``coupled_run.output_dir`` is still ``null`` in the COMPOSED config
        on both sides -- the door's fresh-directory substitution happens
        after this point, in ``run_kwargs``, not in ``.config`` (see
        ``load``'s docstring). So nothing needs excluding today; if that ever
        stops being true, name here exactly what has to be excluded and why,
        rather than loosening this into a partial comparison silently.
        """
        import subprocess
        import sys

        import yaml

        result = subprocess.run(
            [sys.executable, "-m", "jem.main", "+configuration=aquaplanet-slab",
             "--cfg", "job", "--resolve"],
            capture_output=True, text=True, check=True,
        )
        cli_config = yaml.safe_load(result.stdout)
        # Defensive: if a future Hydra/jem change starts printing the
        # `hydra:` node under `--cfg job`, drop it here rather than let it
        # fail this comparison for a reason unrelated to what this test is
        # for (`compose()` on the door's side never includes it either).
        cli_config.pop("hydra", None)

        door_config = configurations.load("aquaplanet-slab").config

        self.assertEqual(cli_config, door_config)

    def test_dotted_value_override_reaches_run_kwargs(self):
        exp = configurations.load(
            "aquaplanet-slab", **{"coupled_run.total_time": "4 days",
                                  "coupled_run.chunk": "2 days"})
        self.assertEqual(exp.run_kwargs["total_time"], "4 days")
        self.assertEqual(exp.run_kwargs["chunk"], "2 days")

    def test_documented_total_time_override_is_a_whole_number_of_chunks(self):
        """The override example ``python_api.md`` documents actually runs.

        A local review of #131 found the ORIGINAL example
        (``load("earth-slab", **{"coupled_run.total_time": 10})``) actually
        raised from ``run_chunked``, because every shipped recipe's own
        ``coupled_run.chunk`` stays its 30-day default and 10 is not a
        multiple of it. The corrected value is read straight out of
        ``docs/source/python_api.md``'s door section -- not copied into this
        test as a second literal -- so a future edit to the docs is what this
        test exercises, and a value that regresses back to something
        non-divisible fails here rather than only at a user's own
        ``run_chunked`` call. It is validated against the same rule
        ``run_chunked`` itself enforces
        (:func:`jem.driver._require_whole_number_of_chunks`, factored out of
        ``run_chunked`` so this calls the real rule rather than a
        reimplementation of it) rather than actually integrating the run, so
        this test stays fast.
        """
        import re

        from jem import driver
        from tests.unit.test_readme_quickstart import PYTHON_API

        text = PYTHON_API.read_text()
        section = text.split("## Validated configurations from Python", 1)[1]
        section = section.split("\n## ", 1)[0]
        match = re.search(
            r'"coupled_run\.total_time":\s*"((?:[^"\\]|\\.)*)"',
            " ".join(section.split()),  # collapse the example's own line wrap
        )
        assert match, (
            f"no `coupled_run.total_time` override example found in "
            f"{PYTHON_API}'s door section"
        )
        documented_total_time = match.group(1)

        exp = configurations.load(
            "earth-slab", **{"coupled_run.total_time": documented_total_time})
        coupling_days = exp.coupler.dt_seconds / 86400
        steps_per_chunk = driver._whole_steps(
            exp.run_kwargs["chunk"], coupling_days, exp.coupler, "chunk")
        total_steps = driver._whole_steps(
            exp.run_kwargs["total_time"], coupling_days, exp.coupler, "total_time")
        # Raises if not a whole number of chunks -- the actual rule
        # `run_chunked` enforces, not a reimplementation of it.
        driver._require_whole_number_of_chunks(
            exp.run_kwargs["total_time"], exp.run_kwargs["chunk"],
            total_steps, steps_per_chunk)

    def test_output_dir_defaults_to_a_fresh_directory_per_call(self):
        """Two successive ``load()`` calls with no ``output_dir`` do not collide.

        A local review of #131 found ``load()`` defaulting ``output_dir`` to
        the literal ``"outputs"`` (``build_run_kwargs``'s own non-door
        fallback): a second ``load()`` + ``run_chunked`` would then resume
        the first's checkpoint rather than starting a fresh run.
        """
        exp1 = configurations.load("aquaplanet-slab")
        exp2 = configurations.load("aquaplanet-slab")
        try:
            self.assertNotEqual(exp1.run_kwargs["output_dir"], exp2.run_kwargs["output_dir"])
            # Neither is a subdirectory (in particular, a checkpoint
            # directory) of the other.
            dir1, dir2 = exp1.run_kwargs["output_dir"], exp2.run_kwargs["output_dir"]
            self.assertFalse(str(dir2).startswith(str(dir1) + "/"))
            self.assertFalse(str(dir1).startswith(str(dir2) + "/"))
        finally:
            for d in (exp1.run_kwargs["output_dir"], exp2.run_kwargs["output_dir"]):
                shutil.rmtree(d, ignore_errors=True)

    def test_explicit_output_dir_override_is_honoured_exactly(self):
        exp = configurations.load(
            "aquaplanet-slab", **{"coupled_run.output_dir": "/tmp/jem-explicit-output-dir"})
        self.assertEqual(exp.run_kwargs["output_dir"], "/tmp/jem-explicit-output-dir")

    def test_config_group_override_selects_an_option(self):
        # `seaice="none"` is a plain Python string, exactly the CLI's bare
        # `seaice=none` -- exercised through the quoting escape hatch and
        # verified to still select the group option, not a literal value.
        exp = configurations.load("aquaplanet-slab", seaice="none")
        self.assertEqual(set(exp.coupler.components), {"atm", "ocn"})
        self.assertEqual(exp.config["seaice"], None)

    def test_earth_slab_applies_its_tuned_overrides(self):
        # Two of the three documented tunables in earth-slab.yaml (the ocean
        # relaxation time and the land tdland) reach the built components,
        # proving the door builds the SAME objects `+configuration=earth-slab`
        # would -- not a lighter approximation of them.
        exp = configurations.load("earth-slab")
        ocn_params = exp.coupler.components["ocn"].params
        lnd_params = exp.coupler.components["lnd"].params
        self.assertAlmostEqual(float(ocn_params.relaxation_time), 2592000.0)
        self.assertAlmostEqual(float(lnd_params.tdland), 86400.0)

    def test_restores_host_hydra_context(self):
        """F3: a host application's own, DIFFERENT Hydra context survives ``load()``.

        The host composes its OWN tiny config from a temp directory, not
        jem's -- if ``load()`` left the wrong context active (e.g. its own,
        not properly cleared and restored), the host's ``compose()`` below
        would either raise or hand back jem's config instead of the host's,
        rather than this test passing vacuously because both sides happen to
        be jem's own context.
        """
        import tempfile
        from pathlib import Path as _Path

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        with tempfile.TemporaryDirectory() as host_dir:
            (_Path(host_dir) / "host_config.yaml").write_text("host_marker: 42\n")
            with initialize_config_dir(version_base=None, config_dir=host_dir):
                self.assertTrue(GlobalHydra.instance().is_initialized())
                configurations.load("aquaplanet-slab")
                # The host's context survived load(): still initialised, and
                # it is still able to compose its OWN config, not jem's.
                self.assertTrue(GlobalHydra.instance().is_initialized())
                host_cfg = compose(config_name="host_config")
                self.assertEqual(host_cfg.host_marker, 42)
                self.assertNotIn("atmosphere", host_cfg)

    def test_constants_override_reaches_the_build_through_the_door(self):
        """A `+atmosphere.constants.*` override reaches the built model.

        Unlike jax-gcm's own door, `load()` applies no separate step for
        this: `build_coupler` -> `build_atmosphere` already applies it (see
        `load`'s docstring). This test is what proves that analysis correct
        rather than merely asserted.
        """
        import jcm.constants as c

        saved = c.physical_constants
        try:
            configurations.load(
                "aquaplanet-slab", **{"+atmosphere.constants.grav": 9.7})
            self.assertAlmostEqual(c.grav, 9.7)
        finally:
            # Required, not decorative: jcm.constants is a process-global
            # singleton that `load()` deliberately never resets (see its
            # docstring's "Constants overrides are process-global" section),
            # so without this restore the override above would leak into
            # every OTHER test in this process that builds a jax-gcm model,
            # not just this one.
            c.set_constants(saved)


class TestOverrideStr(unittest.TestCase):
    def test_quotes_hydra_grammar_values(self):
        # F2: a string value carrying Hydra grammar characters (comma, '=',
        # braces -- ordinary in paths/filenames) must compose back verbatim
        # instead of being read as list/sweep/assignment syntax.
        from hydra.core.override_parser.overrides_parser import OverridesParser

        parser = OverridesParser.create()
        for value in ("/tmp/a,b", "prefix=tag", "/out/{run}/x", "it's",
                      "plain/path"):
            tok = configurations._override_str("coupled_run.output_dir", value)
            self.assertEqual(parser.parse_overrides([tok])[0].value(), value)
        self.assertEqual(
            configurations._override_str("coupled_run.output_averages", None),
            "coupled_run.output_averages=null")
        self.assertEqual(
            configurations._override_str("coupled_run.subsample", 3),
            "coupled_run.subsample=3")

    def test_numeric_looking_string_stays_a_string(self):
        # Unlike the CLI's bare `subsample=3` (composes to the int 3), a
        # Python **overrides value that is itself a numeric-looking `str`
        # composes to the STRING "3" -- it goes through the same quoting as
        # any other string, since `_override_str` has no way to know the
        # caller meant a number rather than, say, a zero-padded code.
        from hydra.core.override_parser.overrides_parser import OverridesParser

        parser = OverridesParser.create()
        tok = configurations._override_str("coupled_run.subsample", "3")
        value = parser.parse_overrides([tok])[0].value()
        self.assertEqual(value, "3")
        self.assertIsInstance(value, str)

    def test_dict_value_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "ocean.params"):
            configurations._override_str("ocean.params", {"forcing_method": "relaxation"})

    def test_tuple_value_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "ocean.params"):
            configurations._override_str("ocean.params", (1, 2))

    def test_tuple_error_suggests_a_list_not_a_dotted_override(self):
        # "one dotted override per field" is meaningless for a tuple (it
        # names no nested config keys the way a dict's do), so its message
        # must say something else -- a plain list, which composes fine.
        with self.assertRaisesRegex(TypeError, r"\blist\b") as ctx:
            configurations._override_str("ocean.params", (1, 2))
        self.assertNotIn("dotted override per field", str(ctx.exception))

    def test_list_containing_none_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "ocean.params"):
            configurations._override_str("ocean.params", [None])

    def test_list_containing_none_nested_raises_type_error(self):
        # The same silent str()->'None' mis-compose applies at any nesting
        # depth, not just the top level of the list.
        with self.assertRaisesRegex(TypeError, "ocean.params"):
            configurations._override_str("ocean.params", [[1, None], 2])

    def test_list_containing_a_dict_raises_type_error(self):
        # A dict nested in a list is unrepresentable for the same reason a
        # top-level dict is. Before, it slipped past the top-level check and
        # reached Hydra's parser as a raw str() token, failing there with an
        # opaque HydraException rather than this clear TypeError.
        for value in ([{"value": None}], [{"value": 1}], [[{"a": 1}]]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, r"ocean\.params.*dict"):
                    configurations._override_str("ocean.params", value)

    def test_list_containing_a_tuple_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, r"ocean\.params.*tuple"):
            configurations._override_str("ocean.params", [(1, 2)])

    def test_list_containing_any_non_literal_raises_type_error(self):
        # Only bool/int/float/str and nested lists round-trip through str()
        # to a Hydra list token; anything else (e.g. a Path, whose str() is a
        # PosixPath(...) repr) must be refused, not handed to the parser.
        from pathlib import Path

        with self.assertRaisesRegex(TypeError, r"ocean\.params.*PosixPath"):
            configurations._override_str("ocean.params", [Path("/tmp/x")])

    def test_every_accepted_list_element_type_round_trips(self):
        # The accepted set must be exactly what composes faithfully: each
        # element type, alone and nested, parses back to the same value.
        from hydra.core.override_parser.overrides_parser import OverridesParser

        parser = OverridesParser.create()
        value = [True, 1, 2.5, "a,b", ["x", [3, False]]]
        tok = configurations._override_str("ocean.params", value)
        self.assertEqual(parser.parse_overrides([tok])[0].value(), value)

    def test_list_without_none_still_composes(self):
        # A plain list -- no None anywhere -- is unaffected by the new check.
        from hydra.core.override_parser.overrides_parser import OverridesParser

        parser = OverridesParser.create()
        tok = configurations._override_str("ocean.params", [1, 2, [3, 4]])
        self.assertEqual(
            parser.parse_overrides([tok])[0].value(), [1, 2, [3, 4]])

    def test_quoted_path_composes_through_load(self):
        # F2 end to end: a grammar-carrying path survives a real compose via
        # `load()`, not just token parsing.
        exp = configurations.load(
            "aquaplanet-slab",
            **{"coupled_run.output_dir": "/tmp/a,b={run}=z",
               "coupled_run.total_time": "2 days", "coupled_run.chunk": "2 days"})
        self.assertEqual(exp.run_kwargs["output_dir"], "/tmp/a,b={run}=z")


def test_module_imports_no_hydra_or_omegaconf_at_top_level():
    """The door's whole point: a caller (this file) never imports hydra."""
    tree = ast.parse(Path(__file__).read_text())
    banned = {"hydra", "omegaconf"}
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.append(node.module.split(".")[0])
    assert not (banned & set(top_level)), \
        f"top-level imports leak hydra/omegaconf: {top_level}"


if __name__ == "__main__":
    unittest.main()
