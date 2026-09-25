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

    def test_matches_cli_composition(self):
        """The door and the CLI path build the identical coupled model.

        Both routes call the SAME ``jem.runners`` builders
        (:func:`jem.runners.build_coupler`, :func:`jem.runners.build_run_kwargs`)
        on the same composed config, so this is really checking that
        ``load()`` does not skip, reorder or duplicate any of that assembly --
        not re-testing the builders themselves (``tests/unit/test_runners.py``
        already does that in depth).
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
        # run_kwargs is exactly what the CLI would have handed run_chunked.
        self.assertEqual(exp.run_kwargs, ref_run_kwargs)

    def test_dotted_value_override_reaches_run_kwargs(self):
        exp = configurations.load(
            "aquaplanet-slab", **{"coupled_run.total_time": "4 days",
                                  "coupled_run.chunk": "2 days"})
        self.assertEqual(exp.run_kwargs["total_time"], "4 days")
        self.assertEqual(exp.run_kwargs["chunk"], "2 days")

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
        """F3: a host application's own Hydra context survives ``load()``."""
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra

        with initialize_config_module(config_module="jem.config", version_base="1.3"):
            self.assertTrue(GlobalHydra.instance().is_initialized())
            configurations.load("aquaplanet-slab")
            # The host's context survived load(): still initialised and it
            # composes without raising.
            self.assertTrue(GlobalHydra.instance().is_initialized())
            self.assertIsNotNone(compose(config_name="config"))

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
