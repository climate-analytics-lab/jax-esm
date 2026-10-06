"""Pin ``01_aquaplanet.ipynb``'s component construction to ``python_api.md``.

``01_basic/01_aquaplanet.ipynb`` builds its atmosphere, its slab grid and its
components the way ``docs/source/python_api.md``'s own quick-start block does
(`tests/unit/test_readme_quickstart.py` already pins that block, in turn, to
the README's copy). This module is the third leg of that chain: it pins the
notebook's construction to the SAME source, so the three (README,
``python_api.md``, the notebook) cannot quietly drift apart on how a
component is built.

What is pinned, and what is not
-------------------------------
The two deliberately part ways at the wiring. ``python_api.md`` couples its
components with the one-line standard table (``default_exchangers``); the
notebook writes the exchange out by hand, as a function, because showing
that function is what the notebook is for. So "the construction" here is the
part the two share: every statement of ``python_api.md``'s first
```python``` block from its first assignment through its ``components``
dict -- the clock, the atmosphere, the grid and the components -- and
nothing after it. Those statements must appear in the notebook's
construction cell, in order, after exactly two substitutions:

- the ``components`` dict gains a ``"seaice": SlabSeaiceModel(grid,
  name="seaice")`` entry (written as a multi-line dict literal purely for
  line length): ``python_api.md``'s block builds an atmosphere/ocean PAIR,
  the smallest coupled model worth showing in the docs, while the notebook
  also couples the slab sea ice;
- the ``Model(...)`` call gains an explicit ``time_step=12`` (minutes):
  without it, ``Model`` picks the physics' own stable time step (30 minutes
  for SPEEDY T31L8) rather than the ``+configuration=aquaplanet-slab``
  recipe's actual step (``atmosphere.run.time_step``, 12 minutes, from
  jax-gcm's ``run/default.yaml``). ``python_api.md``'s own worked example is
  a standalone construction with no configuration to match, so it is not
  wrong to leave this implicit there; the notebook claims to run the same
  model as ``+configuration=aquaplanet-slab``.

Statements are compared, not bytes: comments, blank lines and imports are
the notebook's own (it imports what its exchanger and its later cells need).
That the notebook's hand-written exchanger moves the same fields as the
standard table is not this module's business --
``tests/unit/test_notebook_equivalence.py`` checks it by running both.

If ``python_api.md``'s block changes (a renamed constructor argument, for
instance), this test fails until the notebook's construction cell is updated
to match -- which is the whole point.
"""

import ast
from pathlib import Path

import nbformat

from tests.unit.test_readme_quickstart import PYTHON_API, _first_python_block

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "examples" / "01_basic" / "01_aquaplanet.ipynb"

#: The two substitutions that turn `python_api.md`'s atmosphere/ocean-only,
#: configuration-free construction into what the notebook must build. Each
#: is applied exactly once; see the module docstring for why these two.
_OLD_COMPONENTS = 'components = {"atm": atm, "ocn": SlabOceanModel(grid)}\n'
_NEW_COMPONENTS = (
    'components = {\n'
    '    "atm": atm,\n'
    '    "ocn": SlabOceanModel(grid),\n'
    '    "seaice": SlabSeaiceModel(grid, name="seaice"),\n'
    '}\n'
)
_OLD_MODEL = (
    "atm_model = jcm.model.Model(coords=get_speedy_coords(), start_time=start_date)\n"
)
_NEW_MODEL = (
    "atm_model = jcm.model.Model(\n"
    "    coords=get_speedy_coords(), start_time=start_date, time_step=12\n"
    ")\n"
)
_SUBSTITUTIONS = (
    (_OLD_MODEL, _NEW_MODEL),
    (_OLD_COMPONENTS, _NEW_COMPONENTS),
)


def _statements(source: str) -> list[str]:
    """Return ``source``'s statements, without imports, comments or layout.

    ``ast.unparse`` of each top-level statement: two sources that differ only
    in comments, blank lines or line breaks give the same list.
    """
    return [
        ast.unparse(node) for node in ast.parse(source).body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def _expected_construction() -> list[str]:
    """Return the statements ``01_aquaplanet.ipynb`` must build its components with."""
    block = _first_python_block(PYTHON_API)
    for old, new in _SUBSTITUTIONS:
        count = block.count(old)
        assert count == 1, (
            f"{PYTHON_API} construction no longer contains {old!r} exactly "
            f"once (found {count}); its atmosphere/ocean example changed in "
            "a way this test's substitutions no longer apply to -- update "
            "both this test and the notebook."
        )
        block = block.replace(old, new)

    statements = _statements(block)
    last = [index for index, statement in enumerate(statements)
            if statement.startswith("components = ")]
    assert len(last) == 1, (
        f"{PYTHON_API} no longer assigns `components` exactly once; update "
        "this test's split point to match its new shape."
    )
    return statements[:last[0] + 1]


def _notebook_construction_cell() -> str:
    """Return the one code cell of ``01_aquaplanet.ipynb`` that builds ``coupler``."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    matches = [
        cell.source for cell in nb.cells
        if cell.cell_type == "code" and "coupler = Coupler(" in cell.source
    ]
    assert len(matches) == 1, (
        f"{NOTEBOOK} has {len(matches)} code cells naming `coupler = Coupler(`; "
        "expected exactly one, the notebook's construction cell."
    )
    return matches[0]


def test_notebook_builds_its_components_the_way_python_api_does():
    """The notebook's clock, atmosphere, grid and components are python_api.md's."""
    expected = _expected_construction()
    actual = _statements(_notebook_construction_cell())

    # In order, but not necessarily adjacent: the notebook may put statements
    # of its own between them.
    remaining = iter(actual)
    missing = [statement for statement in expected
               if not any(statement == candidate for candidate in remaining)]
    assert not missing, (
        f"{NOTEBOOK}'s construction cell has drifted from {PYTHON_API}'s "
        "(plus the sea-ice and time-step substitutions). Missing, or out of "
        "order:\n  " + "\n  ".join(missing)
    )


def test_notebook_writes_its_exchange_by_hand():
    """The notebook's wiring is its own function, not the standard table."""
    cell = _notebook_construction_cell()
    tree = ast.parse(cell)
    assert any(isinstance(node, ast.FunctionDef) for node in tree.body), (
        f"{NOTEBOOK}'s construction cell defines no exchanger function."
    )
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "default_exchangers" not in called
