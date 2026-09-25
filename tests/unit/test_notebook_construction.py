"""Pin ``01_aquaplanet.ipynb``'s direct construction to ``python_api.md``.

Issue #131 turned the example notebooks into direct Python instead of a
Hydra composition, and ``01_basic/01_aquaplanet.ipynb`` builds its coupled
model exactly the way ``docs/source/python_api.md``'s own quick-start block
does (`tests/unit/test_readme_quickstart.py` already pins that block, in
turn, to the README's copy). This module is the third leg of that chain: it
pins the notebook's construction cell to the SAME source, so all three
(README, ``python_api.md``, the notebook) can never quietly drift apart.

Defining "construction"
------------------------
The notebook could not be byte-identical to ``python_api.md``'s block as a
whole: that block builds an atmosphere/ocean PAIR (the smallest coupled
model worth showing in the docs), while the notebook runs
``aquaplanet-slab``, which also couples the slab sea-ice component (see the
notebook's own markdown for why that is not optional here). So "the
notebook's construction" is defined precisely as:

1. Take ``python_api.md``'s first ```python``` fenced block, up to and
   including its ``print(repr(coupler))`` line -- i.e. everything that
   BUILDS the model and shows what was built, and none of the ``run_chunked``
   call after it (which the notebook legitimately calls with its own
   ``total_time``/``chunk``/``output_dir``/... -- run *settings*, not
   construction).
2. Apply exactly two substitutions to that prefix, both required by adding
   the sea-ice component ``python_api.md``'s pair-only example leaves out:
   the import line gains ``SlabSeaiceModel``, and the ``components`` dict
   gains a ``"seaice": SlabSeaiceModel(grid, name="seaice")`` entry (written
   as a multi-line dict literal purely for line length -- the mapping it
   builds, and the substituted import, are the only two changes).

The notebook's own code cell that builds ``coupler`` (identified as the one
whose source contains ``"coupler = Coupler("``) must equal the result of
that substitution byte for byte. If ``python_api.md``'s block changes (the
#878 clock migration's ``start_date`` -> ``start_time`` rename, for
instance), this test fails until the notebook's construction cell is
updated to match -- which is the whole point: the two are pinned into
staying in step rather than drifting the way a copy-pasted example would.
"""

from pathlib import Path

import nbformat

from tests.unit.test_readme_quickstart import PYTHON_API, _first_python_block

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "examples" / "01_basic" / "01_aquaplanet.ipynb"

#: The two substitutions that turn `python_api.md`'s atmosphere/ocean-only
#: prefix into what the notebook must build (atmosphere/ocean/sea-ice). Each
#: is applied exactly once; see the module docstring for why these two.
_OLD_IMPORT = "from jem.components import JCMComponent, SlabOceanModel\n"
_NEW_IMPORT = "from jem.components import JCMComponent, SlabOceanModel, SlabSeaiceModel\n"
_OLD_COMPONENTS = 'components = {"atm": atm, "ocn": SlabOceanModel(grid)}\n'
_NEW_COMPONENTS = (
    'components = {\n'
    '    "atm": atm,\n'
    '    "ocn": SlabOceanModel(grid),\n'
    '    "seaice": SlabSeaiceModel(grid, name="seaice"),\n'
    '}\n'
)


def _expected_construction() -> str:
    """Return the construction ``01_aquaplanet.ipynb`` must build, byte for byte."""
    block = _first_python_block(PYTHON_API)
    marker = "print(repr(coupler))\n"
    assert marker in block, (
        f"{PYTHON_API} no longer ends its construction with {marker!r}; "
        "update this test's split point to match its new shape."
    )
    prefix = block[:block.index(marker) + len(marker)]

    for old, new in ((_OLD_IMPORT, _NEW_IMPORT), (_OLD_COMPONENTS, _NEW_COMPONENTS)):
        count = prefix.count(old)
        assert count == 1, (
            f"{PYTHON_API} construction no longer contains {old!r} exactly "
            f"once (found {count}); its atmosphere/ocean example changed in "
            "a way this test's substitutions no longer apply to -- update "
            "both this test and the notebook."
        )
        prefix = prefix.replace(old, new)
    return prefix


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


def test_notebook_construction_matches_python_api_plus_seaice():
    """The notebook's build cell is python_api.md's prefix, plus sea ice."""
    expected = _expected_construction().rstrip("\n")
    actual = _notebook_construction_cell().rstrip("\n")
    if expected != actual:
        import difflib
        diff = "\n".join(difflib.unified_diff(
            expected.splitlines(), actual.splitlines(),
            fromfile=f"{PYTHON_API} (+ sea ice)", tofile=str(NOTEBOOK)))
        raise AssertionError(
            f"{NOTEBOOK}'s construction cell has drifted from "
            f"{PYTHON_API}'s (plus the sea-ice substitutions):\n{diff}"
        )
