"""Pin the README quick start to ``docs/source/python_api.md`` and run it.

The quick start lives in ``docs/source/python_api.md`` -- it is the complete,
executed direct-Python construction -- and ``README.md`` carries a copy of the
same block for a reader who never opens the docs. This module executes the
page's block (so it can never silently go stale) and asserts the two copies
are byte-identical (so they cannot drift apart without a test failing).
"""

import difflib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
PYTHON_API = REPO_ROOT / "docs" / "source" / "python_api.md"

_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.S)


def _first_python_block(path: Path) -> str:
    """Return the first ```python fenced block in ``path``, or raise."""
    match = _PYTHON_FENCE.search(path.read_text())
    assert match, f"{path} has no ```python fence"
    return match.group(1)


def test_python_api_block_is_the_readme_quick_start():
    """The README's Quick Start must be exactly ``python_api.md``'s copy.

    ``docs/source/python_api.md`` is the source; ``README.md`` copies it for a
    reader who never opens the docs. If this fails, update the README's block
    to match ``python_api.md``, not the other way around.
    """
    python_api_block = _first_python_block(PYTHON_API)
    readme_block = _first_python_block(README)
    if python_api_block != readme_block:
        diff = "\n".join(
            difflib.unified_diff(
                python_api_block.splitlines(),
                readme_block.splitlines(),
                fromfile=str(PYTHON_API),
                tofile=str(README),
            )
        )
        raise AssertionError(
            f"{PYTHON_API} (the source) and {README} (the copy) have drifted "
            f"apart; bring {README} back in line with {PYTHON_API}:\n{diff}"
        )


def test_configurations_door_example_compiles_and_names_resolve():
    """``python_api.md``'s door example compiles and every name it uses resolves.

    The ``## Validated configurations from Python`` section's fenced block
    builds a real coupler and would integrate `earth-slab`'s default 30-day
    run -- too expensive for this fast test -- so
    ``jem.configurations.load`` and ``jem.run_chunked`` are patched to
    lightweight stand-ins carrying the same attributes the real objects have
    (``.coupler``/``.config``/``.run_kwargs``), and the block is genuinely
    executed (not merely ``compile()``-d, which only checks syntax) against
    them. This is exactly the gap an earlier draft of the block fell into: it
    called ``run_chunked`` without importing it, a ``NameError`` that
    ``compile()`` alone would not have caught either.
    """
    from types import SimpleNamespace
    from unittest import mock

    import jem
    from jem import configurations as configurations_module

    blocks = _PYTHON_FENCE.findall(PYTHON_API.read_text())
    matches = [b for b in blocks if "configurations.load" in b]
    assert len(matches) == 1, (
        f"expected exactly one door example (naming `configurations.load`) in "
        f"{PYTHON_API}, found {len(matches)}"
    )
    code = matches[0]

    fake_loaded = SimpleNamespace(coupler="<coupler>", config={"ocean": {}}, run_kwargs={})
    run_chunked_calls = []
    with mock.patch.object(configurations_module, "load",
                           return_value=fake_loaded) as fake_load, \
         mock.patch.object(jem, "run_chunked",
                           side_effect=lambda *a, **k: run_chunked_calls.append((a, k))):
        exec(compile(code, str(PYTHON_API), "exec"), {"__name__": "__main__"})

    fake_load.assert_called_once_with("aquaplanet-slab")
    assert run_chunked_calls, (
        f"{PYTHON_API}'s door example never called run_chunked(...)"
    )


@pytest.mark.slow
def test_readme_quickstart_runs(tmp_path, monkeypatch):
    """Run the first ``python`` block of ``python_api.md`` in a scratch directory."""
    code = _first_python_block(PYTHON_API)
    # One coupling step in one chunk is enough to prove the block runs end to
    # end; the page itself keeps the length a reader would actually want.
    shortened = code.replace(
        'total_time="10 days", chunk="5 days"', 'total_time="1 day", chunk="1 day"'
    )
    assert shortened != code, (
        "could not shorten the quick start: the block no longer calls "
        "run_chunked with the expected durations"
    )
    monkeypatch.chdir(tmp_path)
    exec(compile(shortened, str(PYTHON_API), "exec"), {"__name__": "__main__"})

    # The block writes what it says it writes, into the directory it names:
    # one file per component, named after the coupled step its chunk starts
    # at -- which for a shortened one-chunk run is step 0.
    assert sorted(p.name for p in (tmp_path / "output").glob("*.nc")) == [
        "atm-00000000.nc", "ocn-00000000.nc",
    ]
