"""Execute the README quick start so it can never silently go stale again.

The block is extracted verbatim from ``README.md`` (its first ``python`` fence)
and run with the simulation shortened to one coupling step; only the code that
users copy is tested, not a parallel copy of it.
"""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"


@pytest.mark.slow
def test_readme_quickstart_runs(tmp_path, monkeypatch):
    """Run the first ``python`` block of the README in a scratch directory."""
    match = re.search(r"```python\n(.*?)```", README.read_text(), re.S)
    assert match, "README.md has no ```python fence"
    code = match.group(1)
    # One coupling step in one chunk is enough to prove the block runs end to
    # end; the README itself keeps the length a reader would actually want.
    shortened = code.replace(
        'total_time="10 days", chunk="5 days"', 'total_time="1 day", chunk="1 day"'
    )
    assert shortened != code, (
        "could not shorten the README run: the quick start no longer calls "
        "run_chunked with the expected durations"
    )
    monkeypatch.chdir(tmp_path)
    exec(compile(shortened, str(README), "exec"), {"__name__": "__main__"})

    # The block writes what it says it writes, into the directory it names:
    # one file per component, named after the coupled step its chunk starts
    # at -- which for a shortened one-chunk run is step 0.
    assert sorted(p.name for p in (tmp_path / "output").glob("*.nc")) == [
        "atm-00000000.nc", "ocn-00000000.nc",
    ]
