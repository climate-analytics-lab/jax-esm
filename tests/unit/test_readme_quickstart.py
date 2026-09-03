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
    # One coupling step is enough to prove the block runs end to end; the
    # README itself keeps the length a reader would actually want.
    code = re.sub(r"jdt\.to_timedelta\(\d+, \"day\"\)\s*#\s*simulation", 'jdt.to_timedelta(1, "day")  # simulation', code)
    code = code.replace('simulation_interval = jdt.to_timedelta(10, "day")', 'simulation_interval = jdt.to_timedelta(1, "day")')
    assert 'jdt.to_timedelta(1, "day")' in code, "could not shorten the README run to one step"
    monkeypatch.chdir(tmp_path)
    exec(compile(code, str(README), "exec"), {"__name__": "__main__"})
