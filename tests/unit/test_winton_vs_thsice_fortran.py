"""Machine-precision comparison against the compiled MITgcm thsice oracle.

Skips if the standalone Fortran executable has not been built
(tests/reference/thsice/build.sh), unless ``JEM_REQUIRE_THSICE_ORACLE`` is
set, as CI sets it after building the oracle: then a missing executable is a
failure, so the comparison cannot pass by not running.
"""

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXE = REPO / "tests/reference/thsice/thsice_oracle"
sys.path.insert(0, str(REPO / "tests/reference/thsice"))


def test_matches_thsice_fortran_to_roundoff():
    if not EXE.exists():
        if os.environ.get("JEM_REQUIRE_THSICE_ORACLE"):
            pytest.fail(f"JEM_REQUIRE_THSICE_ORACLE is set but {EXE} was not built.")
        pytest.skip("thsice oracle not built (tests/reference/thsice/build.sh)")
    import thsice_compare as TC

    rows = TC.run_all()
    worst = max(rows, key=lambda r: abs(r[3] - r[4]) / max(abs(r[4]), 1.0))
    assert TC.max_relative_error(rows) < 1e-12, worst


if __name__ == "__main__":
    test_matches_thsice_fortran_to_roundoff()
    print("ok test_matches_thsice_fortran_to_roundoff")
