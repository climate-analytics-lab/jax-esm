# thsice reference oracle

A single-column Fortran executable built from the **unmodified** MITgcm `pkg/thsice` routines
`THSICE_SOLVE4TEMP` and `THSICE_CALC_THICKN`, so that `jem.components.slab.winton_seaice_model` can be checked
against the original implementation at machine precision (`tests/unit/test_winton_vs_thsice_fortran.py`,
skipped when the executable is absent).

MITgcm sources are **not** stored here: `build.sh` fetches the four files it needs
(`thsice_solve4temp.F`, `thsice_calc_thickn.F`, `THSICE_PARAMS.h`, `THSICE_SIZE.h`) from
https://github.com/MITgcm/MITgcm at the pinned commit in the script into `upstream/` (git-ignored).
MITgcm is distributed under the MIT License, Copyright (c) 2018 MITgcm Developers and Contributors.

Files in this directory are ours: `thsice_oracle.F` (driver, reads a `key = value` file, prints every output in
full double precision), `thsice_stubs.F` (stand-ins for routines the driver never reaches), `EEPARAMS.h`,
`SIZE.h`, `THSICE_OPTIONS.h` (minimal stub headers: single column, `_RL/_RS = Real*8`, `ALLOW_DBUG_THSICE`
undefined, `THSICE_FRACEN_POWERLAW` defined as upstream), `thsice_oracle.py` (Python wrapper: `solve4temp()`,
`calc_thickn()`, enthalpy conversions), `selftest.py`, and `thsice_compare.py` (the JAX-vs-Fortran comparison
the unit test runs).

```sh
FC=gfortran ./build.sh          # needs gfortran and network access for the first build
python selftest.py              # oracle sanity checks
python -m pytest tests/unit/test_winton_vs_thsice_fortran.py   # from the repository root
```

Notes: the oracle exposes one Newton step of `THSICE_SOLVE4TEMP` per call (`useBulkForce = useEXF = .FALSE.`);
when a step would take the surface above 0 C the routine clamps `tSrf = 0` and returns `dTsrf = 1000` as its
own sentinel. Enthalpies follow thsice: `qIc1 = -cpWater Tmlt1 + cpIce (Tmlt1 - tIc1) + Lfresh (1 - Tmlt1/tIc1)`,
`qIc2 = -cpIce tIc2 + Lfresh`. Parameter defaults compiled into the driver are those of the JAX component
(`rhoi = 905`, `cpIce = 2100`, `kSnow = 0.31`, ...) and every one can be overridden per call.
