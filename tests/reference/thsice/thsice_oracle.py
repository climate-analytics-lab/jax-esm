"""thsice_oracle.py

Python wrapper around the standalone Fortran "oracle" executable
(``thsice_oracle``, built by ``build.sh``) compiled from the
*unmodified* MITgcm pkg/thsice routines THSICE_SOLVE4TEMP
(thsice_solve4temp.F) and THSICE_CALC_THICKN (thsice_calc_thickn.F).
Each call in this module writes a "key = value" input file, runs the
executable once, and parses its "name = value" stdout into a dict of
Python floats -- see README.md for the exact build/compile command
and worked examples.

Two public functions, one per Fortran routine:

    solve4temp(**kwargs) -> dict
    calc_thickn(**kwargs) -> dict

Keyword names match the Fortran dummy-argument names (case-insensitive;
this module lower-cases everything before writing the input file).
Any of the THSICE_PARAMS.h physical/model parameters (rhoi, cpIce,
Lfresh, hIceMin, Terrmax, ... -- see README.md for the full list and
their task-specified defaults) may also be passed as extra keywords to
override the built-in defaults compiled into thsice_oracle.F.
Unrecognised keywords are written to the input file too but silently
ignored by the Fortran driver -- a misspelled key will not raise an
error, so double-check the returned dict if a run looks off.

Four helper functions convert between ice-layer temperature and the
"thsice" ice-enthalpy convention used by qIc1/qIc2 (Winton 1999
2-layer model, as coded at the top/bottom of thsice_solve4temp.F):

    qIc1 = -cpWater*Tmlt1 + cpIce*(Tmlt1 - tIc1) + Lfresh*(1 - Tmlt1/tIc1)
    qIc2 = -cpIce*tIc2 + Lfresh

    qic1_from_tic1(tic1, ...)  qic2_from_tic2(tic2, ...)
    tic1_from_qic1(qic1, ...)  tic2_from_qic2(qic2, ...)  (exact inverses;
        tic1_from_qic1 solves the same quadratic thsice_solve4temp.F
        itself solves to recover tIc1 from qIc1 at the top of the routine)

================================================================
THSICE_SOLVE4TEMP -- solve implicitly for the ice/snow surface and
ice-layer temperatures, given the surface energy balance as a function
of surface temperature (flxExSW), for one time step thSIce_dtTemp.
Called with useBulkForce=useEXF=.FALSE., i.e. surface fluxes always
come from the flxExSW argument (no bulk-formula / EXF coupling).

Inputs (kwargs):
    icmask   sea-ice fractional mask [0-1]                 (use 1.0)
    hice     ice height [m]
    hsnow    snow height [m]                                (0 is fine)
    tfrz     sea-water freezing temperature [oC] (function of salinity)
    flxexsw0 net (minus-SW) surface heat flux (+ = down) [W/m2],
             evaluated at the melting surface temperature (Ts = 0 oC)
    flxexsw1 net (minus-SW) surface heat flux (+ = down) [W/m2],
             evaluated at the current surface temperature tSrf
    flxexsw2 d(flxexsw1)/d(tSrf)                              [W/m2/K]
    flxsw    net Short-Wave flux (+ = down) absorbed at the surface,
             *before* the call                                 [W/m2]
    tsrf     surface (ice or snow) temperature                   [oC]
    qic1     top-layer ice enthalpy, thsice convention          [J/kg]
    qic2     bottom-layer ice enthalpy, thsice convention       [J/kg]
    mytime, myiter, mythid, bi, bj  -- bookkeeping, defaults are fine

Outputs (dict keys, all doubles):
    flxSW    net SW flux passed through the ice into the ocean  [W/m2]
    tSrf     updated surface temperature [oC] (== 0 if melting at surf.)
    qIc1, qIc2  updated ice-layer enthalpies                    [J/kg]
    tIc1, tIc2  ice-layer temperatures                            [oC]
    dTsrf    surface-temperature increment applied this call      [oC]
    sHeat    surface heat flux left over to melt snow/ice        [W/m2]
             (= net atmos. flux - conduction into the ice)
    flxCnB   heat flux conducted through the ice to its base     [W/m2]
    flxAtm   net atmosphere->surface energy flux (+ = down), excludes
             the energy of snow precipitation                   [W/m2]
    evpAtm   evaporation to the atmosphere (> 0 if evaporating) [kg/m2/s]

================================================================
THSICE_CALC_THICKN -- update ice/snow thickness, fraction and
enthalpy for one time step thSIce_deltaT, given the surface/base
energy fluxes (sHeat, flxCnB -- typically the output of solve4temp)
and the ocean mixed-layer state.

Inputs (kwargs):
    icemask  sea-ice fractional mask [0-1]
    tfrz     sea-water freezing temperature                      [oC]
    toce     ocean surface-level (mixed-layer) temperature        [oC]
    v2oc     square of ocean surface-level velocity  [m2/s2] (0: slab ocean)
    snowp    snow precipitation                             [kg/m2/s]
    prcatm   total precipitation from the atmosphere        [kg/m2/s]
    sheat    surf. heating flux left to melt snow/ice (Atmos-conduction),
             typically solve4temp's output "sHeat"              [W/m2]
    flxcnb   heat flux conducted through the ice to its base,
             typically solve4temp's output "flxCnB"              [W/m2]
    icfrac   fraction of grid area covered in ice
    hice     ice height                                            [m]
    hsnow1   snow height                                            [m]
    tsrf     surface temperature                                  [oC]
    qic1, qic2  ice-layer enthalpies                             [J/kg]
    frwatm   evaporation to the atmosphere (> 0 if evaporating) [kg/m2/s]
    fzmloc   ocean mixed-layer freezing/melting potential        [W/m2];
             = (tFrzOce - tOceMxL)*cpWater*rhosw*hOceMxL/dt
             (thsice_step_fwd.F convention; > 0 = freezing potential)
    flx2oc   net heat flux to ocean already accumulated upstream
             (+ = down)                                         [W/m2]

Outputs (dict keys, all doubles):
    icFrac, hIce, hSnow1, tSrf, qIc1, qIc2  -- updated sea-ice state
    frwAtm   updated evap-to-atmosphere diagnostic (Evap - precip)
                                                              [kg/m2/s]
    fzMlOc   updated freezing potential (part used this step
             subtracted off)                                    [W/m2]
    flx2oc   updated net heat flux to ocean (+ = down)           [W/m2]
    frw2oc   total fresh-water flux to ocean (+ = down)       [kg/m2/s]
    fsalt    salt flux to ocean (+ = down)                     [g/m2/s]
    frzSeaWat  seawater freezing rate, as a mass flux           [kg/m2/s]

================================================================
Parameters: see README.md for the full THSICE_PARAMS.h parameter list
and their task-specified defaults (all overridable, e.g. rhoi=900.0).
The convenience key "dt" sets thSIce_dtTemp / thSIce_deltaT /
ocean_deltaT all at once; each may still be overridden individually.
"""

import os
import subprocess
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXE = os.path.join(_HERE, "thsice_oracle")


def _run(routine, kwargs):
    lines = ["routine = %s" % routine]
    for key, val in kwargs.items():
        if isinstance(val, bool):
            val = "T" if val else "F"
        lines.append("%s = %s" % (key, val))
    input_text = "\n".join(lines) + "\n"

    with tempfile.TemporaryDirectory() as d:
        infile = os.path.join(d, "input.txt")
        with open(infile, "w") as f:
            f.write(input_text)
        proc = subprocess.run(
            [_EXE, infile], capture_output=True, text=True
        )

    if proc.returncode != 0:
        raise RuntimeError(
            "thsice_oracle failed (routine=%s, exit=%d)\n"
            "--- input file ---\n%s"
            "--- stdout ---\n%s\n"
            "--- stderr ---\n%s"
            % (routine, proc.returncode, input_text, proc.stdout, proc.stderr)
        )

    out = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        name, val = line.split("=", 1)
        name = name.strip()
        try:
            out[name] = float(val.strip())
        except ValueError:
            # Fortran prints |exponent| >= 100 without the 'E' (e.g. 1.0+180); such values
            # only arise from outputs the routine did not set. Keep the key, mark it unusable.
            out[name] = float("nan")
    return out


def solve4temp(**kwargs):
    """Run THSICE_SOLVE4TEMP once on a single grid column.

    See the module docstring for the full list of input keywords,
    output keys, units and sign conventions.
    """
    kw = {k.lower(): v for k, v in kwargs.items()}
    return _run("solve4temp", kw)


def calc_thickn(**kwargs):
    """Run THSICE_CALC_THICKN once on a single grid column.

    See the module docstring for the full list of input keywords,
    output keys, units and sign conventions.
    """
    kw = {k.lower(): v for k, v in kwargs.items()}
    return _run("calc_thickn", kw)


# ---------------------------------------------------------------------------
# thsice enthalpy <-> temperature conversions (Winton 2-layer convention,
# see the end / start of thsice_solve4temp.F).  All default parameter
# values match the task-specified THSICE_PARAMS.h defaults; pass the
# same overrides given to solve4temp/calc_thickn if a run used
# non-default cpIce/cpWater/Lfresh/mu_Tf/S_winton/Tmlt1.
# ---------------------------------------------------------------------------

def tmlt1(mu_tf=0.054, s_winton=1.0):
    """Winton ice-melting temperature Tmlt1 = -mu_Tf*S_winton [oC]."""
    return -mu_tf * s_winton


def qic1_from_tic1(tic1, cpwater=3990.0, cpice=2100.0, lfresh=3.34e5,
                    mu_tf=0.054, s_winton=1.0, tmlt1_val=None):
    """Top-layer ice enthalpy [J/kg] from top-layer temperature [oC]."""
    t1 = tmlt1_val if tmlt1_val is not None else tmlt1(mu_tf, s_winton)
    return -cpwater * t1 + cpice * (t1 - tic1) + lfresh * (1.0 - t1 / tic1)


def qic2_from_tic2(tic2, cpice=2100.0, lfresh=3.34e5):
    """Bottom-layer ice enthalpy [J/kg] from bottom-layer temperature [oC]."""
    return -cpice * tic2 + lfresh


def tic1_from_qic1(qic1, cpwater=3990.0, cpice=2100.0, lfresh=3.34e5,
                    mu_tf=0.054, s_winton=1.0, tmlt1_val=None):
    """Top-layer temperature [oC] from top-layer enthalpy [J/kg].

    Exact inverse of qic1_from_tic1 -- the same quadratic that
    thsice_solve4temp.F itself solves (lines ~287-290) to recover
    tIc1 from qIc1 at the start of the routine.
    """
    t1 = tmlt1_val if tmlt1_val is not None else tmlt1(mu_tf, s_winton)
    a1 = cpice
    b1 = qic1 + (cpwater - cpice) * t1 - lfresh
    c1 = lfresh * t1
    return 0.5 * (-b1 - (b1 * b1 - 4.0 * a1 * c1) ** 0.5) / a1


def tic2_from_qic2(qic2, cpice=2100.0, lfresh=3.34e5):
    """Bottom-layer temperature [oC] from bottom-layer enthalpy [J/kg]."""
    return (lfresh - qic2) / cpice
