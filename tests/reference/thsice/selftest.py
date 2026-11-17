#!/usr/bin/env python
"""Self-test for the thsice standalone oracle (thsice_oracle.py).

Runs THSICE_SOLVE4TEMP under a cold (no melting) and a strongly-warm
(surface-melting) forcing, and THSICE_CALC_THICKN with the resulting
fluxes, asserting every output is finite and that the surface-melting
clamp actually triggers under warm forcing. Not a framework: plain
asserts, run this file directly.
"""

import math
import sys

from thsice_oracle import (
    solve4temp, calc_thickn,
    qic1_from_tic1, qic2_from_tic2, tic1_from_qic1, tic2_from_qic2,
)


def all_finite(d):
    return all(math.isfinite(v) for v in d.values())


def show(label, d):
    print("--- %s ---" % label)
    for k in sorted(d):
        print("  %-12s = %.10g" % (k, d[k]))


def main():
    dt = 21600.0
    hIce, hSnow = 1.5, 0.0
    tSrf0, tIc1_0, tIc2_0 = -15.0, -8.0, -4.0

    # round-trip check on the enthalpy <-> temperature helpers
    qIc1 = qic1_from_tic1(tIc1_0)
    qIc2 = qic2_from_tic2(tIc2_0)
    assert abs(tic1_from_qic1(qIc1) - tIc1_0) < 1e-9, "tIc1 round-trip"
    assert abs(tic2_from_qic2(qIc2) - tIc2_0) < 1e-9, "tIc2 round-trip"

    # -- cold case: net loss to atmosphere, no surface melting --------
    cold = solve4temp(
        dt=dt, icmask=1.0, hice=hIce, hsnow=hSnow, tfrz=-1.8,
        tsrf=tSrf0, qic1=qIc1, qic2=qIc2, flxsw=0.0,
        flxexsw0=-30.0, flxexsw1=-30.0, flxexsw2=-5.0,
    )
    show("solve4temp: cold case", cold)
    assert all_finite(cold), "cold case produced non-finite output"
    assert cold["tSrf"] < 0.0, "cold case should not reach the melting point"
    assert abs(cold["dTsrf"]) < 500.0, "cold case should not hit the clamp sentinel"

    # -- warm case: strong net surface heating -> surface melting -----
    warm = solve4temp(
        dt=dt, icmask=1.0, hice=hIce, hsnow=hSnow, tfrz=-1.8,
        tsrf=tSrf0, qic1=qIc1, qic2=qIc2, flxsw=0.0,
        flxexsw0=300.0, flxexsw1=300.0, flxexsw2=-5.0,
    )
    show("solve4temp: warm (melting) case", warm)
    assert all_finite(warm), "warm case produced non-finite output"
    assert warm["tSrf"] == 0.0, "warm case should clamp tSrf to 0 degC (melting)"
    # in the non-bulk-force path (useEXF=useBulkForce=.FALSE., as used by
    # this oracle) THSICE_SOLVE4TEMP signals the melting clamp with the
    # sentinel dTsrf = 1000 (see thsice_solve4temp.F, ~line 491)
    assert warm["dTsrf"] == 1000.0, "warm case should set the dTsrf=1000 clamp sentinel"

    # -- feed the cold case's surface fluxes into calc_thickn ---------
    thick = calc_thickn(
        dt=dt, icemask=1.0, tfrz=-1.8, toce=-1.8, v2oc=0.0,
        snowp=0.0, prcatm=0.0,
        sheat=cold["sHeat"], flxcnb=cold["flxCnB"],
        icfrac=1.0, hice=hIce, hsnow1=hSnow,
        tsrf=cold["tSrf"], qic1=cold["qIc1"], qic2=cold["qIc2"],
        frwatm=0.0, fzmloc=0.0, flx2oc=0.0,
    )
    show("calc_thickn: fed from cold solve4temp", thick)
    assert all_finite(thick), "calc_thickn produced non-finite output"
    assert thick["hIce"] > 0.0, "1.5m of ice should not vanish in one 6h step"

    # -- calc_thickn under strong ocean freezing potential -------------
    freeze = calc_thickn(
        dt=dt, icemask=1.0, tfrz=-1.8, toce=-1.8, v2oc=0.0,
        snowp=0.0, prcatm=0.0, sheat=0.0, flxcnb=0.0,
        icfrac=1.0, hice=hIce, hsnow1=hSnow,
        tsrf=-10.0, qic1=qIc1, qic2=qIc2,
        frwatm=0.0, fzmloc=500.0, flx2oc=0.0,
    )
    show("calc_thickn: strong freezing potential", freeze)
    assert all_finite(freeze), "freezing case produced non-finite output"
    assert freeze["hIce"] >= hIce, "freezing potential should grow (or hold) ice"

    print("\nALL SELFTESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("SELFTEST FAILED: %s" % e, file=sys.stderr)
        sys.exit(1)
