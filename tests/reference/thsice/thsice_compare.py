"""Compare the JAX Winton/thsice implementation with the compiled MITgcm thsice oracle.

Both codes are driven with identical inputs; thsice's surface flux enters as
(F at Ts=0, F at current Ts, dF/dTs) and we use a linear flux so a single
linearised solve (n_iter=1) is the same problem in both.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thsice_oracle as O  # noqa: E402
from jem.components.slab.winton_seaice_model import winton_seaice_model as W  # noqa: E402

PARAMS = dict(rhoi=W.RHO_ICE, rhos=W.RHO_SNOW, rhosw=W.RHO_SW, cpice=W.C_ICE, cpwater=W.C_W, lfresh=W.L_ICE,
              kice=W.K_ICE, ksnow=W.K_SNOW, mu_tf=W.MU, s_winton=W.S_ICE, bmeltcoef=W.B_MELT,
              i0swfrac=0.3, ksolar=1.5, hicemin=0.01, himax=1e6, hsmax=1e6, fracenmelt=0.0, fracenfreez=0.0)


def thsice_fbot(toce, fzmloc, tfrz=W.T_FREEZE):
    """thsice basal flux (positive = ice loses heat to ocean), melting branch with soft max (kScal = 1)."""
    if fzmloc >= 0.0:
        return fzmloc
    cpchr = W.C_W * W.RHO_SW * W.B_MELT
    fbot = cpchr * (tfrz - toce) * 5.0e-3
    k = 1.0
    fbot = (fbot * np.exp(k * fbot) + fzmloc * np.exp(k * fzmloc)) / (np.exp(k * fbot) + np.exp(k * fzmloc))
    return min(max(fbot, fzmloc), 0.0)


def compare_temperature(h, hs, T1, T2, Ts, F0, dF, sw, dt):
    """Linear flux F(T) = F0 + dF (T - Ts). Returns (jax, thsice) dicts."""
    fj = lambda T: (F0 + dF * (T - Ts), dF * jnp.ones_like(T))
    T1n, T2n, Tsn, M_s, F_cb, sw_ocn = W.winton_temperature_step(
        jnp.array(h), jnp.array(hs), jnp.array(T1), jnp.array(T2), jnp.array(Ts), fj, jnp.array(sw), dt, n_iter=1)
    j = dict(tIc1=float(T1n), tIc2=float(T2n), tSrf=float(Tsn), sHeat=float(M_s), flxCnB=float(F_cb), flxSW=float(sw_ocn))
    o = O.solve4temp(icmask=1.0, hice=h, hsnow=hs, tfrz=W.T_FREEZE, flxexsw0=F0 + dF * (0.0 - Ts), flxexsw1=F0, flxexsw2=dF,
                     flxsw=sw, tsrf=Ts, qic1=O.qic1_from_tic1(T1), qic2=O.qic2_from_tic2(T2), dt=dt, **PARAMS)
    o = {k: o[k] for k in j}
    return j, o


def compare_mass(h, hs, T1, T2, M_s, F_cb, toce, fzmloc, snowp, dt):
    fbot = thsice_fbot(toce, fzmloc)
    hn, hsn, q1n, q2n, e_ocn, vanished = W.winton_mass_step(
        jnp.array(h), jnp.array(hs), W.q_from_T1(jnp.array(T1)), W.q_from_T2(jnp.array(T2)),
        jnp.array(M_s), jnp.array(-fbot), jnp.array(F_cb), jnp.array(snowp), dt)
    j = dict(hIce=float(hn), hSnow1=float(hsn), qIc1=float(q1n), qIc2=float(q2n), flx2oc=float(e_ocn) / dt + fbot)
    o = O.calc_thickn(icemask=1.0, tfrz=W.T_FREEZE, toce=toce, v2oc=0.0, snowp=snowp, prcatm=snowp, sheat=M_s, flxcnb=F_cb,
                      icfrac=1.0, hice=h, hsnow1=hs, tsrf=0.0 if M_s > 0 else -5.0, qic1=O.qic1_from_tic1(T1),
                      qic2=O.qic2_from_tic2(T2), frwatm=0.0, fzmloc=fzmloc, flx2oc=0.0, dt=dt, **PARAMS)
    o = {k: o[k] for k in j}
    return j, o


TEMPERATURE_CASES = {
    # name: (h, hs, T1, T2, Ts, F0, dF, sw, dt)
    "cold, bare ice": (1.5, 0.0, -8.0, -4.0, -15.0, -60.0, -20.0, 0.0, 21600.0),
    "cold, 20 cm snow": (1.5, 0.2, -8.0, -4.0, -25.0, -40.0, -20.0, 0.0, 21600.0),
    "mild, bare ice, some SW": (1.0, 0.0, -5.0, -3.0, -6.0, 10.0, -25.0, 80.0, 21600.0),
    "warm, melting clamp, bare ice": (1.5, 0.0, -3.0, -2.5, -2.0, 90.0, -20.0, 250.0, 21600.0),
    "warm, melting clamp, snow": (2.0, 0.3, -2.0, -2.0, -1.0, 120.0, -20.0, 300.0, 21600.0),
    "thin ice, 3 h step": (0.15, 0.0, -4.0, -2.5, -10.0, -30.0, -20.0, 0.0, 10800.0),
}
MASS_CASES = {
    # name: (h, hs, T1, T2, M_s, F_cb, toce, fzmloc, snowp, dt)
    "basal growth": (1.5, 0.0, -8.0, -4.0, 0.0, -45.0, -1.8, 0.0, 0.0, 21600.0),
    "surface melt, bare ice": (1.5, 0.0, -1.0, -1.9, 150.0, 5.0, -1.8, 0.0, 0.0, 21600.0),
    "surface melt through snow into ice": (1.0, 0.05, -1.0, -1.9, 400.0, 5.0, -1.8, 0.0, 0.0, 21600.0),
    "basal melt from warm ocean": (1.5, 0.0, -1.5, -1.85, 0.0, 2.0, -0.8, -300.0, 0.0, 21600.0),
    "basal melt, ocean potential binding": (1.5, 0.0, -1.5, -1.85, 0.0, 2.0, 0.5, -40.0, 0.0, 21600.0),
    "snowfall then flooding": (0.5, 0.15, -6.0, -3.0, 0.0, -30.0, -1.8, 0.0, 5e-5, 21600.0),
    "growth with lower layer thicker (re-equalise)": (0.8, 0.0, -6.0, -3.0, 0.0, -200.0, -1.8, 0.0, 0.0, 86400.0),
}


def run_all():
    rows = []
    for name, args in TEMPERATURE_CASES.items():
        j, o = compare_temperature(*args)
        for k in j:
            rows.append(("temperature", name, k, j[k], o[k]))
    for name, args in MASS_CASES.items():
        j, o = compare_mass(*args)
        for k in j:
            rows.append(("mass", name, k, j[k], o[k]))
    return rows


def max_relative_error(rows, scale=1.0):
    return max(abs(a - b) / max(abs(b), scale) for _, _, _, a, b in rows)


if __name__ == "__main__":
    rows = run_all()
    print(f"{'step':12s} {'case':46s} {'variable':8s} {'JAX':>24s} {'thsice':>24s} {'rel. diff':>10s}")
    for step, name, k, a, b in rows:
        print(f"{step:12s} {name:46s} {k:8s} {a:24.16e} {b:24.16e} {abs(a-b)/max(abs(b),1.0):10.1e}")
    print("max relative difference (floor 1):", max_relative_error(rows))
