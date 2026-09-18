"""The JCM adapter's clock: ``carry["date"]`` is exact and ``state.sim_time`` never grows.

``primitive_equations.State.sim_time`` is float32 and the integrator adds ``dt`` to it
every step. Carried across coupling calls it rounds each 1800-s add to +1792 s from
2**27 s (4.25 yr) and to +2048 s from 2**32 s (136 yr): a 366.87-day and then a
321-day year in the insolation. ``make_jem_compatible`` therefore re-zeroes ``sim_time``
every call and carries the calendar date as exact int32 days/seconds, passing it to
``Model.run_from_state(start_date=...)`` as the (traced) date origin.

Uses a stand-in model so the test needs no atmosphere physics; the real
``jcm.model.Model.run_from_state`` signature is what the stub mimics.
"""
from __future__ import annotations

import dataclasses
import datetime

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest

from jem.components.JCM import carry_to_date, date_to_carry, make_jem_compatible

PENTAD_S = 5 * 86400
DT_S = 1800.0


def _dc(cls):
    return jax.tree_util.register_dataclass(dataclasses.dataclass(cls))


@_dc
class _Flux:
    hfluxn: jax.Array
    evap: jax.Array


@_dc
class _Conv:
    precnv: jax.Array


@_dc
class _Cond:
    precls: jax.Array


@_dc
class _Physics:
    surface_flux: _Flux
    convection: _Conv
    condensation: _Cond
    origin_days: jax.Array
    sim_time_in: jax.Array


@_dc
class _Predictions:
    physics: _Physics


@_dc
class _State:
    vorticity: jax.Array
    sim_time: jax.Array

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


class _FakeModel:
    def __init__(self, start_iso: str):
        from dinosaur.scales import units
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        self.start_date = jdt.to_datetime(start_iso)
        self.dt_si = DT_S * units.second
        self.coords = get_speedy_coords(spectral_truncation=31)

    def _prepare_initial_modal_state(self):
        return _State(vorticity=jnp.zeros((3,)), sim_time=jnp.float32(0.0))

    def run_from_state(self, initial_state, forcing, save_interval, total_time,
                       output_averages, start_date=None):
        del forcing, save_interval, output_averages
        t = initial_state.sim_time
        for _ in range(int(round(total_time * 86400 / DT_S))):  # float32, like SIL3
            t = (t + jnp.float32(DT_S)).astype(jnp.float32)
        origin = self.start_date if start_date is None else start_date
        _, nlon, nlat = self.coords.nodal_shape
        z = jnp.zeros((1, nlon, nlat, 3))
        phys = _Physics(
            surface_flux=_Flux(hfluxn=z, evap=z),
            convection=_Conv(precnv=z[..., 0]),
            condensation=_Cond(precls=z[..., 0]),
            origin_days=jnp.asarray(origin.delta.days)[None],
            sim_time_in=jnp.asarray(initial_state.sim_time)[None],
        )
        return initial_state.replace(sim_time=t), _Predictions(physics=phys)


def _adapted(start_iso="2015-01-18"):
    model = _FakeModel(start_iso)
    make_jem_compatible(model, coupling_timestep=jdt.to_timedelta(PENTAD_S, "second"))
    return model


def test_date_carry_round_trip_is_int32_and_exact():
    d = jdt.to_datetime("2151-01-18")
    leaves = date_to_carry(d)
    assert leaves["days"].dtype == jnp.int32 and leaves["seconds"].dtype == jnp.int32
    back = carry_to_date(leaves)
    assert int(back.delta.days) == int(d.delta.days) and int(back.delta.seconds) == 0


def test_initialize_carries_the_start_date_and_zero_sim_time():
    carry = _adapted().initialize()
    assert set(carry) == {"state", "derived", "forcing", "date"}
    assert int(carry["date"]["days"]) == (datetime.date(2015, 1, 18) - datetime.date(1970, 1, 1)).days
    assert int(carry["date"]["seconds"]) == 0
    assert float(carry["state"].sim_time) == 0.0


def test_date_is_exact_past_2p32_seconds_and_sim_time_is_rezeroed():
    model = _adapted()
    step_fn = model.generate_step_function()
    n = 10_000  # 137 yr of atmosphere time: past the float32 counter's 2**32-s cliff

    def body(carry, k):
        new_carry, preds = step_fn(carry, k)
        return new_carry, (new_carry["date"]["days"], new_carry["date"]["seconds"],
                           new_carry["state"].sim_time,
                           preds.physics.origin_days[0], preds.physics.sim_time_in[0])

    carry0 = model.initialize()
    carry0["state"] = carry0["state"].replace(sim_time=jnp.float32(2.0**32))  # poisoned
    _, (days, secs, sim_t, o_days, sim_in) = jax.jit(
        lambda c: jax.lax.scan(body, c, jnp.arange(n)))(carry0)
    start_days = (datetime.date(2015, 1, 18) - datetime.date(1970, 1, 1)).days
    expected = start_days + 5 * (np.arange(n) + 1)
    np.testing.assert_array_equal(np.asarray(days), expected)
    np.testing.assert_array_equal(np.asarray(secs), 0)
    np.testing.assert_array_equal(np.asarray(o_days), expected - 5)
    np.testing.assert_array_equal(np.asarray(sim_in), 0.0)
    np.testing.assert_array_equal(np.asarray(sim_t), 0.0)
    assert 5 * n * 86400 > 2**32


def test_step_function_refuses_a_carry_without_a_date():
    model = _adapted()
    carry = model.initialize()
    del carry["date"]
    with pytest.raises(KeyError, match="date"):
        model.generate_step_function()(carry, 0)
