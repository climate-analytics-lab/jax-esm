"""Tests for run provenance capture (:mod:`jem.provenance`)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf

from jem import Coupler, provenance, run_chunked
from jem.components.slab import (
    SlabOceanModel,
    SlabSeaiceModel,
)
from jem.exchangers import default_exchangers
from jem.output import write_chunk
from tests.unit.slab_test_utils import make_grid


def _two_slabs() -> Coupler:
    """Build a minimal ocean + seaice coupler with no atmosphere or external data."""
    grid = make_grid()
    components = {
        "ocn": SlabOceanModel(grid),
        "seaice": SlabSeaiceModel(grid),
    }
    return Coupler(
        components,
        default_exchangers(components),
        coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=jdt.to_datetime("2000-01-01"),
    )


def test_probe_code_includes_jem():
    code = provenance.probe_code()
    assert "jem" in code
    assert "path" in code["jem"]
    assert "jax" in code and "version" in code["jax"]


def test_probe_environment_fields():
    env = provenance.probe_environment()
    for key in (
        "python",
        "hostname",
        "user",
        "jax_enable_x64",
        "platform",
        "device_kind",
        "device_count",
        "env",
    ):
        assert key in env


def test_describe_input_file_and_hashing(tmp_path, monkeypatch):
    f = tmp_path / "forcing.nc"
    f.write_bytes(b"hello provenance")

    monkeypatch.delenv("JEM_HASH_INPUTS", raising=False)
    monkeypatch.delenv("JCM_HASH_INPUTS", raising=False)
    d = provenance.describe_input(f)
    assert d["size"] == len(b"hello provenance")
    assert "mtime" in d
    assert "sha256" not in d

    expected = hashlib.sha256(b"hello provenance").hexdigest()
    assert provenance.describe_input(f, force_hash=True)["sha256"] == expected

    monkeypatch.setenv("JEM_HASH_INPUTS", "1")
    assert provenance.describe_input(f)["sha256"] == expected

    monkeypatch.delenv("JEM_HASH_INPUTS")
    monkeypatch.setenv("JCM_HASH_INPUTS", "1")
    assert provenance.describe_input(f)["sha256"] == expected


def test_describe_input_missing_file(tmp_path):
    d = provenance.describe_input(tmp_path / "does_not_exist.nc")
    assert d == {"missing": True}


def test_describe_input_checkpoint_directory(tmp_path):
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "carry.msgpack").write_bytes(b"carry-bytes")
    sub = ckpt / "ocn"
    sub.mkdir()
    (sub / "state.bin").write_bytes(b"state-bytes")

    unhashed = provenance.describe_input(ckpt, force_hash=False)
    assert unhashed["kind"] == "directory"
    assert unhashed["files"] == 2
    assert unhashed["size"] == len(b"carry-bytes") + len(b"state-bytes")
    assert "sha256" not in unhashed

    hashed = provenance.describe_input(ckpt, force_hash=True)
    assert hashed["kind"] == "directory"
    assert hashed["files"] == 2
    # Must match the deterministic sorted (rel_path + NUL + bytes) sha256
    # convention used by jcm-monitor's dir_digest.
    h = hashlib.sha256()
    h.update(b"carry.msgpack\0" + b"carry-bytes")
    h.update(b"ocn/state.bin\0" + b"state-bytes")
    assert hashed["sha256"] == h.hexdigest()


def test_record_input_never_downgrades_existing_sha256(tmp_path, monkeypatch):
    monkeypatch.delenv("JEM_HASH_INPUTS", raising=False)
    monkeypatch.delenv("JCM_HASH_INPUTS", raising=False)
    f = tmp_path / "donor.nc"
    f.write_bytes(b"donor-state")

    provenance.start_run()
    provenance.record_input("hf://states/donor.nc", f, force_hash=True)
    provenance.record_input(f)  # unhashed re-record must keep the digest

    entry = provenance.collect()["inputs"][str(f)]
    assert entry["sha256"] == hashlib.sha256(b"donor-state").hexdigest()
    assert entry["requested"] == "hf://states/donor.nc"


def test_collect_run_hash_changes_with_inputs_and_params(tmp_path):
    provenance.start_run(OmegaConf.create({"coupled_run": {"total_time": "2 days"}}))
    prov0 = provenance.collect()
    assert "config_yaml" in prov0
    assert "total_time: 2 days" in prov0["config_yaml"]
    assert len(prov0["run_hash"]) == 12
    assert provenance.collect()["run_hash"] == prov0["run_hash"]

    prov_with_params = provenance.collect(params={"speedy.a": 1.0})
    assert prov_with_params["run_hash"] != prov0["run_hash"]
    assert prov_with_params["params"] == {"speedy.a": 1.0}

    f = tmp_path / "extra.nc"
    f.write_bytes(b"extra")
    provenance.record_input(f)
    assert provenance.collect()["run_hash"] != prov0["run_hash"]


def test_write_sidecar_and_attrs_agree_on_run_hash(tmp_path):
    provenance.start_run(OmegaConf.create({"a": 1}))
    provenance.record_fact("initial_condition", "era5:2000-01-01")
    params = {"term.alpha": 0.5}

    nc_path = tmp_path / "atm-00000000.nc"
    a = provenance.attrs(params, chunk_wall_seconds=12.34)
    sidecar = provenance.write_sidecar(nc_path, params)

    assert sidecar == tmp_path / "atm-00000000.nc.provenance.json"
    assert sidecar.is_file()
    data = json.loads(sidecar.read_text())

    assert a["jem_prov_run_hash"] == data["run_hash"]
    assert a["jem_prov_initial_condition"] == "era5:2000-01-01"
    assert a["jcm_prov_chunk_wall_seconds"] == 12.3
    assert a["jem_prov_chunk_wall_seconds"] == 12.3
    assert data["facts"]["initial_condition"] == "era5:2000-01-01"
    assert data["params"] == params
    assert "config_yaml" in data and "a: 1" in data["config_yaml"]


def test_run_chunked_writes_provenance_sidecars_and_attrs(tmp_path):
    provenance.start_run(OmegaConf.create({"coupled_run": {"total_time": "2 days"}}))
    result = run_chunked(
        _two_slabs(),
        total_time="2 days",
        chunk="2 days",
        output_dir=tmp_path,
        checkpoint_path=None,
    )
    assert result.completed
    assert len(result.paths) == 2

    for nc_path in result.paths:
        sidecar = provenance.sidecar_path(nc_path)
        assert sidecar.is_file()
        prov = json.loads(sidecar.read_text())
        assert "jem" in prov["code"]
        assert "environment" in prov
        assert "created" in prov
        assert len(prov["run_hash"]) == 12
        assert "total_time: 2 days" in prov["config_yaml"]

        with xr.open_dataset(nc_path) as ds:
            assert ds.attrs["jem_prov_run_hash"] == prov["run_hash"]
            assert "jcm_prov_chunk_wall_seconds" in ds.attrs
            assert "jem_prov_created" in ds.attrs
            assert "jem_prov_written" in ds.attrs


def test_run_chunked_warm_start_records_checkpoint_digest_across_resume(tmp_path):
    # Produce a spinup checkpoint first.
    spinup = tmp_path / "spinup"
    run_chunked(
        _two_slabs(),
        total_time="1 day",
        chunk="1 day",
        output_dir=spinup,
    )
    donor = spinup / "checkpoint"
    assert (donor / "carry.msgpack").is_file()

    # Run 1 day of a 2-day warm-started experiment, then resume for day 2.
    exp = tmp_path / "experiment"
    provenance.start_run()
    run_chunked(
        _two_slabs(),
        total_time="1 day",
        chunk="1 day",
        output_dir=exp,
        initial_condition=donor,
    )
    first_prov = json.loads((exp / "ocn-00000000.nc.provenance.json").read_text())

    # Simulate a fresh process resuming from exp/checkpoint.
    provenance.start_run()
    run_chunked(
        _two_slabs(),
        total_time="2 days",
        chunk="1 day",
        output_dir=exp,
        initial_condition=donor,
    )
    second_prov = json.loads((exp / "ocn-00000001.nc.provenance.json").read_text())

    donor_key = str(donor.resolve())
    assert donor_key in first_prov["inputs"]
    assert donor_key in second_prov["inputs"]
    assert "sha256" in first_prov["inputs"][donor_key]
    assert (
        first_prov["inputs"][donor_key]["sha256"]
        == second_prov["inputs"][donor_key]["sha256"]
    )
    assert first_prov["run_hash"] == second_prov["run_hash"]


def test_write_chunk_removes_sidecar_when_removing_empty_dataset(tmp_path):
    ds = xr.Dataset(
        {"t": ("time", np.array([1.0, 2.0]))},
        coords={"time": np.array([0, 1])},
    )
    (nc_path,) = write_chunk({"ocn": ds}, tmp_path, 0)
    sidecar = provenance.write_sidecar(nc_path)
    assert sidecar.is_file()

    empty = ds.isel(time=slice(0, 0))
    written = write_chunk({"ocn": empty}, tmp_path, 0)
    assert written == []
    assert not nc_path.exists()
    assert not sidecar.exists()


def test_merges_jcm_provenance_when_jcm_is_imported(tmp_path):
    jcm_provenance = pytest.importorskip("jcm.provenance")

    jcm_input = tmp_path / "bc.nc"
    jcm_input.write_bytes(b"boundary-data")
    jem_input = tmp_path / "grid.nc"
    jem_input.write_bytes(b"grid-data")

    cfg = OmegaConf.create({"coupled_run": {"total_time": "1 day"}})
    jcm_provenance.start_run(cfg)
    provenance.start_run(cfg)
    jcm_provenance.record_input(str(jcm_input))
    jcm_provenance.record_fact("ozone_source", "prescribed:/x/ozone.nc")
    provenance.record_input(str(jem_input))
    provenance.record_fact("jem_only", "yes")

    prov = provenance.collect()
    assert "jem" in prov["code"]
    assert "jcm" in prov["code"]
    assert str(jcm_input) in prov["inputs"]
    assert str(jem_input) in prov["inputs"]
    assert prov["facts"]["ozone_source"] == "prescribed:/x/ozone.nc"
    assert prov["facts"]["jem_only"] == "yes"
    assert "data_mirror_revision" in prov["facts"]
    assert "ozone=prescribed:/x/ozone.nc" in provenance.summary()


def test_importing_jem_provenance_does_not_import_components():
    """Importing jem and jem.provenance must not drag in jem.components."""
    source = (
        "import sys, jem, jem.provenance;"
        " assert 'jem.components.jcm.component' not in sys.modules,"
        " sorted(m for m in sys.modules if m.startswith('jem'))"
    )
    subprocess.run([sys.executable, "-c", source], check=True)
