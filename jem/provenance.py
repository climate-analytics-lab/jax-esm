"""Run provenance capture for JAX-ESM, matching jax-gcm's schema.

Records what actually produced a coupled output file — code versions (git SHA,
branch and dirty state of every editable install *actually imported*, including
``jem``, ``jcm`` and ``veros``), precision and hardware, the resolved input
files and warm-start checkpoints, run-shaping facts, the atmosphere's effective
parameter values, and a single ``run_hash`` — and writes it both as netCDF
global attributes and as a ``<output>.provenance.json`` sidecar carrying the
fully composed config.

When ``jcm`` is imported (any run with a JCM atmosphere), this module merges
everything :mod:`jcm.provenance` recorded during ``jcm.runners.build_model``
and ``jcm.runners.build_forcing`` — boundary files, data-mirror revision,
``ozone_source``, ``dated_input_persistence`` — with JAX-ESM's own entries,
while keeping ``jem``'s top-level import free of ``jcm`` for slab-only runs
and unit tests.

Schema compatibility with :mod:`jcm.provenance` is deliberate: downstream
tools such as ``jcm-monitor`` discover ``*.provenance.json`` sidecars and read
``code.jcm.sha``, ``created``, ``run_hash``, ``inputs`` and
``jcm_prov_chunk_wall_seconds`` directly, without special-casing JAX-ESM.

Lifecycle: :func:`start_run` at run start (captures the composed config and
resets the registries), :func:`record_input` / :func:`record_fact` during
model build and warm-start setup, :func:`summary` for the one-line run log,
and :func:`attrs` / :func:`write_sidecar` at chunk-write time.

:func:`start_run` is the only thing that empties the registries. A run driven
from Python that never calls it still gets a complete record: the record opens
at the first thing recorded (:func:`ensure_started`, which
:func:`jem.driver.run_chunked` calls before it records a warm start) and keeps
everything recorded from then on. The registries are per process, so a script
that makes more than one run in one process calls :func:`start_run` before
building each model; otherwise the later run's record carries the earlier
run's inputs, facts and creation time.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Libraries whose working tree changes what a coupled run computes. Probed
#: only when already imported — an absent entry means "not part of this run".
_EDITABLE_LIBS = (
    "jem",
    "jcm",
    "dinosaur",
    "rrtmgp",
    "mam4_jax",
    "jax_cosp",
    "pyses",
    "veros",
)

#: Module-name -> distribution-name where they differ.
_DIST_NAMES = {
    "jem": "jax-esm",
    "rrtmgp": "jax-rrtmgp",
    "mam4_jax": "mam4-jax",
    "jax_cosp": "jax-cosp",
}

#: Packaged dependencies recorded by version only.
_VERSION_LIBS = ("jax", "jaxlib", "numpy", "xarray", "flax")

#: Environment variables that silently change precision or compilation.
_ENV_FLAGS = ("JAX_ENABLE_X64", "MAM4_JAX_ENABLE_X64", "XLA_FLAGS", "JCM_CACHE_DIR")

#: Environment switches enabling full content sha256 hashing of all inputs.
#: Both the JEM name and JCM's name are honoured so a user running both repos
#: needs only one knob.
_HASH_INPUTS_ENVS = ("JEM_HASH_INPUTS", "JCM_HASH_INPUTS")

_state: dict[str, Any] = {"base": None, "inputs": {}, "facts": {}}


def _git(path: str, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", path, *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _hash_inputs_enabled() -> bool:
    return any(os.environ.get(var) == "1" for var in _HASH_INPUTS_ENVS)


def _probe_module(name: str) -> dict[str, Any] | None:
    """Return path + version + git SHA/branch/dirty for one imported library."""
    mod = sys.modules.get(name)
    mod_file = getattr(mod, "__file__", None) if mod is not None else None
    if not mod_file:
        return None
    path = str(Path(mod_file).resolve().parent)
    entry: dict[str, Any] = {"path": path}
    for dist in (_DIST_NAMES.get(name, name), name.replace("_", "-")):
        try:
            entry["version"] = metadata.version(dist)
            break
        except metadata.PackageNotFoundError:
            pass
    top = _git(path, "rev-parse", "--show-toplevel")
    if top:
        entry["sha"] = _git(top, "rev-parse", "HEAD")
        entry["branch"] = _git(top, "rev-parse", "--abbrev-ref", "HEAD")
        status = _git(top, "status", "--porcelain")
        entry["dirty"] = bool(status)
        if status:
            # Distinguish two dirty trees without shipping the diff.
            diff = _git(top, "diff", "HEAD") or ""
            entry["dirty_diff_sha"] = hashlib.sha256(diff.encode()).hexdigest()[:12]
    return entry


def probe_code() -> dict[str, dict[str, Any]]:
    """Return path, version and git state for every imported key library."""
    code: dict[str, dict[str, Any]] = {}
    jcm_prov = sys.modules.get("jcm.provenance")
    if jcm_prov is not None:
        # Start from jcm's own probe so any library jax-gcm adds is picked up
        # automatically, then add JAX-ESM and any other coupled libraries.
        code.update(jcm_prov.probe_code())
    for name in _EDITABLE_LIBS:
        if name in code:
            continue
        entry = _probe_module(name)
        if entry is not None:
            code[name] = entry
    for name in _VERSION_LIBS:
        if name in code:
            continue
        try:
            code[name] = {"version": metadata.version(name)}
        except metadata.PackageNotFoundError:
            pass
    return code


def probe_environment() -> dict[str, Any]:
    """Return precision flags, devices and host — the silent run-shapers."""
    import jax

    devices = jax.devices()
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "jax_enable_x64": bool(getattr(jax.config, "jax_enable_x64", False)),
        "env": {k: os.environ[k] for k in _ENV_FLAGS if k in os.environ},
        "platform": devices[0].platform,
        "device_kind": devices[0].device_kind,
        "device_count": len(devices),
    }
    try:
        # On GPU this carries the CUDA runtime/driver versions.
        from jax.extend import backend as jax_backend

        env["platform_version"] = jax_backend.get_backend().platform_version
    except Exception:  # noqa: BLE001 — best-effort, backend-dependent
        pass
    return env


def _dir_digest(root: Path) -> tuple[str | None, int, int]:
    """Return ``(sha256_hex, file_count, total_bytes)`` for a checkpoint directory.

    Walks ``root`` following symlinks (cycle-safe) and hashes every file's
    POSIX relative path plus bytes in sorted order, matching the directory
    digest convention used by ``jcm-monitor``. Returns ``sha256_hex=None`` if
    the directory is empty or contains a non-regular entry such as a dangling
    symlink.
    """
    files: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []  # cycle/alias: do not recurse again
        seen.add(real)
        dirnames.sort()
        for name in filenames:
            p = Path(dirpath) / name
            if not p.is_file():
                return None, 0, 0
            files.append((p.relative_to(root).as_posix(), p))
    if not files:
        return None, 0, 0
    h = hashlib.sha256()
    total_bytes = 0
    for rel, p in sorted(files):
        h.update(rel.encode() + b"\0")
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                total_bytes += len(block)
                h.update(block)
    return h.hexdigest(), len(files), total_bytes


def describe_input(path: str | Path, *, force_hash: bool = False) -> dict[str, Any]:
    """Describe an input file or checkpoint directory by size/mtime and optional sha256.

    Content hashing is opt-in by default (``JEM_HASH_INPUTS=1`` or
    ``JCM_HASH_INPUTS=1``) because boundary-condition files can run to several
    GB. ``force_hash=True`` hashes this one call regardless of the env switch —
    used for warm-start initial conditions (``atmosphere.init.file`` and
    ``coupled_run.initial_condition``), whose identity must survive copying
    outputs to another host.
    """
    p = Path(path)
    d: dict[str, Any] = {}
    try:
        st = os.stat(p)
        d["mtime"] = datetime.fromtimestamp(
            st.st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds")
    except OSError:
        d["missing"] = True
        return d

    should_hash = force_hash or _hash_inputs_enabled()
    if p.is_dir():
        d["kind"] = "directory"
        if should_hash:
            digest, file_count, total_bytes = _dir_digest(p)
            d["files"] = file_count
            d["size"] = total_bytes
            if digest is not None:
                d["sha256"] = digest
        else:
            file_count = 0
            total_bytes = 0
            for dirpath, _, filenames in os.walk(p, followlinks=True):
                for name in filenames:
                    fp = Path(dirpath) / name
                    try:
                        total_bytes += fp.stat().st_size
                        file_count += 1
                    except OSError:
                        pass
            d["files"] = file_count
            d["size"] = total_bytes
        return d

    d["size"] = st.st_size
    if should_hash:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        d["sha256"] = h.hexdigest()
    return d


def _new_base(cfg: Any = None) -> dict[str, Any]:
    """Return the part of a record that is fixed when the record opens.

    That is the time it opened and the composed config. Code and environment
    are not in it: they are probed on every :func:`collect`, so libraries
    imported during model construction and any later ``jax_enable_x64`` flip
    are captured accurately.
    """
    config_yaml = None
    if cfg is not None:
        from omegaconf import OmegaConf
        from omegaconf.errors import OmegaConfBaseException

        try:
            config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        except OmegaConfBaseException:
            # hydra:-runtime interpolations only resolve inside a live
            # Hydra app; the unresolved form still records every choice.
            config_yaml = OmegaConf.to_yaml(cfg, resolve=False)
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_yaml": config_yaml,
    }


def start_run(cfg: Any = None) -> None:
    """Reset the registries and capture the composed config at run start.

    This is the one call that discards what has been recorded, here and in
    :mod:`jcm.provenance` when it is imported, so it belongs before the model
    is built: an input recorded during construction is part of the run.
    """
    _state["base"] = _new_base(cfg)
    _state["inputs"] = {}
    _state["facts"] = {}
    jcm_prov = sys.modules.get("jcm.provenance")
    if jcm_prov is not None:
        jcm_prov.start_run(cfg)


def ensure_started() -> None:
    """Open the run's record if :func:`start_run` has not, keeping what is recorded.

    A run driven from Python need not call :func:`start_run`, and its inputs
    are recorded before the first chunk is written. Opening the record must
    therefore never empty a registry, here or in :mod:`jcm.provenance`: a warm
    start recorded by :func:`jem.driver.run_chunked` would drop out of the
    sidecars, and two runs differing only in their donor checkpoint would
    share a ``run_hash``. The record opened this way has no composed config
    (``config_yaml`` is ``None``), and its ``created`` is the time of this
    call.
    """
    if _state["base"] is None:
        _state["base"] = _new_base()


def record_input(
    requested: str | Path,
    resolved: str | Path | None = None,
    *,
    force_hash: bool = False,
) -> None:
    """Record a resolved input file or checkpoint directory (deduped by path).

    A path already recorded with a ``sha256`` digest keeps it when a later call
    records the same path with ``force_hash=False``, so an unhashed lookup
    cannot silently downgrade a warm-start digest.
    """
    ensure_started()
    resolved_str = str(resolved if resolved is not None else requested)
    if not (os.path.isfile(resolved_str) or os.path.isdir(resolved_str)):
        return
    existing = _state["inputs"].get(resolved_str)
    if existing is not None and "sha256" in existing and not force_hash:
        return
    entry = describe_input(resolved_str, force_hash=force_hash)
    if str(requested) != resolved_str:
        entry["requested"] = str(requested)
    _state["inputs"][resolved_str] = entry


def record_fact(key: str, value: Any) -> None:
    """Record a run-shaping fact (for example, an initial-condition source)."""
    ensure_started()
    _state["facts"][key] = value


def _merge_inputs(
    base: Mapping[str, dict[str, Any]],
    overlay: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge two input maps, never downgrading an entry that already has a sha256."""
    merged = {k: dict(v) for k, v in base.items()}
    for path, entry in overlay.items():
        existing = merged.get(path)
        if existing is not None and "sha256" in existing and "sha256" not in entry:
            continue
        merged[path] = dict(entry)
    return merged


def collect(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the full provenance record for the current run.

    Code and environment are probed fresh on every call so libraries imported
    lazily during model build and any post-start precision flips are reflected.
    When :mod:`jcm.provenance` is imported in this process, its recorded
    inputs, facts and data-mirror revision are merged automatically.
    """
    ensure_started()
    prov = dict(_state["base"])

    jcm_prov_mod = sys.modules.get("jcm.provenance")
    if jcm_prov_mod is not None:
        jcm_prov = jcm_prov_mod.collect()
        prov["code"] = probe_code()
        prov["environment"] = jcm_prov["environment"]
        prov["inputs"] = _merge_inputs(jcm_prov["inputs"], _state["inputs"])
        prov["facts"] = {**jcm_prov["facts"], **_state["facts"]}
    else:
        prov["code"] = probe_code()
        prov["environment"] = probe_environment()
        prov["inputs"] = {k: dict(v) for k, v in _state["inputs"].items()}
        prov["facts"] = dict(_state["facts"])

    if params:
        prov["params"] = params

    hash_material: dict[str, Any] = {
        # A dirty tree is a different code state than its HEAD: fold the
        # working-tree diff fingerprint into the identity.
        "code": {
            k: (v.get("sha", v.get("version")), v.get("dirty_diff_sha"))
            for k, v in prov["code"].items()
        },
        "config": prov["config_yaml"],
        "inputs": {
            k: v.get("sha256", (v.get("size"), v.get("mtime")))
            for k, v in prov["inputs"].items()
        },
        "x64": prov["environment"]["jax_enable_x64"],
        "params": params or None,
    }
    if "data_mirror_revision" in prov["facts"]:
        hash_material["data_mirror_revision"] = prov["facts"]["data_mirror_revision"]
    prov["run_hash"] = hashlib.sha256(
        json.dumps(hash_material, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return prov


def summary() -> str:
    """Return a one-line summary of SHAs, precision and devices (probed fresh)."""
    code = probe_code()
    env = probe_environment()
    shas = ", ".join(
        f"{k}={v['sha'][:8]}{'+dirty' if v.get('dirty') else ''}"
        for k, v in code.items()
        if "sha" in v
    )
    parts = [
        shas or "no git trees",
        f"x64={env.get('jax_enable_x64')}",
        f"{env.get('device_count')}x{env.get('device_kind')}",
    ]
    ozone = _state["facts"].get("ozone_source")
    if ozone is None:
        jcm_prov = sys.modules.get("jcm.provenance")
        if jcm_prov is not None:
            ozone = jcm_prov.collect()["facts"].get("ozone_source")
    if ozone is not None:
        parts.append(f"ozone={ozone}")
    return "; ".join(parts)


def read_params_from_datasets(datasets: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract the atmosphere's parameter record from a chunk's datasets, if present.

    :meth:`jem.components.jcm.component.JCMComponent.to_xarray` delegates to
    ``ModelPredictions.to_xarray``, which already stamps ``jcm_prov_params``
    onto the atmosphere dataset. Reading it back here keeps the sidecar's
    ``params`` and ``run_hash`` consistent with the atmosphere netCDF file
    without re-walking the physics tree on every chunk.
    """
    for ds in datasets.values():
        blob = getattr(ds, "attrs", {}).get("jcm_prov_params")
        if blob:
            parsed = json.loads(blob)
            if isinstance(parsed, dict) and parsed:
                return parsed
    return None


def attrs(
    params: dict[str, Any] | None = None,
    *,
    chunk_wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Return flat netCDF-safe global attributes (nested parts as JSON strings)."""
    prov = collect(params)
    out: dict[str, Any] = {
        "jem_prov_created": prov["created"],
        # attrs() runs at output time, so this stamps the write — with
        # ``created`` it brackets the run without a separate end hook.
        "jem_prov_written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jem_prov_run_hash": prov["run_hash"],
        "jem_prov_code": json.dumps(prov["code"], sort_keys=True),
        "jem_prov_environment": json.dumps(prov["environment"], sort_keys=True),
        "jem_prov_inputs": json.dumps(prov["inputs"], sort_keys=True),
    }
    if chunk_wall_seconds is not None:
        # Both keys: `jcm_prov_chunk_wall_seconds` is the attribute name
        # `jcm-monitor` reads for chunk throughput when no log is present.
        rounded = round(chunk_wall_seconds, 1)
        out["jcm_prov_chunk_wall_seconds"] = rounded
        out["jem_prov_chunk_wall_seconds"] = rounded
    for key, value in prov["facts"].items():
        out[f"jem_prov_{key}"] = str(value)
    if prov["config_yaml"] is not None:
        # The composed config is too large for attributes (it goes in the
        # sidecar); the hash makes "same config" checkable from attrs.
        out["jem_prov_config_sha"] = hashlib.sha256(
            prov["config_yaml"].encode()
        ).hexdigest()[:12]
    return out


def sidecar_path(output_path: Path | str) -> Path:
    """Return the ``<output>.provenance.json`` path for ``output_path``."""
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".provenance.json")


def write_sidecar(
    output_path: Path | str,
    params: dict[str, Any] | None = None,
) -> Path:
    """Write ``<output>.provenance.json`` with the full record and composed config.

    Pass the same ``params`` given to :func:`attrs` for the same file so the
    sidecar's ``run_hash`` matches the one stamped in the netCDF attributes.
    """
    sidecar = sidecar_path(output_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(collect(params), indent=1, sort_keys=True, default=str)
    )
    return sidecar
