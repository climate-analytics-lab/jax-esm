#!/usr/bin/env python3
"""Build one JEM image from the root Dockerfile and launch one or more GKE Jobs."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shlex
import string
import sys
import tarfile
import tempfile
import time
import uuid

from google import auth as google_auth
from google.api_core import client_options
import google.auth.transport.requests as auth_requests
from google.cloud import container_v1, storage
from google.cloud.devtools import cloudbuild_v1
from kubernetes import client as k8s_client
from kubernetes.client.rest import ApiException
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")
PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")

DEFAULT_PROJECT = "fluid-script-453604-u5"
DEFAULT_CLUSTER = "climatebench-spot"

GPU_PROFILES = {
    "l4": {"accelerator": "nvidia-l4", "cpu": "8", "memory": "32Gi", "ephemeral_storage": "30Gi"},
    "a100-40gb": {
        "accelerator": "nvidia-tesla-a100",
        "cpu": "8",
        "memory": "56Gi",
        "ephemeral_storage": "40Gi",
    },
    "a100-80gb": {
        "accelerator": "nvidia-a100-80gb",
        "cpu": "8",
        "memory": "112Gi",
        "ephemeral_storage": "40Gi",
    },
}


def default_config(project: str = DEFAULT_PROJECT, cluster: str = DEFAULT_CLUSTER) -> dict:
    """Return GKE infrastructure settings matching `climate-analytics-lab-terraform`."""
    runs_bucket = f"{project}-jax-gcm-runs"
    return {
        "project": project,
        "region": "us-central1",
        "cluster": cluster,
        "namespace": "jax-gcm",
        "ksa": "jax-gcm-runner",
        "runner_service_account": f"jax-gcm-gke-runner@{project}.iam.gserviceaccount.com",
        "builder_service_account": f"jax-gcm-gke-builder@{project}.iam.gserviceaccount.com",
        "staging_bucket": f"{project}-gke-staging",
        "dataset_bucket": "climatebench",
        "default_output_bucket": runs_bucket,
        "cache_bucket": runs_bucket,
        "image_repository": f"us-central1-docker.pkg.dev/{project}/jax-gcm/jax-gcm-runtime",
    }


def parse_target_bucket(bucket_arg: str, prefix_arg: str | None, run_id: str) -> tuple[str, str]:
    """Validate and split a target `gs://bucket[/subpath]` URI and run prefix."""
    raw = bucket_arg.strip().removeprefix("gs://").strip("/")
    bucket, _, subpath = raw.partition("/")
    if not BUCKET_RE.match(bucket) or ".." in bucket or bucket.endswith("-tfstate"):
        raise ValueError(f"Invalid or disallowed GCS bucket: {bucket_arg!r}")
    if prefix_arg:
        prefix = f"{subpath}/{prefix_arg.strip('/')}" if subpath else prefix_arg.strip("/")
    else:
        prefix = f"{subpath}/{run_id}" if subpath else f"runs/{run_id}"
    if not prefix or not PREFIX_RE.match(prefix) or ".." in Path(prefix).parts:
        raise ValueError(f"Invalid GCS prefix: {prefix!r}")
    return bucket, prefix


SKIP_DIRS = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
    "output",
    "outputs",
    "checkpoint",
    "runs",
    "multirun",
    "jax-gcm",
    "veros-jittable",
}


def archive_source(source: Path, archive_path: Path) -> None:
    """Create a `.tar.gz` archive of `source` for Cloud Build."""
    if not (source / "Dockerfile").is_file():
        raise ValueError(f"Expected Dockerfile in {source}")
    with tarfile.open(archive_path, "w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(source):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in SKIP_DIRS
                and not d.endswith(".egg-info")
                and not d.startswith("output_T")
                and d != ".claude"
            )
            rel_dir = Path(dirpath).relative_to(source)
            for name in sorted(filenames):
                if name == "uv.lock" or name.endswith((".pyc", ".pyo", ".ckpt", ".msgpack", ".h5", ".log")):
                    continue
                path = Path(dirpath) / name
                if path.is_symlink() or not path.is_file():
                    continue
                rel = (rel_dir / name).as_posix()
                if name.endswith(".nc") and not rel.startswith("jem/data/"):
                    continue
                tar.add(path, arcname=rel, recursive=False)


def build_image(
    creds,
    storage_client,
    config: dict,
    source: Path,
    image_uri: str,
    run_id: str,
    extras: str,
) -> None:
    """Upload the source archive and build the root `Dockerfile` once in Cloud Build."""
    with tempfile.TemporaryDirectory(prefix="jem-gke-") as tmp:
        archive = Path(tmp) / "source.tar.gz"
        archive_source(source, archive)
        obj_name = f"cloudbuild/{run_id}.tar.gz"
        storage_client.bucket(config["staging_bucket"]).blob(obj_name).upload_from_filename(
            str(archive)
        )

    print(f"Building {image_uri} via Cloud Build...", flush=True)
    cb = cloudbuild_v1.CloudBuildClient(
        credentials=creds,
        client_options=client_options.ClientOptions(
            api_endpoint=f"{config['region']}-cloudbuild.googleapis.com"
        ),
    )
    build_args = ["build", "--platform", "linux/amd64"]
    if extras:
        build_args.extend(["--build-arg", f"EXTRAS={extras}"])
    build_args.extend(["-t", image_uri, "."])

    op = cb.create_build(
        request=cloudbuild_v1.CreateBuildRequest(
            parent=f"projects/{config['project']}/locations/{config['region']}",
            project_id=config["project"],
            build=cloudbuild_v1.Build(
                source=cloudbuild_v1.Source(
                    storage_source=cloudbuild_v1.StorageSource(
                        bucket=config["staging_bucket"],
                        object_=obj_name,
                    )
                ),
                steps=[
                    cloudbuild_v1.BuildStep(
                        name="gcr.io/cloud-builders/docker",
                        args=build_args,
                    )
                ],
                images=[image_uri],
                service_account=(
                    f"projects/{config['project']}/serviceAccounts/{config['builder_service_account']}"
                ),
                options=cloudbuild_v1.BuildOptions(
                    logging=cloudbuild_v1.BuildOptions.LoggingMode.CLOUD_LOGGING_ONLY,
                    machine_type=cloudbuild_v1.BuildOptions.MachineType.E2_HIGHCPU_8,
                ),
                timeout="3600s",
            ),
        )
    )
    build_meta = getattr(getattr(op, "metadata", None), "build", None)
    log_url = getattr(build_meta, "log_url", "")
    if log_url:
        print(f"Cloud Build logs: {log_url}", flush=True)
    res = op.result(timeout=3600)
    if getattr(res, "status", None) not in (None, cloudbuild_v1.Build.Status.SUCCESS):
        raise RuntimeError(f"Cloud Build failed with status {res.status}")


def render_job(
    config: dict,
    *,
    job_name: str,
    image: str,
    output_bucket: str,
    output_prefix: str,
    command: str,
    gpu: str = "l4",
    spot: bool = True,
) -> dict:
    """Render `deploy/gke/job.yaml` for a single command."""
    profile = GPU_PROFILES[gpu]
    rendered = string.Template((HERE / "job.yaml").read_text()).safe_substitute(
        JEM_JOB_NAME=job_name,
        JEM_KSA=config["ksa"],
        JEM_ACCELERATOR=profile["accelerator"],
        JEM_IMAGE=image,
        JEM_COMMAND="",
        JEM_OUTPUT_PREFIX=output_prefix,
        JEM_DATASET_BUCKET=config["dataset_bucket"],
        JEM_OUTPUT_BUCKET=output_bucket,
    )
    job: dict = yaml.safe_load(rendered)
    job["metadata"]["namespace"] = config["namespace"]

    pod_spec = job["spec"]["template"]["spec"]
    if not spot:
        pod_spec["nodeSelector"].pop("cloud.google.com/gke-spot", None)

    cache_bucket = config["cache_bucket"]
    separate_cache = cache_bucket != output_bucket
    jax_cache_dir = "/mnt/cache/.jem-cache/jax" if separate_cache else "/mnt/output/.jem-cache/jax"

    container = pod_spec["containers"][0]
    container["args"] = [command]
    container["env"].append({"name": "JAX_COMPILATION_CACHE_DIR", "value": jax_cache_dir})
    if separate_cache:
        container["volumeMounts"].append({"name": "cache", "mountPath": "/mnt/cache"})
        pod_spec["volumes"].append(
            {
                "name": "cache",
                "csi": {
                    "driver": "gcsfuse.csi.storage.gke.io",
                    "readOnly": False,
                    "volumeAttributes": {"bucketName": cache_bucket},
                },
            }
        )
    for section in ("requests", "limits"):
        container["resources"][section]["cpu"] = profile["cpu"]
        container["resources"][section]["memory"] = profile["memory"]
        container["resources"][section]["ephemeral-storage"] = profile["ephemeral_storage"]
        for k, v in list(container["resources"][section].items()):
            container["resources"][section][k] = str(v)
    return job


@contextmanager
def connect_k8s(creds, config: dict):
    """Yield an authenticated Kubernetes `ApiClient` for the GKE cluster."""
    cluster = container_v1.ClusterManagerClient(credentials=creds).get_cluster(
        name=(
            f"projects/{config['project']}/locations/"
            f"{config['region']}/clusters/{config['cluster']}"
        )
    )
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".crt") as ca:
        ca.write(base64.b64decode(cluster.master_auth.cluster_ca_certificate))
        ca.flush()
        cfg = k8s_client.Configuration()
        cfg.host = f"https://{cluster.endpoint}"
        cfg.ssl_ca_cert = ca.name
        req = auth_requests.Request()

        def refresh(c):
            if not creds.valid or creds.expired:
                creds.refresh(req)
            c.api_key = {"authorization": creds.token}
            c.api_key_prefix = {"authorization": "Bearer"}

        refresh(cfg)
        cfg.refresh_api_key_hook = refresh
        api = k8s_client.ApiClient(cfg)
        try:
            yield api
        finally:
            api.close()


def _ensure_namespace_and_ksa(core, config: dict) -> None:
    """Ensure the Kubernetes Namespace and Workload Identity KSA exist."""
    ns = config["namespace"]
    try:
        core.create_namespace(
            body={"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": ns}}
        )
    except ApiException as exc:
        if exc.status != 409:
            raise

    ksa_body = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {
            "name": config["ksa"],
            "annotations": {"iam.gke.io/gcp-service-account": config["runner_service_account"]},
        },
    }
    try:
        core.create_namespaced_service_account(namespace=ns, body=ksa_body)
    except ApiException as exc:
        if exc.status != 409:
            raise


def _wait_for_jobs(batch, namespace: str, job_names: list[str]) -> None:
    """Wait until every submitted Job succeeds or raise if any fails."""
    pending = set(job_names)
    failures: list[str] = []
    while pending:
        for job_name in list(pending):
            status = batch.read_namespaced_job_status(name=job_name, namespace=namespace).status
            for cond in (status.conditions if status else None) or []:
                if cond.status == "True" and cond.type == "Complete":
                    print(f"[{job_name}] Complete", flush=True)
                    pending.remove(job_name)
                    break
                if cond.status == "True" and cond.type == "Failed":
                    msg = f"{job_name}: {cond.reason} {cond.message or ''}".strip()
                    print(f"[{job_name}] Failed ({msg})", flush=True)
                    failures.append(msg)
                    pending.remove(job_name)
                    break
        if pending:
            time.sleep(15)
    if failures:
        raise RuntimeError("One or more Jobs failed: " + "; ".join(failures))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--cmd",
        action="append",
        default=[],
        help="Shell command to run in a Job (repeatable to launch multiple Jobs from one image)",
    )
    parser.add_argument(
        "--image",
        help="Existing container image URI (skips Cloud Build when set)",
    )
    parser.add_argument(
        "--extras",
        default="",
        help="Comma-separated pyproject extras for Dockerfile build (e.g. 'plot'; default: none)",
    )
    parser.add_argument(
        "--bucket",
        help="Target GCS bucket (gs://bucket[/subpath]; defaults to cluster's output bucket)",
    )
    parser.add_argument(
        "--prefix",
        help="Object prefix inside --bucket (default: runs/<run-id>)",
    )
    parser.add_argument(
        "--gpu",
        default="l4",
        choices=tuple(GPU_PROFILES),
        help="GKE Autopilot GPU class (default: %(default)s)",
    )
    parser.add_argument(
        "--no-spot",
        action="store_true",
        help="Schedule on standard (on-demand) GKE Autopilot nodes instead of Spot",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT") or DEFAULT_PROJECT,
        help="GCP project ID (default: $GOOGLE_CLOUD_PROJECT or %(default)s)",
    )
    parser.add_argument(
        "--cluster",
        default=DEFAULT_CLUSTER,
        help="GKE cluster name (default: %(default)s)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT,
        help="Local jax-esm checkout to build (default: %(default)s)",
    )
    parser.add_argument("--wait", action="store_true", help="Wait for all launched Jobs to finish")
    parser.add_argument("--dry-run", action="store_true", help="Print Job YAML(s) without cloud calls")
    parser.add_argument(
        "trailing_cmd",
        nargs=argparse.REMAINDER,
        help="Optional single command after `--`",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build once and submit one or more GKE Jobs."""
    parser = build_parser()
    args = parser.parse_args(argv)

    commands = list(args.cmd)
    trailing = args.trailing_cmd[1:] if args.trailing_cmd[:1] == ["--"] else args.trailing_cmd
    if trailing:
        commands.append(trailing[0] if len(trailing) == 1 else shlex.join(trailing))
    if not commands:
        parser.error("Provide at least one `--cmd '...'` or a command after `--`.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    config = default_config(args.project, args.cluster)
    image_uri = args.image or f"{config['image_repository'].split(':', 1)[0]}:{run_id}"
    output_bucket, base_prefix = parse_target_bucket(
        args.bucket or f"gs://{config['default_output_bucket']}",
        args.prefix,
        run_id,
    )
    planned = [
        (
            f"jem-{run_id}" if len(commands) == 1 else f"jem-{run_id}-{i}",
            base_prefix if len(commands) == 1 else f"{base_prefix}/job-{i}",
            cmd,
        )
        for i, cmd in enumerate(commands)
    ]
    jobs = [
        render_job(
            config,
            job_name=job_name,
            image=image_uri,
            output_bucket=output_bucket,
            output_prefix=job_prefix,
            command=cmd,
            gpu=args.gpu,
            spot=not args.no_spot,
        )
        for job_name, job_prefix, cmd in planned
    ]

    if args.dry_run:
        print(yaml.safe_dump_all(jobs, sort_keys=False), end="")
        return 0

    creds, _ = google_auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not args.image:
        sc = storage.Client(project=config["project"], credentials=creds)
        build_image(
            creds,
            sc,
            config,
            args.source.resolve(),
            image_uri,
            run_id,
            args.extras.strip(),
        )

    ns = config["namespace"]
    with connect_k8s(creds, config) as api:
        core, batch = k8s_client.CoreV1Api(api), k8s_client.BatchV1Api(api)
        _ensure_namespace_and_ksa(core, config)
        for (job_name, job_prefix, _), job in zip(planned, jobs, strict=True):
            batch.create_namespaced_job(namespace=ns, body=job)
            print(
                f"Launched {ns}/{job_name} -> gs://{output_bucket}/{job_prefix}/",
                flush=True,
            )

        if args.wait:
            _wait_for_jobs(batch, ns, [name for name, _, _ in planned])
    return 0


if __name__ == "__main__":
    sys.exit(main())
