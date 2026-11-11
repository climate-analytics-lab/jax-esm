# Running JAX-ESM on Kubernetes (GKE Autopilot)

This directory holds a manifest and a launcher for running JAX-ESM (`jem`) on
the **GKE Autopilot** GPU cluster (`climatebench-spot`) provisioned by
[`climate-analytics-lab-terraform`](https://github.com/climate-analytics-lab/climate-analytics-lab-terraform),
building from the root [`Dockerfile`](../../Dockerfile).

## Files

| File | Purpose |
| --- | --- |
| [`job.yaml`](./job.yaml) | Batch `Job` template for a coupled JEM simulation run on a GKE Spot GPU (mounts GCS buckets via GCS FUSE + Workload Identity). |
| [`launch.py`](./launch.py) | Builds one image from the root [`Dockerfile`](../../Dockerfile) in Cloud Build and submits one or more `Job`s with different commands. |
| [`pyproject.toml`](./pyproject.toml) | Host-side Python project and dependencies for `launch.py`. |
| [`launch_test.py`](./launch_test.py) | Unit tests for `launch.py`. |

## Prerequisites

1. **Authenticate to GCP:**
   ```bash
   gcloud auth application-default login
   ```
2. **Launcher dependencies:** `uv run deploy/gke/launch.py ...` installs them
   automatically from [`pyproject.toml`](./pyproject.toml), or install into a
   virtualenv with `pip install ./deploy/gke`.

## Building Once & Launching Multiple Jobs (`launch.py`)

`launch.py` packages the local checkout, runs a single Cloud Build of the root
`Dockerfile`, and launches one Kubernetes `Job` per `--cmd` using that shared
image:

```bash
uv run deploy/gke/launch.py \
  --prefix coupled-sweep \
  --cmd 'exec python -m jem.main +configuration=aquaplanet-slab coupled_run=short_run hydra.run.dir="$OUTPUT_DIR"' \
  --cmd 'exec python -m jem.main +configuration=earth-slab coupled_run=short_run hydra.run.dir="$OUTPUT_DIR"' \
  --cmd 'exec python -m jem.main +configuration=veros-earth coupled_run=short_run hydra.run.dir="$OUTPUT_DIR"'
```

Setting `hydra.run.dir="$OUTPUT_DIR"` directs each component's chunked NetCDF
output (`<component>-<step>.nc`), Hydra's `.hydra/` metadata, and the default
relative `checkpoint/` directory into `$OUTPUT_DIR` — so if GKE preempts a Spot
GPU pod, the retried pod automatically resumes from the last completed chunk.

When multiple `--cmd` flags are given, each Job writes to its own subdirectory
(`$OUTPUT_DIR` = `/mnt/output/<prefix>/job-0`, `job-1`, …) on the cluster's
Hierarchical Namespace GCS bucket (`gs://fluid-script-453604-u5-jax-gcm-runs` by
default, or any `--bucket gs://...`).

For a single Job, you can also pass the command after `--`:

```bash
uv run deploy/gke/launch.py \
  --prefix earth-slab-long \
  -- 'exec python -m jem.main +configuration=earth-slab coupled_run=long_run hydra.run.dir="$OUTPUT_DIR"'
```

Useful flags:

- `--image <uri>` — skip Cloud Build and launch Jobs from an already-built image.
- `--jcm-source <path>` — bundle a local `jax-gcm` checkout (e.g. `../jax-gcm`) into the Cloud Build image to test local uncommitted `jcm` changes.
- `--jcm-rev <ref>` — build against a pushed `jax-gcm` branch, tag, or commit SHA instead of the default `JCM_REV` in `Dockerfile`.
- `--gpu {l4,a100-40gb,a100-80gb}` — pick GKE Autopilot GPU class (default `l4` Spot; pass `--no-spot` for on-demand).
- `--extras plot` — optional `pyproject.toml` extras passed to `Dockerfile` (default `""` for runtime-only).
- `--dry-run` — print rendered `Job` YAML(s) locally without cloud API calls.
- `--wait` — wait until all submitted Jobs complete.

## Submitting Directly with `envsubst` + `kubectl`

`job.yaml` also works standalone with `kubectl` once cluster credentials are
configured
(`gcloud container clusters get-credentials climatebench-spot --region us-central1 --project fluid-script-453604-u5`):

```bash
export JEM_IMAGE=us-central1-docker.pkg.dev/fluid-script-453604-u5/jax-gcm/jax-gcm-runtime:<tag>
export JEM_JOB_NAME=jem-earth-slab-$(date +%s)
export JEM_KSA=jax-gcm-runner
export JEM_ACCELERATOR=nvidia-l4
export JEM_DATASET_BUCKET=climatebench
export JEM_OUTPUT_BUCKET=fluid-script-453604-u5-jax-gcm-runs
export JEM_OUTPUT_PREFIX=earth-slab-smoke
export JEM_COMMAND='exec python -m jem.main +configuration=earth-slab coupled_run=short_run hydra.run.dir="$OUTPUT_DIR"'

envsubst '$JEM_IMAGE $JEM_JOB_NAME $JEM_KSA $JEM_ACCELERATOR $JEM_DATASET_BUCKET $JEM_OUTPUT_BUCKET $JEM_OUTPUT_PREFIX $JEM_COMMAND' \
  < deploy/gke/job.yaml | kubectl -n jax-gcm apply -f -
```
