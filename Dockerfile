# GPU build of the JEM image. Uses the slim `cuda:base` image (just the
# driver-compat shim and CUDA repo metadata, no preinstalled CUDA/cuDNN
# libs) and lets `jax[cuda12]` pull its own bundled NVIDIA wheels —
# cuBLAS, cuDNN, NCCL, etc. live under `nvidia-*-cu12` in pip and JAX
# loads them from there at import time. Going from `cudnn-runtime` to
# `base` cuts the compressed registry size from ~6.0 GiB to ~3.8 GiB
# (about 36%, measured 2026-05-12). Savings are smaller than you might
# expect because JAX detects system-installed CUDA libs and skips its
# bundled wheels when they're available — so the old base wasn't pure
# overhead, it was substituting for the wheels.
#
# JAX falls back to CPU if no GPU is visible at runtime, so the same
# image works on hosts without `--gpus all`, just without acceleration.
#
# Build:
#     docker build -t jem .
#
# Run on a GPU host:
#     docker run --rm --gpus all jem +configuration=veros-earth coupled_run=short_run
FROM nvidia/cuda:12.6.3-base-ubuntu22.04

# Ubuntu 22.04 ships Python 3.10; pull 3.11 from deadsnakes to satisfy
# JEM's >=3.11 requirement. DEBIAN_FRONTEND=noninteractive prevents tzdata
# (a transitive dep of build-essential) from prompting at build time.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
        libffi-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install jax-gcm at JCM_SUPPORTED_REV (jem/components/jcm/contract.py) first,
# since `jcm>=3.0.0rc1` is not published on PyPI. Keeping this before `COPY .`
# caches jax-gcm's dependencies across local edits; when `launch.py
# --jcm-source` stages a local checkout at `_deps/jax-gcm`, the step after
# `COPY . /app` replaces `/opt/jax-gcm` with it.
ARG JCM_REV=f378069090b127badbb158b06e2767f3be8f436c
RUN git clone https://github.com/climate-analytics-lab/jax-gcm.git /opt/jax-gcm \
    && (git -C /opt/jax-gcm checkout "origin/${JCM_REV}" 2>/dev/null \
        || git -C /opt/jax-gcm checkout "${JCM_REV}") \
    && pip install --no-cache-dir -e /opt/jax-gcm

# Install the jittable Veros fork (matching .github/workflows/tests.yml) so
# Veros configurations (`+configuration=veros-earth` / `veros-double-drake`)
# work out of the box.
ARG VEROS_REF=dev/naive-fix-AD-needs-physical-justification
RUN git clone --branch "${VEROS_REF}" --single-branch \
        https://github.com/meteorologytoday/veros-jittable.git /opt/veros-jittable \
    && pip install --no-cache-dir -r /opt/veros-jittable/requirements.txt
ENV PYTHONPATH="/opt/veros-jittable"

COPY . /app
RUN if [ -d /app/_deps/jax-gcm ]; then \
        rm -rf /opt/jax-gcm \
        && mv /app/_deps/jax-gcm /opt/jax-gcm \
        && rmdir /app/_deps \
        && pip install --no-cache-dir -e /opt/jax-gcm; \
    fi
ARG EXTRAS=""
RUN pip install --no-cache-dir -e ".${EXTRAS:+[$EXTRAS]}"

# Replace the CPU JAX pulled in above with the CUDA 12 build. The cuda12
# extra bundles the NVIDIA wheels JAX needs at runtime. The version range
# MUST stay below 0.11.2 (see pyproject.toml): jax 0.11.2 removed
# `jax.experimental.hijax.HiPrimitive`, breaking released flax at import time.
RUN pip install --no-cache-dir --upgrade "jax[cuda12]>=0.10,<0.11"

# Run the JEM Hydra CLI by default. Anything passed after the image name on
# `docker run` is forwarded as arguments to `python -m jem.main`, so Hydra
# overrides (e.g. `+configuration=veros-earth coupled_run=short_run`) work
# out of the box. Override the entrypoint (`--entrypoint bash`) to drop into a shell.
ENTRYPOINT ["python", "-m", "jem.main"]
CMD []
