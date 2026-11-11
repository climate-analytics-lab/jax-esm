"""Unit tests for `deploy/gke/launch.py`."""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock

import yaml


class _K8sApiError(Exception):
    """Stand-in for `kubernetes.client.rest.ApiException`."""

    def __init__(self, status: int = 500):
        """Initialize with HTTP status code."""
        super().__init__(status)
        self.status = status


def _cloud_import_stubs() -> dict[str, types.ModuleType]:
    """Build scoped import stubs when optional GKE host SDKs are absent in CI."""
    required = (
        "kubernetes",
        "google.cloud.storage",
        "google.cloud.container_v1",
        "google.cloud.devtools.cloudbuild_v1",
    )
    if all(importlib.util.find_spec(mod) for mod in required):
        return {}

    stubs: dict[str, types.ModuleType] = {}
    if "google" not in sys.modules:
        google_pkg = types.ModuleType("google")
        google_pkg.__path__ = []
        stubs["google"] = google_pkg

    api_core = types.ModuleType("google.api_core")
    api_core.client_options = types.SimpleNamespace(
        ClientOptions=lambda **kw: types.SimpleNamespace(**kw)
    )
    auth = types.ModuleType("google.auth")
    auth.default = lambda scopes=None: (None, "proj")
    auth_transport = types.ModuleType("google.auth.transport")
    auth_requests = types.ModuleType("google.auth.transport.requests")
    auth_requests.Request = lambda: object()
    auth.transport = auth_transport
    auth_transport.requests = auth_requests

    cloud = types.ModuleType("google.cloud")
    cloud.__path__ = []
    cloud.container_v1 = types.SimpleNamespace(ClusterManagerClient=None)
    cloud.storage = types.SimpleNamespace(Client=None)
    devtools = types.ModuleType("google.cloud.devtools")
    build_cls = type(
        "_StubBuild",
        (),
        {
            "Status": types.SimpleNamespace(SUCCESS=3),
            "__init__": lambda self, **kw: self.__dict__.update(kw),
        },
    )
    options_cls = type(
        "_StubBuildOptions",
        (),
        {
            "LoggingMode": types.SimpleNamespace(CLOUD_LOGGING_ONLY=1),
            "MachineType": types.SimpleNamespace(E2_HIGHCPU_8=1),
            "__init__": lambda self, **kw: self.__dict__.update(kw),
        },
    )
    devtools.cloudbuild_v1 = types.SimpleNamespace(
        CloudBuildClient=None,
        Build=build_cls,
        BuildOptions=options_cls,
        Source=lambda **kw: types.SimpleNamespace(**kw),
        StorageSource=lambda **kw: types.SimpleNamespace(**kw),
        BuildStep=lambda **kw: types.SimpleNamespace(**kw),
        CreateBuildRequest=lambda **kw: types.SimpleNamespace(**kw),
    )

    k8s = types.ModuleType("kubernetes")
    k8s.__path__ = []
    k8s_client = types.ModuleType("kubernetes.client")
    k8s_client.Configuration = object
    k8s_client.ApiClient = object
    k8s_client.CoreV1Api = object
    k8s_client.BatchV1Api = object
    k8s_rest = types.ModuleType("kubernetes.client.rest")
    k8s_rest.ApiException = _K8sApiError
    k8s.client = k8s_client

    stubs.update(
        {
            "google.api_core": api_core,
            "google.auth": auth,
            "google.auth.transport": auth_transport,
            "google.auth.transport.requests": auth_requests,
            "google.cloud": cloud,
            "google.cloud.devtools": devtools,
            "kubernetes": k8s,
            "kubernetes.client": k8s_client,
            "kubernetes.client.rest": k8s_rest,
        }
    )
    return stubs


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
with mock.patch.dict(sys.modules, _cloud_import_stubs()):
    if "google" in sys.modules and "google.auth" in sys.modules and not hasattr(sys.modules["google"], "auth"):
        with mock.patch.object(sys.modules["google"], "auth", sys.modules["google.auth"], create=True):
            import deploy.gke.launch as launch
    else:
        import deploy.gke.launch as launch

DEFAULT_TEST_PROJECT = "fluid-script-453604-u5"


def _fake_cluster():
    """Build a mock GKE cluster endpoint object."""
    return types.SimpleNamespace(
        endpoint="198.51.100.10",
        master_auth=types.SimpleNamespace(
            cluster_ca_certificate=base64.b64encode(b"FAKE-CA").decode()
        ),
    )


class GkeLauncherTest(unittest.TestCase):
    """Verify multi-job manifest rendering and single-image Cloud Build launch."""

    def setUp(self):
        """Create a minimal source directory with a Dockerfile and scratch artifacts."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "jax-esm"
        self.source.mkdir()
        (self.source / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (self.source / "jem.py").write_text("x = 1\n")
        (self.source / "jem" / "data").mkdir(parents=True)
        (self.source / "jem" / "data" / "terrain_JCM_T31.nc").write_text("bc\n")
        (self.source / ".git").mkdir()
        (self.source / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (self.source / "jax-gcm").mkdir()
        (self.source / "jax-gcm" / "setup.py").write_text("skip\n")
        (self.source / "veros-jittable").mkdir()
        (self.source / "veros-jittable" / "setup.py").write_text("skip\n")
        (self.source / "outputs").mkdir()
        (self.source / "outputs" / "atm-00000000.nc").write_text("skip\n")
        (self.source / "output_T31").mkdir()
        (self.source / "output_T31" / "atm-00000000.nc").write_text("skip\n")
        (self.source / "checkpoint").mkdir()
        (self.source / "checkpoint" / "carry.msgpack").write_text("skip\n")
        (self.source / "stray.nc").write_text("skip\n")
        (self.source / "carry.msgpack").write_text("skip\n")
        (self.source / "veros.restart.h5").write_text("skip\n")
        (self.source / "run.ckpt").write_text("skip\n")

        self.jcm_source = Path(self.temp.name) / "sibling-jax-gcm"
        self.jcm_source.mkdir()
        (self.jcm_source / "pyproject.toml").write_text("[project]\nname = 'jcm'\n")
        (self.jcm_source / "jcm" / "data" / "bc").mkdir(parents=True)
        (self.jcm_source / "jcm" / "__init__.py").write_text("__version__ = '3.1.0'\n")
        (self.jcm_source / "jcm" / "data" / "bc" / "terrain.nc").write_text("bc\n")
        (self.jcm_source / ".git").mkdir()
        (self.jcm_source / ".git" / "HEAD").write_text("ref: refs/heads/dev\n")
        (self.jcm_source / "outputs").mkdir()
        (self.jcm_source / "outputs" / "state.nc").write_text("skip\n")
        (self.jcm_source / "stray.nc").write_text("skip\n")

    @contextlib.contextmanager
    def _patched_cloud(self):
        """Patch cloud and Kubernetes clients on `launch` and record calls."""
        uploaded = []
        builds = []
        jobs = []

        class _FakeStorageClient:
            def __init__(self, **kwargs):
                """Record client keyword arguments."""
                self.kwargs = kwargs

            def bucket(self, name: str):
                def _upload(blob_name: str, path: str) -> None:
                    with tarfile.open(path) as tar:
                        members = sorted(tar.getnames())
                    uploaded.append((name, blob_name, Path(path).stat().st_size, members))

                return types.SimpleNamespace(
                    blob=lambda blob_name: types.SimpleNamespace(
                        upload_from_filename=lambda path: _upload(blob_name, path)
                    )
                )

        class _FakeCloudBuildClient:
            def __init__(self, **kwargs):
                """Record client keyword arguments."""
                self.kwargs = kwargs

            def create_build(self, request):
                builds.append(request)
                return types.SimpleNamespace(
                    metadata=types.SimpleNamespace(build=types.SimpleNamespace(log_url="")),
                    result=lambda timeout=3600: types.SimpleNamespace(
                        status=launch.cloudbuild_v1.Build.Status.SUCCESS
                    ),
                )

        fake_creds = types.SimpleNamespace(valid=True, expired=False, token="tok")
        with (
            mock.patch.object(launch.google_auth, "default", return_value=(fake_creds, DEFAULT_TEST_PROJECT)),
            mock.patch.object(launch.storage, "Client", _FakeStorageClient),
            mock.patch.object(launch.cloudbuild_v1, "CloudBuildClient", _FakeCloudBuildClient),
            mock.patch.object(
                launch.container_v1,
                "ClusterManagerClient",
                return_value=types.SimpleNamespace(get_cluster=lambda name: _fake_cluster()),
            ),
            mock.patch.object(launch.k8s_client, "Configuration", lambda: types.SimpleNamespace()),
            mock.patch.object(
                launch.k8s_client,
                "ApiClient",
                lambda cfg: types.SimpleNamespace(close=lambda: None),
            ),
            mock.patch.object(
                launch.k8s_client,
                "CoreV1Api",
                lambda api: types.SimpleNamespace(
                    create_namespace=lambda body: None,
                    create_namespaced_service_account=lambda namespace, body: None,
                ),
            ),
            mock.patch.object(
                launch.k8s_client,
                "BatchV1Api",
                lambda api: types.SimpleNamespace(
                    create_namespaced_job=lambda namespace, body: jobs.append((namespace, body))
                ),
            ),
        ):
            yield types.SimpleNamespace(uploaded=uploaded, builds=builds, jobs=jobs)

    def test_archive_source_skips_git_and_run_artifacts(self):
        """Pack source and packaged `jem/data` files while skipping `.git`, checkpoints, and outputs."""
        archive = Path(self.temp.name) / "src.tar.gz"
        launch.archive_source(self.source, archive)
        with tarfile.open(archive) as tar:
            self.assertEqual(
                sorted(tar.getnames()),
                ["Dockerfile", "jem.py", "jem/data/terrain_JCM_T31.nc"],
            )

    def test_archive_source_bundles_local_jcm_source(self):
        """Stage `--jcm-source` under `_deps/jax-gcm/` with packaged `jcm/data` NetCDFs."""
        archive = Path(self.temp.name) / "src-with-jcm.tar.gz"
        launch.archive_source(self.source, archive, jcm_source=self.jcm_source)
        with tarfile.open(archive) as tar:
            self.assertEqual(
                sorted(tar.getnames()),
                [
                    "Dockerfile",
                    "_deps/jax-gcm/jcm/__init__.py",
                    "_deps/jax-gcm/jcm/data/bc/terrain.nc",
                    "_deps/jax-gcm/pyproject.toml",
                    "jem.py",
                    "jem/data/terrain_JCM_T31.nc",
                ],
            )

    def test_archive_source_rejects_invalid_jcm_source(self):
        """Fail fast when `--jcm-source` does not point at a `jax-gcm` checkout."""
        bad_dir = Path(self.temp.name) / "not-jcm"
        bad_dir.mkdir()
        archive = Path(self.temp.name) / "bad.tar.gz"
        with self.assertRaises(ValueError):
            launch.archive_source(self.source, archive, jcm_source=bad_dir)

    def test_dry_run_multi_job_shares_single_image(self):
        """Render multiple Job manifests sharing one image URI with isolated output prefixes."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = launch.main(
                [
                    "--dry-run",
                    "--prefix",
                    "sweep",
                    "--gpu",
                    "a100-40gb",
                    "--no-spot",
                    "--cmd",
                    'exec python -m jem.main +configuration=aquaplanet-slab coupled_run=short_run hydra.run.dir="$OUTPUT_DIR"',
                    "--cmd",
                    'exec python -m jem.main +configuration=earth-slab coupled_run=short_run hydra.run.dir="$OUTPUT_DIR"',
                ]
            )
        self.assertEqual(rc, 0)
        docs = list(yaml.safe_load_all(buf.getvalue()))
        self.assertEqual(len(docs), 2)

        self.assertTrue(docs[0]["metadata"]["name"].startswith("jem-"))
        self.assertTrue(docs[1]["metadata"]["name"].startswith("jem-"))
        self.assertEqual(docs[0]["metadata"]["labels"]["app"], "jem")
        self.assertEqual(
            docs[0]["spec"]["podFailurePolicy"]["rules"][1]["onExitCodes"]["containerName"],
            "jem",
        )

        img_0 = docs[0]["spec"]["template"]["spec"]["containers"][0]["image"]
        img_1 = docs[1]["spec"]["template"]["spec"]["containers"][0]["image"]
        self.assertEqual(img_0, img_1)
        self.assertNotEqual(docs[0]["metadata"]["name"], docs[1]["metadata"]["name"])

        c0 = docs[0]["spec"]["template"]["spec"]["containers"][0]
        c1 = docs[1]["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(c0["name"], "jem")
        self.assertIn("aquaplanet-slab", c0["args"][0])
        self.assertIn("earth-slab", c1["args"][0])
        env0 = {e["name"]: e["value"] for e in c0["env"]}
        env1 = {e["name"]: e["value"] for e in c1["env"]}
        self.assertEqual(env0["OUTPUT_DIR"], "/mnt/output/sweep/job-0")
        self.assertEqual(env1["OUTPUT_DIR"], "/mnt/output/sweep/job-1")
        self.assertEqual(env0["JAX_COMPILATION_CACHE_DIR"], "/mnt/output/.jem-cache/jax")
        for section in ("requests", "limits"):
            for val in c0["resources"][section].values():
                self.assertIsInstance(val, str)
        self.assertEqual(
            docs[0]["spec"]["template"]["spec"]["nodeSelector"],
            {"cloud.google.com/gke-accelerator": "nvidia-tesla-a100"},
        )

    def test_build_once_and_launch_multiple_jobs(self):
        """Build one image in Cloud Build and submit every `--cmd` Job with that image."""
        with self._patched_cloud() as rec, contextlib.redirect_stdout(io.StringIO()):
            rc = launch.main(
                [
                    "--source",
                    str(self.source),
                    "--cmd",
                    'python -m jem.main +configuration=aquaplanet-slab coupled_run.total_time="1 day"',
                    "--cmd",
                    'python -m jem.main +configuration=aquaplanet-slab coupled_run.total_time="2 days"',
                    "--cmd",
                    'python -m jem.main +configuration=aquaplanet-slab coupled_run.total_time="3 days"',
                ]
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(rec.builds), 1)
        self.assertEqual(len(rec.jobs), 3)
        build_args = list(rec.builds[0].build.steps[0].args)
        self.assertNotIn("--build-arg", build_args)

        images = {
            job["spec"]["template"]["spec"]["containers"][0]["image"]
            for _, job in rec.jobs
        }
        self.assertEqual(len(images), 1)

    def test_extras_flag_forwards_dockerfile_build_arg(self):
        """Pass `--build-arg EXTRAS=...` to Cloud Build when `--extras` is non-empty."""
        with self._patched_cloud() as rec, contextlib.redirect_stdout(io.StringIO()):
            rc = launch.main(
                [
                    "--source",
                    str(self.source),
                    "--extras",
                    "plot",
                    "--",
                    "python -m jem.main +configuration=aquaplanet-slab",
                ]
            )
        self.assertEqual(rc, 0)
        build_args = list(rec.builds[0].build.steps[0].args)
        self.assertIn("--build-arg", build_args)
        self.assertIn("EXTRAS=plot", build_args)

    def test_jcm_source_flag_uploads_staged_jcm_tree(self):
        """Upload `_deps/jax-gcm/` in the Cloud Build source archive when `--jcm-source` is set."""
        with self._patched_cloud() as rec, contextlib.redirect_stdout(io.StringIO()):
            rc = launch.main(
                [
                    "--source",
                    str(self.source),
                    "--jcm-source",
                    str(self.jcm_source),
                    "--",
                    "python -m jem.main +configuration=aquaplanet-slab",
                ]
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(rec.uploaded), 1)
        _, _, _, members = rec.uploaded[0]
        self.assertIn("_deps/jax-gcm/jcm/__init__.py", members)
        self.assertIn("_deps/jax-gcm/jcm/data/bc/terrain.nc", members)

    def test_jcm_rev_flag_forwards_dockerfile_build_arg(self):
        """Pass `--build-arg JCM_REV=...` to Cloud Build when `--jcm-rev` is provided."""
        with self._patched_cloud() as rec, contextlib.redirect_stdout(io.StringIO()):
            rc = launch.main(
                [
                    "--source",
                    str(self.source),
                    "--jcm-rev",
                    "feat/custom-physics",
                    "--",
                    "python -m jem.main +configuration=aquaplanet-slab",
                ]
            )
        self.assertEqual(rc, 0)
        build_args = list(rec.builds[0].build.steps[0].args)
        self.assertIn("--build-arg", build_args)
        self.assertIn("JCM_REV=feat/custom-physics", build_args)

    def test_conflicting_jcm_flags_rejected(self):
        """Reject `--jcm-source` combined with `--jcm-rev` or `--image`."""
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.main(
                [
                    "--source",
                    str(self.source),
                    "--jcm-source",
                    str(self.jcm_source),
                    "--jcm-rev",
                    "dev",
                    "--",
                    "python -m jem.main",
                ]
            )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.main(
                [
                    "--image",
                    "us-central1-docker.pkg.dev/proj/jax-gcm/jax-gcm-runtime:prebuilt",
                    "--jcm-rev",
                    "dev",
                    "--",
                    "python -m jem.main",
                ]
            )

    def test_separate_output_bucket_mounts_cache_bucket(self):
        """Mount the cluster's cache bucket separately when `--bucket` overrides output."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = launch.main(
                [
                    "--dry-run",
                    "--bucket",
                    "gs://custom-output-bucket",
                    "--",
                    "python -m jem.main +configuration=aquaplanet-slab",
                ]
            )
        self.assertEqual(rc, 0)
        raw = buf.getvalue()
        self.assertNotIn("${", raw)
        job = yaml.safe_load(raw)
        container = job["spec"]["template"]["spec"]["containers"][0]
        env = {e["name"]: e["value"] for e in container["env"]}
        self.assertEqual(env["JAX_COMPILATION_CACHE_DIR"], "/mnt/cache/.jem-cache/jax")
        mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
        self.assertEqual(mounts["cache"], "/mnt/cache")

    def test_existing_image_flag_skips_cloud_build(self):
        """Skip Cloud Build altogether when `--image` is provided."""
        prebuilt = "us-central1-docker.pkg.dev/proj/jax-gcm/jax-gcm-runtime:prebuilt"
        with self._patched_cloud() as rec, contextlib.redirect_stdout(io.StringIO()):
            rc = launch.main(["--image", prebuilt, "--", "python", "-m", "jem.main"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(rec.builds), 0)
        self.assertEqual(len(rec.jobs), 1)
        _, job = rec.jobs[0]
        self.assertEqual(job["spec"]["template"]["spec"]["containers"][0]["image"], prebuilt)

    def test_bucket_guard_rejects_tfstate(self):
        """Refuse `-tfstate` buckets and parent-directory traversal prefixes."""
        with self.assertRaises(ValueError):
            launch.parse_target_bucket("gs://fluid-script-453604-u5-tfstate", None, "r1")
        with self.assertRaises(ValueError):
            launch.parse_target_bucket("gs://valid-bucket", "../bad", "r1")


if __name__ == "__main__":
    unittest.main()
