from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "docker" / "rfdetr" / "build_auto.sh"
RUN = REPO / "docker" / "rfdetr" / "kcd-rfdetr"


def _run(script, *args, **env_overrides):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    proc = subprocess.run(
        [str(script), *map(str, args)],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return proc.stdout


def test_rfdetr_auto_prefers_stable_cu130_on_ampere():
    out = _run(
        BUILD,
        HOST_CUDA_VERSION="13.2",
        HOST_COMPUTE_CAP="8.6",
        KCD_DOCKER_DRYRUN="1",
    )
    assert "Selected RF-DETR profile: cu130" in out
    assert "nvidia/cuda:13.0.1-runtime-ubuntu24.04" in out
    assert "download.pytorch.org/whl/cu130" in out
    assert "rfdetr-cu130" in out
    assert "nightly/cu132" not in out


def test_rfdetr_auto_uses_cu132_for_blackwell():
    out = _run(
        BUILD,
        HOST_CUDA_VERSION="13.2",
        HOST_COMPUTE_CAP="12.0",
        KCD_DOCKER_DRYRUN="1",
    )
    assert "Selected RF-DETR profile: cu132" in out
    assert "nvidia/cuda:13.2.0-runtime-ubuntu24.04" in out
    assert "download.pytorch.org/whl/nightly/cu132" in out


def test_rfdetr_runner_defaults_to_gpu0_and_preserves_key_value_args():
    out = _run(
        RUN,
        "predict",
        "--model=/tmp/model.zip",
        "--windowed=true",
        "--batch-size=16",
        KCD_DOCKER_DRYRUN="1",
    )
    assert "--gpus device=0" in out
    assert "kwcoco-detector-kit" in out
    assert "predict" in out
    assert "--model=/tmp/model.zip" in out
    assert "--windowed=true" in out
    assert "--batch-size=16" in out


def test_rfdetr_runner_gpu_override():
    out = _run(
        RUN,
        "image-info",
        KCD_DOCKER_DRYRUN="1",
        KCD_RFDETR_GPU="1",
    )
    assert "--gpus device=1" in out


def test_rfdetr_dockerfile_keeps_python_out_of_root_home():
    text = (REPO / "docker" / "rfdetr" / "Dockerfile").read_text()
    assert "UV_PYTHON_INSTALL_DIR=/opt/uv-python" in text
    assert "uv python install --install-dir ${UV_PYTHON_INSTALL_DIR}" in text
    assert 'kcd.runtime_uid_safe="1"' in text


def test_rfdetr_runner_checks_uid_safe_image_label():
    text = RUN.read_text()
    assert "kcd.runtime_uid_safe" in text
    assert "predates the UID-safe runtime contract" in text


def test_rfdetr_runner_supports_external_data_mounts(tmp_path):
    data_root = tmp_path / "external-data"
    data_root.mkdir()
    out = _run(
        RUN,
        "predict",
        f"--src={data_root / 'dataset.kwcoco.zip'}",
        f"--dst={data_root / 'pred.kwcoco.zip'}",
        KCD_DOCKER_DRYRUN="1",
        KCD_RFDETR_EXTRA_MOUNTS=str(data_root),
    )
    mount = f"--volume {data_root}:{data_root}"
    assert mount in out
