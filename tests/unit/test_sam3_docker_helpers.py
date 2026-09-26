from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "docker" / "sam3" / "build.sh"
RUN = REPO / "docker" / "sam3" / "kcd-sam3"


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


def test_sam3_build_dryrun_uses_ampere_compatible_stack():
    out = _run(BUILD, KCD_DOCKER_DRYRUN="1")
    assert "nvidia/cuda:12.6.3-runtime-ubuntu24.04" in out
    assert "download.pytorch.org/whl/cu126" in out
    assert "TORCH_VERSION=2.7.1" in out
    assert "kwcoco-detector-kit:sam3-auto" in out


def test_sam3_runner_admission_preserves_kwconf_arguments():
    out = _run(
        RUN,
        "admission",
        "--src=/tmp/data.kwcoco.zip",
        "--out_dpath=/tmp/sam3-out",
        "--windowed=true",
        "--review=true",
        KCD_DOCKER_DRYRUN="1",
    )
    assert "--gpus device=0" in out
    assert "kwcoco-detector-kit" in out
    assert "sam3-admission" in out
    assert "--src=/tmp/data.kwcoco.zip" in out
    assert "--windowed=true" in out
    assert "--review=true" in out
    assert "--volume /:/" not in out


def test_sam3_runner_gpu_and_hf_overrides():
    out = _run(
        RUN,
        "image-info",
        KCD_DOCKER_DRYRUN="1",
        KCD_SAM3_GPU="1",
        HF_TOKEN="test-token",
        HF_HOME="/tmp/hf-home-does-not-exist-yet",
    )
    assert "--gpus device=1" in out
    assert "HF_TOKEN=test-token" in out
    assert "HF_HOME=/tmp/hf-home-does-not-exist-yet" in out


def test_sam3_dockerfile_uses_pinned_fork_and_does_not_bake_checkpoint():
    text = (REPO / "docker" / "sam3" / "Dockerfile").read_text()
    assert "COPY tpl/sam3 ./tpl/sam3" in text
    assert "uv pip install -e ./tpl/sam3" in text
    assert "torch==${TORCH_VERSION}" in text
    assert "torchvision==${TORCHVISION_VERSION}" in text
    assert "hf_hub_download" not in text
    assert "facebook/sam3.1" not in text
    assert 'kcd.runtime_uid_safe="1"' in text
    assert 'kcd.sam3_admission="1"' in text
    assert 'kcd.kwimage_ext_rust="1"' in text
    assert 'docker/requirements-kwstack.txt' in text
    assert 'KWIMAGE_EXT_FORCE_RUST=1' in text
    assert "backend['kind'] == 'rust'" in text


def test_shared_kwstack_includes_released_rust_kwimage_ext():
    text = (REPO / "docker" / "requirements-kwstack.txt").read_text()
    assert "kwimage_ext==0.4.1" in text
    rfdetr = (REPO / "docker" / "rfdetr" / "Dockerfile").read_text()
    sam3 = (REPO / "docker" / "sam3" / "Dockerfile").read_text()
    assert "docker/requirements-kwstack.txt" in rfdetr
    assert "docker/requirements-kwstack.txt" in sam3


def test_sam3_image_info_reports_rust_backend():
    out = _run(RUN, "image-info", KCD_DOCKER_DRYRUN="1")
    assert "kwimage-ext" in out
    assert "cpu_nms.backend_metadata" in out


def test_sam3_exec_keeps_stdin_open():
    out = _run(RUN, "exec", "python", "-", KCD_DOCKER_DRYRUN="1")
    assert "--interactive" in out
