"""
Post-hoc model package builder and loader.

Packages are ordinary directories or archives (``.zip``, ``.tar``,
``.tar.gz``, ``.tgz``) with a root ``package.yaml`` manifest. All artifact
paths inside the manifest are relative package members so packages can be
rsynced or archived without preserving their original absolute run paths.
"""
from __future__ import annotations

import getpass
import json
import os
import shutil
import socket
import tarfile
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

import scriptconfig as scfg
import yaml


MANIFEST_NAME = "package.yaml"


def _now_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def _safe_read_json(fpath: Path) -> dict:
    try:
        return json.loads(fpath.read_text())
    except Exception:
        return {}


def _copy_optional(src: Optional[Path], package_root: Path, rel: str,
                   missing: list[dict], *, required: bool = False) -> Optional[str]:
    if not src:
        if required:
            missing.append({"path": None, "role": rel, "reason": "not specified"})
        return None
    src = Path(src).expanduser()
    if not src.exists():
        missing.append({"path": str(src), "role": rel, "reason": "missing"})
        if required:
            raise FileNotFoundError(src)
        return None
    dst = package_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return rel


def _find_checkpoint(workdir: Path) -> Optional[Path]:
    for pattern in ["best*.pth", "checkpoint*.pth", "*.pth"]:
        cands = sorted(workdir.glob(pattern))
        if cands:
            return cands[-1]
    return None


def _find_first(root: Path, patterns: Iterable[str]) -> Optional[Path]:
    for pattern in patterns:
        cands = sorted(root.glob(pattern))
        if cands:
            return cands[0]
    return None


def _write_yaml(fpath: Path, data: dict) -> None:
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(yaml.safe_dump(data, sort_keys=False, width=100))


def suggest_package_out(
    *,
    out_root: str | Path,
    dataset_slug: str,
    experiment_slug: str,
    variant: str,
    run_id: str,
    username: Optional[str] = None,
    hostname: Optional[str] = None,
    archive_ext: str = ".zip",
) -> Path:
    """Build a sync-friendly package path with separate user/host components."""
    username = username or getpass.getuser()
    hostname = hostname or socket.gethostname()
    archive_ext = archive_ext if archive_ext.startswith(".") else f".{archive_ext}"
    return (
        Path(out_root).expanduser()
        / str(dataset_slug)
        / str(experiment_slug)
        / "users"
        / str(username)
        / "hosts"
        / str(hostname)
        / str(run_id)
        / f"{variant}{archive_ext}"
    )


def _archive_dpath(src_dpath: Path, out_fpath: Path) -> Path:
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    name = out_fpath.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(out_fpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fpath in sorted(src_dpath.rglob("*")):
                if fpath.is_file():
                    zf.write(fpath, fpath.relative_to(src_dpath))
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(out_fpath, "w:gz") as tf:
            tf.add(src_dpath, arcname=".")
    elif name.endswith(".tar"):
        with tarfile.open(out_fpath, "w") as tf:
            tf.add(src_dpath, arcname=".")
    else:
        raise ValueError(f"unknown archive extension for {out_fpath}")
    return out_fpath


def _safe_extract_zip(zf: zipfile.ZipFile, dst: Path) -> None:
    dst = dst.resolve()
    for member in zf.infolist():
        target = (dst / member.filename).resolve()
        if os.path.commonpath([str(dst), str(target)]) != str(dst):
            raise RuntimeError(f"unsafe zip member path: {member.filename}")
    zf.extractall(dst)


def _safe_extract_tar(tf: tarfile.TarFile, dst: Path) -> None:
    dst = dst.resolve()
    for member in tf.getmembers():
        target = (dst / member.name).resolve()
        if os.path.commonpath([str(dst), str(target)]) != str(dst):
            raise RuntimeError(f"unsafe tar member path: {member.name}")
    tf.extractall(dst)


def _is_archive(fpath: Path) -> bool:
    name = fpath.name
    return name.endswith((".zip", ".tar", ".tar.gz", ".tgz"))


def _copytree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def build_model_package(
    *,
    workdir: Path,
    out: Path,
    trainer: str,
    variant: Optional[str] = None,
    category_names: Optional[Sequence[str]] = None,
    dataset_slug: Optional[str] = None,
    experiment_slug: Optional[str] = None,
    run_id: Optional[str] = None,
    train_kwcoco: Optional[str] = None,
    vali_kwcoco: Optional[str] = None,
    test_kwcoco: Optional[str] = None,
    metrics_fpath: Optional[Path] = None,
    allow_missing_weights: bool = False,
    score_thresh: float = 0.30,
    nms_iou_thresh: float = 0.50,
    username: Optional[str] = None,
    hostname: Optional[str] = None,
) -> Path:
    """Create a package directory or archive from an existing trainer workdir."""
    workdir = Path(workdir).expanduser().resolve()
    out = Path(out).expanduser()
    if not workdir.exists():
        raise FileNotFoundError(workdir)

    policy = _safe_read_json(workdir / "policy.json")
    variant = variant or policy.get("variant") or trainer
    if category_names is None:
        category_names = policy.get("category_names")
    if not category_names:
        # Fall back to label_list in policy if present, else the catch-all "object".
        category_names = policy.get("label_list") or ["object"]
    category_names = list(category_names)
    dataset_slug = dataset_slug or "unknown_dataset"
    experiment_slug = experiment_slug or str(policy.get("candidate_id") or variant)
    run_id = run_id or os.environ.get("SLURM_JOB_ID") or _now_id()
    username = username or getpass.getuser()
    hostname = hostname or socket.gethostname()

    tmp_ctx = tempfile.TemporaryDirectory() if _is_archive(out) else None
    if tmp_ctx is None:
        package_root = out
        if package_root.exists() and package_root.is_file():
            raise FileExistsError(package_root)
        package_root.mkdir(parents=True, exist_ok=True)
    else:
        package_root = Path(tmp_ctx.name)

    missing: list[dict] = []
    ckpt = _find_checkpoint(workdir)
    cfg = _find_first(workdir / "generated_configs", ["*.py", "*.yml", "*.yaml"])
    datasets_json = workdir / "detector_prepared" / "datasets.json"
    policy_fpath = workdir / "policy.json"
    train_log = workdir / "train.log"
    export_onnx = _find_first(workdir / "export", ["*.onnx"])
    modelspec = export_onnx.with_suffix(".modelspec.json") if export_onnx else None
    if modelspec and not modelspec.exists():
        modelspec = None
    metrics = Path(metrics_fpath).expanduser() if metrics_fpath else None

    artifacts: Dict[str, Any] = {
        "checkpoint": _copy_optional(
            ckpt, package_root, "weights/checkpoint.pth", missing,
            required=not allow_missing_weights,
        ),
        "train_config": _copy_optional(cfg, package_root, f"training_config/{cfg.name}" if cfg else "", missing),
        "datasets_json": _copy_optional(datasets_json, package_root, "training_config/datasets.json", missing),
        "policy": _copy_optional(policy_fpath, package_root, "training_config/policy.json", missing),
        "train_log": _copy_optional(train_log, package_root, "logs/train.log", missing),
        "metrics": _copy_optional(metrics, package_root, "eval/detect_metrics.json", missing),
    }
    if export_onnx:
        artifacts["onnx"] = _copy_optional(export_onnx, package_root, f"exports/{export_onnx.name}", missing)
    if modelspec:
        artifacts["modelspec"] = _copy_optional(modelspec, package_root, f"exports/{modelspec.name}", missing)

    labels = policy.get("label_list") or list(category_names)
    (package_root / "labels.json").write_text(json.dumps({"labels": labels}, indent=2))
    artifacts["labels"] = "labels.json"

    manifest = {
        "schema": "kwcoco_detector_kit.package.v1",
        "backend": "trainer_checkpoint",
        "trainer": str(trainer),
        "variant": str(variant),
        "category_names": list(category_names),
        "dataset_slug": str(dataset_slug),
        "experiment_slug": str(experiment_slug),
        "run_id": str(run_id),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "username": username,
            "hostname": hostname,
            "cwd": str(Path.cwd()),
            "source_workdir": str(workdir),
            "git_commit": _git_commit(Path.cwd()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "docker_image": os.environ.get("IMAGE_TAG"),
        },
        "training_data": {
            "train_kwcoco": str(train_kwcoco) if train_kwcoco else None,
            "vali_kwcoco": str(vali_kwcoco) if vali_kwcoco else None,
            "test_kwcoco": str(test_kwcoco) if test_kwcoco else None,
        },
        "postprocess": {
            "score_thresh": float(score_thresh),
            "nms_iou_thresh": float(nms_iou_thresh),
        },
        "artifacts": artifacts,
        "missing_optional": missing,
        "policy": policy,
    }
    _write_yaml(package_root / MANIFEST_NAME, manifest)

    examples = package_root / "inference_examples"
    examples.mkdir(exist_ok=True)
    (examples / "predict_kwcoco.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "kwcoco-detector-kit predict --package \"$(dirname \"$0\")/../package.yaml\" "
        "--src \"$1\" --dst \"${2:-pred.kwcoco.zip}\" --device \"${DEVICE:-cpu}\"\n"
    )
    (examples / "python_api.py").write_text(
        "from kwcoco_detector_kit.predict import predict_kwcoco\n"
        "predict_kwcoco(package='package.yaml', src='input.kwcoco.zip', dst='pred.kwcoco.zip')\n"
    )

    if tmp_ctx is not None:
        try:
            return _archive_dpath(package_root, out)
        finally:
            tmp_ctx.cleanup()
    return package_root


def _git_commit(cwd: Path) -> Optional[str]:
    import subprocess
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


@contextmanager
def open_package(package: str | Path) -> Iterator[tuple[Path, dict]]:
    """Yield ``(package_root, manifest)`` for a directory or archive package."""
    package = Path(package).expanduser()
    tmp_ctx = None
    if package.is_dir():
        root = package
    elif _is_archive(package):
        tmp_ctx = tempfile.TemporaryDirectory()
        root = Path(tmp_ctx.name)
        if package.name.endswith(".zip"):
            with zipfile.ZipFile(package, "r") as zf:
                _safe_extract_zip(zf, root)
        else:
            with tarfile.open(package, "r:*") as tf:
                _safe_extract_tar(tf, root)
    else:
        root = package.parent

    manifest_fpath = root / MANIFEST_NAME
    if not manifest_fpath.exists() and package.is_file() and package.suffix in {".yaml", ".yml"}:
        manifest_fpath = package
    if not manifest_fpath.exists():
        raise FileNotFoundError(f"could not find {MANIFEST_NAME} in {package}")
    manifest = yaml.safe_load(manifest_fpath.read_text()) or {}
    try:
        yield root, manifest
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def materialize_workdir(package_root: Path, manifest: dict, out_dpath: Path) -> Path:
    """Build the minimal trainer workdir expected by trainer.build_predictor."""
    out_dpath.mkdir(parents=True, exist_ok=True)
    artifacts = manifest.get("artifacts", {})
    for key in ["checkpoint"]:
        rel = artifacts.get(key)
        if rel:
            shutil.copy2(package_root / rel, out_dpath / Path(rel).name)
    cfg_rel = artifacts.get("train_config")
    if cfg_rel:
        cfg_dst = out_dpath / "generated_configs" / Path(cfg_rel).name
        cfg_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_root / cfg_rel, cfg_dst)
    policy_rel = artifacts.get("policy")
    if policy_rel:
        shutil.copy2(package_root / policy_rel, out_dpath / "policy.json")
    datasets_rel = artifacts.get("datasets_json")
    if datasets_rel:
        ds_dst = out_dpath / "detector_prepared" / "datasets.json"
        ds_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_root / datasets_rel, ds_dst)
    return out_dpath


class PackageBuildConfig(scfg.DataConfig):
    """Build a model package from an existing training workdir."""

    workdir = scfg.Value(None, required=True, help="trained candidate workdir")
    out = scfg.Value(None, help="output package directory or archive; defaults under --out-root")
    out_root = scfg.Value(None, help="root for automatic user/host-separated package paths")
    trainer = scfg.Value(None, required=True, help="trainer name")
    variant = scfg.Value(None, help="model variant")
    category_names = scfg.Value(None, help="comma-separated category names in train order")
    dataset_slug = scfg.Value(None, help="dataset identity")
    experiment_slug = scfg.Value(None, help="experiment identity")
    run_id = scfg.Value(None, help="run id")
    train_kwcoco = scfg.Value(None)
    vali_kwcoco = scfg.Value(None)
    test_kwcoco = scfg.Value(None)
    metrics = scfg.Value(None, help="optional detect_metrics.json")
    allow_missing_weights = scfg.Value(False, isflag=True)
    score_thresh = scfg.Value(0.30, type=float, help="default package inference score threshold")
    nms_iou_thresh = scfg.Value(0.50, type=float, help="recorded NMS IoU threshold")
    username = scfg.Value(None, help="provenance username; defaults to current user")
    hostname = scfg.Value(None, help="provenance hostname; defaults to current host")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        workdir = Path(config.workdir)
        policy = _safe_read_json(workdir / "policy.json")
        variant = config.variant or policy.get("variant") or str(config.trainer)
        dataset_slug = config.dataset_slug or "unknown_dataset"
        experiment_slug = config.experiment_slug or str(policy.get("candidate_id") or variant)
        run_id = config.run_id or os.environ.get("SLURM_JOB_ID") or _now_id()
        username = config.username or getpass.getuser()
        hostname = config.hostname or socket.gethostname()
        if config.out:
            out = Path(config.out)
        else:
            out_root = Path(config.out_root) if config.out_root else workdir.parent / "packages"
            out = suggest_package_out(
                out_root=out_root,
                dataset_slug=str(dataset_slug),
                experiment_slug=str(experiment_slug),
                variant=str(variant),
                run_id=str(run_id),
                username=str(username),
                hostname=str(hostname),
            )
        if config.category_names is None:
            cli_category_names = None
        elif isinstance(config.category_names, (list, tuple)):
            cli_category_names = [str(n).strip() for n in config.category_names if str(n).strip()]
        else:
            cli_category_names = [s.strip() for s in str(config.category_names).split(",") if s.strip()]
        out = build_model_package(
            workdir=workdir,
            out=out,
            trainer=str(config.trainer),
            variant=variant,
            category_names=cli_category_names,
            dataset_slug=dataset_slug,
            experiment_slug=experiment_slug,
            run_id=run_id,
            train_kwcoco=config.train_kwcoco,
            vali_kwcoco=config.vali_kwcoco,
            test_kwcoco=config.test_kwcoco,
            metrics_fpath=Path(config.metrics) if config.metrics else None,
            allow_missing_weights=bool(config.allow_missing_weights),
            score_thresh=float(config.score_thresh),
            nms_iou_thresh=float(config.nms_iou_thresh),
            username=str(username),
            hostname=str(hostname),
        )
        print(f"wrote package: {out}")
        return 0


def build_package_yaml(*, out_fpath: str | Path, **kwargs) -> Path:
    """Compatibility helper for older ONNX-centric callers.

    New code should call :func:`build_model_package`. This helper keeps the
    old import surface alive by writing a minimal package manifest from the
    provided keyword data.
    """
    out_fpath = Path(out_fpath).expanduser()
    manifest = {
        "schema": "kwcoco_detector_kit.package.v1",
        "backend": kwargs.pop("backend", "legacy"),
        **kwargs,
    }
    _write_yaml(out_fpath, manifest)
    return out_fpath


__cli__ = PackageBuildConfig
