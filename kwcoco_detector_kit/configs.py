"""
Environment and dataset config helpers.

The kit can infer many training parameters from model family, image scale,
GPU memory, and the kwcoco bundle. The two inputs that are least portable
between users are:

* environment config: where and how jobs run.
* dataset config: which kwcoco files define the training problem.

Both configs are ordinary YAML so users can edit them directly, but this
module provides small CLI helpers for initialization, inspection, and a
simple terminal editor with introspected suggestions.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import scriptconfig as scfg
import yaml


YamlDict = Dict[str, Any]


def _run_capture(cmd: List[str], *, timeout: float = 2.0) -> Optional[str]:
    """Return stdout for a short-lived command, or None if it cannot run."""
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _as_path_text(value: Optional[str]) -> Optional[str]:
    if value in {None, ""}:
        return None
    return str(Path(str(value)).expanduser())


def _coerce_yaml_scalar(text: str) -> Any:
    """Parse CLI scalar values with YAML semantics."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _split_sets(values) -> List[Tuple[str, Any]]:
    """Parse ``key=value`` CLI overrides."""
    if values is None or values == "":
        return []
    if isinstance(values, str):
        items = [values]
    else:
        items = list(values)
    parsed = []
    for item in items:
        text = str(item)
        if "=" not in text:
            raise ValueError(f"--set item must be key=value, got {text!r}")
        key, value = text.split("=", 1)
        parsed.append((key.strip(), _coerce_yaml_scalar(value.strip())))
    return parsed


def get_dotted(data: Mapping[str, Any], key: str, default=None) -> Any:
    cur: Any = data
    for part in key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_dotted(data: YamlDict, key: str, value: Any) -> None:
    cur = data
    parts = key.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def flatten_dotted(data: Mapping[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    rows: List[Tuple[str, Any]] = []
    for key in sorted(data.keys()):
        value = data[key]
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            rows.extend(flatten_dotted(value, dotted))
        elif isinstance(value, list):
            rows.append((dotted, value))
        else:
            rows.append((dotted, value))
    return rows


def read_yaml(fpath: Path) -> YamlDict:
    with Path(fpath).expanduser().open("r") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {fpath}, got {type(data).__name__}")
    return data


def write_yaml(fpath: Path, data: Mapping[str, Any]) -> Path:
    fpath = Path(fpath).expanduser()
    fpath.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        dict(data),
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
    fpath.write_text(text)
    return fpath


def introspect_environment() -> YamlDict:
    """Collect portable facts and best-effort host suggestions."""
    env = os.environ
    docker_version = _run_capture(["docker", "--version"]) if shutil.which("docker") else None
    nvcc_text = _run_capture(["nvcc", "--version"]) if shutil.which("nvcc") else None
    smi_text = None
    if shutil.which("nvidia-smi"):
        smi_text = _run_capture([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ])
    torch_info: YamlDict = {}
    try:
        import torch  # type: ignore
    except Exception as ex:
        torch_info = {"import_error": f"{type(ex).__name__}: {ex}"}
    else:
        torch_info = {
            "version": str(getattr(torch, "__version__", None)),
            "cuda": str(getattr(torch.version, "cuda", None)),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }

    nvcc_cuda = None
    if nvcc_text:
        match = re.search(r"release\s+([0-9]+\.[0-9]+)", nvcc_text)
        if match:
            nvcc_cuda = match.group(1)

    gpu_names: List[str] = []
    gpu_memory_gb: List[float] = []
    driver_version = None
    if smi_text:
        for line in smi_text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpu_names.append(parts[0])
                mem_match = re.search(r"([0-9.]+)\s*MiB", parts[1])
                if mem_match:
                    gpu_memory_gb.append(round(float(mem_match.group(1)) / 1024.0, 2))
                driver_version = parts[2]

    visible = env.get("CUDA_VISIBLE_DEVICES")
    if not visible and gpu_names:
        visible = ",".join(str(i) for i in range(len(gpu_names)))

    slurm_present = any(k.startswith("SLURM_") for k in env)
    return {
        "paths": {
            "cwd": str(Path.cwd()),
            "kit_root": str(_repo_root()),
            "python_bin": sys.executable,
            "scratch": env.get("SCRATCH"),
            "home": env.get("HOME"),
        },
        "executables": {
            "docker": shutil.which("docker"),
            "nvidia_smi": shutil.which("nvidia-smi"),
            "nvcc": shutil.which("nvcc"),
            "sbatch": shutil.which("sbatch"),
            "srun": shutil.which("srun"),
        },
        "versions": {
            "docker": docker_version,
            "nvcc_cuda": nvcc_cuda,
            "nvidia_driver": driver_version,
            "torch": torch_info,
        },
        "gpu": {
            "names": gpu_names,
            "memory_gb": gpu_memory_gb,
            "count": len(gpu_names) or int(torch_info.get("device_count", 0) or 0),
            "visible_devices": visible,
        },
        "slurm": {
            "detected": slurm_present,
            "job_id": env.get("SLURM_JOB_ID"),
            "job_gpus": env.get("SLURM_JOB_GPUS"),
            "partition": env.get("SLURM_JOB_PARTITION"),
        },
    }


def default_environment_config(*, execution: Optional[str] = None,
                               docker_image: Optional[str] = None) -> YamlDict:
    info = introspect_environment()
    scratch = get_dotted(info, "paths.scratch") or "/tmp"
    gpu_count = int(get_dotted(info, "gpu.count", 0) or 0)
    if execution is None:
        execution = "slurm-docker" if get_dotted(info, "executables.sbatch") else "docker"
        if not get_dotted(info, "executables.docker"):
            execution = "host"
    if docker_image is None:
        docker_image = "kwcoco-detector-kit:ogdino-cu132-arisia"
    gres = f"gpu:{gpu_count}" if gpu_count else "gpu:1"
    return {
        "version": 1,
        "kind": "kwcoco_detector_kit.environment",
        "environment": {
            "profile_name": os.environ.get("KCD_PROFILE", "local"),
            "execution": execution,
            "work_root": os.environ.get("KCD_ROOT", str(Path(scratch) / "kwcoco_detector_kit")),
            "cache_root": os.environ.get(
                "KCD_CACHE_ROOT",
                str(Path(scratch) / "kwcoco_detector_kit_cache"),
            ),
            "python_bin": get_dotted(info, "paths.python_bin"),
            "kit_root": get_dotted(info, "paths.kit_root"),
            "opengroundingdino_repo": os.environ.get(
                "KCD_OPENGROUNDINGDINO_REPO_DPATH",
                str(_repo_root() / "tpl" / "Open-GroundingDino"),
            ),
            "docker": {
                "image": docker_image,
                "shm_size": "32g",
                "mounts": [],
            },
            "slurm": {
                "partition": None,
                "account": None,
                "gres": gres,
                "nodes": 1,
                "ntasks_per_node": 1,
                "cpus_per_task": 32,
                "mem": "192G",
                "time": "72:00:00",
            },
            "cuda": {
                "visible_devices": get_dotted(info, "gpu.visible_devices"),
                "expected_gpu_count": gpu_count or 1,
                "expected_gpu_memory_gb": get_dotted(info, "gpu.memory_gb"),
            },
        },
        "suggestions": {
            "introspection": info,
        },
    }


def _load_coco_summary(fpath: Optional[str]) -> YamlDict:
    if not fpath:
        return {"path": None, "exists": False}
    path = Path(str(fpath)).expanduser()
    summary: YamlDict = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return summary
    try:
        import kwcoco
        dset = kwcoco.CocoDataset(str(path))
        cat_names = sorted([cat.get("name") for cat in dset.dataset.get("categories", []) if cat.get("name")])
        gids = list(dset.images())
        aids = list(dset.annots())
        summary.update({
            "n_images": len(gids),
            "n_annotations": len(aids),
            "n_categories": len(cat_names),
            "categories": cat_names,
            "bundle_dpath": str(Path(dset.bundle_dpath).resolve()),
        })
    except Exception as ex:
        summary["error"] = f"{type(ex).__name__}: {ex}"
    return summary


def introspect_dataset(*, train_kwcoco: Optional[str] = None,
                       vali_kwcoco: Optional[str] = None,
                       test_kwcoco: Optional[str] = None) -> YamlDict:
    train = _load_coco_summary(train_kwcoco)
    vali = _load_coco_summary(vali_kwcoco)
    test = _load_coco_summary(test_kwcoco)
    all_cats: List[str] = []
    for item in [train, vali, test]:
        all_cats.extend(item.get("categories", []) or [])
    categories = sorted(set(all_cats))
    return {
        "train": train,
        "validation": vali,
        "test": test,
        "categories": categories,
    }


def default_dataset_config(*, name: Optional[str] = None,
                           train_kwcoco: Optional[str] = None,
                           vali_kwcoco: Optional[str] = None,
                           test_kwcoco: Optional[str] = None,
                           category_names=None) -> YamlDict:
    info = introspect_dataset(
        train_kwcoco=train_kwcoco,
        vali_kwcoco=vali_kwcoco,
        test_kwcoco=test_kwcoco,
    )
    if name is None:
        train_path = Path(str(train_kwcoco)).expanduser() if train_kwcoco else None
        name = train_path.parent.name if train_path else "my_kwcoco_dataset"
    categories = info.get("categories", [])
    if not category_names:
        # Default to whatever the source kwcoco contains (any count).
        category_names = list(categories) or ["object"]
    return {
        "version": 1,
        "kind": "kwcoco_detector_kit.dataset",
        "dataset": {
            "name": name,
            "task": "detection",
            "category_names": list(category_names),
            "channels": "r|g|b",
            "train_kwcoco": _as_path_text(train_kwcoco),
            "vali_kwcoco": _as_path_text(vali_kwcoco),
            "test_kwcoco": _as_path_text(test_kwcoco),
            "image_root": None,
            "tiling": {
                "enabled": True,
                "mode": "multiscale",
                "tile_size": 800,
                "source_scales": [1.0, 0.5, 0.25],
                "stride_frac": 0.75,
                "keep_negative": False,
            },
        },
        "suggestions": {
            "introspection": info,
        },
    }


def make_suggestion_table(env_cfg: Optional[Mapping[str, Any]] = None,
                          dataset_cfg: Optional[Mapping[str, Any]] = None) -> List[Tuple[str, Any, str]]:
    rows: List[Tuple[str, Any, str]] = []
    if env_cfg:
        env_info = get_dotted(env_cfg, "suggestions.introspection", {}) or {}
        rows.extend([
            ("environment.execution", get_dotted(env_cfg, "environment.execution"), "runtime mode"),
            ("environment.docker.image", get_dotted(env_cfg, "environment.docker.image"), "docker image"),
            ("environment.slurm.gres", get_dotted(env_cfg, "environment.slurm.gres"), "GPU request"),
            ("environment.cuda.visible_devices", get_dotted(env_cfg, "environment.cuda.visible_devices"), "CUDA devices"),
            ("suggestions.introspection.gpu.names", get_dotted(env_info, "gpu.names"), "detected GPUs"),
            ("suggestions.introspection.versions.nvcc_cuda", get_dotted(env_info, "versions.nvcc_cuda"), "detected CUDA toolkit"),
            ("suggestions.introspection.versions.torch.cuda", get_dotted(env_info, "versions.torch.cuda"), "torch CUDA ABI"),
        ])
    if dataset_cfg:
        rows.extend([
            ("dataset.name", get_dotted(dataset_cfg, "dataset.name"), "dataset label"),
            ("dataset.category_names", get_dotted(dataset_cfg, "dataset.category_names"), "ordered class names"),
            ("dataset.train_kwcoco", get_dotted(dataset_cfg, "dataset.train_kwcoco"), "training kwcoco"),
            ("suggestions.introspection.train.n_images", get_dotted(dataset_cfg, "suggestions.introspection.train.n_images"), "train images"),
            ("suggestions.introspection.train.n_annotations", get_dotted(dataset_cfg, "suggestions.introspection.train.n_annotations"), "train annotations"),
            ("suggestions.introspection.categories", get_dotted(dataset_cfg, "suggestions.introspection.categories"), "categories"),
        ])
    return rows


def format_suggestions(rows: Iterable[Tuple[str, Any, str]]) -> str:
    lines = []
    for key, value, note in rows:
        lines.append(f"{key:55s} = {value!r}  # {note}")
    return "\n".join(lines)


def interactive_edit(configs: List[Tuple[str, Path, YamlDict]]) -> int:
    """Small curses-free textual editor for config YAML."""
    fields: List[Tuple[str, Path, YamlDict, str, Any]] = []
    for label, fpath, data in configs:
        for key, value in flatten_dotted(data):
            if key.startswith("suggestions."):
                continue
            fields.append((label, fpath, data, key, value))

    while True:
        print("\nkwcoco-detector-kit config editor")
        print("=" * 40)
        for idx, (label, _fpath, _data, key, value) in enumerate(fields, start=1):
            print(f"{idx:2d}. [{label}] {key} = {value!r}")
        print("\nCommands: number to edit, s show suggestions, w write, q quit")
        choice = input("> ").strip()
        if choice.lower() == "q":
            return 1
        if choice.lower() == "s":
            env_cfg = next((d for lbl, _p, d in configs if lbl == "env"), None)
            dataset_cfg = next((d for lbl, _p, d in configs if lbl == "dataset"), None)
            print(format_suggestions(make_suggestion_table(env_cfg, dataset_cfg)))
            continue
        if choice.lower() == "w":
            for _label, fpath, data in configs:
                write_yaml(fpath, data)
                print(f"wrote {fpath}")
            return 0
        try:
            idx = int(choice) - 1
            label, fpath, data, key, old = fields[idx]
        except (ValueError, IndexError):
            print("unrecognized selection")
            continue
        new_text = input(f"{label}.{key} [{old!r}] > ").strip()
        if not new_text:
            continue
        new_value = _coerce_yaml_scalar(new_text)
        set_dotted(data, key, new_value)
        fields[idx] = (label, fpath, data, key, new_value)


class ConfigInitConfig(scfg.DataConfig):
    """Write editable environment and dataset YAML configs."""

    env = scfg.Value("kcd.environment.yaml", help="output environment config YAML")
    dataset = scfg.Value("kcd.dataset.yaml", help="output dataset config YAML")
    train_kwcoco = scfg.Value(None, help="training kwcoco path")
    vali_kwcoco = scfg.Value(None, help="validation kwcoco path")
    test_kwcoco = scfg.Value(None, help="test kwcoco path")
    name = scfg.Value(None, help="dataset name")
    category_names = scfg.Value(None, help="comma-separated category names (train order)")
    execution = scfg.Value(None, help="host, docker, or slurm-docker")
    docker_image = scfg.Value(None, help="docker image tag to put in the environment config")
    overwrite = scfg.Value(False, isflag=True, help="replace existing YAML files")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        return init_configs(config)


class ConfigInspectConfig(scfg.DataConfig):
    """Show current config values and introspected suggestions."""

    env = scfg.Value("kcd.environment.yaml", help="environment config YAML")
    dataset = scfg.Value("kcd.dataset.yaml", help="dataset config YAML")
    refresh = scfg.Value(False, isflag=True, help="refresh suggestions from host and kwcoco files")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        return inspect_configs(config)


class ConfigEditConfig(scfg.DataConfig):
    """Edit YAML configs with a small text UI or dotted --set overrides."""

    env = scfg.Value("kcd.environment.yaml", help="environment config YAML")
    dataset = scfg.Value("kcd.dataset.yaml", help="dataset config YAML")
    set = scfg.Value([], nargs="*", help="dotted overrides, e.g. environment.slurm.gres=gpu:4")
    non_interactive = scfg.Value(False, isflag=True, help="apply --set and exit without prompting")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        return edit_configs(config)


def init_configs(config) -> int:
    env_fpath = Path(config.env).expanduser()
    dataset_fpath = Path(config.dataset).expanduser()
    for fpath in [env_fpath, dataset_fpath]:
        if fpath.exists() and not bool(config.overwrite):
            raise FileExistsError(f"{fpath} exists; pass --overwrite to replace it")
    env_cfg = default_environment_config(
        execution=config.execution,
        docker_image=config.docker_image,
    )
    if config.category_names is None:
        cli_cats = None
    elif isinstance(config.category_names, (list, tuple)):
        cli_cats = [str(n).strip() for n in config.category_names if str(n).strip()]
    else:
        cli_cats = [s.strip() for s in str(config.category_names).split(",") if s.strip()]
    dataset_cfg = default_dataset_config(
        name=config.name,
        train_kwcoco=config.train_kwcoco,
        vali_kwcoco=config.vali_kwcoco,
        test_kwcoco=config.test_kwcoco,
        category_names=cli_cats,
    )
    write_yaml(env_fpath, env_cfg)
    write_yaml(dataset_fpath, dataset_cfg)
    print(f"wrote {env_fpath}")
    print(f"wrote {dataset_fpath}")
    return 0


def _refresh_dataset_suggestions(dataset_cfg: YamlDict) -> None:
    dataset = dataset_cfg.get("dataset", {})
    dataset_cfg.setdefault("suggestions", {})["introspection"] = introspect_dataset(
        train_kwcoco=dataset.get("train_kwcoco"),
        vali_kwcoco=dataset.get("vali_kwcoco"),
        test_kwcoco=dataset.get("test_kwcoco"),
    )


def inspect_configs(config) -> int:
    env_cfg = read_yaml(Path(config.env))
    dataset_cfg = read_yaml(Path(config.dataset))
    if bool(config.refresh):
        env_cfg.setdefault("suggestions", {})["introspection"] = introspect_environment()
        _refresh_dataset_suggestions(dataset_cfg)
    print(format_suggestions(make_suggestion_table(env_cfg, dataset_cfg)))
    return 0


def edit_configs(config) -> int:
    env_fpath = Path(config.env).expanduser()
    dataset_fpath = Path(config.dataset).expanduser()
    env_cfg = read_yaml(env_fpath)
    dataset_cfg = read_yaml(dataset_fpath)
    configs = [("env", env_fpath, env_cfg), ("dataset", dataset_fpath, dataset_cfg)]
    overrides = _split_sets(config.set)
    for key, value in overrides:
        if key.startswith("dataset."):
            set_dotted(dataset_cfg, key, value)
        elif key.startswith("environment.") or key in {"version", "kind"}:
            set_dotted(env_cfg, key, value)
        else:
            raise KeyError(
                f"Ambiguous config override {key!r}; prefix it with dataset. or environment."
            )
    if bool(config.non_interactive) or overrides:
        write_yaml(env_fpath, env_cfg)
        write_yaml(dataset_fpath, dataset_cfg)
        print(f"wrote {env_fpath}")
        print(f"wrote {dataset_fpath}")
        return 0
    return interactive_edit(configs)
