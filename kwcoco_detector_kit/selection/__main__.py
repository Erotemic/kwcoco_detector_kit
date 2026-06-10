"""Selection-worker CLI: tail a training run's journal and decide.

Run beside (or after — the journal is durable and the worker resumes) a
DEIMv2 training whose generated config carries ``kcd_journal_dir``::

    python -m kwcoco_detector_kit.selection \\
        --workdir       $KCD_ROOT/runs/<candidate_id> \\
        --vali_kwcoco   .../scheme_applied/vali.kwcoco.zip \\
        --category_names pup,nonpup_sealion \\
        --distractor_classes northern_fur_seal \\
        --trains_on_tiles True \\
        --train_input_hw "[640, 640]" \\
        --num_epochs 30 \\
        --device cuda

The resolved plan (fingerprints, buckets, probe_id, disabled buckets) is
materialized to ``<workdir>/journal/selection_plan.json`` — the visible,
never-magic record of what this run's selection means.
"""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import scriptconfig as scfg


class SelectionWorkerConfig(scfg.DataConfig):
    """Run the checkpoint-selection worker over a training run's journal."""

    workdir = scfg.Value(None, required=True, help=(
        "the run dir: holds journal/, staging/, generated_configs/"))
    vali_kwcoco = scfg.Value(None, required=True, help=(
        "FULL validation kwcoco (selection never touches test)"))
    category_names = scfg.Value(None, required=True, type=str, help=(
        "ordered CSV == train-time class indices"))
    distractor_classes = scfg.Value("", type=str, help=(
        "CSV of distractor class names (the scheme-owned list half of the "
        "protocol's exclude_distractors rule)"))
    trains_on_tiles = scfg.Value(True, isflag=True, help=(
        "project-type default: tiled probe + whole lens (True) or whole "
        "lens only (False); ignored when --selection_config is given"))
    selection_config = scfg.Value(None, help=(
        "optional JSON/YAML SelectionConfig overriding the derived default"))
    train_input_hw = scfg.Value("[640, 640]", help="train input (H, W)")
    num_epochs = scfg.Value(None, required=True, type=int)
    device = scfg.Value("cuda", help="scoring device")
    poll_s = scfg.Value(30.0, type=float, help="journal poll interval")
    timeout_s = scfg.Value(None, type=float, help="optional worker timeout")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        return run_selection_worker(config)


def _git_sha(repo_dpath: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_dpath), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _class_support_of_kwcoco(fpath) -> dict:
    import kwcoco
    dset = kwcoco.CocoDataset.coerce(str(fpath))
    cat_name = {c["id"]: c["name"] for c in dset.dataset.get("categories", [])}
    support: dict = {}
    for ann in dset.dataset.get("annotations", []):
        name = cat_name.get(ann.get("category_id"))
        if name is not None:
            support[name] = support.get(name, 0) + 1
    return support


def run_selection_worker(config) -> int:
    from kwcoco_detector_kit.eval.protocols import dataset_id_of_file
    from kwcoco_detector_kit.selection.config import (
        SelectionConfig, default_selection_config, resolve_plan,
    )
    from kwcoco_detector_kit.selection.journal import RunJournal
    from kwcoco_detector_kit.selection.probe import build_probe
    from kwcoco_detector_kit.selection.scoring import KitScorer
    from kwcoco_detector_kit.selection.worker import SelectionWorker

    workdir = Path(config.workdir)
    vali_fpath = str(config.vali_kwcoco)
    category_names = [
        s.strip() for s in str(config.category_names).split(",") if s.strip()
    ]
    distractors = [
        s.strip() for s in str(config.distractor_classes or "").split(",")
        if s.strip()
    ]
    input_hw = config.train_input_hw
    if isinstance(input_hw, str):
        input_hw = json.loads(input_hw)
    input_hw = tuple(int(v) for v in input_hw)

    if config.selection_config:
        import yaml
        raw = yaml.safe_load(Path(config.selection_config).read_text())
        sel_config = SelectionConfig.from_dict(
            raw.get("checkpoint_selection", raw))
    else:
        sel_config = default_selection_config(
            trains_on_tiles=bool(config.trains_on_tiles))

    journal = RunJournal(workdir)
    journal.journal_dpath.mkdir(parents=True, exist_ok=True)

    # ---- datasets: full vali + (when configured) the frozen probe ----
    vali_id = dataset_id_of_file(vali_fpath)
    dataset_fpaths = {"vali": vali_fpath, "vali_full": vali_fpath}
    dataset_ids = {"vali": vali_id, "vali_full": vali_id}
    class_support = {}
    roles_used = {e["dataset"] for e in [*sel_config.inloop, *sel_config.buckets]}
    if "probe" in roles_used:
        probe = build_probe(
            vali_fpath,
            journal.journal_dpath / "probe",
            frames=int(sel_config.probe.get("frames", 50)),
            seed=int(sel_config.probe.get("seed", 0)),
            empty_frac=float(sel_config.probe.get("empty_frac", 0.1)),
            source_id=vali_id,
        )
        dataset_fpaths["probe"] = str(probe.probe_kwcoco_fpath)
        dataset_ids["probe"] = probe.probe_id
        class_support["probe"] = probe.manifest["class_support"]
    class_support["vali"] = _class_support_of_kwcoco(vali_fpath)
    class_support["vali_full"] = class_support["vali"]

    plan = resolve_plan(
        sel_config,
        train_input_hw=input_hw,
        dataset_fpaths=dataset_fpaths,
        dataset_ids=dataset_ids,
        num_epochs=int(config.num_epochs),
        class_support=class_support,
    )

    # materialize the resolved plan — visible, never magic
    kit_root = Path(__file__).resolve().parents[2]
    circumstances = {
        "kit_sha": _git_sha(kit_root),
        "deimv2_sha": _git_sha(kit_root / "tpl" / "DEIMv2"),
        "host": platform.node(),
    }
    plan_fpath = journal.journal_dpath / "selection_plan.json"
    plan_fpath.write_text(json.dumps({
        "config": sel_config.to_jsonable(),
        "resolved": plan.to_jsonable(),
        "datasets": {
            r: {"fpath": dataset_fpaths[r], "dataset_id": dataset_ids[r]}
            for r in dataset_fpaths
        },
        "circumstances": circumstances,
    }, indent=2, sort_keys=True))
    print(f"[selection] resolved plan -> {plan_fpath}")

    scorer = KitScorer(
        workdir=workdir,
        train_workdir=workdir,
        category_names=category_names,
        distractor_classes=distractors,
        device=str(config.device),
    )
    worker = SelectionWorker(
        workdir, plan, scorer, circumstances=circumstances)
    done = worker.run(
        poll_s=float(config.poll_s),
        timeout_s=(float(config.timeout_s) if config.timeout_s else None),
    )
    return 0 if done else 1


__cli__ = SelectionWorkerConfig

if __name__ == "__main__":
    raise SystemExit(SelectionWorkerConfig.main())
