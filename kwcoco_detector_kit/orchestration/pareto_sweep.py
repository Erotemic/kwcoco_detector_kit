"""
Pareto sweep — iterate (variant, input_hw, train_policy) cells; for each
cell run train -> export -> eval -> bench. Record per-cell status to a TSV
index. Aggregates into the eligibility manifest at the end.

Failure #12: per-stage exit-code state machine. Each stage records its
own pass/fail; ``status="ok"`` is NEVER the default. If a stage raises,
the cell's status becomes ``fail_<stage>`` and the sweep optionally
continues to the next cell.
"""
from __future__ import annotations

import csv
import json
import os
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import scriptconfig as scfg

from kwcoco_detector_kit.trainers._registry import get_trainer


# ---------------------------------------------------------------------------
# Cell + matrix
# ---------------------------------------------------------------------------


class SweepConfig(scfg.DataConfig):
    """Drive a Pareto sweep across (variant x input_hw x train_policy) cells."""

    train_kwcoco = scfg.Value(None, required=True, help="training kwcoco bundle")
    vali_kwcoco = scfg.Value(None, required=True, help="validation kwcoco bundle")
    test_kwcoco = scfg.Value(None, required=True, help="test kwcoco bundle (for eval stage)")
    kcd_root = scfg.Value(None, help="root for runs/, eval/, sweeps/. Defaults to $KCD_ROOT")

    trainer = scfg.Value("mock_tiny", help="trainer plugin name")
    matrix = scfg.Value(
        None,
        help=(
            "YAML/JSON file or inline string describing the sweep matrix. "
            "Format: list of {variant, input_hw: [H, W], train_policy} dicts. "
            "If unset, sweeps a single cell using --variant/--input_hw/--train_policy."
        ),
    )
    variant = scfg.Value("mock_tiny", help="single-cell fallback")
    input_hw = scfg.Value([256, 256], help="single-cell fallback (HxW)")
    train_policy = scfg.Value("fixed", help="single-cell fallback")

    num_epochs = scfg.Value(2)
    batch_size = scfg.Value(2)
    val_batch_size = scfg.Value(2)
    num_classes = scfg.Value(1)
    category_name = scfg.Value("widget")
    lr = scfg.Value(1e-2)
    backbone_lr = scfg.Value(1e-2)
    use_amp = scfg.Value(False)
    scale_tier = scfg.Value("S")
    num_gpus = scfg.Value(1)
    distributed = scfg.Value(False, isflag=True, help="enable torch.distributed.run for num_gpus > 1")

    keep_going = scfg.Value(True, isflag=True, help="continue past failed cells")
    do_export = scfg.Value(True, isflag=True, help="run ONNX export per cell")
    do_eval = scfg.Value(True, isflag=True, help="run kwcoco eval per cell")
    do_bench = scfg.Value(True, isflag=True, help="run ONNX desktop bench per cell")
    force_train = scfg.Value(False, isflag=True, help="re-run training even if best_*.pth exists")
    force_export = scfg.Value(False, isflag=True, help="re-run export even if a plausible .onnx exists")
    force_eval = scfg.Value(False, isflag=True, help="re-run eval even if detect_metrics.json exists")
    force_bench = scfg.Value(False, isflag=True, help="re-run bench even if *.bench.json exists")
    retry_failed = scfg.Value(
        None,
        help=(
            "prior sweep index.tsv; skip cells whose prior status is ok or "
            "ok_resumed, and run only missing/failed cells"
        ),
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _load_matrix(config) -> List[dict]:
    """Returns a list of cell dicts: {variant, input_hw, train_policy}."""
    if config.matrix:
        import yaml
        text = Path(str(config.matrix)).read_text() if Path(str(config.matrix)).exists() else str(config.matrix)
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, list):
            raise ValueError("matrix must be a list of cell dicts")
        return [dict(c) for c in parsed]
    # Single-cell fallback
    return [{
        "variant": str(config.variant),
        "input_hw": list(config.input_hw),
        "train_policy": str(config.train_policy),
    }]


def _candidate_id(cell) -> str:
    H, W = int(cell["input_hw"][0]), int(cell["input_hw"][1])
    return f"{cell['variant']}_{H}x{W}_{cell.get('train_policy', 'fixed')}"


def _filter_retry_failed(matrix: List[dict], prior_index) -> List[dict]:
    """Drop cells that were already ok in a prior sweep index."""
    if not prior_index:
        return matrix
    prior_index = Path(str(prior_index))
    if not prior_index.exists():
        raise FileNotFoundError(f"retry_failed index does not exist: {prior_index}")

    prior_status = {}
    with prior_index.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            prior_status[row.get("candidate_id", "")] = row.get("status", "")

    kept = []
    skipped = 0
    for cell in matrix:
        status = prior_status.get(_candidate_id(cell), "")
        if status in {"ok", "ok_resumed"}:
            skipped += 1
        else:
            kept.append(cell)
    print(
        f"[sweep] retry_failed kept {len(kept)} of {len(matrix)} cells "
        f"(skipped {skipped} already-ok cells from {prior_index})"
    )
    return kept


def _has_best_checkpoint(workdir: Path) -> bool:
    return (workdir / "best_stg2.pth").exists() or (workdir / "best_stg1.pth").exists()


def _find_plausible_onnx(workdir: Path) -> Optional[Path]:
    export_dpath = workdir / "export"
    for fpath in sorted(export_dpath.glob("*.onnx")):
        if fpath.stat().st_size >= 262144:
            return fpath
    return None


def _find_bench_json(workdir: Path) -> Optional[Path]:
    export_dpath = workdir / "export"
    found = sorted(export_dpath.glob("*.bench.json"))
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Per-stage runners
# ---------------------------------------------------------------------------


def _run_train(trainer, *, config, cell, workdir: Path, candidate_id: str) -> Path:
    # Propagate the sweep's candidate identity into the trainer's
    # policy.json so the eligibility manifest joins the sweep index +
    # policy + eval metrics on the same key.
    os.environ["KCD_CANDIDATE_ID"] = candidate_id
    cfg_fpath = trainer.generate_config(
        train_kwcoco_fpath=str(config.train_kwcoco),
        vali_kwcoco_fpath=str(config.vali_kwcoco),
        workdir=workdir,
        variant=str(cell["variant"]),
        input_hw=tuple(cell["input_hw"]),
        train_policy=str(cell.get("train_policy", "fixed")),
        num_classes=int(config.num_classes),
        batch_size=int(config.batch_size),
        val_batch_size=int(config.val_batch_size),
        num_epochs=int(config.num_epochs),
        lr=float(config.lr),
        backbone_lr=float(config.backbone_lr),
        use_amp=bool(config.use_amp),
        channels="r|g|b",
        scale_tier=str(config.scale_tier),
        num_gpus=int(config.num_gpus),
        data_format="kwcoco",
        extra={"category_name": str(config.category_name),
               "candidate_id": candidate_id},
    )
    trainer.launch(
        cfg_fpath,
        num_gpus=int(config.num_gpus),
        distributed=bool(config.distributed),
    )
    return workdir


def _run_export(trainer, *, workdir: Path, cell, force: bool = False) -> Path:
    from kwcoco_detector_kit.export.onnx import export_onnx
    return export_onnx(
        trainer=trainer,
        workdir=workdir,
        input_hw=tuple(cell["input_hw"]),
        force=force,
    )


def _run_eval(trainer, *, workdir: Path, test_kwcoco: str, kcd_root: Path,
              candidate_id: str, category_name: str, score_thresh: float = 0.30,
              force: bool = False) -> Path:
    from kwcoco_detector_kit.eval.kwcoco_eval import run_kwcoco_eval
    return run_kwcoco_eval(
        trainer=trainer,
        workdir=workdir,
        test_kwcoco=test_kwcoco,
        kcd_root=kcd_root,
        candidate_id=candidate_id,
        category_name=category_name,
        score_thresh=score_thresh,
        force=force,
    )


def _run_bench(*, workdir: Path) -> Path:
    from kwcoco_detector_kit.eval.bench import run_onnx_bench
    return run_onnx_bench(workdir=workdir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(config):
    kcd_root = Path(
        config.kcd_root or os.environ.get("KCD_ROOT")
        or (Path.home() / "data" / "kcd")
    )
    kcd_root.mkdir(parents=True, exist_ok=True)
    runs_root = kcd_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    sweeps_root = kcd_root / "sweeps"
    sweeps_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    sweep_dpath = sweeps_root / ts
    sweep_dpath.mkdir(parents=True, exist_ok=True)
    index_fpath = sweep_dpath / "index.tsv"

    trainer = get_trainer(str(config.trainer))
    matrix = _filter_retry_failed(_load_matrix(config), config.retry_failed)

    index_rows: List[dict] = []
    for cell in matrix:
        H, W = int(cell["input_hw"][0]), int(cell["input_hw"][1])
        candidate_id = _candidate_id(cell)
        workdir = runs_root / candidate_id
        workdir.mkdir(parents=True, exist_ok=True)

        row = {
            "candidate_id": candidate_id,
            "workdir": str(workdir),
            "variant": cell["variant"],
            "input_hw": f"{H}x{W}",
            "train_policy": cell.get("train_policy", "fixed"),
            "status": "",         # NEVER default to ok (#12)
            "stage_failed": "",
            "error": "",
        }
        did_any_stage = False
        enabled_stages = 1
        if bool(config.do_export):
            enabled_stages += 1
        if bool(config.do_eval):
            enabled_stages += 1
        if bool(config.do_bench):
            enabled_stages += 1

        if _has_best_checkpoint(workdir) and not bool(config.force_train):
            print(f"[sweep] {candidate_id}: skip train; best_*.pth already exists")
        else:
            did_any_stage = True
            try:
                _run_train(
                    trainer, config=config, cell=cell, workdir=workdir,
                    candidate_id=candidate_id,
                )
            except Exception as ex:
                row["status"] = "fail_train"
                row["stage_failed"] = "train"
                row["error"] = f"{type(ex).__name__}: {ex}"
                print(f"\n[sweep] {candidate_id} FAILED at train: {ex}\n{traceback.format_exc()}")
                index_rows.append(row)
                if not bool(config.keep_going):
                    break
                continue

        if bool(config.do_export):
            existing_onnx = _find_plausible_onnx(workdir)
            if existing_onnx and not bool(config.force_export):
                print(f"[sweep] {candidate_id}: skip export; {existing_onnx} already exists")
            else:
                did_any_stage = True
                try:
                    _run_export(
                        trainer, workdir=workdir, cell=cell,
                        force=bool(config.force_export),
                    )
                except Exception as ex:
                    row["status"] = "fail_export"
                    row["stage_failed"] = "export"
                    row["error"] = f"{type(ex).__name__}: {ex}"
                    print(f"\n[sweep] {candidate_id} FAILED at export: {ex}\n{traceback.format_exc()}")
                    index_rows.append(row)
                    if not bool(config.keep_going):
                        break
                    continue

        if bool(config.do_eval):
            metrics_fpath = kcd_root / "eval" / candidate_id / "eval" / "detect_metrics.json"
            if metrics_fpath.exists() and not bool(config.force_eval):
                print(f"[sweep] {candidate_id}: skip eval; {metrics_fpath} already exists")
            else:
                did_any_stage = True
                try:
                    _run_eval(
                        trainer, workdir=workdir, test_kwcoco=str(config.test_kwcoco),
                        kcd_root=kcd_root, candidate_id=candidate_id,
                        category_name=str(config.category_name),
                        force=bool(config.force_eval),
                    )
                except Exception as ex:
                    row["status"] = "fail_eval"
                    row["stage_failed"] = "eval"
                    row["error"] = f"{type(ex).__name__}: {ex}"
                    print(f"\n[sweep] {candidate_id} FAILED at eval: {ex}\n{traceback.format_exc()}")
                    index_rows.append(row)
                    if not bool(config.keep_going):
                        break
                    continue

        if bool(config.do_bench):
            bench_json = _find_bench_json(workdir)
            if bench_json and not bool(config.force_bench):
                print(f"[sweep] {candidate_id}: skip bench; {bench_json} already exists")
            else:
                did_any_stage = True
                try:
                    _run_bench(workdir=workdir)
                except Exception as ex:
                    row["status"] = "fail_bench"
                    row["stage_failed"] = "bench"
                    row["error"] = f"{type(ex).__name__}: {ex}"
                    print(f"\n[sweep] {candidate_id} FAILED at bench: {ex}\n{traceback.format_exc()}")
                    index_rows.append(row)
                    if not bool(config.keep_going):
                        break
                    continue

        row["status"] = "ok" if did_any_stage or enabled_stages == 0 else "ok_resumed"
        index_rows.append(row)
        print(f"\n[sweep] {candidate_id} {row['status']}\n")

    # Write the sweep index
    fields = ["candidate_id", "workdir", "variant", "input_hw", "train_policy",
              "status", "stage_failed", "error"]
    with index_fpath.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in index_rows:
            w.writerow(row)
    print(f"\nsweep index -> {index_fpath}")

    # Exit non-zero when any cell failed, even with --keep_going. The
    # index TSV records the cell-level pass/fail; the process exit code
    # reflects the aggregate so CI / smoke drivers can rely on it.
    failed = [r for r in index_rows if r.get("status", "").startswith("fail_")]
    if failed:
        import sys as _sys
        print(
            f"\n[sweep] {len(failed)} cell(s) failed: "
            + ", ".join(f"{r['candidate_id']} ({r['stage_failed']})" for r in failed),
            file=_sys.stderr,
        )
        # Use a clamped non-zero exit code so the shell can read it.
        _sys.exit(min(255, len(failed)))

    return index_fpath


__cli__ = SweepConfig
