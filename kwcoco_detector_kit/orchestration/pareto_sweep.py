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

import kwconf

from kwcoco_detector_kit.trainers._registry import get_trainer


# ---------------------------------------------------------------------------
# Cell + matrix
# ---------------------------------------------------------------------------


class SweepConfig(kwconf.Config):
    """Drive a Pareto sweep across (variant x input_hw x train_policy) cells."""

    train_kwcoco = kwconf.Value(None, required=True, help="training kwcoco bundle")
    vali_kwcoco = kwconf.Value(None, required=True, help="validation kwcoco bundle")
    test_kwcoco = kwconf.Value(None, required=True, help="test kwcoco bundle (for eval stage)")
    kcd_root = kwconf.Value(None, help="root for runs/, eval/, sweeps/. Defaults to $KCD_ROOT")

    trainer = kwconf.Value("mock_tiny", help="trainer plugin name")
    matrix = kwconf.Value(
        None,
        help=(
            "YAML/JSON file or inline string describing the sweep matrix. "
            "Format: list of {variant, input_hw: [H, W], train_policy} dicts. "
            "If unset, sweeps a single cell using --variant/--input_hw/--train_policy."
        ),
    )
    variant = kwconf.Value("mock_tiny", help="single-cell fallback")
    # parser=str keeps the raw string (kwconf never comma-splits). Parsing
    # happens via kwutil.Yaml.coerce in _normalize_input_hw so a YAML-style
    # "[H, W]" or a bare scalar "S" all resolve to a [H, W] pair downstream.
    input_hw = kwconf.Value("[256, 256]", parser=str, help="single-cell fallback (HxW); YAML-style list")
    train_policy = kwconf.Value("fixed", help="single-cell fallback")

    num_epochs = kwconf.Value(2)
    batch_size = kwconf.Value(2)
    val_batch_size = kwconf.Value(2)
    train_num_workers = kwconf.Value(4, help="DataLoader workers for the train loop")
    val_num_workers = kwconf.Value(2, help="DataLoader workers for the val loop")
    train_wds_shards_dpath = kwconf.Value(
        None,
        help=(
            "Optional path to a directory of kwcoco_dataloader WebDataset "
            "shards (output of `kwcoco_dataloader build_detection_webdataset`). "
            "When set, the train_dataloader's dataset becomes type "
            "WebDatasetCocoDetection (streaming from tar shards) instead of "
            "the MSCOCO CocoDetection (random-access from disk). Vali stays "
            "MSCOCO either way."
        ),
    )
    train_wds_epoch_length = kwconf.Value(
        0,
        help=(
            "Nominal samples per epoch for the WebDataset train loader. 0 = "
            "drain the shards once per epoch and stop. Only used when "
            "train_wds_shards_dpath is set."
        ),
    )
    train_wds_source_to_target = kwconf.Value(
        None,
        # parser=str keeps the raw string (kwconf never comma-splits), so a
        # JSON object string like {"B":"x","S":"y"} survives intact. We
        # json.loads() ourselves in _run_train.
        parser=str,
        help=(
            "JSON string mapping raw source category names (as written into "
            "shards by build_detection_webdataset's `source_category` field) "
            "to target class names (one of --category_names). The shard "
            "reader uses this to apply the scheme collapse on the fly. "
            "Source classes absent from the mapping are dropped from each "
            "sample's annotations."
        ),
    )
    train_wds_bucket_weights = kwconf.Value(
        None,
        parser=str,
        help=(
            "JSON object mapping bucket directory name (one of the "
            "<shards>/dominant_raw_class_EQ_*/ subdirs) to a float weight "
            "for kwcoco_dataloader.WeightedChunkMix. Unmentioned buckets "
            "fall back to the footer-derived default. Set 0.0 to exclude "
            "a bucket from training (e.g. NFS, DN). Experiment-defining: "
            "set in submit script, NOT via env."
        ),
    )
    balance_weights_fpath = kwconf.Value(
        None,
        help=(
            "Optional balance_weights.json (from "
            "kwcoco_detector_kit.data.balanced_sampler) enabling "
            "dataloader-level weighted sampling instead of the "
            "file-duplication balance. The weights must have been computed "
            "from the SAME annotation file passed as train_kwcoco "
            "(the solver hard-fails on a count mismatch). "
            "Experiment-defining: set in submit script, NOT via env."
        ),
    )
    balance_epoch_length = kwconf.Value(
        0, parser=int,
        help=(
            "Total samples per epoch (across ranks) for sampler-mode "
            "balance. 0 -> len(dataset). Weighted draws are with "
            "replacement, so epoch length is a free knob."
        ),
    )
    balance_seed = kwconf.Value(
        0, parser=int,
        help="Seed for the weighted sampler's (seed, epoch, rank) streams.",
    )
    train_wds_skip_empty = kwconf.Value(
        False,
        help=(
            "If True, drop samples whose post-scheme-collapse annotation "
            "list is empty. Default False: empty tiles are valuable "
            "negative signal for detection AP. Only set True to reproduce "
            "the pre-gen003 contract (see journal 2026-06-01). "
            "Experiment-defining: set in submit script, NOT via env."
        ),
    )
    category_names = kwconf.Value(
        "widget",
        help=(
            "comma-separated category names to train on. Order determines "
            "the class index assigned in the trained detector, which the "
            "predictor returns at eval time. num_classes is derived from "
            "the length of this list."
        ),
    )
    lr = kwconf.Value(1e-2)
    backbone_lr = kwconf.Value(1e-2)
    use_amp = kwconf.Value(False)
    scale_tier = kwconf.Value("S")
    num_gpus = kwconf.Value(1)
    distributed = kwconf.Value(False, isflag=True, help="enable torch.distributed.run for num_gpus > 1")

    init_checkpoint = kwconf.Value(
        None,
        help=(
            "Optional path to a pretrained detector checkpoint to fine-tune "
            "from (passed to the trainer's launch() as init_checkpoint). For "
            "DEIMv2 cells, this should be the variant-matched "
            "deimv2_<variant>_coco.pth file -- training from scratch on small "
            "data typically loses 5-10 AP vs. fine-tuning from a COCO init."
        ),
    )
    resume = kwconf.Value(
        None,
        help=(
            "Optional path to a *.pth checkpoint produced by a prior incomplete "
            "training run (typically last.pth). Resumes the full training state "
            "-- optimizer, epoch counter, LR scheduler, EMA -- as opposed to "
            "init_checkpoint which only loads model weights and restarts at "
            "epoch 0. Use this after a slurm walltime kill."
        ),
    )
    distractor_classes = kwconf.Value(
        None,
        help=(
            "Comma-separated list of CATEGORY NAMES that the model learns to "
            "detect (so it can discriminate them) but that the mission treats "
            "as non-targets. Classes in this list are pruned from both GT and "
            "predictions before computing detection AP, in a second eval pass "
            "that writes a sidecar detect_metrics.<distractors>.json next to "
            "the standard metrics. Eligibility selects on the sidecar when "
            "present (so model selection runs on the mission metric, not the "
            "with-distractors metric). Per-class AP on distractors is still "
            "reported in the original metrics file as a diagnostic. "
            "Use case (sea-lion project): pass northern_fur_seal so NFS "
            "predictions don't count as positive sea-lion detections."
        ),
    )
    tiled_eval = kwconf.Value(
        False, isflag=True,
        help=(
            "Run EVAL with windowed (tiled) inference instead of resizing each "
            "whole image to the model input. Slides a native-resolution window "
            "(default = model eval_spatial_size, i.e. the training tile size) "
            "over each full image and merges per-window detections with NMS. "
            "Closes the train/eval resolution gap for small objects (pups), "
            "where whole-image resize shrinks them below detectability. Costs "
            "more eval time (one forward pass per window) but does not change "
            "training. See eval/tiled_predictor.py."
        ),
    )
    tiled_eval_window = kwconf.Value(
        None, parser=int,
        help="tiled-eval window size (square, source px). Default: model eval_spatial_size.",
    )
    tiled_eval_overlap = kwconf.Value(
        0.25, help="tiled-eval fractional window overlap in [0, 0.9].")
    tiled_eval_nms_thresh = kwconf.Value(
        0.5, help="tiled-eval cross-window NMS IoU threshold.")
    tiled_eval_keep_full = kwconf.Value(
        True, isflag=True,
        help="tiled-eval: also run one whole-image pass and merge it in "
             "(protects large-object recall).")
    eval_device = kwconf.Value(
        "cpu",
        help="device for the eval inference predictor ('cpu' or 'cuda'). "
             "Default 'cpu' preserves prior behavior; set 'cuda' for tiled "
             "eval, which runs many windows per image and is impractical on "
             "CPU at scale.")
    tiled_eval_batch = kwconf.Value(
        64, help="tiled-eval: windows scored per GPU forward pass. Raise to "
                 "fill GPU memory (eval is often <2GB at 16); lower if OOM.")
    eval_read_workers = kwconf.Value(
        4, help="background threads that decode upcoming eval images so the "
                "GPU isn't starved by HDD/JPEG decode between inference "
                "spikes. 0 = sequential read.")
    keep_going = kwconf.Value(True, isflag=True, help="continue past failed cells")
    selection_journal = kwconf.Value(
        False, isflag=True,
        help=(
            "Stage EVERY epoch's checkpoint into <workdir>/staging and append "
            "a selection journal, so a detached worker can rerank them under "
            "the true-tiled protocol after training. Without it only DEIMv2's "
            "own best_stg*.pth survive, and the best checkpoint under the "
            "DEPLOYMENT geometry may not be the one in-loop validation "
            "picked. Costs ~1 checkpoint of disk per epoch."
        ),
    )
    do_export = kwconf.Value(True, isflag=True, help="run ONNX export per cell")
    do_eval = kwconf.Value(True, isflag=True, help="run kwcoco eval per cell")
    do_bench = kwconf.Value(True, isflag=True, help="run ONNX desktop bench per cell")
    force_train = kwconf.Value(False, isflag=True, help="re-run training even if best_*.pth exists")
    force_export = kwconf.Value(False, isflag=True, help="re-run export even if a plausible .onnx exists")
    force_eval = kwconf.Value(False, isflag=True, help="re-run eval even if detect_metrics.json exists")
    force_bench = kwconf.Value(False, isflag=True, help="re-run bench even if *.bench.json exists")
    retry_failed = kwconf.Value(
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


def _normalize_input_hw(value) -> List[int]:
    """Coerce --input_hw to [H, W].

    Accepts a YAML-style ``"[H, W]"`` string, a bare scalar ``"S"`` /
    ``S`` (interpreted as a square ``[S, S]``), or already-parsed
    list/tuple. Falls back to shorthand parsing for ``"HxW"`` and
    ``"H,W"`` so legacy wrappers keep working.

    Implementation uses ``kwutil.Yaml.coerce`` for the canonical case
    (YAML list) — keeps the kit's "you should be able to pass anything
    YAML-ish" convention. Falls back to pyyaml.safe_load when kwutil
    isn't installed (e.g. in a stripped-down local dev env); the
    YAML-parse semantics are identical for the strings we accept.
    """
    parsed = value
    if isinstance(value, str):
        try:
            from kwutil import Yaml
            parsed = Yaml.coerce(value)
        except ImportError:
            import yaml as _yaml
            try:
                parsed = _yaml.safe_load(value)
            except Exception:
                parsed = value
        except Exception:
            parsed = value
        # If YAML returned the string unchanged (e.g. "320x320" or
        # "320,320" — neither valid YAML), try our shorthand split.
        if isinstance(parsed, str):
            normed = parsed.replace("x", ",").replace(" ", ",")
            parsed = [int(p) for p in normed.split(",") if p.strip()]

    if isinstance(parsed, int):
        items = [parsed, parsed]
    elif isinstance(parsed, (list, tuple)):
        items = list(parsed)
    else:
        raise TypeError(
            f"input_hw must coerce to int or list; got {type(parsed).__name__}: {parsed!r}"
        )

    if len(items) == 1:
        items = [items[0], items[0]]
    if len(items) != 2:
        raise ValueError(
            f"input_hw must resolve to two values; got {items!r} from {value!r}"
        )
    return [int(items[0]), int(items[1])]


def _load_matrix(config) -> List[dict]:
    """Returns a list of cell dicts: {variant, input_hw, train_policy}."""
    if config.matrix:
        import yaml
        if Path(str(config.matrix)).exists():
            # Expand ${VAR}/${VAR:-default} for host-portable matrix files (KCD-CFG-01).
            from kwcoco_detector_kit.configs import expand_env_vars
            text = expand_env_vars(
                Path(str(config.matrix)).read_text(), source=str(config.matrix))
        else:
            text = str(config.matrix)
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, list):
            raise ValueError("matrix must be a list of cell dicts")
        return [dict(c) for c in parsed]
    # Single-cell fallback
    return [{
        "variant": str(config.variant),
        "input_hw": _normalize_input_hw(config.input_hw),
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


_TRAIN_COMPLETE_MARKER = ".train_complete"


def _train_complete(workdir: Path) -> bool:
    """True iff a prior _run_train in this workdir ran to completion.

    Distinguishes a finished checkpoint from one left behind by a crashed or
    killed (scancel / SIGKILL / power loss) training. DEIMv2 writes
    best_stg2.pth right after the FIRST eval, so even an epoch-0 kill leaves a
    best_*.pth — `_has_best_checkpoint` alone can't tell the two apart. The
    marker is written only after _run_train returns without raising, so a
    partial checkpoint never satisfies it (and the cell retrains instead of
    evaluating garbage).
    """
    return (workdir / _TRAIN_COMPLETE_MARKER).exists()


def _mark_train_complete(workdir: Path) -> None:
    try:
        (workdir / _TRAIN_COMPLETE_MARKER).write_text("ok\n")
    except OSError:
        pass


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


def _parse_category_names(raw) -> List[str]:
    if isinstance(raw, (list, tuple)):
        names = [str(n).strip() for n in raw]
    else:
        names = [s.strip() for s in str(raw).split(",")]
    names = [n for n in names if n]
    if not names:
        raise ValueError("category_names must contain at least one name")
    return names


def _run_train(trainer, *, config, cell, workdir: Path, candidate_id: str) -> Path:
    # Propagate the sweep's candidate identity into the trainer's
    # policy.json so the eligibility manifest joins the sweep index +
    # policy + eval metrics on the same key.
    os.environ["KCD_CANDIDATE_ID"] = candidate_id
    # Per-cell init_checkpoint overrides the sweep-level setting. We
    # resolve here so generate_config records it in policy.json AND
    # launch() picks it up via the same value.
    init_ckpt = cell.get("init_checkpoint") or config.init_checkpoint
    if init_ckpt:
        init_ckpt = str(init_ckpt)
        if not Path(init_ckpt).exists():
            raise FileNotFoundError(
                f"init_checkpoint {init_ckpt!r} does not exist. "
                f"Either remove the recipe's init_checkpoint to train from "
                f"scratch (expect 5-10 AP loss vs. a COCO init), or bind-mount "
                f"the path into the container."
            )
    category_names = _parse_category_names(config.category_names)
    cfg_fpath = trainer.generate_config(
        train_kwcoco_fpath=str(config.train_kwcoco),
        vali_kwcoco_fpath=str(config.vali_kwcoco),
        workdir=workdir,
        variant=str(cell["variant"]),
        input_hw=tuple(cell["input_hw"]),
        train_policy=str(cell.get("train_policy", "fixed")),
        num_classes=len(category_names),
        batch_size=int(config.batch_size),
        val_batch_size=int(config.val_batch_size),
        num_epochs=int(config.num_epochs),
        lr=float(config.lr),
        backbone_lr=float(config.backbone_lr),
        use_amp=bool(config.use_amp),
        init_checkpoint=init_ckpt,
        channels="r|g|b",
        scale_tier=str(config.scale_tier),
        num_gpus=int(config.num_gpus),
        data_format="kwcoco",
        extra={"category_names": category_names,
               "candidate_id": candidate_id,
               "init_checkpoint": init_ckpt or "",
               # The trainer has honoured this since the selection subsystem
               # landed; the sweep simply never passed it, so per-epoch staging
               # was unreachable from a normal run.
               "selection_journal_dpath": (
                   str(workdir / "staging")
                   if bool(getattr(config, "selection_journal", False))
                   else None
               ),
               "train_num_workers": int(config.train_num_workers),
               "val_num_workers": int(config.val_num_workers),
               "train_wds_shards_dpath": (
                   str(config.train_wds_shards_dpath)
                   if config.train_wds_shards_dpath else None
               ),
               "train_wds_category_names": category_names,
               "train_wds_source_to_target": (
                   json.loads(config.train_wds_source_to_target)
                   if config.train_wds_source_to_target else None
               ),
               "train_wds_epoch_length": int(config.train_wds_epoch_length or 0),
               "train_wds_bucket_weights": (
                   json.loads(config.train_wds_bucket_weights)
                   if config.train_wds_bucket_weights else None
               ),
               "train_wds_skip_empty": bool(config.train_wds_skip_empty),
               "balance_weights_fpath": (
                   str(config.balance_weights_fpath)
                   if config.balance_weights_fpath else None
               ),
               "balance_epoch_length": int(config.balance_epoch_length or 0),
               "balance_seed": int(config.balance_seed or 0)},
    )
    # init_ckpt was already resolved + validated above.
    resume_ckpt = config.resume
    if resume_ckpt is not None:
        resume_ckpt = str(resume_ckpt)
        if not Path(resume_ckpt).exists():
            raise FileNotFoundError(
                f"--resume points at {resume_ckpt!r} which does not exist."
            )
        # Resume and init_checkpoint are mutually exclusive at the
        # DEIMv2 layer (its train.py asserts not all([tuning, resume]):
        # the resume ckpt already carries the fine-tuned weights that
        # were originally produced from the init_checkpoint). When both
        # are set, resume wins. Don't pass the init_checkpoint downstream.
        if init_ckpt:
            print(
                f"[pareto_sweep] --resume set; ignoring init_checkpoint "
                f"({init_ckpt}). The resume checkpoint already carries "
                "the fine-tuned weights."
            )
            init_ckpt = None
    trainer.launch(
        cfg_fpath,
        init_checkpoint=init_ckpt,
        resume=resume_ckpt,
        num_gpus=int(config.num_gpus),
        distributed=bool(config.distributed),
    )
    return workdir


def _run_export(trainer, *, workdir: Path, cell, category_names, force: bool = False) -> Path:
    from kwcoco_detector_kit.export.onnx import export_onnx
    return export_onnx(
        trainer=trainer,
        workdir=workdir,
        input_hw=tuple(cell["input_hw"]),
        category_names=category_names,
        force=force,
    )


def _run_eval(trainer, *, workdir: Path, test_kwcoco: str, kcd_root: Path,
              candidate_id: str, category_names, score_thresh: float = 0.001,
              force: bool = False, distractor_classes=None,
              tiled_eval: bool = False, tiled_window=None,
              tiled_overlap: float = 0.25, tiled_nms_thresh: float = 0.5,
              tiled_keep_full: bool = True, tiled_batch: int = 64,
              read_workers: int = 4, device: str = "cpu") -> Path:
    from kwcoco_detector_kit.eval.kwcoco_eval import run_kwcoco_eval
    return run_kwcoco_eval(
        trainer=trainer,
        workdir=workdir,
        test_kwcoco=test_kwcoco,
        kcd_root=kcd_root,
        candidate_id=candidate_id,
        category_names=category_names,
        score_thresh=score_thresh,
        force=force,
        distractor_classes=distractor_classes,
        tiled_eval=tiled_eval,
        tiled_window=tiled_window,
        tiled_overlap=tiled_overlap,
        tiled_nms_thresh=tiled_nms_thresh,
        tiled_keep_full=tiled_keep_full,
        tiled_batch=tiled_batch,
        read_workers=read_workers,
        device=device,
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

        if _has_best_checkpoint(workdir) and _train_complete(workdir) \
                and not bool(config.force_train):
            print(f"[sweep] {candidate_id}: skip train; completed best_*.pth already exists")
        else:
            if _has_best_checkpoint(workdir) and not _train_complete(workdir):
                # A best_*.pth with no completion marker means a prior training
                # crashed or was killed (DEIMv2 writes best_stg2 after the first
                # eval, so even an epoch-0 kill leaves one). Retrain rather than
                # skip-to-eval on that partial checkpoint — evaluating a crashed
                # model is never what we want.
                print(f"[sweep] {candidate_id}: best_*.pth present but no completion "
                      f"marker (prior train crashed/killed) -> retraining")
            did_any_stage = True
            try:
                _run_train(
                    trainer, config=config, cell=cell, workdir=workdir,
                    candidate_id=candidate_id,
                )
                _mark_train_complete(workdir)
            except Exception as ex:
                row["status"] = "fail_train"
                row["stage_failed"] = "train"
                row["error"] = f"{type(ex).__name__}: {ex}"
                print(f"\n[sweep] {candidate_id} FAILED at train: {ex}\n{traceback.format_exc()}")
                index_rows.append(row)
                if not bool(config.keep_going):
                    break
                continue

        # Eval BEFORE export: eval loads the .pth via the trainer and does
        # not depend on the .onnx, whereas the ONNX export can fail on deploy-
        # only bugs (e.g. torch's dynamo ONNX exporter tripping on a DEIM op
        # pattern). Running eval first guarantees the scientific metric is
        # produced even when export dies — the deploy artifact must never
        # block the science. (bench stays after export; it needs the .onnx.)
        if bool(config.do_eval):
            metrics_fpath = kcd_root / "eval" / candidate_id / "eval" / "detect_metrics.json"
            if metrics_fpath.exists() and not bool(config.force_eval):
                print(f"[sweep] {candidate_id}: skip eval; {metrics_fpath} already exists")
            else:
                did_any_stage = True
                try:
                    distractors = None
                    if config.distractor_classes:
                        distractors = [
                            s.strip() for s in str(config.distractor_classes).split(",")
                            if s.strip()
                        ]
                    _run_eval(
                        trainer, workdir=workdir, test_kwcoco=str(config.test_kwcoco),
                        kcd_root=kcd_root, candidate_id=candidate_id,
                        category_names=_parse_category_names(config.category_names),
                        force=bool(config.force_eval),
                        distractor_classes=distractors,
                        tiled_eval=bool(config.tiled_eval),
                        tiled_window=config.tiled_eval_window,
                        tiled_overlap=float(config.tiled_eval_overlap),
                        tiled_nms_thresh=float(config.tiled_eval_nms_thresh),
                        tiled_keep_full=bool(config.tiled_eval_keep_full),
                        tiled_batch=int(config.tiled_eval_batch),
                        read_workers=int(config.eval_read_workers),
                        device=str(config.eval_device),
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

        if bool(config.do_export):
            existing_onnx = _find_plausible_onnx(workdir)
            if existing_onnx and not bool(config.force_export):
                print(f"[sweep] {candidate_id}: skip export; {existing_onnx} already exists")
            else:
                did_any_stage = True
                try:
                    _run_export(
                        trainer, workdir=workdir, cell=cell,
                        category_names=_parse_category_names(config.category_names),
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

        if bool(config.do_bench):
            bench_json = _find_bench_json(workdir)
            if bench_json and not bool(config.force_bench):
                print(f"[sweep] {candidate_id}: skip bench; {bench_json} already exists")
            elif _find_plausible_onnx(workdir) is None:
                # Bench benchmarks the ONNX model; with no .onnx (export
                # disabled, skipped, or failed) there is nothing to bench.
                # Skip rather than hard-fail — a missing deploy artifact must
                # not mark the whole cell FAILED when train+eval succeeded.
                print(f"[sweep] {candidate_id}: skip bench; no .onnx (export disabled/failed)")
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
