"""
mock_tiny — a deliberately small torch detector that exercises the kit's
pipeline end-to-end on CPU in seconds, without depending on DEIMv2 or
any other heavy submodule.

Why this exists
---------------
The kit's pipeline (sweep -> train -> export -> eval -> bench -> manifest)
is normally driven by a real trainer subprocess (DEIMv2 train.py +
OpenGroundingDINO train_dist.sh) that costs GPU-hours per cell. To
smoke-test the *plumbing* — generated config structure, checkpoint
discovery, ONNX export, eval driver, modelspec sidecar, manifest
aggregation — the kit registers a tiny CPU detector that:

* Has DEIMv2-shaped IO: inputs (images Nx3xHxW float32, orig_target_sizes
  Nx2 int64), outputs (labels NxK int64, boxes NxKx4 float, scores NxK
  float). Boxes in pixel coords w.r.t. orig_target_sizes.
* Trains on CPU in <60s on 8 toy images.
* Has measurable loss decrease over a handful of iterations
  (verifies gradients flow, not just code paths).
* Exports cleanly to ONNX with a fixed input shape.
* Writes the same on-disk artifacts a real DEIMv2 run does
  (``best_stg2.pth`` so the kit's checkpoint discovery works,
  ``policy.json``, ``generated_configs/train.yml``).

Architecture
------------
Hard-coded oracle priors (from the training set's GT bboxes) + a
single learnable scalar gate. The image flows through a vestigial
3x3-stride-16 conv stem; its global-pooled feature contributes a
small additive bias to the gate, so gradients still touch the image,
but the dominant control is the scalar. A few steps of "matched
queries should have score 1" pushes the gate above the score
threshold and produces real AP > 0 against the test set without
needing real backbone capacity.

This is *not* a serious detector. It exists so CI can run the entire
kit's pipeline on a 1-CPU machine in <90 s.

The trainer-plugin / predictor-plugin pair is exported via
:func:`register_trainer` at module-import time.
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import scriptconfig as scfg
import yaml

from kwcoco_detector_kit.trainers._registry import register_trainer


NUM_QUERIES_DEFAULT = 16
NUM_CLASSES_DEFAULT = 1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _build_model(num_queries: int = NUM_QUERIES_DEFAULT, prior_boxes_norm=None):
    """Build the oracle-priors + scalar-gate detector.

    ``prior_boxes_norm`` is a (K, 4) tensor in normalised [0, 1] xyxy. These
    are typically derived from the training set's GT boxes (see
    :func:`_collect_prior_boxes`). They are baked in as a non-learnable
    buffer; the only learnable parameter of consequence is the single
    scalar ``gate_logit``.
    """
    import torch
    import torch.nn as nn

    K = int(num_queries)
    if prior_boxes_norm is None:
        side = max(1, int(round(K ** 0.5)))
        ys, xs = torch.meshgrid(
            torch.linspace(0.20, 0.80, side),
            torch.linspace(0.20, 0.80, side),
            indexing="ij",
        )
        cxcy = torch.stack([xs.flatten(), ys.flatten()], dim=-1)
        if cxcy.shape[0] < K:
            cxcy = torch.cat([cxcy, torch.full((K - cxcy.shape[0], 2), 0.5)], dim=0)
        cxcy = cxcy[:K]
        wh = torch.full_like(cxcy, 0.20)
        xy0 = (cxcy - wh / 2).clamp(0, 1)
        xy1 = (cxcy + wh / 2).clamp(0, 1)
        prior_boxes_norm = torch.cat([xy0, xy1], dim=-1)

    prior_boxes_norm = prior_boxes_norm.float().clamp(0, 1)
    if prior_boxes_norm.shape != (K, 4):
        raise ValueError(
            f"prior_boxes_norm must be shape ({K}, 4), got {tuple(prior_boxes_norm.shape)}"
        )

    class MockTinyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = K
            self.register_buffer("priors_xyxy_norm", prior_boxes_norm.clone())
            self.gate_logit = nn.Parameter(torch.tensor(-2.5))
            self.stem = nn.Conv2d(3, 1, kernel_size=3, stride=16, padding=1)
            self.image_bias_scale = nn.Parameter(torch.tensor(0.05))

        def forward(self, images, orig_target_sizes):
            import torch
            import torch.nn.functional as F

            N = images.shape[0]
            feat = F.relu(self.stem(images))
            img_bias = F.adaptive_avg_pool2d(feat, 1).flatten(1).squeeze(-1)  # N
            gate = self.gate_logit + self.image_bias_scale * img_bias        # N
            scores_scalar = torch.sigmoid(gate)                              # N
            scores = scores_scalar.unsqueeze(-1).expand(N, self.K)           # NxK
            sizes_f = orig_target_sizes.float()                              # Nx2 [W, H]
            scale = torch.stack(
                [sizes_f[:, 0], sizes_f[:, 1], sizes_f[:, 0], sizes_f[:, 1]],
                dim=-1,
            )                                                                 # Nx4
            boxes = self.priors_xyxy_norm.unsqueeze(0) * scale.unsqueeze(1)  # NxKx4
            labels = torch.zeros_like(scores, dtype=torch.long)
            return labels, boxes, scores

    return MockTinyDetector()


def _collect_prior_boxes(kwcoco_fpath, category_name: str = "widget",
                         num_queries: int = NUM_QUERIES_DEFAULT):
    """Derive K oracle prior boxes in normalised xyxy from a kwcoco bundle."""
    import kwcoco
    import torch

    K = int(num_queries)
    dset = kwcoco.CocoDataset.coerce(str(kwcoco_fpath))
    cats_by_name = {c["name"]: c["id"] for c in dset.dataset.get("categories", [])}
    target_cid = cats_by_name.get(category_name)

    priors = []
    for ann in dset.annots().objs:
        if target_cid is not None and ann.get("category_id") != target_cid:
            continue
        bbox = ann.get("bbox")
        if not bbox:
            continue
        gid = ann["image_id"]
        img = dset.imgs[gid]
        W = float(img.get("width", 0))
        H = float(img.get("height", 0))
        if W <= 0 or H <= 0:
            continue
        bx, by, bw, bh = bbox
        x1 = max(0.0, min(1.0, bx / W))
        y1 = max(0.0, min(1.0, by / H))
        x2 = max(0.0, min(1.0, (bx + bw) / W))
        y2 = max(0.0, min(1.0, (by + bh) / H))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        priors.append([x1, y1, x2, y2])
        if len(priors) >= K:
            break

    while len(priors) < K:
        priors.append([0.30, 0.30, 0.70, 0.70])

    return torch.tensor(priors[:K], dtype=torch.float32)


def _matched_loss(pred_boxes_xyxy, pred_scores, gt_boxes_xyxy, gt_present, orig_sizes):
    """Image-level objectness BCE — what the scalar-gate model can learn."""
    import torch
    import torch.nn.functional as F

    has_gt = gt_present.any(dim=1).float()  # N
    img_scores = pred_scores[:, 0].clamp(1e-6, 1 - 1e-6)
    obj_loss = F.binary_cross_entropy(img_scores, has_gt, reduction="mean")
    box_loss = torch.zeros((), device=pred_boxes_xyxy.device)
    return obj_loss + box_loss, float(box_loss.item()), float(obj_loss.item())


def _coco_to_batches(kwcoco_fpath, category_name, input_h, input_w,
                     batch_size=2, max_gt=8, shuffle=True, seed=0):
    """Yield (images, orig_sizes, gt_boxes, gt_present) per batch."""
    import kwcoco
    import kwimage
    import numpy as np
    import torch

    rng = np.random.RandomState(int(seed))
    dset = kwcoco.CocoDataset.coerce(str(kwcoco_fpath))
    cats_by_name = {c["name"]: c["id"] for c in dset.dataset.get("categories", [])}
    target_cid = cats_by_name.get(category_name)
    img_ids = list(dset.images())
    if shuffle:
        rng.shuffle(img_ids)

    for start in range(0, len(img_ids), batch_size):
        batch_gids = img_ids[start:start + batch_size]
        if not batch_gids:
            break
        imgs, sizes, gts, gts_present = [], [], [], []
        for gid in batch_gids:
            try:
                arr = dset.coco_image(gid).imdelay().finalize()
            except Exception:
                continue
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.shape[2] == 4:
                arr = arr[..., :3]
            orig_h, orig_w = arr.shape[:2]
            try:
                resized = kwimage.imresize(arr, dsize=(input_w, input_h), interpolation="area")
            except NotImplementedError:
                resized = kwimage.imresize(arr, dsize=(input_w, input_h), interpolation="linear")
            chw = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
            imgs.append(chw)
            sizes.append([orig_w, orig_h])
            anns = [
                a for a in dset.annots(gid=gid).objs
                if (target_cid is None or a.get("category_id") == target_cid)
                and a.get("bbox") is not None
            ]
            gt_xyxy = np.zeros((max_gt, 4), dtype=np.float32)
            gt_pres = np.zeros((max_gt,), dtype=np.bool_)
            for k, ann in enumerate(anns[:max_gt]):
                bx, by, bw, bh = ann["bbox"]
                gt_xyxy[k] = [bx, by, bx + bw, by + bh]
                gt_pres[k] = True
            gts.append(gt_xyxy)
            gts_present.append(gt_pres)
        if not imgs:
            continue
        yield (
            torch.from_numpy(np.stack(imgs)),
            torch.tensor(sizes, dtype=torch.int64),
            torch.from_numpy(np.stack(gts)),
            torch.from_numpy(np.stack(gts_present)),
        )


# ---------------------------------------------------------------------------
# Predictor adapter (kwcoco_detector_kit.predictors._interface.DetectorPredictor)
# ---------------------------------------------------------------------------


class MockTinyPredictor:
    """Inference adapter for a mock_tiny checkpoint."""

    def __init__(self, ckpt_fpath, config_fpath=None, device: str = "cpu"):
        import torch

        ckpt = torch.load(str(ckpt_fpath), map_location=device, weights_only=False)
        meta = ckpt.get("meta", {})
        self._H = int(meta.get("input_h", 256))
        self._W = int(meta.get("input_w", 256))
        K = int(meta.get("num_queries", NUM_QUERIES_DEFAULT))
        priors = meta.get("prior_boxes_norm")
        priors_t = (torch.tensor(priors, dtype=torch.float32) if priors is not None else None)
        self._model = _build_model(num_queries=K, prior_boxes_norm=priors_t)
        self._model.load_state_dict(ckpt["model"])
        self._model.to(device).eval()
        self._device = device
        self._score_thresh = float(meta.get("score_thresh", 0.30))

    @property
    def eval_spatial_size(self) -> Tuple[int, int]:
        return (self._H, self._W)

    def predict_image(self, image_np, orig_size):
        import kwimage
        import numpy as np
        import torch

        if image_np.ndim == 2:
            image_np = np.repeat(image_np[..., None], 3, axis=-1)
        if image_np.shape[2] == 4:
            image_np = image_np[..., :3]
        try:
            resized = kwimage.imresize(image_np, dsize=(self._W, self._H), interpolation="area")
        except NotImplementedError:
            resized = kwimage.imresize(image_np, dsize=(self._W, self._H), interpolation="linear")
        chw = torch.from_numpy(
            (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        ).to(self._device)
        W, H = int(orig_size[0]), int(orig_size[1])
        sz = torch.tensor([[W, H]], dtype=torch.int64, device=self._device)
        with torch.no_grad():
            labels, boxes, scores = self._model(chw, sz)
        out: List[dict] = []
        b = boxes[0].cpu().numpy()
        s = scores[0].cpu().numpy()
        l = labels[0].cpu().numpy()
        for k in range(b.shape[0]):
            score = float(s[k])
            if score < self._score_thresh:
                continue
            x1, y1, x2, y2 = [float(v) for v in b[k]]
            out.append({
                "label": int(l[k]),
                "bbox_xyxy": [x1, y1, x2, y2],
                "score": score,
            })
        return out


# ---------------------------------------------------------------------------
# Trainer plugin
# ---------------------------------------------------------------------------


@register_trainer
class MockTinyTrainer:
    """CPU-only mock detector trainer plugin — wire into `kwcoco_detector_kit run-all`."""

    name = "mock_tiny"
    variants = {"mock_tiny": {"description": "CPU smoke detector with oracle priors + scalar gate"}}
    supports_onnx_export = True

    # ---------------- Protocol methods ----------------

    def generate_config(
        self,
        train_kwcoco_fpath,
        vali_kwcoco_fpath,
        workdir,
        *,
        variant: str = "mock_tiny",
        input_hw: Tuple[int, int] = (256, 256),
        train_policy: str = "fixed",
        num_classes: int = 1,
        batch_size: int = 2,
        val_batch_size: int = 2,
        num_epochs: int = 2,
        lr: float = 1e-2,
        backbone_lr: float = 1e-2,
        use_amp: bool = False,
        channels: str = "r|g|b",
        scale_tier: str = "S",
        num_gpus: int = 1,
        data_format: str = "kwcoco",
        extra: Optional[dict] = None,
    ) -> Path:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        gen_dpath = workdir / "generated_configs"
        gen_dpath.mkdir(parents=True, exist_ok=True)
        cfg = {
            "trainer": self.name,
            "variant": str(variant),
            "candidate_kind": "smoke",
            "category_name": (extra or {}).get("category_name", "widget"),
            "num_classes": int(num_classes),
            "num_queries": (extra or {}).get("num_queries", NUM_QUERIES_DEFAULT),
            "input_hw": [int(input_hw[0]), int(input_hw[1])],
            "eval_spatial_size": [int(input_hw[0]), int(input_hw[1])],
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "val_batch_size": int(val_batch_size),
            "lr": float(lr),
            "backbone_lr": float(backbone_lr),
            "use_amp": bool(use_amp),
            "scale_tier": str(scale_tier),
            "num_gpus": int(num_gpus),
            "distributed": False,
            "data_format": str(data_format),
            "channels": str(channels),
            "train_kwcoco": str(train_kwcoco_fpath),
            "vali_kwcoco": str(vali_kwcoco_fpath),
            "train_policy": str(train_policy),
            "score_thresh": (extra or {}).get("score_thresh", 0.30),
            "seed": (extra or {}).get("seed", 0),
        }
        cfg_fpath = gen_dpath / "train.yml"
        cfg_fpath.write_text(yaml.safe_dump(cfg, sort_keys=True))

        # Resolved-effective-config sidecar (mirror of v4 pattern):
        # for mock_tiny the resolved view is identical, but the file exists
        # so downstream tools that look for it find something.
        (gen_dpath / "resolved_effective_config.yml").write_text(
            yaml.safe_dump(cfg, sort_keys=True)
        )
        return cfg_fpath

    def launch(
        self,
        config_fpath,
        *,
        init_checkpoint=None,
        resume=None,
        num_gpus: int = 1,
        distributed: bool = False,
    ) -> Path:
        """Train in-process (no subprocess); CPU only. Returns workdir."""
        import torch

        cfg_fpath = Path(config_fpath)
        cfg = yaml.safe_load(cfg_fpath.read_text())
        workdir = cfg_fpath.parent.parent  # workdir/generated_configs/train.yml -> workdir
        _train_inproc(cfg, workdir, resume=resume)
        return workdir

    def find_checkpoint(self, workdir) -> Path:
        workdir = Path(workdir)
        ckpt = workdir / "best_stg2.pth"
        if ckpt.exists():
            return ckpt
        cands = sorted(workdir.glob("checkpoint*.pth"))
        if cands:
            return cands[-1]
        raise FileNotFoundError(f"no mock_tiny checkpoint in {workdir}")

    def supports_dynamic_input(self, variant: str) -> bool:
        return True  # the mock is shape-agnostic

    def memory_tier_default_batch(self, variant, input_hw, total_vram_gb) -> int:
        return 2  # CPU-friendly default

    def supports_webdataset_input(self) -> bool:
        return False

    def build_predictor(self, workdir, *, device: str = "cpu"):
        workdir = Path(workdir)
        ckpt = self.find_checkpoint(workdir)
        return MockTinyPredictor(ckpt, device=device)


# ---------------------------------------------------------------------------
# In-process training (called from launch())
# ---------------------------------------------------------------------------


def _train_inproc(cfg: dict, workdir: Path, *, resume=None) -> Path:
    """Run a tiny training loop given the parsed config dict."""
    import time as _time
    import torch

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg.get("seed", 0)))

    input_h, input_w = cfg["input_hw"]
    category_name = cfg.get("category_name", "widget")
    num_queries = int(cfg.get("num_queries", NUM_QUERIES_DEFAULT))

    prior_boxes = _collect_prior_boxes(cfg["train_kwcoco"], category_name, num_queries)
    model = _build_model(num_queries=num_queries, prior_boxes_norm=prior_boxes)
    if resume is not None and Path(str(resume)).exists():
        state = torch.load(str(resume), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])

    # torch 2.10+ lazily imports torch._dynamo from inside `add_param_group`
    # on first optimizer construction. That chain pulls in sympy (slow on
    # a cold filesystem cache, ~20-30s). Print a visible "loading" line so
    # users don't mistake the wait for a hang.
    print("  loading torch optimizer machinery (one-time)... ", end="", flush=True)
    _t0 = _time.perf_counter()
    try:
        import torch._dynamo  # noqa: F401
    except Exception:
        pass
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    print(f"ok ({_time.perf_counter() - _t0:.1f}s)")
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"mock_tiny train: {n_params} parameters")

    history: List[Tuple[int, int, float, float]] = []
    for epoch in range(int(cfg["num_epochs"])):
        for step, (imgs, sizes, gt_xyxy, gt_present) in enumerate(_coco_to_batches(
            cfg["train_kwcoco"], category_name, int(input_h), int(input_w),
            batch_size=int(cfg["batch_size"]), shuffle=True,
            seed=int(cfg.get("seed", 0)) + epoch,
        )):
            opt.zero_grad()
            _, pred_boxes, pred_scores = model(imgs, sizes)
            loss, lbox, lobj = _matched_loss(pred_boxes, pred_scores, gt_xyxy, gt_present, sizes)
            loss.backward()
            opt.step()
            history.append((epoch, step, float(loss.item()), lbox))
            print(f"  epoch={epoch} step={step} loss={loss.item():.4f} "
                  f"(box={lbox:.4f} obj={lobj:.4f})")

    model.eval()
    vali_losses = []
    with torch.no_grad():
        for imgs, sizes, gt_xyxy, gt_present in _coco_to_batches(
            cfg["vali_kwcoco"], category_name, int(input_h), int(input_w),
            batch_size=int(cfg["val_batch_size"]), shuffle=False,
            seed=int(cfg.get("seed", 0)),
        ):
            _, pred_boxes, pred_scores = model(imgs, sizes)
            loss, _, _ = _matched_loss(pred_boxes, pred_scores, gt_xyxy, gt_present, sizes)
            vali_losses.append(float(loss.item()))
    vali_mean = sum(vali_losses) / max(len(vali_losses), 1)
    print(f"  vali_mean_loss = {vali_mean:.4f}")

    ckpt_fpath = workdir / "best_stg2.pth"
    torch.save({
        "model": model.state_dict(),
        "meta": {
            "kind": "mock_tiny",
            "num_queries": num_queries,
            "input_h": int(input_h),
            "input_w": int(input_w),
            "score_thresh": float(cfg.get("score_thresh", 0.30)),
            "history": history,
            "vali_mean_loss": vali_mean,
            "prior_boxes_norm": prior_boxes.tolist(),
        },
    }, ckpt_fpath)
    print(f"  saved {ckpt_fpath}")

    candidate_id = os.environ.get(
        "KCD_CANDIDATE_ID",
        f"mock_tiny_{int(input_h)}x{int(input_w)}",
    )
    policy = {
        "candidate_id": candidate_id,
        "variant": cfg.get("variant", "mock_tiny"),
        "candidate_kind": "smoke",
        "run_tag": os.environ.get("KCD_RUN_TAG", "mock"),
        "export_input_h": int(input_h),
        "export_input_w": int(input_w),
        "train_resolution_policy": cfg.get("train_policy", "fixed"),
        "requested_train_resolution_min": int(input_h),
        "requested_train_resolution_max": int(input_h),
        "multiscale_base_size": int(input_h),
        "multiscale_repeat": 0,
        "multiscale_stop_epoch": int(cfg["num_epochs"]),
        "tile_training_policy": cfg.get("tile_training_policy", ""),
        "train_batch": int(cfg["batch_size"]),
        "val_batch": int(cfg["val_batch_size"]),
        "num_epochs": int(cfg["num_epochs"]),
        "lr": float(cfg["lr"]),
        "backbone_lr": float(cfg.get("backbone_lr", cfg["lr"])),
        "use_amp": bool(cfg.get("use_amp", False)),
        "init_ckpt": "",
        "generated_train_cfg": str(workdir / "generated_configs" / "train.yml"),
        "effective_train_scales": [int(input_h)],
        "effective_train_scale_min": int(input_h),
        "effective_train_scale_max": int(input_h),
    }
    (workdir / "policy.json").write_text(json.dumps(policy, indent=2))
    print(f"  wrote {workdir / 'policy.json'}")

    if len(history) >= 2:
        first, last = history[0][2], history[-1][2]
        print(f"  first_loss={first:.4f} last_loss={last:.4f} delta={(first - last):+.4f}")

    return workdir
