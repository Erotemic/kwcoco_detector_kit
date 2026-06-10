"""The detached selection worker: a fold-apply loop over the run journal.

Each :meth:`SelectionWorker.step`:

1. reads the journal and folds it (``boards.fold``) into the current
   state — leaderboards, retention, anchors, pending work;
2. scores pending ``(epoch, fingerprint)`` pairs via the injected scorer,
   appending a ``score_record`` per result (crash loses at most the
   in-flight eval; everything appended is durable);
3. applies safe GC: deletes epochs no board wants (fail-retentive — only
   after they are scored under every in-loop fingerprint), strips
   optimizer state from retained-but-not-anchor epochs;
4. after ``train_complete`` and a drained queue: scores the retained
   union under the re-rank (full-validation) bindings, builds the
   candidate × axis matrix, selects, and appends ``rerank_result``.

The worker is deliberately stateless between steps — every step re-folds
from the journal, so kills/restarts/lag need no recovery logic. The
scorer is injected so all of this is testable without a GPU.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from kwcoco_detector_kit.selection.boards import fold
from kwcoco_detector_kit.selection.config import Binding, ResolvedPlan
from kwcoco_detector_kit.selection.journal import RunJournal
from kwcoco_detector_kit.selection.rerank import build_matrix, select

__all__ = ["SelectionWorker", "strip_optimizer_state", "ckpt_hash_of"]


def ckpt_hash_of(fpath: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def strip_optimizer_state(fpath: Path) -> None:
    """Rewrite a staged checkpoint keeping only deploy/eval-relevant keys.

    AdamW moments are ~2x model params; retained checkpoints are for
    eval/deploy, never resume (``last.pth`` and the anchors keep that
    job), so the optimizer/scheduler/scaler states go. Atomic replace.
    """
    import torch
    state = torch.load(fpath, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return
    keep = {"model", "ema", "epoch", "last_epoch", "date"}
    slim = {k: v for k, v in state.items() if k in keep}
    if not slim:
        return
    tmp = fpath.with_suffix(".pth.tmp")
    torch.save(slim, tmp)
    tmp.replace(fpath)


class SelectionWorker:

    def __init__(
        self,
        workdir: Path,
        plan: ResolvedPlan,
        scorer: Callable[[Path, Binding], Dict[str, float]],
        *,
        weights_kind: str = "ema",
        strip_fn: Callable[[Path], None] = strip_optimizer_state,
        circumstances: Optional[Dict[str, Any]] = None,
        log=print,
    ):
        self.journal = RunJournal(workdir)
        self.plan = plan
        self.scorer = scorer
        self.weights_kind = weights_kind
        self.strip_fn = strip_fn
        self.circumstances = circumstances or {}
        self.log = log
        self._bindings = {
            b.fingerprint: b
            for b in [*plan.inloop_bindings, *plan.rerank_bindings]
        }
        for fp, binding in self._bindings.items():
            self.journal.register_definition(fp, binding.definition())

    # ------------------------------------------------------------------

    def _fold(self):
        return fold(
            self.journal.read_all(),
            buckets=self.plan.buckets,
            inloop_fingerprints=[b.fingerprint for b in self.plan.inloop_bindings],
            anchor_bucket=self.plan.anchor_bucket,
            anchor_top_m=self.plan.anchor_top_m,
            min_epoch=self.plan.min_epoch,
        )

    def _score_one(self, epoch: int, fp: str, ckpt_relpath: str) -> bool:
        binding = self._bindings[fp]
        ckpt_fpath = self.journal.resolve_ckpt(ckpt_relpath)
        if not ckpt_fpath.exists():
            self.log(f"[selection] epoch {epoch}: staged ckpt missing "
                     f"({ckpt_fpath}); skipping this step")
            return False
        try:
            measures = self.scorer(ckpt_fpath, binding)
        except Exception:
            self.log(f"[selection] scoring failed for epoch {epoch} under "
                     f"{binding.label}:\n{traceback.format_exc()}")
            return False
        self.journal.append_worker({
            "event": "score_record",
            "fingerprint": fp,
            "epoch": int(epoch),
            "weights_kind": self.weights_kind,
            "ckpt_hash": ckpt_hash_of(ckpt_fpath),
            "measures": {k: float(v) for k, v in measures.items()},
            "circumstances": self.circumstances,
        })
        self.log(f"[selection] scored epoch {epoch} under {binding.label}: "
                 f"{ {k: round(v, 4) for k, v in list(measures.items())[:4]} }")
        return True

    # ------------------------------------------------------------------

    def step(self) -> bool:
        """One fold-apply cycle. Returns True if any progress was made."""
        state = self._fold()
        progressed = False

        # 2. score pending in-loop work (oldest epoch first)
        for epoch, fp in state.pending:
            if self._score_one(epoch, fp, state.staged[epoch]):
                progressed = True

        # 3. GC — refold so this step's scores inform the decisions
        state = self._fold()
        for epoch in state.deletable:
            ckpt = self.journal.resolve_ckpt(state.staged[epoch])
            if ckpt.exists():
                ckpt.unlink()
            self.journal.append_worker(
                {"event": "gc", "action": "delete", "epoch": int(epoch)})
            self.log(f"[selection] gc delete epoch {epoch}")
            progressed = True
        for epoch in state.strippable:
            ckpt = self.journal.resolve_ckpt(state.staged[epoch])
            if ckpt.exists():
                try:
                    self.strip_fn(ckpt)
                except Exception:
                    self.log(f"[selection] strip failed for epoch {epoch}:\n"
                             f"{traceback.format_exc()}")
                    continue
            self.journal.append_worker(
                {"event": "gc", "action": "strip", "epoch": int(epoch)})
            self.log(f"[selection] gc strip optimizer state epoch {epoch}")
            progressed = True

        # 4. final re-rank
        state = self._fold()
        if state.train_complete and not state.pending and not state.rerank_done:
            self._rerank(state)
            progressed = True

        return progressed

    def _rerank(self, state) -> None:
        candidates = sorted(state.retained)
        if not candidates:
            self.log("[selection] rerank: no retained candidates")
            self.journal.append_worker({
                "event": "rerank_result", "winner_epoch": None,
                "policy": self.plan.rerank_policy,
                "primary_axis": self.plan.rerank_primary.axis_id,
                "matrix": {}, "frontier": [],
                "provenance": {"note": "no retained candidates"},
            })
            return

        # score the union under the full-validation bindings (cache-aware:
        # a (epoch, fingerprint) already in the journal is never re-scored)
        scores = dict(state.scores)
        for binding in self.plan.rerank_bindings:
            for epoch in candidates:
                key = (epoch, binding.fingerprint)
                if key in scores:
                    continue
                relpath = state.staged.get(epoch)
                if relpath is None:
                    continue
                if self._score_one(epoch, binding.fingerprint, relpath):
                    refolded = self._fold()
                    scores = dict(refolded.scores)

        matrix = build_matrix(
            scores, candidates, self.plan.rerank_axes,
            derived_inputs=self.plan.derived_inputs,
        )
        result = select(
            matrix,
            axes=self.plan.rerank_axes,
            policy=self.plan.rerank_policy,
            primary=self.plan.rerank_primary,
        )
        provenance = {
            "candidates": candidates,
            "buckets": [
                {"label": s.label, "fingerprint": s.fingerprint,
                 "metric": s.metric, "k": s.k}
                for s in self.plan.buckets
            ],
            "axis_labels": {a.axis_id: a.label for a in self.plan.rerank_axes},
            "weights_kind": self.weights_kind,
            **self.circumstances,
        }
        payload = {
            "event": "rerank_result",
            **result.to_jsonable(),
            "primary_axis": result.primary_axis_id,
            "provenance": provenance,
        }
        self.journal.append_worker(payload)
        out_fpath = self.journal.journal_dpath / "rerank.json"
        out_fpath.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self.log(f"[selection] rerank winner: epoch {result.winner_epoch} "
                 f"(policy={result.policy}, frontier={result.frontier}) "
                 f"-> {out_fpath}")

    # ------------------------------------------------------------------

    def run(self, *, poll_s: float = 30.0, timeout_s: Optional[float] = None) -> bool:
        """Loop until the re-rank lands. Returns True on completion."""
        t0 = time.time()
        while True:
            progressed = self.step()
            state = self._fold()
            if state.rerank_done:
                return True
            if timeout_s is not None and (time.time() - t0) > timeout_s:
                self.log("[selection] worker timeout; exiting (journal is "
                         "durable — rerun to resume)")
                return False
            if not progressed:
                time.sleep(poll_s)
