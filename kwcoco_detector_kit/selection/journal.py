"""Append-only run journal: the single source of truth for a run.

Layout under ``<workdir>/journal/``::

    train.jsonl        # written ONLY by the training process
    worker.jsonl       # written ONLY by the selection worker
    definitions.json   # content-addressed fingerprint -> definition store

Single-writer-per-file keeps appends safe without locking. Every event
carries ``event`` (kind), ``seq`` (per-file monotonic), and ``time``.

Event kinds (``event`` field):

train.jsonl
    ``epoch_staged``    {epoch, ckpt}    ckpt is workdir-relative
    ``train_complete``  {}

worker.jsonl
    ``score_record``    {fingerprint, epoch, weights_kind, ckpt_hash,
                         measures, circumstances}
    ``gc``              {action: "delete"|"strip", epoch}
    ``rerank_result``   {winner_epoch, policy, primary_axis, matrix,
                         frontier, provenance}

The journal is the provenance record; leaderboards, retention state, and
the re-rank are all *folds* over it (see ``boards.py``) and are always
recomputable. Scores are immutable: a (fingerprint, epoch) pair is scored
at most once.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


class RunJournal:

    def __init__(self, workdir: Union[str, Path]):
        self.workdir = Path(workdir)
        self.journal_dpath = self.workdir / "journal"
        self.train_fpath = self.journal_dpath / "train.jsonl"
        self.worker_fpath = self.journal_dpath / "worker.jsonl"
        self.definitions_fpath = self.journal_dpath / "definitions.json"
        self.staging_dpath = self.workdir / "staging"

    # -- writing ------------------------------------------------------------

    def _append(self, fpath: Path, event: Dict[str, Any]) -> Dict[str, Any]:
        self.journal_dpath.mkdir(parents=True, exist_ok=True)
        seq = sum(1 for _ in self._iter_file(fpath))
        row = {"seq": seq, "time": time.time(), **event}
        with open(fpath, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        return row

    def append_train(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return self._append(self.train_fpath, event)

    def append_worker(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return self._append(self.worker_fpath, event)

    def register_definition(self, fp: str, definition: Dict[str, Any]) -> None:
        """Idempotent insert into the content-addressed definition store."""
        defs = self.definitions()
        if fp in defs:
            return
        defs[fp] = definition
        self.journal_dpath.mkdir(parents=True, exist_ok=True)
        tmp = self.definitions_fpath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(defs, indent=2, sort_keys=True))
        tmp.replace(self.definitions_fpath)

    # -- reading ------------------------------------------------------------

    @staticmethod
    def _iter_file(fpath: Path) -> Iterator[Dict[str, Any]]:
        if not fpath.exists():
            return
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def read_train(self) -> List[Dict[str, Any]]:
        return list(self._iter_file(self.train_fpath))

    def read_worker(self) -> List[Dict[str, Any]]:
        return list(self._iter_file(self.worker_fpath))

    def read_all(self) -> List[Dict[str, Any]]:
        """All events; fold logic must not depend on cross-file order."""
        return self.read_train() + self.read_worker()

    def definitions(self) -> Dict[str, Any]:
        if self.definitions_fpath.exists():
            return json.loads(self.definitions_fpath.read_text())
        return {}

    # -- staging paths -------------------------------------------------------

    @staticmethod
    def staged_ckpt_name(epoch: int) -> str:
        return f"epoch_{epoch:04d}.pth"

    def staged_ckpt_fpath(self, epoch: int) -> Path:
        return self.staging_dpath / self.staged_ckpt_name(epoch)

    def resolve_ckpt(self, relpath: str) -> Path:
        return self.workdir / relpath
