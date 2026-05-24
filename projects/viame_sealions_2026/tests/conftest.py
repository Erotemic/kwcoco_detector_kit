"""Shared pytest helpers for viame_sealions_2026.

Convention: tests/expensive/* read the real training_ready_v1 bundles.
The REPO_ROOT fixture resolves to <repo>/, which is the parent of this
tests/ directory. Tests that need the real kwcoco files should
@pytest.mark.skipif on their existence so the suite still runs on a
fresh checkout without DVC pulls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Make `import xxx` work for scripts/*.py — they're plain modules, not a
# package.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def training_ready_dir(repo_root) -> Path:
    return repo_root / "training_ready_v1"
