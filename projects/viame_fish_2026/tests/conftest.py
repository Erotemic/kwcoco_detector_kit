"""Shared pytest helpers for viame_fish_2026.

Convention: tests/unit/* run anywhere off synthetic fixtures;
tests/expensive/* touch the real FishTrack23 mirror and skip when it is
not on disk. The kit's own pytest run does not collect these — invoke
pytest from inside this project subtree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# scripts/*.py are plain modules, not a package.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT
