"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make the package importable when running from the tests/ directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _deterministic():
    """Make every test deterministic by default."""
    torch.manual_seed(0)
    yield
