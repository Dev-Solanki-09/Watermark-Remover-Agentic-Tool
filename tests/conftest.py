"""pytest configuration and fixtures for wmclean test suite."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Directory for generated synthetic test fixtures."""
    tmp = Path(tempfile.mkdtemp(prefix="wmclean_fixtures_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def generated_fixtures(fixtures_dir: Path) -> dict:
    """Generate all synthetic test fixtures once for the test session."""
    from tests.fixture_gen import generate_all_fixtures

    return generate_all_fixtures(fixtures_dir)
