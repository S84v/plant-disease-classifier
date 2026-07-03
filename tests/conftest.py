from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app

TEST_ASSETS_DIR = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def assets_dir():
    return TEST_ASSETS_DIR


@pytest.fixture(scope="session")
def healthy_image(assets_dir):
    return assets_dir / "healthy_leaf.jpg"


@pytest.fixture(scope="session")
def diseased_image(assets_dir):
    return assets_dir / "diseased_leaf.jpg"


@pytest.fixture(scope="session")
def fake_image(assets_dir):
    return assets_dir / "fake.jpg"


@pytest.fixture(scope="session")
def invalid_file(assets_dir):
    return assets_dir / "notes.txt"
