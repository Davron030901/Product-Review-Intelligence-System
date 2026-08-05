"""Shared fixtures."""
import pytest


@pytest.fixture(scope="module")
def monkeypatch_module():
    """monkeypatch is function-scoped; module-scoped fixtures need this."""
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
