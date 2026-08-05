"""The paths that only show up in production.

A service that works on a healthy machine and leaks a stack trace the moment
something breaks is not finished. These tests cover the states a deployed
instance actually reaches: no model on disk, a corrupt artefact, and an
unexpected exception mid-request.
"""
import joblib
import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.main import app
from src.inference.predictor import ModelNotTrained, ReviewPredictor

client = TestClient(app)


# --- 503: the service is up but has no model -------------------------------

def test_analyze_returns_503_when_no_model_is_trained(monkeypatch):
    def boom():
        raise ModelNotTrained("No model found. Run train_baseline.")
    monkeypatch.setattr(api_main, "get_predictor", boom)

    r = client.post("/analyze", json={"text": "Arrived late."})
    assert r.status_code == 503
    assert "train_baseline" in r.json()["detail"]


def test_batch_returns_503_when_no_model_is_trained(monkeypatch):
    def boom():
        raise ModelNotTrained("No model found. Run train_baseline.")
    monkeypatch.setattr(api_main, "get_predictor", boom)

    assert client.post("/analyze/batch", json={"reviews": ["x"]}).status_code == 503


def test_health_reports_degraded_rather_than_failing(monkeypatch):
    """Health must answer even when the model is missing, or the platform
    health check kills a container that is merely misconfigured."""
    def boom():
        raise ModelNotTrained("nope")
    monkeypatch.setattr(api_main, "get_predictor", boom)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["model_loaded"] is False


def test_taxonomy_still_works_without_a_model(monkeypatch):
    def boom():
        raise ModelNotTrained("nope")
    monkeypatch.setattr(api_main, "get_predictor", boom)
    assert client.get("/taxonomy").status_code == 200


# --- 500: something unexpected broke ---------------------------------------

def test_unexpected_error_returns_a_safe_500(monkeypatch):
    """No stack trace, no internal detail, but still valid JSON."""
    class Exploding:
        def predict(self, *a, **k):
            raise RuntimeError("secret internal detail: /home/user/model.joblib")

    monkeypatch.setattr(api_main, "get_predictor", lambda: Exploding())
    safe_client = TestClient(app, raise_server_exceptions=False)

    r = safe_client.post("/analyze", json={"text": "Arrived late."})
    assert r.status_code == 500
    body = r.json()
    assert "secret internal detail" not in str(body)
    assert "Traceback" not in str(body)
    assert body["detail"]


def test_error_response_names_the_path_for_debugging(monkeypatch):
    class Exploding:
        def predict(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(api_main, "get_predictor", lambda: Exploding())
    safe_client = TestClient(app, raise_server_exceptions=False)
    assert safe_client.post("/analyze", json={"text": "x"}).json()["path"] == "/analyze"


# --- model loading ---------------------------------------------------------

def test_missing_model_dir_raises_with_run_instructions(tmp_path):
    with pytest.raises(ModelNotTrained) as excinfo:
        ReviewPredictor(models_dir=tmp_path)
    message = str(excinfo.value)
    assert "build_dataset" in message and "train_baseline" in message


def test_corrupt_baseline_artefact_is_not_silently_ignored(tmp_path):
    (tmp_path / "baseline.joblib").write_bytes(b"this is not a joblib file")
    with pytest.raises(Exception):
        ReviewPredictor(models_dir=tmp_path)


def test_broken_transformer_falls_back_to_the_baseline(tmp_path, capsys):
    """A transformer that fails to load must not take the service down when a
    working baseline is sitting right next to it."""
    from src.config import MODELS_DIR

    real = MODELS_DIR / "baseline.joblib"
    if not real.exists():
        pytest.skip("no trained baseline available")
    joblib.dump(joblib.load(real), tmp_path / "baseline.joblib")

    tdir = tmp_path / "transformer"
    tdir.mkdir()
    (tdir / "model.pt").write_bytes(b"corrupt")
    (tdir / "meta.json").write_text("{}")

    predictor = ReviewPredictor(models_dir=tmp_path)
    assert predictor.backend == "baseline"
    assert "transformer load failed" in capsys.readouterr().out


def test_baseline_is_preferred_when_no_transformer_exists(tmp_path):
    from src.config import MODELS_DIR

    real = MODELS_DIR / "baseline.joblib"
    if not real.exists():
        pytest.skip("no trained baseline available")
    joblib.dump(joblib.load(real), tmp_path / "baseline.joblib")

    predictor = ReviewPredictor(models_dir=tmp_path)
    assert predictor.backend == "baseline"
    assert predictor.model_version


# --- transformer training script -------------------------------------------

def test_transformer_script_fails_with_install_instructions_when_torch_is_absent():
    """The optional dependency must produce guidance, not an ImportError."""
    pytest.importorskip  # noqa
    try:
        import torch  # noqa: F401
        pytest.skip("torch is installed; the guard cannot be exercised")
    except ImportError:
        pass

    from src.training.train_transformer import _require_torch
    with pytest.raises(SystemExit) as excinfo:
        _require_torch()
    assert "pip install" in str(excinfo.value)


def test_transformer_module_imports_without_torch():
    """Importing must be safe: predictor.py imports build_model lazily and the
    API would fail to boot if the module needed torch at import time."""
    import src.training.train_transformer as tt
    assert hasattr(tt, "train")
    assert tt.DEFAULT_MODEL
