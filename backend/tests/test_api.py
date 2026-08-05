"""Every endpoint, every documented failure mode.

The API is the only part of this system another program talks to, so its
contract is tested harder than anything else: shape, validation, edge cases,
and the errors it is supposed to return rather than crash on.
"""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import ISSUE_LABELS, SENTIMENT_LABELS

client = TestClient(app)

RESULT_KEYS = {
    "sentiment", "issues", "low_confidence", "reasons", "input_category",
    "word_count", "truncated", "model_version", "model_backend", "processed_at",
}


# --- GET /health -----------------------------------------------------------

def test_health_returns_200():
    assert client.get("/health").status_code == 200


def test_health_reports_model_state():
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert isinstance(body["model_loaded"], bool)


def test_health_names_the_backend_when_a_model_is_loaded():
    body = client.get("/health").json()
    if body["model_loaded"]:
        assert body["model_backend"] in ("baseline", "transformer")
        assert body["model_version"]


# --- GET /taxonomy ---------------------------------------------------------

def test_taxonomy_returns_the_full_label_space():
    body = client.get("/taxonomy").json()
    assert set(body["issue_labels"]) == set(ISSUE_LABELS)
    assert set(body["sentiment_labels"]) == set(SENTIMENT_LABELS)


def test_taxonomy_matches_what_analyze_can_return():
    """If these drift the frontend renders a category it has no colour for."""
    allowed = set(client.get("/taxonomy").json()["issue_labels"])
    body = client.post("/analyze", json={
        "text": "Late delivery, crushed box, broken zip, rude service, too expensive, runs small."
    }).json()
    assert {i["category"] for i in body["issues"]} <= allowed


# --- POST /analyze: success ------------------------------------------------

def test_analyze_returns_the_full_contract():
    body = client.post("/analyze", json={"text": "Arrived late and crushed."}).json()
    assert RESULT_KEYS <= set(body)


def test_analyze_sentiment_shape():
    body = client.post("/analyze", json={"text": "I love this."}).json()
    assert set(body["sentiment"]) == {"label", "confidence"}
    assert 0.0 <= body["sentiment"]["confidence"] <= 1.0


def test_analyze_issue_shape():
    body = client.post("/analyze", json={
        "text": "Arrived late and the box was crushed."}).json()
    for issue in body["issues"]:
        assert set(issue) == {"category", "confidence"}


def test_analyze_echoes_the_category():
    body = client.post("/analyze", json={"text": "Nice", "category": "Dresses"}).json()
    assert body["input_category"] == "Dresses"


def test_analyze_accepts_a_null_category():
    assert client.post("/analyze", json={"text": "Nice", "category": None}).status_code == 200


def test_analyze_detects_a_delivery_complaint():
    body = client.post("/analyze", json={
        "text": "The parcel arrived three weeks late and tracking never updated."}).json()
    assert "delivery" in [i["category"] for i in body["issues"]]


def test_response_includes_a_timing_header():
    r = client.post("/analyze", json={"text": "Good."})
    assert "X-Response-Time-ms" in r.headers
    assert float(r.headers["X-Response-Time-ms"]) >= 0


# --- POST /analyze: edge cases that must not crash -------------------------

@pytest.mark.parametrize("text", [
    "",
    "   ",
    "meh",
    "😡😡😡",
    "!!!???",
    "<script>alert('xss')</script>",
    "'; DROP TABLE reviews; --",
    "{{7*7}}",
    "../../etc/passwd",
    "\x00\x01 broken",
    "Bu mahsulot juda yomon keldi",
    "很好的产品非常满意",
    "a" * 30000,
    "word " * 8000,
])
def test_difficult_input_returns_200_with_a_valid_result(text):
    r = client.post("/analyze", json={"text": text})
    assert r.status_code == 200
    assert RESULT_KEYS <= set(r.json())


def test_empty_text_is_flagged_with_a_reason():
    body = client.post("/analyze", json={"text": ""}).json()
    assert body["low_confidence"] is True
    assert body["sentiment"]["label"] == "unknown"
    assert body["reasons"]


def test_oversized_input_is_marked_truncated():
    assert client.post("/analyze", json={"text": "broken " * 9000}).json()["truncated"] is True


# --- POST /analyze: validation ---------------------------------------------

def test_missing_text_field_is_422():
    assert client.post("/analyze", json={"category": "Tops"}).status_code == 422


def test_wrong_type_for_text_is_422():
    assert client.post("/analyze", json={"text": 123}).status_code == 422


def test_null_text_is_422():
    assert client.post("/analyze", json={"text": None}).status_code == 422


def test_malformed_json_is_422():
    r = client.post("/analyze", content=b"{not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_empty_body_is_422():
    assert client.post("/analyze", content=b"").status_code == 422


def test_get_on_analyze_is_405():
    assert client.get("/analyze").status_code == 405


def test_unknown_route_is_404():
    assert client.get("/does-not-exist").status_code == 404


def test_extra_unknown_fields_are_ignored():
    assert client.post("/analyze",
                       json={"text": "Good.", "nonsense": True, "id": 9}).status_code == 200


# --- POST /analyze/batch ---------------------------------------------------

def test_batch_returns_one_result_per_review():
    body = client.post("/analyze/batch", json={
        "reviews": ["broken on arrival", "love it", ""]}).json()
    assert body["count"] == 3
    assert len(body["results"]) == 3


def test_batch_results_have_the_full_contract():
    body = client.post("/analyze/batch", json={"reviews": ["late", "great"]}).json()
    assert all(RESULT_KEYS <= set(r) for r in body["results"])


def test_batch_preserves_input_order():
    body = client.post("/analyze/batch", json={
        "reviews": ["", "The item arrived completely broken and useless."]}).json()
    assert body["results"][0]["sentiment"]["label"] == "unknown"
    assert body["results"][1]["sentiment"]["label"] != "unknown"


def test_batch_agrees_with_single_analyze():
    text = "Arrived late and the box was crushed."
    single = client.post("/analyze", json={"text": text}).json()
    batched = client.post("/analyze/batch", json={"reviews": [text]}).json()["results"][0]
    assert single["sentiment"] == batched["sentiment"]
    assert single["issues"] == batched["issues"]


def test_batch_applies_the_category_to_every_review():
    body = client.post("/analyze/batch", json={
        "reviews": ["a", "b"], "category": "Home"}).json()
    assert all(r["input_category"] == "Home" for r in body["results"])


def test_empty_batch_is_rejected():
    assert client.post("/analyze/batch", json={"reviews": []}).status_code == 422


def test_missing_reviews_field_is_422():
    assert client.post("/analyze/batch", json={"category": "Home"}).status_code == 422


def test_batch_over_the_size_limit_is_rejected():
    assert client.post("/analyze/batch", json={"reviews": ["x"] * 501}).status_code == 422


def test_batch_at_the_size_limit_is_accepted():
    r = client.post("/analyze/batch", json={"reviews": ["ok"] * 500})
    assert r.status_code == 200
    assert r.json()["count"] == 500


def test_batch_with_wrong_element_type_is_422():
    assert client.post("/analyze/batch", json={"reviews": [1, 2]}).status_code == 422


# --- CORS ------------------------------------------------------------------

def test_local_dev_origin_is_allowed():
    r = client.post("/analyze", json={"text": "Good."},
                    headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_unknown_origin_gets_no_allow_header():
    r = client.post("/analyze", json={"text": "Good."},
                    headers={"Origin": "https://not-my-frontend.example"})
    assert "access-control-allow-origin" not in r.headers


def test_preflight_request_succeeds():
    r = client.options("/analyze", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert r.status_code == 200


# --- docs ------------------------------------------------------------------

def test_openapi_schema_is_served():
    schema = client.get("/openapi.json").json()
    for path in ("/analyze", "/analyze/batch", "/health", "/taxonomy"):
        assert path in schema["paths"]


def test_docs_page_loads():
    assert client.get("/docs").status_code == 200
