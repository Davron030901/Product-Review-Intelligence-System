"""FastAPI service.

Run:  uvicorn src.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import ISSUE_LABELS, SENTIMENT_LABELS
from src.api.schemas import (AnalyzeRequest, AnalyzeResponse, BatchRequest,
                             BatchResponse, HealthResponse, TaxonomyResponse)
from src.inference.predictor import ModelNotTrained, get_predictor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("review-intel")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model once at boot so the first request is not slow."""
    try:
        p = get_predictor()
        log.info("model loaded: backend=%s version=%s", p.backend, p.model_version)
    except ModelNotTrained as e:
        log.warning("starting WITHOUT a model: %s", e)
    yield


app = FastAPI(
    title="Product Review Intelligence API",
    description="Turns a raw product review into structured sentiment and issue categories.",
    version="1.0.0",
    lifespan=lifespan,
)

# Origins come from the environment so the deployed frontend can be allowed
# without a code change. Local dev ports are always permitted.
#   ALLOWED_ORIGINS=https://my-app.vercel.app,https://my-app-git-main.vercel.app
_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_ENV_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS + _ENV_ORIGINS,
    # Vercel preview deployments get a new subdomain per commit, so match them
    # by pattern rather than listing every one.
    allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX") or None,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
log.info("CORS origins: %s (regex=%s)", _DEV_ORIGINS + _ENV_ORIGINS,
         os.getenv("ALLOWED_ORIGIN_REGEX") or "none")


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Response-Time-ms"] = str(round((time.time() - t0) * 1000, 1))
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """Never leak a stack trace to the client, never return an empty 500 body."""
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={
        "detail": "The service could not process this request.",
        "path": request.url.path,
    })


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    try:
        p = get_predictor()
        return HealthResponse(status="ok", model_loaded=True,
                              model_backend=p.backend, model_version=p.model_version)
    except ModelNotTrained:
        return HealthResponse(status="degraded", model_loaded=False)


@app.get("/taxonomy", response_model=TaxonomyResponse, tags=["meta"])
def taxonomy():
    """The label space, so the frontend does not hardcode it."""
    return TaxonomyResponse(issue_labels=ISSUE_LABELS, sentiment_labels=SENTIMENT_LABELS)


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(req: AnalyzeRequest):
    try:
        predictor = get_predictor()
    except ModelNotTrained as e:
        raise HTTPException(status_code=503, detail=str(e))
    return predictor.predict(req.text, req.category)


@app.post("/analyze/batch", response_model=BatchResponse, tags=["analysis"])
def analyze_batch(req: BatchRequest):
    if not req.reviews:
        raise HTTPException(status_code=422, detail="Send at least one review.")
    try:
        predictor = get_predictor()
    except ModelNotTrained as e:
        raise HTTPException(status_code=503, detail=str(e))
    results = predictor.predict_batch(req.reviews, req.category)
    return BatchResponse(results=results, count=len(results))
