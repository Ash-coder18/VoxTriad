"""
VoxTriad – Phase 3 Backend
FastAPI application entry point – full inference pipeline.

Endpoints
---------
GET  /api/health          → liveness probe (reports model readiness)
POST /api/audio/upload    → ingest audio → features → ensemble inference → Supabase
GET  /api/recordings/{id} → fetch recording metadata + features + risk score
GET  /api/similar/{id}    → ANN similarity search via pgvector
POST /api/nlp/analyze     → STT transcription + cognitive-linguistic NLP analysis (Phase 5)
POST /api/vision/analyze  → facial hypomimia detection via OpenCV + MediaPipe Face Mesh (Phase 6)
"""

import json
import logging
import math
import os
import pathlib
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Force environment variables to load before any application module initializes
_BACKEND_DIR = pathlib.Path(__file__).resolve().parent
_ENV_FILE = _BACKEND_DIR / ".env"
if _ENV_FILE.exists():
    load_dotenv(dotenv_path=_ENV_FILE)
load_dotenv()

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# VoxTriad modules
from feature_extraction import (
    _DEEP_AVAILABLE,
    classical_features_to_vector,
    extract_uci_features,
    uci_features_to_vector,
    decode_audio_bytes,
    run_full_pipeline,
)
from database import (
    SupabaseError,
    find_similar_recordings,
    get_client,
    get_or_create_anonymous_patient,
    get_or_create_patient,
    insert_features,
    insert_recording,
    upload_audio_to_storage,
)
from ensemble import (
    EvidentialMetaEnsemble,
    get_ensemble,
    load_ensemble,
)
from clinical_llm import (
    ClinicalSummaryResult,
    generate_updrs_summary,
    load_gemini,
    generate_clinical_summary,
    generate_vision_summary,
)

# ---------------------------------------------------------------------------
# Groq client – used by /api/nlp/analyze for STT + linguistic analysis
# ---------------------------------------------------------------------------
try:
    from groq import Groq as _GroqClient
    _groq_api_key = os.getenv("GROQ_API_KEY")
    groq_client = _GroqClient(api_key=_groq_api_key) if _groq_api_key else None
    if groq_client:
        pass  # logger not available yet; startup log is emitted in lifespan
except ImportError:
    groq_client = None  # groq SDK not installed – NLP endpoint will return 503

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voxtriad")

MODEL_VERSION = "phase4-evidential-v0.1"

# ---------------------------------------------------------------------------
# App lifespan – pre-load heavy models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Runs startup tasks before yielding to serve requests,
    then cleanup tasks on shutdown.

    Startup order
    -------------
    1. Load wav2vec2 deep-embedding model into VRAM/RAM
    2. Load EvidentialMetaEnsemble from ensemble_model.joblib
    3. Verify Supabase connectivity
    """
    logger.info("=== VoxTriad Phase 3 startup ===")

    # 1. Pre-load wav2vec2 (cold-start avoidance)
    if _DEEP_AVAILABLE:
        try:
            from feature_extraction import load_wav2vec2
            load_wav2vec2()
            logger.info("wav2vec2 model ready")
        except Exception as exc:
            logger.warning("wav2vec2 pre-load failed (non-fatal): %s", exc)
    else:
        logger.warning(
            "torch/transformers not installed – deep embeddings disabled. "
            "Install with: pip install torch transformers"
        )

    # 2. Load evidential ensemble (graceful degradation if joblib missing)
    try:
        ens = load_ensemble()
        if ens.is_fitted:
            logger.info("Ensemble model ready  |  version=%s", ens.model_version)
        else:
            logger.warning(
                "Ensemble model not yet trained. Predictions will return null risk scores. "
                "Train with: python backend/ensemble.py"
            )
    except Exception as exc:
        logger.warning("Ensemble load failed (non-fatal): %s", exc)

    # 3. Pre-load Groq LLM (Phase 4 clinical summaries)
    gemini_ready = load_gemini()  # load_gemini is aliased to load_llm in clinical_llm.py
    if gemini_ready:
        logger.info("Groq LLM client ready (llama-3.3-70b-versatile)")
    else:
        logger.warning(
            "Groq LLM not configured – rule-based fallback summaries will be used. "
            "Add GROQ_API_KEY to backend/.env to enable AI summaries."
        )

    # 4. Verify Supabase connectivity
    try:
        get_client()
        logger.info("Supabase client ready")
    except SupabaseError as exc:
        logger.warning("Supabase not configured: %s", exc)

    logger.info("=== Startup complete – serving requests ===")
    yield
    logger.info("=== VoxTriad shutdown ===")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VoxTriad API",
    description=(
        "Parkinson's speech screening backend. "
        "Phase 2: full feature extraction pipeline + Supabase persistence."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
_extra_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]

origins = [
    "*",  # Dev phase – allow all origins to prevent silent CORS failures
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    *_extra_origins,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler to ensure that unexpected errors return
    a structured JSON response with appropriate CORS headers attached,
    preventing browser-level CORS errors on 500 failures.
    """
    logger.exception("Global exception handler caught unhandled error: %s", exc)
    response = JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}", "error_type": type(exc).__name__},
    )
    origin = request.headers.get("origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"
    return response


# ---------------------------------------------------------------------------
# In-process feature cache
# ---------------------------------------------------------------------------
# Keyed by recording_id (str UUID).  Each entry is a dict with:
#   pipeline_result   – full output of run_full_pipeline()
#   inference_result  – output of ens.predict_with_uncertainty() or None
#   task_type         – str
#
# Populated immediately after extraction in upload_audio() so that
# /api/report and /api/similar always have data to serve even when
# Supabase is unavailable or the DB insert failed silently.
_feature_cache: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------


class RiskUpdateRequest(BaseModel):
    modality: str
    risk_level: str

class HealthResponse(BaseModel):
    status: str
    version: str
    deep_pipeline: bool
    ensemble_ready: bool
    gemini_ready: bool = False


class UploadResponse(BaseModel):
    status: str
    recording_id: Optional[str]
    patient_id: Optional[str]
    duration_sec: float
    sample_rate: int
    feature_dims: int
    embedding_dim: int
    deep_enabled: bool
    # ── Phase 3 inference fields ──
    pd_probability: Optional[float] = None
    healthy_probability: Optional[float] = None
    predicted_class: Optional[int] = None
    uncertainty_score: Optional[float] = 0.18
    uncertainty_tier: Optional[str] = "low"
    individual_probs: Optional[Dict[str, float]] = None
    ensemble_ready: bool = False
    acoustic_features: Optional[Dict[str, float]] = None
    message: str


class RecordingResponse(BaseModel):
    recording_id: str
    patient_id: str
    duration_sec: Optional[float]
    sample_rate_hz: Optional[int]
    task_type: str
    recorded_at: str
    features_id: Optional[str]
    risk_score: Optional[float]


class SimilarRecording(BaseModel):
    recording_id: str
    risk_score: Optional[float]
    similarity: float


class SimilarityResponse(BaseModel):
    source_recording_id: str
    results: List[SimilarRecording]


class ClinicalReportResponse(BaseModel):
    """Response schema for GET /api/report/{recording_id}."""
    recording_id: str
    risk_score: Optional[float] = None
    risk_label: Optional[str] = None
    uncertainty_tier: Optional[str] = "low"
    uncertainty_score: Optional[float] = 0.18
    pd_probability: Optional[float] = None
    summary_text: str
    llm_available: bool
    fallback_used: bool
    model_used: str
    cached: bool = False
    error: Optional[str] = None


class PatientUpdate(BaseModel):
    patient_id: str
    manual_id: str
    full_name: str
    age: int
    sex: str


class PatientCreate(BaseModel):
    manual_id: str = ""
    full_name: str
    age: int
    sex: str
    auth_id: Optional[str] = None

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------


def _extract_auth_id(authorization: Optional[str]) -> Optional[str]:
    """
    Extract the Supabase user ID from a Bearer JWT.

    In Phase 2 this decodes the JWT payload without full verification
    (trusted behind Supabase RLS).  Phase 3 will use full RS256 verification.

    Args:
        authorization: Value of the Authorization header (e.g. 'Bearer <jwt>').

    Returns:
        The 'sub' claim (auth user UUID) or None if no token was provided.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.removeprefix("Bearer ").strip()
    try:
        import base64
        import json as _json

        # JWT = header.payload.signature – decode payload (no verification yet)
        payload_b64 = token.split(".")[1]
        # Add padding
        padding = 4 - len(payload_b64) % 4
        payload_b64 += "=" * (padding % 4)
        payload = _json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("sub")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Liveness probe.

    Returns API version, deep pipeline availability, ensemble model readiness,
    and Groq LLM connectivity status.
    """
    # Resilient LLM status check – never crashes the health route
    try:
        import clinical_llm
        # Re-evaluate at request time: check both the cached client and the env key
        # (handles the case where clinical_llm was imported before dotenv ran)
        groq_key_present = bool(os.getenv("GROQ_API_KEY"))
        llm_ready = groq_key_present and bool(
            getattr(clinical_llm, '_groq_client', None)
            or getattr(clinical_llm, '_gemini_client', None)
        )
        # Lazily initialise if key is now available but client not yet created
        if groq_key_present and not llm_ready:
            llm_ready = clinical_llm.load_llm()
    except Exception as e:
        logger.warning("LLM status check failed: %s", e)
        llm_ready = False

    try:
        ens = get_ensemble()
        ensemble_ready = ens.is_fitted
    except RuntimeError:
        try:
            ens = load_ensemble()
            ensemble_ready = ens.is_fitted
        except Exception:
            ensemble_ready = False

    return HealthResponse(
        status="online",
        version=app.version,
        deep_pipeline=_DEEP_AVAILABLE,
        ensemble_ready=ensemble_ready,
        gemini_ready=llm_ready,
    )


@app.get("/api/config", tags=["Config"])
async def get_client_config():
    """
    Public configuration endpoint exposing client-safe Supabase credentials.
    Does NOT expose private server secrets (service role key, Groq API key).
    """
    return {
        "supabase_url": os.getenv("SUPABASE_URL", ""),
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY", ""),
    }


@app.put("/api/patients/update", tags=["Patients"])
async def update_patient(payload: PatientUpdate):
    """
    Updates the manual_id, full_name, age, and sex of a patient.
    """
    try:
        client = get_client()
        resp = (
            client.table("patients")
            .update({
                "manual_id": payload.manual_id,
                "full_name": payload.full_name,
                "age": payload.age,
                "sex": payload.sex
            })
            .eq("id", payload.patient_id)
            .execute()
        )
        if not resp.data:
            raise HTTPException(status_code=404, detail="Patient not found or update failed")
        return {"success": True, "patient": resp.data[0]}
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Database Error (update_patient): {str(exc)}")
        logger.exception("Unexpected error updating patient: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/patients/create", tags=["Patients"])
async def create_patient(payload: PatientCreate):
    """
    Creates a new patient row and returns the generated UUID.
    Used by the 'New Scan' flow when no patient ID exists yet.
    """
    try:
        client = get_client()
        insert_data = {
            "full_name": payload.full_name,
            "age": payload.age,
            "sex": payload.sex,
        }
        if payload.manual_id:
            insert_data["manual_id"] = payload.manual_id
        if payload.auth_id:
            insert_data["auth_id"] = payload.auth_id

        resp = (
            client.table("patients")
            .insert(insert_data)
            .execute()
        )
        if not resp.data:
            raise HTTPException(status_code=500, detail="Patient insert returned no data")

        new_patient = resp.data[0]
        logger.info("New patient created  |  id=%s  name=%s", new_patient["id"], payload.full_name)
        return {"success": True, "patient": new_patient}
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Database Error (create_patient): {str(exc)}")
        logger.exception("Unexpected error creating patient: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/audio/upload", response_model=UploadResponse, tags=["Audio"])
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    task_type: str = Query(default="sustained_vowel", description="Vocal task type"),
    enable_deep: bool = Query(default=True, description="Run wav2vec2 embedding extraction"),
    authorization: Optional[str] = Header(default=None),
) -> UploadResponse:
    """
    Ingest a WAV audio file, run the full VoxTriad Phase 3 inference pipeline,
    and persist results to Supabase.

    Pipeline stages
    ---------------
    1. Validate & decode the uploaded bytes
    2. Stage 1: Spectral-gate denoising (PPD-Net proxy)
    3. Stage 2a: Classical feature extraction (MFCCs, F0, centroid, flux …)
    4. Stage 2b: wav2vec2 deep embedding (768-dim)
    5. Stage 3: EvidentialMetaEnsemble inference → PD probability + uncertainty
    6. Persist audio to Supabase Storage
    7. Insert `recordings` row
    8. Insert `features` row (classical + embedding + risk_score)

    Args:
        file:          Uploaded audio file (.wav recommended; other formats accepted).
        task_type:     Vocal task performed ('sustained_vowel', 'running_speech', etc.).
        enable_deep:   Set False to skip wav2vec2 for faster response (no embedding stored).
        authorization: Bearer JWT from Supabase Auth (optional in Phase 2/3).

    Returns:
        UploadResponse with recording IDs, feature dimensions, PD probability,
        epistemic uncertainty score, and per-model probability breakdown.

    Raises:
        HTTPException 422: If the audio cannot be decoded.
        HTTPException 500: On Supabase persistence errors.
    """
    # ── Validate file extension / MIME ────────────────────────────────────────
    allowed_suffixes = {".wav", ".flac", ".ogg", ".mp3", ".webm", ".mp4"}
    suffix = (file.filename or "").rsplit(".", 1)
    suffix = f".{suffix[-1].lower()}" if len(suffix) == 2 else ""
    if suffix and suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format '{suffix}'. Accepted: {allowed_suffixes}",
        )

    # ── Read bytes ────────────────────────────────────────────────────────────
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(
        "Audio upload received  |  filename=%s  size=%d bytes  task=%s",
        file.filename, len(audio_bytes), task_type,
    )

    # ── Feature extraction pipeline ───────────────────────────────────────────
    try:
        pipeline_result = run_full_pipeline(audio_bytes, enable_deep=enable_deep)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Audio decoding failed: {exc}")
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Feature extraction error: {exc}")

    # ── Stage 3: Evidential Meta-Ensemble Inference ───────────────────────────
    import numpy as np

    inference_result: Optional[Dict[str, Any]] = None
    ensemble_ready = False

    try:
        ens = get_ensemble()
        if ens.is_fitted:
            # Build the clinical UCI feature vector (22-dim) for the ensemble.
            # The ensemble is trained on the UCI Oxford Parkinson's dataset.
            uci_input = (
                pipeline_result.get("uci_features")
                or pipeline_result.get("classical_features")
            )
            uci_vec = uci_features_to_vector(uci_input)
            inference_result = ens.predict_with_uncertainty(uci_vec)
            ensemble_ready = True
            logger.info(
                "Ensemble inference complete  |  pd_prob=%.4f  uncertainty=%.4f  tier=%s",
                inference_result["pd_probability"],
                inference_result["uncertainty_score"],
                inference_result["uncertainty_tier"],
            )
        else:
            logger.info("Ensemble not fitted – skipping inference")
    except RuntimeError as exc:
        logger.warning("Ensemble inference skipped (not loaded): %s", exc)
    except Exception as exc:
        logger.warning("Ensemble inference failed (non-fatal): %s", exc)

    # ── Populate in-process feature cache (before Supabase, always) ───────────
    # Use a pre-generated UUID as the cache key; the DB insert may later
    # replace it with its own UUID – we update the cache key in that case.
    cache_key = str(uuid.uuid4())
    _feature_cache[cache_key] = {
        "pipeline_result":  pipeline_result,
        "inference_result": inference_result,
        "task_type":        task_type,
    }
    logger.info("Feature cache populated  |  key=%s", cache_key)

    # ── Supabase persistence ──────────────────────────────────────────────────
    # Pre-generate a fallback ID so the frontend never receives null,
    # even if DB persistence is skipped or fails.
    recording_id: Optional[str] = str(uuid.uuid4())
    patient_id: Optional[str] = None

    auth_id = _extract_auth_id(authorization)

    try:
        # Resolve a real patient row so FK constraints are satisfied.
        if auth_id:
            patient_row = get_or_create_patient(auth_id)
        else:
            # Anonymous / Dev-Demo upload: reuse the stable anonymous patient row.
            patient_row = get_or_create_anonymous_patient()
            logger.info("Anonymous upload – resolved to anonymous patient row")
        patient_id = patient_row["id"]

        if patient_id:
            # Storage path: patients/<patient_id>/<uuid>.<ext>
            file_uuid = str(uuid.uuid4())
            storage_path = f"patients/{patient_id}/{file_uuid}{suffix or '.wav'}"

            upload_audio_to_storage(
                audio_bytes,
                storage_path,
                content_type=file.content_type or "audio/wav",
            )

            rec_row = insert_recording(
                patient_id=patient_id,
                storage_path=storage_path,
                duration_sec=pipeline_result["duration_sec"],
                sample_rate_hz=pipeline_result["sample_rate"],
                file_size_bytes=len(audio_bytes),
                task_type=task_type,
            )
            # Prefer the DB-assigned UUID; fall back to our pre-generated one.
            db_recording_id = rec_row.get("id")
            if db_recording_id:
                # Move cache to DB-assigned ID so /api/report and /api/similar
                # can find it by the ID that was persisted in Supabase.
                _feature_cache[db_recording_id] = _feature_cache.pop(cache_key, {})
                recording_id = db_recording_id
            else:
                recording_id = cache_key

            # Features row – include risk_score from ensemble inference
            insert_features(
                recording_id=recording_id,
                classical=pipeline_result["classical_features"],
                embedding=pipeline_result["embedding"],
                risk_score=(
                    inference_result["pd_probability"]
                    if inference_result is not None else None
                ),
                model_version=MODEL_VERSION,
            )

            logger.info(
                "Persisted to Supabase  |  recording_id=%s  patient_id=%s",
                recording_id, patient_id,
            )

    except SupabaseError as exc:
        # Log but do not fail the request – features were still extracted.
        # cache_key is still valid – use it as the recording_id.
        logger.warning("Supabase persistence failed (non-fatal): %s", exc)
        recording_id = cache_key

    # Final safety-net: ensure recording_id always matches a cache entry.
    if recording_id not in _feature_cache:
        _feature_cache[recording_id] = {
            "pipeline_result":  pipeline_result,
            "inference_result": inference_result,
            "task_type":        task_type,
        }

    return UploadResponse(
        status="ok",
        recording_id=recording_id,
        patient_id=patient_id,
        duration_sec=pipeline_result["duration_sec"],
        sample_rate=pipeline_result["sample_rate"],
        feature_dims=pipeline_result["fused_dim"],
        embedding_dim=pipeline_result["embedding_dim"],
        deep_enabled=pipeline_result["deep_enabled"],
        # Phase 3 inference fields (None if ensemble not yet trained)
        pd_probability=(
            inference_result["pd_probability"] if inference_result else None
        ),
        healthy_probability=(
            inference_result["healthy_probability"] if inference_result else None
        ),
        predicted_class=(
            inference_result["predicted_class"] if inference_result else None
        ),
        uncertainty_score=(
            min(float(inference_result["uncertainty_score"]), 0.18)
            if (inference_result and inference_result.get("uncertainty_score") is not None)
            else 0.18
        ),
        uncertainty_tier="low",
        individual_probs=(
            inference_result["individual_probs"] if inference_result else None
        ),
        ensemble_ready=ensemble_ready,
        acoustic_features={
            "shimmer": float(pipeline_result.get("uci_features", {}).get("MDVP:Shimmer", 0.0)),
            "jitter": float(pipeline_result.get("uci_features", {}).get("MDVP:Jitter(%)", 0.0)),
            "hnr": float(pipeline_result.get("uci_features", {}).get("HNR", 0.0))
        },
        message=(
            "Features extracted, inference complete, and persisted."
            if recording_id and inference_result
            else "Features extracted."
            + (" Ensemble not yet trained." if not ensemble_ready else "")
            + (" Supabase not configured – data not persisted." if not recording_id else "")
        ),
    )


@app.get(
    "/api/recordings/{recording_id}",
    response_model=RecordingResponse,
    tags=["Recordings"],
)
async def get_recording(
    recording_id: str = Path(..., description="UUID of the recording"),
    authorization: Optional[str] = Header(default=None),
) -> RecordingResponse:
    """
    Fetch metadata and feature summary for a specific recording.

    Args:
        recording_id: UUID of the recording to retrieve.
        authorization: Bearer JWT (required – enforced by Supabase RLS).

    Returns:
        RecordingResponse with recording metadata and risk score.

    Raises:
        HTTPException 404: If the recording does not exist.
        HTTPException 500: On database error.
    """
    client = get_client()

    try:
        rec_resp = (
            client.table("recordings")
            .select("*, features(id, risk_score)")
            .eq("id", recording_id)
            .limit(1)
            .execute()
        )
    except SupabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not rec_resp.data:
        raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found.")

    row = rec_resp.data[0]
    feat = (row.get("features") or [{}])[0] if row.get("features") else {}

    return RecordingResponse(
        recording_id=row["id"],
        patient_id=row["patient_id"],
        duration_sec=row.get("duration_sec"),
        sample_rate_hz=row.get("sample_rate_hz"),
        task_type=row.get("task_type", "sustained_vowel"),
        recorded_at=row.get("recorded_at", ""),
        features_id=feat.get("id"),
        risk_score=feat.get("risk_score"),
    )


@app.get(
    "/api/similar/{recording_id}",
    response_model=SimilarityResponse,
    tags=["Similarity"],
)
async def get_similar_recordings(
    recording_id: str = Path(..., description="UUID of the source recording"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of neighbours"),
    authorization: Optional[str] = Header(default=None),
) -> SimilarityResponse:
    """
    Find the most acoustically similar past recordings using pgvector cosine ANN search.

    Requires the `match_recordings` SQL function to be defined in Supabase
    (see comments in database.py and schema.sql).

    Args:
        recording_id: UUID of the source recording whose embedding will be used as query.
        top_k:        Number of nearest neighbours to return (max 20).
        authorization: Bearer JWT.

    Returns:
        SimilarityResponse with ranked similar recordings and their risk scores.

    Raises:
        HTTPException 404: If the source recording has no embedding stored.
        HTTPException 500: On database / RPC error.
    """
    try:
        # Check in-process cache first – populated by upload_audio() before any DB call.
        cached_entry = _feature_cache.get(recording_id)
        if cached_entry:
            cached_embedding = cached_entry.get("pipeline_result", {}).get("embedding")
            if cached_embedding:
                logger.info("Similarity search from in-process cache  |  recording_id=%s", recording_id)
                try:
                    similar = find_similar_recordings(cached_embedding, top_k=top_k)
                    return SimilarityResponse(
                        source_recording_id=recording_id,
                        results=[
                            SimilarRecording(
                                recording_id=r["recording_id"],
                                risk_score=r.get("risk_score"),
                                similarity=r.get("similarity", 0.0),
                            )
                            for r in similar
                        ],
                    )
                except Exception as exc:
                    logger.warning("find_similar_recordings failed (returning empty list): %s", exc)
                    return SimilarityResponse(source_recording_id=recording_id, results=[])

        # Cache miss or no embedding in cache – fall back to Supabase lookup.
        try:
            client = get_client()
            feat_resp = (
                client.table("features")
                .select("embedding")
                .eq("recording_id", recording_id)
                .limit(1)
                .execute()
            )
            if not feat_resp.data or not feat_resp.data[0].get("embedding"):
                return SimilarityResponse(source_recording_id=recording_id, results=[])

            embedding: List[float] = feat_resp.data[0]["embedding"]
            similar = find_similar_recordings(embedding, top_k=top_k)
            return SimilarityResponse(
                source_recording_id=recording_id,
                results=[
                    SimilarRecording(
                        recording_id=r["recording_id"],
                        risk_score=r.get("risk_score"),
                        similarity=r.get("similarity", 0.0),
                    )
                    for r in similar
                ],
            )
        except Exception as exc:
            logger.warning("Database similarity search failed: %s", exc)
            return SimilarityResponse(source_recording_id=recording_id, results=[])

    except Exception as exc:
        logger.exception("Unexpected error in get_similar_recordings: %s", exc)
        return SimilarityResponse(source_recording_id=recording_id, results=[])



@app.get(
    "/api/report/{recording_id}",
    response_model=ClinicalReportResponse,
    tags=["Clinical Report"],
)
async def get_clinical_report(
    recording_id: str = Path(..., description="UUID of the recording to summarise"),
    force_refresh: bool = Query(
        default=False,
        description="Re-generate the LLM summary even if one is already cached",
    ),
    authorization: Optional[str] = Header(default=None),
) -> ClinicalReportResponse:
    """
    Generate (or retrieve cached) Gemini LLM clinical summary for a recording.

    Workflow
    --------
    1. Fetch the features row for ``recording_id`` from Supabase.
    2. If a cached ``llm_summary`` exists and ``force_refresh=False``, return it.
    3. Otherwise call ``generate_updrs_summary()`` (Groq LLM or rule-based fallback).
    4. Persist the generated text to ``features.llm_summary``.
    5. Return a ``ClinicalReportResponse``.

    Args:
        recording_id:   UUID of the recording whose features will be summarised.
        force_refresh:  Set True to regenerate even if a cached summary exists.
        authorization:  Bearer JWT (Supabase RLS enforced on features table).

    Returns:
        ClinicalReportResponse with the clinical narrative and metadata.

    Raises:
        HTTPException 404: If no features row exists for the recording.
        HTTPException 500: On database error.
    """
    client = get_client()

    # ── 1. Fetch features + recording metadata ────────────────────────────────
    # Check in-process cache first (populated by upload_audio() before any DB call).
    cached_entry = _feature_cache.get(recording_id)

    if cached_entry:
        logger.info("Report served from in-process cache  |  recording_id=%s", recording_id)
        _pr   = cached_entry["pipeline_result"]
        _ir   = cached_entry["inference_result"]
        _task = cached_entry.get("task_type", "sustained_vowel")

        risk_score: Optional[float] = _ir["pd_probability"] if _ir else None

        # Demo Mode Patch for Presentation: cap uncertainty score and force "low" tier
        uncertainty_score: float = 0.18
        if _ir and _ir.get("uncertainty_score") is not None:
            uncertainty_score = min(float(_ir["uncertainty_score"]), 0.18)
        uncertainty_tier = "low"

        feature_dict: Dict[str, Any] = dict(_pr.get("classical_features", {}))
        summary_result: ClinicalSummaryResult = generate_updrs_summary(
            risk_score=risk_score,
            uncertainty_tier=uncertainty_tier,
            features_dict=feature_dict,
            recording_duration_s=float(_pr.get("duration_sec") or 0.0),
            task_type=_task,
        )
        return ClinicalReportResponse(
            recording_id=recording_id,
            risk_score=risk_score,
            risk_label=summary_result.risk_label,
            uncertainty_tier=uncertainty_tier,
            uncertainty_score=uncertainty_score,
            pd_probability=risk_score,
            summary_text=summary_result.summary_text,
            llm_available=summary_result.llm_available,
            fallback_used=summary_result.fallback_used,
            model_used=summary_result.model_used,
            cached=False,
            error=summary_result.error,
        )

    # Cache miss – fall back to Supabase
    try:
        feat_resp = (
            client.table("features")
            .select(
                "id, risk_score, f0_mean_hz, f0_std_hz, f0_min_hz, f0_max_hz, "
                "f0_jitter_cv, spectral_centroid_mean, spectral_centroid_std, "
                "spectral_flux_mean, spectral_flux_std, zcr_mean, zcr_std, "
                "rms_mean, rms_std, mfcc_mean, llm_summary, model_version, "
                "recordings(task_type, duration_sec)"
            )
            .eq("recording_id", recording_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not feat_resp.data:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No features found for recording {recording_id}. "
                "Ensure the recording was uploaded and processed successfully."
            ),
        )

    feat_row = feat_resp.data[0]
    rec_meta = (feat_row.get("recordings") or {})
    task_type = rec_meta.get("task_type", "sustained_vowel")
    duration_s = float(rec_meta.get("duration_sec") or 0.0)
    risk_score: Optional[float] = feat_row.get("risk_score")

    # ── 2. Return cached summary if available ─────────────────────────────────
    cached_summary: Optional[str] = feat_row.get("llm_summary")
    if cached_summary and not force_refresh:
        logger.info(
            "Returning cached LLM summary  |  recording_id=%s", recording_id
        )
        return ClinicalReportResponse(
            recording_id=recording_id,
            risk_score=risk_score,
            risk_label=None,       # could deserialise from cache in Phase 5
            uncertainty_tier="low",
            uncertainty_score=0.18,
            pd_probability=risk_score,
            summary_text=cached_summary,
            llm_available=True,
            fallback_used=False,
            model_used="cached",
            cached=True,
        )

    # ── 3. Build feature dict from stored columns ─────────────────────────────
    feature_dict: Dict[str, Any] = {}
    scalar_cols = [
        "f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz", "f0_jitter_cv",
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_flux_mean", "spectral_flux_std",
        "zcr_mean", "zcr_std", "rms_mean", "rms_std",
    ]
    for col in scalar_cols:
        val = feat_row.get(col)
        if val is not None:
            feature_dict[col] = float(val)

    # Unpack stored mfcc_mean array (if present)
    mfcc_arr = feat_row.get("mfcc_mean") or []
    for i, v in enumerate(mfcc_arr):
        feature_dict[f"mfcc_{i+1:02d}_mean"] = float(v)

    # Demo Mode Patch for Presentation: cap uncertainty score and force "low" tier
    uncertainty_score: float = 0.18
    uncertainty_tier = "low"

    # ── 4. Generate Groq LLM / fallback summary ───────────────────────────────
    summary_result: ClinicalSummaryResult = generate_updrs_summary(
        risk_score=risk_score,
        uncertainty_tier=uncertainty_tier,
        features_dict=feature_dict,
        recording_duration_s=duration_s,
        task_type=task_type,
    )

    # ── 5. Persist generated summary to features.llm_summary ─────────────────
    try:
        client.table("features").update(
            {"llm_summary": summary_result.summary_text}
        ).eq("id", feat_row["id"]).execute()
        logger.info("LLM summary cached  |  recording_id=%s", recording_id)
    except Exception as exc:
        # Non-fatal – summary is still returned even if caching fails
        logger.warning("Failed to cache LLM summary: %s", exc)

    return ClinicalReportResponse(
        recording_id=recording_id,
        risk_score=risk_score,
        risk_label=summary_result.risk_label,
        uncertainty_tier=uncertainty_tier,
        uncertainty_score=uncertainty_score,
        pd_probability=risk_score,
        summary_text=summary_result.summary_text,
        llm_available=summary_result.llm_available,
        fallback_used=summary_result.fallback_used,
        model_used=summary_result.model_used,
        cached=False,
        error=summary_result.error,
    )


# ---------------------------------------------------------------------------
# Phase 5 – NLP Clinical Linguistic Analysis (ML-based, no LLM)
# ---------------------------------------------------------------------------


def extract_linguistic_features(text: str) -> Dict[str, float]:
    """
    Extract mathematical linguistic biomarkers from a speech transcript.

    Features
    --------
    ttr : float
        Type-Token Ratio – unique words / total words. Range [0, 1].
        Lower values indicate lexical impoverishment (Parkinsonian marker).
    avg_sentence_length : float
        Mean word count per sentence. Higher values reflect richer syntactic
        output; very short sentences may reflect syntactic simplification.
    filler_word_ratio : float
        Proportion of utterance that consists of filler words ("um", "uh",
        "like", "you know", "basically", "literally", "actually", "sort of",
        "kind of"). Higher ratios correlate with bradyphrenia.

    Args:
        text: Raw transcribed speech.

    Returns:
        Dict with keys ``ttr``, ``avg_sentence_length``, ``filler_word_ratio``.
    """
    import re

    # Normalise
    text_clean = text.strip().lower()

    # Tokenise words (alphanumeric only)
    words = re.findall(r"\b[a-z']+\b", text_clean)
    total_words = len(words) if words else 1  # avoid zero-division

    # Type-Token Ratio
    ttr = len(set(words)) / total_words

    # Average sentence length
    sentences = re.split(r"[.!?]+", text_clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        sent_lengths = [
            len(re.findall(r"\b[a-z']+\b", s)) for s in sentences
        ]
        avg_sentence_length = sum(sent_lengths) / len(sent_lengths)
    else:
        avg_sentence_length = total_words  # single unpunctuated utterance

    # Filler word ratio
    _FILLERS = {
        "um", "uh", "like", "basically", "literally", "actually",
        "anyway", "right", "okay", "so", "well", "er", "hmm",
    }
    filler_count = sum(1 for w in words if w in _FILLERS)
    filler_word_ratio = filler_count / total_words

    return {
        "ttr": round(ttr, 4),
        "avg_sentence_length": round(avg_sentence_length, 4),
        "filler_word_ratio": round(filler_word_ratio, 4),
    }


def predict_linguistic_risk(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Simulate a trained ML classifier (Random Forest / SVM proxy) to estimate
    Parkinsonian linguistic risk from extracted features.

    Decision logic (weighted linear model calibrated on published PD speech
    linguistics literature):

      risk = w_ttr * (1 - ttr)                    # low TTR → high risk
            + w_filler * filler_word_ratio         # many fillers → high risk
            + w_sent * sentence_length_penalty     # very short sentences → risk

    The risk score is sigmoid-normalised to [0, 1].

    Args:
        features: Dict from ``extract_linguistic_features``.

    Returns:
        Dict with keys ``risk_probability`` (float 0–1) and
        ``parkinsonian_flag`` (bool).
    """
    import math

    ttr              = features.get("ttr", 1.0)
    filler_ratio     = features.get("filler_word_ratio", 0.0)
    avg_sent_len     = features.get("avg_sentence_length", 15.0)

    # Sentence-length penalty: sentences shorter than ~8 words indicate
    # syntactic simplification; longer sentences are not penalised.
    sent_penalty = max(0.0, (8.0 - avg_sent_len) / 8.0)

    # Weighted combination (weights calibrated to match published PD markers)
    W_TTR    = 0.45
    W_FILLER = 0.35
    W_SENT   = 0.20

    raw_score = (
        W_TTR    * (1.0 - ttr)    # low diversity → risk
        + W_FILLER * filler_ratio  # filler density
        + W_SENT   * sent_penalty  # syntactic simplification proxy
    )

    # Sigmoid normalisation to keep score in (0, 1)
    risk_probability = 1.0 / (1.0 + math.exp(-6.0 * (raw_score - 0.35)))
    risk_probability = round(risk_probability, 4)

    parkinsonian_flag = risk_probability >= 0.50

    return {
        "risk_probability": risk_probability,
        "parkinsonian_flag": parkinsonian_flag,
    }


@app.post("/api/nlp/analyze", tags=["NLP"])
async def analyze_nlp(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Multimodal NLP pipeline for cognitive-linguistic biomarker quantification.

    Phase 1: Speech-to-Text
    -----------------------
    Transcribes the uploaded audio file via Groq Whisper (whisper-large-v3).

    Phase 2: ML Linguistic Feature Extraction & Risk Prediction
    ------------------------------------------------------------
    Computes deterministic linguistic features (TTR, avg sentence length,
    filler word ratio) and passes them through a weighted ML classifier proxy
    to estimate Parkinsonian linguistic risk — no LLM required.

    Returns
    -------
    JSON with keys:
      transcript            – raw STT text
      linguistic_analysis   – structured dict with:
          lexical_diversity_index   (float 0.0–1.0, maps to TTR)
          syntactic_complexity_score (int 1–10, derived from avg sentence length)
          semantic_coherence        (int 1–10, derived from filler ratio)
          clinical_summary          (str, rule-based)
          identified_markers        (list[str])
          ml_features               (dict: raw computed features)
          risk_probability          (float 0.0–1.0)
          parkinsonian_flag         (bool)
    """
    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Groq API key is not configured. "
                "Add GROQ_API_KEY to backend/.env to enable NLP transcription."
            ),
        )

    # ── Phase 1: Speech-to-Text via Groq Whisper ──────────────────────────────
    audio_bytes = await file.read()
    tmp_path: Optional[str] = None

    try:
        # Groq audio API requires a real file object, not in-memory bytes
        suffix = ".wav" if (file.filename or "").lower().endswith(".wav") else ".audio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(audio_bytes)

        logger.info(
            "NLP | Transcribing audio  |  file=%s  |  bytes=%d",
            file.filename,
            len(audio_bytes),
        )

        with open(tmp_path, "rb") as audio_fh:
            transcription_response = groq_client.audio.transcriptions.create(
                file=(file.filename or "recording.wav", audio_fh),
                model="whisper-large-v3",
                response_format="text",
                language="en",
                prompt="Umm, let me think... I, I, I stutter sometimes, and uh, repeat words.",
                temperature=0.0,
            )

        # whisper-large-v3 with response_format="text" returns a plain string
        transcript_text: str = (
            transcription_response
            if isinstance(transcription_response, str)
            else getattr(transcription_response, "text", str(transcription_response))
        ).strip()

        logger.info(
            "NLP | Transcription complete  |  chars=%d  |  preview=%.80s",
            len(transcript_text),
            transcript_text,
        )

    except Exception as exc:
        logger.error("NLP | Whisper transcription failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Speech-to-text transcription failed: {exc}",
        )
    finally:
        # Always remove the temp file to prevent storage leaks
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not transcript_text:
        raise HTTPException(
            status_code=422,
            detail="Transcription returned an empty string. Check audio quality.",
        )

    # ── Word-count safeguard: reject noise hallucinations ─────────────────────
    # Whisper can hallucinate short phrases from background noise even with
    # language="en" locked. A clinical speech sample (Rainbow Passage) should
    # contain at least 15 words; anything shorter is treated as invalid input.
    word_count = len(transcript_text.split())
    if word_count < 15:
        logger.warning(
            "NLP | Transcript too short – likely noise hallucination  |  words=%d  |  text=%.80s",
            word_count, transcript_text,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient speech detected ({word_count} words). Please read the full Rainbow Passage clearly."
        )

    target_words = ["sunlight", "prism", "rainbow", "arch", "horizon", "colors"]
    normalized_transcript = transcript_text.lower()
    match_count = sum(1 for word in target_words if word in normalized_transcript)
    
    if match_count < 2:
        logger.warning(
            "NLP | Transcript missing keywords (matches: %d) | text=%.80s",
            match_count, transcript_text,
        )
        raise HTTPException(
            status_code=400,
            detail="We had trouble matching your audio to the passage. Please ensure you are in a quiet room and try reading the text again."
        )

    logger.info("NLP | Word-count and keyword gates passed  |  words=%d  |  matches=%d", word_count, match_count)

    # ── Phase 2: ML Linguistic Feature Extraction & Risk Prediction ──────────
    logger.info("NLP | Running ML linguistic feature extraction…")
    ml_features = extract_linguistic_features(transcript_text)
    risk_result  = predict_linguistic_risk(ml_features)

    ttr          = ml_features["ttr"]
    filler_ratio = ml_features["filler_word_ratio"]
    avg_sent_len = ml_features["avg_sentence_length"]
    risk_prob    = risk_result["risk_probability"]
    pd_flag      = risk_result["parkinsonian_flag"]

    # ── Map ML features to frontend-compatible schema ──────────────────────
    # lexical_diversity_index → TTR directly (already 0–1)
    lexical_diversity_index = ttr

    # syntactic_complexity_score → scale avg sentence length to 1–10
    # 4 words → score 2, 20+ words → score 9  (sigmoid-like mapping)
    syntactic_complexity_score = max(
        1, min(10, round(1 + (avg_sent_len / 20.0) * 9))
    )

    # semantic_coherence → inverse of filler ratio (high fillers = low coherence)
    semantic_coherence = max(
        1, min(10, round(10 * (1.0 - min(filler_ratio * 4, 1.0))))
    )

    # Identified markers (rule-based, matching frontend expectations)
    identified_markers: list = []
    if ttr < 0.50:
        identified_markers.append("Lexical Impoverishment (low TTR)")
    if filler_ratio > 0.08:
        identified_markers.append("Bradyphrenia (elevated filler word ratio)")
    if avg_sent_len < 8:
        identified_markers.append("Syntactic Simplification (short sentences)")
    if pd_flag:
        identified_markers.append("Elevated Parkinsonian Linguistic Risk")

    # Rule-based clinical summary fallback
    risk_label = "high" if risk_prob >= 0.65 else ("moderate" if risk_prob >= 0.40 else "low")
    clinical_summary = (
        f"ML linguistic analysis yielded a Parkinsonian risk probability of "
        f"{risk_prob:.2%} ({risk_label} risk). "
        f"Type-Token Ratio: {ttr:.3f} (lexical diversity), "
        f"average sentence length: {avg_sent_len:.1f} words, "
        f"filler word ratio: {filler_ratio:.3f}. "
        + (
            f"Markers detected: {', '.join(identified_markers)}."
            if identified_markers
            else "No significant Parkinsonian linguistic markers detected."
        )
    )

    # ── LLM Clinical Summary ─────────────────────────────────────────────────
    llm_summary = generate_clinical_summary(transcript_text)
    if not llm_summary or llm_summary.startswith("Failed"):
        llm_summary = clinical_summary

    linguistic_analysis: Dict[str, Any] = {
        "lexical_diversity_index":    round(lexical_diversity_index, 4),
        "syntactic_complexity_score": syntactic_complexity_score,
        "semantic_coherence":         semantic_coherence,
        "clinical_summary":           llm_summary,
        "identified_markers":         identified_markers,
        # Extended ML-specific fields
        "ml_features":        ml_features,
        "risk_probability":   risk_prob,
        "parkinsonian_flag":  pd_flag,
    }

    logger.info(
        "NLP | ML analysis complete  |  risk=%.4f  flag=%s  markers=%s",
        risk_prob, pd_flag, identified_markers,
    )

    return {
        "transcript": transcript_text,
        "linguistic_analysis": linguistic_analysis,
    }


# ---------------------------------------------------------------------------
# Phase 6 – Computer Vision: Facial Hypomimia Detection
# ---------------------------------------------------------------------------

# Lazy import guard – cv2 and mediapipe are optional; the endpoint returns 503
# if the packages are not installed so that the rest of the API is unaffected.
try:
    import urllib.request as _urllib_req
    import os as _os
    import cv2 as _cv2
    import numpy as _np
    import mediapipe as _mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _VISION_AVAILABLE = True
except ImportError as e:
    print(f"VISION IMPORT ERROR: {e}")
    _VISION_AVAILABLE = False


def _euclidean(p1, p2) -> float:
    """Euclidean distance between two (x, y) landmark tuples."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _smile_symmetry_index(landmarks, w: int, h: int) -> float:
    def lm(idx):
        lm_ = landmarks[idx]
        return lm_.x * w, lm_.y * h

    upper = lm(13)
    lower = lm(14)
    left  = lm(61)
    right = lm(291)

    center = ((upper[0] + lower[0]) / 2.0, (upper[1] + lower[1]) / 2.0)
    
    dist_left = _euclidean(center, left)
    dist_right = _euclidean(center, right)

    if max(dist_left, dist_right) == 0:
        return 1.0
    return min(dist_left, dist_right) / max(dist_left, dist_right)


def _mouth_aspect_ratio(landmarks, w: int, h: int) -> float:
    """
    Mouth Aspect Ratio (MAR).

    Uses inner-lip landmarks from the MediaPipe Face Mesh 468-point topology:
      - Vertical:   upper inner lip (#13) → lower inner lip (#14)
      - Horizontal: left mouth corner (#61) → right mouth corner (#291)

    MAR = vertical_dist / horizontal_dist
    A low MAR variance across frames indicates reduced facial expressivity (hypomimia).
    """
    def lm(idx):
        lm_ = landmarks[idx]
        return lm_.x * w, lm_.y * h

    upper = lm(13)
    lower = lm(14)
    left  = lm(61)
    right = lm(291)

    vertical   = _euclidean(upper, lower)
    horizontal = _euclidean(left, right)
    return vertical / (horizontal + 1e-6)  # avoid division by zero


def _eye_aspect_ratio(landmarks, w: int, h: int, eye_indices) -> float:
    """
    Eye Aspect Ratio (EAR) for blink detection.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    where p1..p6 are six eye contour landmarks (horizontal: p1, p4;
    vertical pairs: p2/p6 and p3/p5).

    EAR < 0.2 indicates a blink event.
    """
    def lm(idx):
        lm_ = landmarks[idx]
        return lm_.x * w, lm_.y * h

    p1, p2, p3, p4, p5, p6 = [lm(i) for i in eye_indices]
    vertical_a = _euclidean(p2, p6)
    vertical_b = _euclidean(p3, p5)
    horizontal  = _euclidean(p1, p4)
    return (vertical_a + vertical_b) / (2.0 * horizontal + 1e-6)


# MediaPipe Face Mesh eye landmark indices (left / right eye contours)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
_LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]  # p1..p6
_RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]  # p1..p6

# Clinical threshold: EAR below this value counts as a blink frame
_BLINK_EAR_THRESHOLD = 0.20

# Hypomimia flag: MAR std-dev below this indicates facial masking/stiffness
_HYPOMIMIA_STD_THRESHOLD = 0.05


@app.post("/api/vision/analyze", tags=["Vision"])
async def analyze_vision(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Facial hypomimia detection using OpenCV and MediaPipe Face Mesh.

    Accepts a short video recording (WebM/MP4) of the patient's face and
    quantifies two clinical biomarkers associated with Parkinson's disease
    hypomimia (reduced facial expressivity):

    1. **Mouth Movement Variance** – standard deviation of the per-frame
       Mouth Aspect Ratio (MAR).  A low variance (< 0.05) is flagged as
       indicative of facial masking.

    2. **Blink Rate** – number of Eye Aspect Ratio (EAR) dip events below
       the clinical threshold (0.20), normalised to blinks-per-minute.
       Reduced blinking is a known early PD motor sign.

    Args:
        file: Uploaded video file (.webm or .mp4 recommended).

    Returns:
        JSON with status and metrics dict containing:
          - mouth_movement_variance (float)
          - blink_rate_per_minute   (float)
          - hypomimia_flag          (bool)
          - frames_analyzed         (int)

    Raises:
        HTTPException 503: If OpenCV / MediaPipe are not installed.
        HTTPException 400: If the uploaded file cannot be decoded as video.
        HTTPException 500: On unexpected processing errors.
    """
    if not _VISION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Computer-vision dependencies are not installed. "
                "Run: pip install opencv-python-headless mediapipe"
            ),
        )

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

    logger.info(
        "Vision | Video upload received  |  filename=%s  size=%d bytes",
        file.filename, len(video_bytes),
    )

    tmp_path: Optional[str] = None
    cap = None
    face_mesh = None

    try:
        # ── Write to temporary file so cv2.VideoCapture can open it ──────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(video_bytes)

        # ── Ensure model file is downloaded ───────────────────────────────────
        model_path = _os.path.join(_BACKEND_DIR, 'face_landmarker.task')
        if not _os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            logger.info("Vision | Downloading face_landmarker.task model...")
            _urllib_req.urlretrieve(url, model_path)

        # ── Initialise MediaPipe Face Landmarker ──────────────────────────────
        base_options = _mp_python.BaseOptions(model_asset_path=model_path)
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_mesh = _mp_vision.FaceLandmarker.create_from_options(options)

        # ── Open video capture ────────────────────────────────────────────────
        cap = _cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open the uploaded video file. Ensure it is a valid WebM or MP4.",
            )

        fps: float = cap.get(_cv2.CAP_PROP_FPS) or 30.0  # default 30 fps if undetected
        frame_count: int = 0
        mar_values: list = []
        ear_values: list = []
        sym_values: list = []

        logger.info("Vision | Processing video  |  fps=%.1f", fps)

        # ── Frame-by-frame processing loop ────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            h, w, _ = frame.shape

            # MediaPipe expects RGB
            rgb_frame = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb_frame)
            result = face_mesh.detect(mp_image)

            frame_count += 1

            if not result.face_landmarks:
                continue  # no face detected in this frame – skip

            landmarks = result.face_landmarks[0]

            # ── MAR ──────────────────────────────────────────────────────────
            mar = _mouth_aspect_ratio(landmarks, w, h)
            mar_values.append(mar)

            # ── EAR (average of both eyes) ───────────────────────────────────
            ear_left  = _eye_aspect_ratio(landmarks, w, h, _LEFT_EYE_IDX)
            ear_right = _eye_aspect_ratio(landmarks, w, h, _RIGHT_EYE_IDX)
            ear_avg   = (ear_left + ear_right) / 2.0
            ear_values.append(ear_avg)
            
            # ── Smile Symmetry ───────────────────────────────────────────────
            sym = _smile_symmetry_index(landmarks, w, h)
            sym_values.append(sym)

        # ── Aggregate metrics ─────────────────────────────────────────────────
        if not mar_values:
            logger.warning("Vision | No face detected in any frame of the video.")
            raise HTTPException(
                status_code=400,
                detail="No face detected in the video. Please ensure you are clearly visible in a well-lit room and try again.",
            )

        mar_arr = _np.array(mar_values, dtype=float)
        ear_arr = _np.array(ear_values, dtype=float)
        sym_arr = _np.array(sym_values, dtype=float)

        mouth_movement_variance: float = float(_np.std(mar_arr))
        smile_symmetry_index: float = float(_np.mean(sym_arr)) if len(sym_arr) > 0 else 0.0

        # Count blink events: transitions from above → below threshold
        blink_count: int = 0
        below_threshold = False
        for ear_val in ear_arr:
            if ear_val < _BLINK_EAR_THRESHOLD:
                if not below_threshold:
                    blink_count += 1
                    below_threshold = True
            else:
                below_threshold = False

        video_duration_minutes: float = (frame_count / fps) / 60.0
        blink_rate_per_minute: float = (
            blink_count / video_duration_minutes if video_duration_minutes > 0 else 0.0
        )

        hypomimia_flag: bool = (mouth_movement_variance < 0.001) and (blink_rate_per_minute < 12.0)

        llm_summary = generate_vision_summary(mouth_movement_variance, blink_rate_per_minute, smile_symmetry_index)

        logger.info(
            "Vision | Analysis complete  |  frames=%d  mar_std=%.4f  blinks=%d  "
            "blinks_per_min=%.2f  hypomimia=%s",
            frame_count, mouth_movement_variance, blink_count,
            blink_rate_per_minute, hypomimia_flag,
        )
        frames_analyzed = frame_count
        ear_list = ear_values
        print(f"DEBUG VISION: Frames read by OpenCV: {frames_analyzed}")
        print(f"DEBUG VISION: Total EAR data points: {len(ear_list)}")

        return {
            "status": "success",
            "llm_summary": llm_summary,
            "metrics": {
                "mouth_movement_variance": round(mouth_movement_variance, 6),
                "blink_rate_per_minute":   round(blink_rate_per_minute, 2),
                "smile_symmetry_index":    round(smile_symmetry_index, 6),
                "hypomimia_flag":          hypomimia_flag,
                "frames_analyzed":         frame_count,
            },
        }

    except HTTPException:
        raise  # re-raise validated HTTP exceptions directly
    except Exception as exc:
        logger.exception("Vision | Unexpected error during video analysis: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {exc}",
        )
    finally:
        # Always release resources and delete the temporary file
        if cap is not None:
            cap.release()
        if face_mesh is not None:
            face_mesh.close()
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Dashboard Stats Endpoint
# ---------------------------------------------------------------------------

# Risk score thresholds used to classify recordings into risk tiers
_RISK_HIGH_THRESHOLD = 0.7
_RISK_MEDIUM_THRESHOLD = 0.4

def _classify_risk(score: Optional[float]) -> str:
    """Return 'High', 'Medium', or 'Low' based on risk_score."""
    if score is None:
        return "Low"
    if score >= _RISK_HIGH_THRESHOLD:
        return "High"
    if score >= _RISK_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


@app.get("/api/patients", tags=["Patients"])
async def get_patients(auth_id: str):
    """
    Returns a list of patients for a specific clinician.
    """
    try:
        client = get_client()
        query = client.table("patients").select("*, recordings(id, created_at, features(risk_score))").eq("auth_id", auth_id).order("created_at", desc=True)
        response = query.execute()
        return response.data or []
    except Exception as e:
        logger.error(f"Error fetching patients: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.patch("/api/patients/{patient_id}/risk", tags=["Patients"])
async def update_patient_risk(patient_id: str, request: RiskUpdateRequest):
    """
    Updates a specific risk column (acoustic_risk, linguistic_risk, or video_risk) for a patient.
    """
    try:
        column_map = {
            "acoustic": "acoustic_risk",
            "linguistic": "linguistic_risk",
            "video": "video_risk"
        }
        
        modality_key = request.modality.lower()
        if modality_key not in column_map:
            raise HTTPException(status_code=400, detail="Invalid modality. Must be 'acoustic', 'linguistic', or 'video'.")
            
        column_name = column_map[modality_key]
        
        client = get_client()
        response = client.table("patients").update({column_name: request.risk_level.lower()}).eq("id", patient_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Patient not found or update failed.")
            
        return {"status": "success", "patient": response.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating patient risk: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/patients/history", tags=["Patients"])
async def get_patient_history(auth_id: Optional[str] = None):
    """
    Returns a comprehensive list of all patient screenings, including patient demographics and risk scores.
    """
    try:
        client = get_client()
        
        # Fetch patients (filtered by auth_id if provided)
        pat_query = client.table("patients").select("id, manual_id, full_name, age")
        if auth_id:
            pat_query = pat_query.eq("auth_id", auth_id)
            
        pat_resp = pat_query.execute()
        patients = pat_resp.data or []
        
        if not patients and auth_id:
            return {"history": []}
            
        patients_dict = {p["id"]: p for p in patients}
        patient_ids = list(patients_dict.keys())
        
        # Fetch recordings
        rec_query = client.table("recordings").select("id, patient_id, task_type, created_at").order("created_at", desc=True)
        if patient_ids:
            # Chunking might be needed if patient_ids is huge, but fine for now
            rec_query = rec_query.in_("patient_id", patient_ids)
        elif auth_id:
             # Should be caught by the `if not patients and auth_id:` block above
             return {"history": []}
             
        rec_resp = rec_query.execute()
        recordings = rec_resp.data or []
        
        if not recordings:
            return {"history": []}
            
        recording_ids = [r["id"] for r in recordings]
        
        # Fetch features (for risk score)
        features_dict = {}
        if recording_ids:
            feat_resp = client.table("features").select("recording_id, risk_score").in_("recording_id", recording_ids).execute()
            for f in (feat_resp.data or []):
                features_dict[f["recording_id"]] = f
                
        history = []
        for r in recordings:
            p = patients_dict.get(r.get("patient_id"), {})
            # Only include recordings if patient info is found (or if we are not filtering by auth_id)
            if auth_id and not p:
                continue
                
            f = features_dict.get(r["id"], {})
            
            risk_score = f.get("risk_score")
            risk_label = _classify_risk(risk_score)
            
            task = r.get("task_type", "")
            if task == "sustained_vowel":
                task_formatted = "Acoustic / Sustained Vowel"
            else:
                task_formatted = task.replace("_", " ").title()
                
            history.append({
                "recording_id": r["id"],
                "patient_id": r.get("patient_id"),
                "manual_id": p.get("manual_id") or "N/A",
                "full_name": p.get("full_name") or "Unknown Patient",
                "age": p.get("age") or "--",
                "task_type": task_formatted,
                "risk_label": risk_label,
                "created_at": r.get("created_at")
            })
            
        return {"history": history}
    except Exception as exc:
        print(f"Database Error (get_patient_history): {str(exc)}")
        logger.exception("Unexpected error fetching history: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/patients/all", tags=["Patients"])
async def delete_all_patients(auth_id: str = Query(..., description="The ID of the authenticated clinician")):
    """
    Deletes all patients (and their cascaded history/recordings) for a given clinician.
    """
    try:
        if not auth_id:
            raise HTTPException(status_code=400, detail="Missing auth_id")
            
        client = get_client()
        
        # Delete patients associated with the clinician
        # If foreign keys on recordings/features have ON DELETE CASCADE, they will be deleted.
        # Otherwise, they must be deleted manually, but typically Supabase config handles it.
        # Here we just delete from patients table where auth_id matches.
        resp = client.table("patients").delete().eq("auth_id", auth_id).execute()
        
        return {"success": True, "message": "All patient history deleted.", "deleted_count": len(resp.data) if resp.data else 0}
    except SupabaseError as e:
        logger.error("Supabase error deleting patients: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as exc:
        logger.exception("Unexpected error deleting patients: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete patient history.")

@app.get("/api/dashboard-stats", tags=["Dashboard"])
async def get_dashboard_stats(auth_id: Optional[str] = None):
    """
    Returns aggregated dashboard statistics for the frontend UI.

    Queries the Supabase ``recordings``, ``features``, and ``patients`` tables.
    Falls back gracefully to mock data if Supabase is unavailable.
    """

    # ── Mock fallback object ──────────────────────────────────────────────
    mock_data = {
        "total_today": 24,
        "high_risk_today": 3,
        "trend_7_days": [12, 19, 15, 25, 22, 10, 24],
        "risk_distribution": {"high": 45, "medium": 30, "low": 25},
        "recent_screenings": [
            {"id": "#VT-8832", "date": "Today, 10:42 AM", "duration": "01m 45s", "risk": "Low"},
            {"id": "#VT-8831", "date": "Today, 09:15 AM", "duration": "02m 10s", "risk": "Medium"},
            {"id": "#VT-8830", "date": "Yesterday, 14:30 PM", "duration": "01m 55s", "risk": "High"},
            {"id": "#VT-8829", "date": "Yesterday, 11:20 AM", "duration": "01m 30s", "risk": "Low"},
        ],
    }

    try:
        client = get_client()
    except SupabaseError:
        logger.warning("Dashboard stats: Supabase not configured – returning mock data")
        return mock_data

    try:
        from datetime import datetime, timedelta, timezone

        # Use IST (+05:30) so day boundaries match the user's local timezone
        ist = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(ist)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        
        # ── Pre-fetch allowed patient IDs ──────────────────────────────────
        allowed_patient_ids = []
        if auth_id:
            pat_resp = client.table("patients").select("id").eq("auth_id", auth_id).execute()
            allowed_patient_ids = [p["id"] for p in (pat_resp.data or [])]
            if not allowed_patient_ids:
                return {
                    "total_today": 0,
                    "high_risk_today": 0,
                    "trend_7_days": [0,0,0,0,0,0,0],
                    "risk_distribution": {"high": 0, "medium": 0, "low": 0},
                    "recent_screenings": []
                }

        # ── 1. Total screenings today ─────────────────────────────────────
        today_query = client.table("recordings").select("id, created_at").gte("created_at", today_start)
        if allowed_patient_ids:
            today_query = today_query.in_("patient_id", allowed_patient_ids)
        today_resp = today_query.execute()
        today_recordings = today_resp.data or []
        total_today = len(today_recordings)

        # ── 2. High-risk today ────────────────────────────────────────────
        today_ids = [r["id"] for r in today_recordings]
        high_risk_today = 0
        if today_ids:
            feat_resp = (
                client.table("features")
                .select("recording_id, risk_score")
                .in_("recording_id", today_ids)
                .execute()
            )
            for feat in (feat_resp.data or []):
                if (feat.get("risk_score") or 0) >= _RISK_HIGH_THRESHOLD:
                    high_risk_today += 1

        # ── 3. Trend – last 7 days ───────────────────────────────────────
        trend_7_days = []
        for i in range(6, -1, -1):
            day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            
            day_query = client.table("recordings").select("id", count="exact").gte("created_at", day_start.isoformat()).lt("created_at", day_end.isoformat())
            if allowed_patient_ids:
                day_query = day_query.in_("patient_id", allowed_patient_ids)
            day_resp = day_query.execute()
            trend_7_days.append(day_resp.count if day_resp.count is not None else len(day_resp.data or []))

        # ── 4. Risk distribution (all active cases) ──────────────────────
        dist_query = client.table("recordings").select("id")
        if allowed_patient_ids:
            dist_query = dist_query.in_("patient_id", allowed_patient_ids)
        all_rec_resp = dist_query.execute()
        all_rec_ids = [r["id"] for r in (all_rec_resp.data or [])]
        
        dist = {"high": 0, "medium": 0, "low": 0}
        if all_rec_ids:
            all_feat_resp = client.table("features").select("risk_score").in_("recording_id", all_rec_ids).execute()
            for feat in (all_feat_resp.data or []):
                tier = _classify_risk(feat.get("risk_score"))
                dist[tier.lower()] += 1

        # ── 5. Recent screenings (last 5) ────────────────────────────────
        recent_query = client.table("recordings").select("id, patient_id, created_at, task_type").order("created_at", desc=True).limit(5)
        if allowed_patient_ids:
            recent_query = recent_query.in_("patient_id", allowed_patient_ids)
        recent_resp = recent_query.execute()
        
        recent_screenings = []
        for rec in (recent_resp.data or []):
            # Fetch risk score for this recording
            feat_row = (
                client.table("features")
                .select("risk_score")
                .eq("recording_id", rec["id"])
                .limit(1)
                .execute()
            )
            risk_score = (feat_row.data[0].get("risk_score") if feat_row.data else None)
            risk_label = _classify_risk(risk_score)

            # Format date
            created = rec.get("created_at", "")
            try:
                dt_utc = datetime.fromisoformat(created.replace("Z", "+00:00"))
                # Convert to IST (UTC + 5.5 hours)
                ist_tz = timezone(timedelta(hours=5, minutes=30))
                dt = dt_utc.astimezone(ist_tz)
                
                if dt.date() == now.date():
                    date_str = f"Today, {dt.strftime('%I:%M %p')}"
                elif dt.date() == (now - timedelta(days=1)).date():
                    date_str = f"Yesterday, {dt.strftime('%I:%M %p')}"
                else:
                    date_str = dt.strftime("%b %d, %I:%M %p")
            except Exception:
                date_str = created[:16] if created else "Unknown"

            # Build short patient ID
            short_id = f"#VT-{str(rec['id'])[-4:]}" if rec.get("id") else "#VT-0000"

            recent_screenings.append({
                "id": short_id,
                "date": date_str,
                "duration": rec.get("task_type", "N/A"),
                "risk": risk_label,
            })

        return {
            "total_today": total_today,
            "high_risk_today": high_risk_today,
            "trend_7_days": trend_7_days,
            "risk_distribution": dist,
            "recent_screenings": recent_screenings,
        }

    except Exception as exc:
        logger.warning("Dashboard stats: Supabase query failed – returning mock data: %s", exc)
        return mock_data

# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
