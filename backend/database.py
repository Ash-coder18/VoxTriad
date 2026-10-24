"""
VoxTriad – Supabase Database Operations
Phase 2 Backend

Thin async wrapper around the Supabase Python client providing:
  - Patient lookup / upsert
  - Recording row insert
  - Feature row insert (including pgvector embedding)
  - Similarity search via pgvector cosine distance

All functions raise SupabaseError (a plain RuntimeError subclass) so the
FastAPI handlers can catch them cleanly and return structured 500 responses.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()

from supabase import Client, create_client

logger = logging.getLogger(__name__)


# =============================================================================
# Client singleton
# =============================================================================

_supabase_client: Optional[Client] = None


class SupabaseError(RuntimeError):
    """Raised when a Supabase operation fails."""


def get_client() -> Client:
    """
    Return (and lazily initialise) the Supabase client singleton.

    Reads SUPABASE_URL and SUPABASE_ANON_KEY from the environment.
    Call load_dotenv() before this in main.py so .env is respected.

    Returns:
        Authenticated Supabase client.

    Raises:
        SupabaseError: If credentials are missing from the environment.
    """
    global _supabase_client

    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL", "")
        # Prefer service role key on backend to bypass RLS for server-side operations
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")

        if not url or not key:
            raise SupabaseError(
                "SUPABASE_URL and SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY) must be set in the environment. "
                "Copy backend/.env.example to backend/.env and fill in your credentials."
            )

        _supabase_client = create_client(url, key)
        logger.info("Supabase client initialised  |  project: %s", url)

    return _supabase_client


# =============================================================================
# Patient operations
# =============================================================================


def get_or_create_patient(auth_id: str) -> Dict[str, Any]:
    """
    Fetch an existing patient row or create a new one for the given auth user.

    Args:
        auth_id: The Supabase Auth user UUID (from JWT sub claim).

    Returns:
        The patient row as a dict.

    Raises:
        SupabaseError: On any database error.
    """
    client = get_client()

    try:
        resp = (
            client.table("patients")
            .select("*")
            .eq("auth_id", auth_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise SupabaseError(f"patients lookup failed: {exc}") from exc

    if resp.data:
        return resp.data[0]

    # Create a new patient row
    try:
        resp = (
            client.table("patients")
            .insert({"auth_id": auth_id})
            .execute()
        )
    except Exception as exc:
        raise SupabaseError(f"patients insert failed: {exc}") from exc

    if not resp.data:
        raise SupabaseError("patients insert returned no data")

    logger.info("New patient created  |  auth_id=%s", auth_id)
    return resp.data[0]


def get_or_create_anonymous_patient() -> Dict[str, Any]:
    """
    Fetch or create a stable 'Anonymous' patient row used for unauthenticated
    uploads.  The row is keyed on a fixed sentinel auth_id so it is only ever
    created once and reused for all anonymous sessions.

    Returns:
        The patient row dict (always contains a valid 'id' UUID).

    Raises:
        SupabaseError: On any database error.
    """
    ANON_SENTINEL = "voxtriad-anonymous-uploads"
    return get_or_create_patient(ANON_SENTINEL)


# =============================================================================
# Recording operations
# =============================================================================


def insert_recording(
    patient_id: str,
    storage_path: str,
    duration_sec: float,
    sample_rate_hz: int,
    file_size_bytes: int,
    task_type: str = "sustained_vowel",
) -> Dict[str, Any]:
    """
    Insert a new row into the `recordings` table.

    Args:
        patient_id:      UUID of the patient row.
        storage_path:    Object key in the Supabase Storage 'recordings' bucket.
        duration_sec:    Recording duration in seconds.
        sample_rate_hz:  Audio sample rate.
        file_size_bytes: Raw byte size of the uploaded file.
        task_type:       Vocal task type (default 'sustained_vowel').

    Returns:
        The newly created recording row dict (including its UUID).

    Raises:
        SupabaseError: On database error.
    """
    client = get_client()

    payload = {
        "patient_id": patient_id,
        "storage_path": storage_path,
        "task_type": task_type,
        "duration_sec": duration_sec,
        "sample_rate_hz": sample_rate_hz,
        "channels": 1,
        "file_size_bytes": file_size_bytes,
    }

    try:
        resp = client.table("recordings").insert(payload).execute()
    except Exception as exc:
        raise SupabaseError(f"recordings insert failed: {exc}") from exc

    if not resp.data:
        raise SupabaseError("recordings insert returned no data")

    logger.info("Recording row inserted  |  id=%s", resp.data[0].get("id"))
    return resp.data[0]


def upload_audio_to_storage(
    audio_bytes: bytes,
    storage_path: str,
    content_type: str = "audio/wav",
) -> str:
    """
    Upload raw audio bytes to the 'recordings' Supabase Storage bucket.

    Args:
        audio_bytes:   Raw bytes to upload.
        storage_path:  Destination key inside the bucket (e.g. 'patient_uuid/rec_uuid.wav').
        content_type:  MIME type of the file.

    Returns:
        The public storage path (same as `storage_path`).

    Raises:
        SupabaseError: On upload failure.
    """
    client = get_client()

    try:
        client.storage.from_("recordings").upload(
            path=storage_path,
            file=audio_bytes,
            file_options={"content-type": content_type},
        )
    except Exception as exc:
        raise SupabaseError(f"Storage upload failed: {exc}") from exc

    logger.info("Audio uploaded to storage  |  path=%s  bytes=%d", storage_path, len(audio_bytes))
    return storage_path


# =============================================================================
# Feature operations
# =============================================================================


def insert_features(
    recording_id: str,
    classical: Dict[str, float],
    embedding: Optional[List[float]] = None,
    risk_score: Optional[float] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Insert extracted acoustic features into the `features` table.

    The `embedding` list is stored in the pgvector `vector(768)` column.
    Supabase / PostgREST accepts the vector as a plain Python list of floats.

    Args:
        recording_id:  UUID of the parent recording row.
        classical:     Dict from extract_classical_features().
        embedding:     768-element list from extract_deep_embeddings().tolist()
                       or None if deep extraction was skipped.
        risk_score:    Model-predicted PD risk probability [0, 1] or None.
        model_version: Identifier for the model version used.

    Returns:
        The newly created features row dict (including its UUID).

    Raises:
        SupabaseError: On database error.
    """
    client = get_client()

    payload: Dict[str, Any] = {
        "recording_id": recording_id,
        # Unpack classical feature dict into individual columns
        "f0_mean_hz":             classical.get("f0_mean_hz"),
        "f0_std_hz":              classical.get("f0_std_hz"),
        "f0_min_hz":              classical.get("f0_min_hz"),
        "f0_max_hz":              classical.get("f0_max_hz"),
        "f0_jitter_cv":           classical.get("f0_jitter_cv"),
        "spectral_centroid_mean": classical.get("spectral_centroid_mean"),
        "spectral_centroid_std":  classical.get("spectral_centroid_std"),
        "spectral_flux_mean":     classical.get("spectral_flux_mean"),
        "spectral_flux_std":      classical.get("spectral_flux_std"),
        "zcr_mean":               classical.get("zcr_mean"),
        "zcr_std":                classical.get("zcr_std"),
        "rms_mean":               classical.get("rms_mean"),
        "rms_std":                classical.get("rms_std"),
        "mfcc_mean":              [v for k, v in classical.items() if k.endswith("_mean") and k.startswith("mfcc")],
        "mfcc_std":               [v for k, v in classical.items() if k.endswith("_std")  and k.startswith("mfcc")],
    }

    if embedding is not None:
        # pgvector expects the vector as a string "[0.1, 0.2, ...]" or plain list
        payload["embedding"] = embedding

    if risk_score is not None:
        payload["risk_score"] = risk_score

    if model_version is not None:
        payload["model_version"] = model_version

    try:
        resp = client.table("features").insert(payload).execute()
    except Exception as exc:
        raise SupabaseError(f"features insert failed: {exc}") from exc

    if not resp.data:
        raise SupabaseError("features insert returned no data")

    logger.info("Features row inserted  |  id=%s", resp.data[0].get("id"))
    return resp.data[0]


# =============================================================================
# Similarity search (pgvector)
# =============================================================================


def find_similar_recordings(
    embedding: List[float],
    top_k: int = 5,
    distance_fn: str = "cosine",
) -> List[Dict[str, Any]]:
    """
    Find the most acoustically similar past recordings using pgvector ANN search.

    Uses a raw PostgREST RPC call to a Supabase database function named
    `match_recordings` (defined below – add this to schema.sql in Phase 3).

    Expected SQL function in Supabase:
    ```sql
    CREATE OR REPLACE FUNCTION match_recordings(
        query_embedding vector(768),
        match_count     int DEFAULT 5
    )
    RETURNS TABLE (
        recording_id    uuid,
        risk_score      real,
        similarity      float
    )
    LANGUAGE sql STABLE AS $$
        SELECT
            f.recording_id,
            f.risk_score,
            1 - (f.embedding <=> query_embedding) AS similarity
        FROM features f
        WHERE f.embedding IS NOT NULL
        ORDER BY f.embedding <=> query_embedding
        LIMIT match_count;
    $$;
    ```

    Args:
        embedding:    768-element float list (query vector).
        top_k:        Number of nearest neighbours to return.
        distance_fn:  'cosine' (only option currently supported).

    Returns:
        List of row dicts with keys: recording_id, risk_score, similarity.

    Raises:
        SupabaseError: On RPC failure.
    """
    client = get_client()

    try:
        resp = client.rpc(
            "match_recordings",
            {"query_embedding": embedding, "match_count": top_k},
        ).execute()
    except Exception as exc:
        raise SupabaseError(f"match_recordings RPC failed: {exc}") from exc

    return resp.data or []
