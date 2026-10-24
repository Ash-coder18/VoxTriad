-- =============================================================================
-- VoxTriad – PostgreSQL Schema (Phase 1)
-- Designed for Supabase (PostgreSQL 15+ with pgvector extension).
-- Run this in the Supabase SQL Editor or via psql.
-- =============================================================================

-- Enable the pgvector extension (must be run as superuser / via Supabase dashboard)
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation helper
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- 1. patients
--    Links an application-level patient record to a Supabase Auth user.
-- =============================================================================
CREATE TABLE IF NOT EXISTS patients (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    auth_id     UUID        NOT NULL UNIQUE,   -- references auth.users(id) in Supabase
    full_name   TEXT,
    date_of_birth DATE,
    sex         TEXT        CHECK (sex IN ('male', 'female', 'other', 'prefer_not_to_say')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  patients          IS 'Clinical patient profiles, one row per registered user.';
COMMENT ON COLUMN patients.auth_id  IS 'Foreign key to Supabase Auth users table.';

-- Automatically bump updated_at on every row update
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_patients_updated_at
    BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- 2. recordings
--    One row per submitted audio capture session.
-- =============================================================================
CREATE TABLE IF NOT EXISTS recordings (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID        NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    -- Path/key inside Supabase Storage bucket 'recordings'
    storage_path    TEXT        NOT NULL,
    -- e.g. 'sustained_vowel', 'running_speech', 'diadochokinesis'
    task_type       TEXT        NOT NULL DEFAULT 'sustained_vowel',
    duration_sec    REAL,
    sample_rate_hz  INTEGER,
    channels        SMALLINT    DEFAULT 1,
    file_size_bytes INTEGER,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  recordings              IS 'Raw audio recording metadata; audio bytes live in Supabase Storage.';
COMMENT ON COLUMN recordings.storage_path IS 'Object key within the recordings Storage bucket.';
COMMENT ON COLUMN recordings.task_type    IS 'Vocal task performed during the recording session.';

CREATE INDEX IF NOT EXISTS idx_recordings_patient_id ON recordings(patient_id);
CREATE INDEX IF NOT EXISTS idx_recordings_recorded_at ON recordings(recorded_at DESC);

-- =============================================================================
-- 3. features
--    Acoustic features extracted per recording, plus a pgvector embedding
--    for similarity search and downstream ML inference.
-- =============================================================================
CREATE TABLE IF NOT EXISTS features (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    recording_id    UUID        NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,

    -- -------------------------------------------------------------------------
    -- Acoustic / clinical features (nullable – filled as pipeline matures)
    -- -------------------------------------------------------------------------
    -- Fundamental frequency statistics
    f0_mean_hz      REAL,
    f0_std_hz       REAL,
    f0_min_hz       REAL,
    f0_max_hz       REAL,

    -- Jitter (cycle-to-cycle frequency perturbation)
    f0_jitter_cv    REAL,   -- Coefficient of variation (proxy)
    jitter_local    REAL,   -- %
    jitter_rap      REAL,
    jitter_ppq5     REAL,

    -- Shimmer (cycle-to-cycle amplitude perturbation)
    shimmer_local   REAL,   -- %
    shimmer_apq3    REAL,
    shimmer_apq5    REAL,

    -- Spectral and energy features
    spectral_centroid_mean REAL,
    spectral_centroid_std  REAL,
    spectral_flux_mean     REAL,
    spectral_flux_std      REAL,
    zcr_mean               REAL,
    zcr_std                REAL,
    rms_mean               REAL,
    rms_std                REAL,

    -- Harmonics-to-noise ratio
    hnr_db          REAL,

    -- MFCC statistics (13 coefficients × mean + std = 26 values stored as arrays)
    mfcc_mean       REAL[],
    mfcc_std        REAL[],

    -- RPDE & DFA nonlinear measures (common in Parkinson's research)
    rpde            REAL,
    dfa             REAL,

    -- -------------------------------------------------------------------------
    -- pgvector embedding – 768-dim (e.g. from a speech foundation model)
    -- -------------------------------------------------------------------------
    embedding       vector(768),

    -- -------------------------------------------------------------------------
    -- ML & LLM output (Phase 3 & Phase 4)
    -- -------------------------------------------------------------------------
    risk_score      REAL    CHECK (risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 1)),
    llm_summary     TEXT,
    model_version   TEXT,

    -- -------------------------------------------------------------------------
    -- Metadata
    -- -------------------------------------------------------------------------
    extracted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  features           IS 'Acoustic features and pgvector embeddings extracted from each recording.';
COMMENT ON COLUMN features.embedding IS '768-dimensional vector embedding for ANN similarity search via pgvector.';
COMMENT ON COLUMN features.risk_score IS 'Model-predicted Parkinson''s risk probability [0, 1]; NULL until Phase 3.';

CREATE INDEX IF NOT EXISTS idx_features_recording_id ON features(recording_id);

-- IVFFlat index for approximate nearest-neighbour search on the embedding column.
-- Adjust lists= based on expected row count (sqrt(n_rows) is a good heuristic).
-- Build this index AFTER the table has been populated with at least a few hundred rows.
-- CREATE INDEX idx_features_embedding ON features USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================================================
-- Row-Level Security (RLS) – patients may only access their own data
-- =============================================================================
ALTER TABLE patients   ENABLE ROW LEVEL SECURITY;
ALTER TABLE recordings ENABLE ROW LEVEL SECURITY;
ALTER TABLE features   ENABLE ROW LEVEL SECURITY;

-- patients: users can read/update their own row only
CREATE POLICY patients_self_access ON patients
    USING (auth_id = auth.uid());

-- recordings: users can read recordings linked to their patient row
CREATE POLICY recordings_patient_access ON recordings
    USING (
        patient_id IN (
            SELECT id FROM patients WHERE auth_id = auth.uid()
        )
    );

-- features: users can read features linked to their recordings
CREATE POLICY features_patient_access ON features
    USING (
        recording_id IN (
            SELECT r.id
            FROM recordings r
            JOIN patients p ON p.id = r.patient_id
            WHERE p.auth_id = auth.uid()
        )
    );

-- =============================================================================
-- Phase 2 Additions
-- =============================================================================

-- Additional acoustic columns added to features table (run ALTER TABLE if schema
-- already exists in your Supabase project)
ALTER TABLE features
    ADD COLUMN IF NOT EXISTS jitter_local   REAL,
    ADD COLUMN IF NOT EXISTS jitter_rap     REAL,
    ADD COLUMN IF NOT EXISTS shimmer_local  REAL,
    ADD COLUMN IF NOT EXISTS shimmer_apq3   REAL,
    ADD COLUMN IF NOT EXISTS hnr_db         REAL,
    ADD COLUMN IF NOT EXISTS zcr_mean       REAL,
    ADD COLUMN IF NOT EXISTS zcr_std        REAL,
    ADD COLUMN IF NOT EXISTS rms_mean       REAL,
    ADD COLUMN IF NOT EXISTS rms_std        REAL,
    ADD COLUMN IF NOT EXISTS spectral_centroid_mean REAL,
    ADD COLUMN IF NOT EXISTS spectral_centroid_std  REAL,
    ADD COLUMN IF NOT EXISTS spectral_flux_mean     REAL,
    ADD COLUMN IF NOT EXISTS spectral_flux_std      REAL,
    ADD COLUMN IF NOT EXISTS f0_jitter_cv           REAL,
    -- Phase 4: cached Gemini LLM clinical narrative (avoids re-generation cost)
    ADD COLUMN IF NOT EXISTS llm_summary    TEXT;

COMMENT ON COLUMN features.llm_summary IS
    'Gemini-generated UPDRS-III clinical narrative. Cached after first generation.';

-- =============================================================================
-- pgvector ANN Search – match_recordings()
--
-- Called by GET /api/similar/{recording_id} via PostgREST RPC.
-- Returns the top-K most similar recordings by cosine distance on embeddings.
--
-- Expose via: Supabase Dashboard → SQL Editor → Run
-- =============================================================================
CREATE OR REPLACE FUNCTION match_recordings(
    query_embedding vector(768),
    match_count     int DEFAULT 5
)
RETURNS TABLE (
    recording_id    uuid,
    patient_id      uuid,
    task_type       text,
    risk_score      real,
    similarity      float
)
LANGUAGE sql STABLE SECURITY DEFINER AS $$
    SELECT
        r.id                                         AS recording_id,
        r.patient_id,
        r.task_type,
        f.risk_score,
        (1 - (f.embedding <=> query_embedding))::float AS similarity
    FROM features f
    JOIN recordings r ON r.id = f.recording_id
    WHERE f.embedding IS NOT NULL
    ORDER BY f.embedding <=> query_embedding
    LIMIT match_count;
$$;

COMMENT ON FUNCTION match_recordings IS
    'pgvector cosine ANN search over feature embeddings. Used by the /api/similar/{id} endpoint.';

-- Grant execute to authenticated users (Supabase RLS on the joined tables
-- still applies inside the function body via SECURITY INVOKER if changed)
GRANT EXECUTE ON FUNCTION match_recordings TO authenticated;

-- =============================================================================
-- Phase 3 TODO
-- =============================================================================
-- 1. Build IVFFlat index once ~500+ rows exist:
--    CREATE INDEX idx_features_embedding
--        ON features USING ivfflat (embedding vector_cosine_ops)
--        WITH (lists = 100);
--
-- 2. Add HNSW index (faster but more memory):
--    CREATE INDEX idx_features_embedding_hnsw
--        ON features USING hnsw (embedding vector_cosine_ops)
--        WITH (m = 16, ef_construction = 64);
--
-- 3. Add periodic_assessments table for longitudinal tracking
-- 4. Add clinical_reports table linking Gemini LLM output to recordings
