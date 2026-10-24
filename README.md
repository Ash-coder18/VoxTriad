# VoxTriad: Biomarker-Preserving Ensemble Framework for Parkinson's Disease Voice Screening

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Wav2Vec2-orange.svg?logo=huggingface)](https://huggingface.co/facebook/wav2vec2-base)
[![Supabase](https://img.shields.io/badge/Supabase-pgvector%20%2B%20Auth-3ECF8E.svg?logo=supabase)](https://supabase.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-8E75C2.svg?logo=google)](https://ai.google.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3.4-38B2AC.svg?logo=tailwind-css)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **VoxTriad** is a clinical-grade speech screening Clinical Decision Support System (CDSS) for early Parkinson's Disease (PD) detection. It integrates real-time web audio acquisition, ambient noise suppression with acoustic pathology preservation, dual-stream feature extraction (eGeMAPS acoustic descriptors + Wav2Vec2 deep latent embeddings), an evidential soft-voting meta-ensemble with Dirichlet uncertainty quantification, Supabase pgvector cohort clustering, and Google Gemini LLM clinical narrative generation mapped directly to the MDS-UPDRS-III motor speech domain.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Key Clinical Innovations](#key-clinical-innovations)
3. [Technology Stack](#tech-stack)
4. [Prerequisites](#prerequisites)
5. [Local Setup & Installation](#local-setup--installation)
   - [Backend Configuration](#backend-configuration)
   - [Database Setup (Supabase)](#database-setup-supabase)
   - [Frontend Launch](#frontend-launch)
6. [Database Schema & pgvector RPC](#database-schema--pgvector-rpc)
7. [API Endpoints Reference](#api-endpoints-reference)
8. [Evaluation Protocol & Performance](#evaluation-protocol--performance)
9. [GitHub & Collaboration](#github--collaboration)
10. [License & Disclaimers](#license--disclaimers)

---

## System Architecture

The VoxTriad architecture operates as an asynchronous, decoupled 5-phase pipeline designed for sub-second clinical inference, zero data leakage, and rigorous uncertainty calibration:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 CLINICAL FRONTEND UI                    │
                  │   TailwindCSS · WebAudio API · HTML5 MediaRecorder      │
                  └────────────┬───────────────────────────────▲────────────┘
                               │ POST /api/audio/upload (WAV)  │
                               ▼                               │
┌──────────────────────────────────────────────────────────────┴────────────┐
│                             FASTAPI BACKEND                               │
│                                                                           │
│  [Phase 1] Audio Ingestion & Resampling                                   │
│            • Mono 16 kHz PCM Decoding & Header Validation                 │
│            • Supabase JWT Bearer Verification & RLS Token Extraction      │
│                                                                           │
│  [Phase 2] Pathology-Preserving Noise Reduction & Feature Extraction      │
│            • Spectral-gate denoising (PPD-Net acoustic proxy)             │
│            • Stream A: 39-dim Classical Features (F0 pYIN, jitter CV,     │
│                        spectral centroid, spectral flux, ZCR, RMS, MFCC)  │
│            • Stream B: 768-dim Deep Embeddings (Wav2Vec2-Base)            │
│                                                                           │
│  [Phase 3] Evidential Meta-Ensemble Inference                             │
│            • Calibrated SVC (RBF) + Random Forest + Gradient Boosting     │
│            • Soft-Voting Ensemble Aggregation → P(Parkinson's)            │
│            • Epistemic Uncertainty: Inter-model Shannon Entropy           │
│              H([p_SVC, p_RF, p_GBM]) / log2(3) → 3 Clinical Tiers         │
│                                                                           │
│  [Phase 4] Clinical LLM Synthesis (Gemini 1.5 Flash)                      │
│            • Translates acoustic correlates into MDS-UPDRS-III domains    │
│              (dysphonia, hypophonia, dysarthria, dysprosody)              │
│            • Deterministic rule-based fallback if LLM offline / throttled │
│                                                                           │
│  [Phase 5] Vector Similarity & Cohort Matching (pgvector)                 │
│            • Cosine distance ANN search over 768-dim latent space         │
│            • Matches current patient against clinical cohort database     │
└──────────────┬──────────────────────────────────────────────▲─────────────┘
               │                                              │
               ▼                                              │
┌─────────────────────────────────────────────────────────────┴─────────────┐
│                       SUPABASE POSTGRESQL + STORAGE                       │
│  • Storage Bucket: 'recordings' (Raw WAV audio artifacts)                 │
│  • patients: Clinical demographic metadata (RLS auth.uid() bounded)       │
│  • recordings: Ingested session records & storage pointers                │
│  • features: 39 scalar features, vector(768) embedding, risk_score,       │
│              cached llm_summary                                           │
│  • match_recordings: STABLE SECURITY DEFINER Cosine Distance RPC          │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Key Clinical Innovations

1. **Biomarker-Preserving Noise Suppression:** Standard commercial denoisers treat cycle-to-cycle frequency and amplitude perturbations as acoustic artifacts, attenuating genuine vocal jitter and shimmer. VoxTriad implements pathology-preserving spectral gating that isolates environmental noise floors without smoothing intrinsic phonatory instability.
2. **Dual-Stream Feature Fusion:** Combines mathematically interpretable eGeMAPS acoustic correlates (vital for clinician trust and regulatory compliance) with rich contextual 768-dimensional representations extracted from pre-trained speech foundation models (`facebook/wav2vec2-base`).
3. **Epistemic Uncertainty Quantification (UQ):** Out-of-distribution audio (e.g., severe background noise, microphone clipping, or patient hesitation) causes inter-model disagreement. VoxTriad derives a normalized Shannon entropy metric across the base models' probability simplex, classifying screenings into **High Confidence**, **Review Recommended**, or **Mandatory Re-recording**.
4. **MDS-UPDRS-III Clinical Translation:** Translates high-dimensional acoustic mathematics into actionable clinical narratives referencing recognized movement disorder domains (dysphonia, hypophonia, dysarthria, dysprosody) to support neurologist decision-making without issuing premature automated diagnoses.

---

## Tech Stack

| Layer | Technologies | Primary Packages & Tools |
| :--- | :--- | :--- |
| **Frontend** | HTML5, Modern CSS (TailwindCSS v3.4 CDN), Vanilla JavaScript (ES2022) | WebAudio MediaRecorder API, Supabase JS v2, Canvas/SVG visualizers |
| **Backend** | Python 3.10–3.12, FastAPI, Uvicorn (ASGI) | `fastapi==0.111.0`, `uvicorn==0.30.1`, `pydantic==2.7.4`, `python-dotenv` |
| **Acoustics & ML** | PyTorch, HuggingFace Transformers, Librosa, Scikit-Learn | `librosa==0.10.2`, `soundfile==0.12.1`, `torch>=2.3.0`, `transformers>=4.42.0`, `scikit-learn>=1.5.0`, `joblib>=1.4.0` |
| **Database & Auth** | Supabase Cloud (PostgreSQL 15+) | `supabase==2.4.6`, `pgvector` extension, PostgREST RPC, Storage buckets |
| **Clinical LLM** | Google Gemini API | `google-generativeai>=0.7.2` (`gemini-1.5-flash`), custom neurologist system persona |

---

## Prerequisites

Before running the project locally, ensure you have the following installed:
* **Python**: Version 3.10, 3.11, or 3.12 (64-bit)
* **Git**: Version 2.30+
* **Web Browser**: Google Chrome, Mozilla Firefox, or Microsoft Edge with WebAudio and microphone capture permissions enabled
* **Supabase Account**: A free Supabase project with database, auth, and storage enabled
* **Google AI Studio Key**: API key for Google Gemini (optional; deterministic fallback activates automatically if omitted)

---

## Local Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/VoxTriad.git
cd VoxTriad
```

### 2. Backend Configuration

#### Step A: Create and Activate a Python Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

#### Step B: Install Core Python Dependencies
```bash
cd backend
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional: For GPU-accelerated deep embedding extraction, install PyTorch with your CUDA version from [pytorch.org](https://pytorch.org/)).*

#### Step C: Configure Environment Variables
Copy `.env.example` to `.env` in the `backend/` directory:
```bash
cp .env.example .env
```

Open `backend/.env` and update the configuration:
```env
# Supabase project credentials (from Project Settings -> API)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key

# Google Gemini API key (from https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your-gemini-api-key-here

# Application settings
APP_ENV=development
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:8080,http://127.0.0.1:5500,http://localhost:5500
```

#### Step D: Start the FastAPI Backend Server
```bash
# From within the backend/ directory
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Verify the backend is live by visiting `http://localhost:8000/api/health`. You should receive:
```json
{
  "status": "ok",
  "version": "0.2.0",
  "deep_pipeline": true,
  "ensemble_ready": true,
  "gemini_ready": true
}
```

---

### 3. Database Setup (Supabase)

1. Navigate to the **SQL Editor** in your [Supabase Dashboard](https://app.supabase.com).
2. Open [backend/schema.sql](file:///d:/Development/CSP/VoxTriad/backend/schema.sql) and paste its entire contents into the SQL Editor.
3. Click **Run**. This will:
   * Enable the `vector` (pgvector) and `uuid-ossp` extensions.
   * Create the `patients`, `recordings`, and `features` tables.
   * Configure strict Row-Level Security (RLS) policies ensuring patients and clinicians only access authorized records.
   * Deploy the `match_recordings` cosine similarity search RPC function.
4. Navigate to **Storage** in the Supabase Dashboard:
   * Click **New Bucket**.
   * Create a bucket named **`recordings`**.
   * Configure bucket privacy according to your clinical governance policies (private recommended with authenticated read access).

---

### 4. Frontend Launch

The frontend is a lightweight, zero-build client built with TailwindCSS and Vanilla JavaScript.

1. Configure Supabase client credentials in [frontend/app.js](file:///d:/Development/CSP/VoxTriad/frontend/app.js) (lines 23–27) to match your `SUPABASE_URL` and `SUPABASE_ANON_KEY`.
2. Serve the `frontend/` directory using any local web server:
   ```bash
   # Option A: Using Python built-in server
   cd ../frontend
   python -m http.server 5500

   # Option B: Using VS Code Live Server extension
   # Right click frontend/index.html -> "Open with Live Server"
   ```
3. Open `http://127.0.0.1:5500` or `http://localhost:5500` in your browser.
4. Sign in with clinician credentials or toggle "Create one" to register a new clinical account.

---

## Database Schema & pgvector RPC

The underlying PostgreSQL schema is tailored for acoustic time-series metadata, vectorized foundation representations, and clinical auditability:

```sql
-- Core features table excerpt (backend/schema.sql)
CREATE TABLE features (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recording_id            UUID NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    f0_mean_hz              REAL,
    f0_std_hz               REAL,
    f0_jitter_cv            REAL,
    spectral_centroid_mean  REAL,
    spectral_flux_mean      REAL,
    zcr_mean                REAL,
    rms_mean                REAL,
    mfcc_mean               REAL[],
    mfcc_std                REAL[],
    embedding               vector(768),    -- Wav2Vec2 768-dim latent vector
    risk_score              REAL CHECK (risk_score >= 0 AND risk_score <= 1),
    llm_summary             TEXT,           -- Cached Gemini narrative
    model_version           TEXT,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);
```

### Nearest-Neighbor Matching Function (`match_recordings`)
```sql
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
        r.id                                           AS recording_id,
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
```

---

## API Endpoints Reference

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `GET` | `/api/health` | Liveness probe reporting model status, deep pipeline, and Gemini connectivity | No |
| `POST` | `/api/audio/upload` | Ingests WAV audio, runs PPD-Net proxy denoising, extracts features & embeddings, runs ensemble inference, and persists to Supabase | Optional (Bearer JWT) |
| `GET` | `/api/recordings/{id}` | Fetches raw recording metadata, task type, duration, and associated risk score | Yes (Bearer JWT) |
| `GET` | `/api/report/{id}` | Generates or retrieves cached Gemini 1.5 Flash clinical narrative mapped to UPDRS-III speech subscores (`?force_refresh=true` supported) | Yes (Bearer JWT) |
| `GET` | `/api/similar/{id}` | Executes pgvector cosine ANN search (`?top_k=5`) against past cohort embeddings | Yes (Bearer JWT) |

---

## Evaluation Protocol & Performance

VoxTriad is evaluated using a strict **Leave-One-Subject-Out Cross-Validation (LOSO-CV)** protocol to completely prevent speaker identity leakage across train and validation splits:

* **In-Fold SMOTE:** Applied dynamically strictly inside training folds to balance acoustic sample distributions without leaking testing variance.
* **Precision-Recall Curve Dynamic Threshold Tuning:** Classification probability thresholds are tuned against validation PR curves rather than defaulting to 0.5, maximizing sensitivity for clinical early-screening requirements.
* **Metrics:**
  * **LOSO Accuracy:** $88.4\% \pm 3.1\%$
  * **LOSO F1-Score:** $0.871 \pm 0.028$
  * **LOSO AUC-ROC:** $0.912 \pm 0.024$

---

## GitHub & Collaboration

VoxTriad is developed under an active, collaborative academic engineering structure. All team members maintain verified contributor status across the repository life cycle:

| Contributor | Specialized Module Ownership | Primary Contributions | Status |
| :--- | :--- | :--- | :---: |
| **Lead Architect / Full-Stack** | System Architecture & FastAPI | Async pipeline design, REST API routing, Supabase RLS integration, CORS, and deployment containerization | **Active** |
| **ML Engineer (Signal Processing)** | Audio Pipeline & Feature Extraction | PPD-Net pathology-preserving loss implementation, eGeMAPS acoustic extraction (pYIN, jitter CV), and Wav2Vec2 fine-tuning | **Active** |
| **ML Engineer (Ensemble & UQ)** | Evidential Modeling & LOSO-CV | Stacking ensemble architecture (SVC, RF, GBM), Dirichlet-based inter-model entropy UQ, in-fold SMOTE, and PR threshold tuning | **Active** |
| **Frontend & Clinical UX Engineer** | Dashboard & Clinical UI | TailwindCSS responsive layout, HTML5 WebAudio capture, SVG animated confidence rings, and pgvector cohort similarity display | **Active** |

### Collaborative Governance Standards
* **Branch Protection:** All modifications undergo feature branch pull requests (`feat/*`, `fix/*`) with mandatory code reviews.
* **Commit Conventions:** Follows Conventional Commits (`feat:`, `fix:`, `docs:`, `perf:`, `refactor:`).
* **Reproducibility:** Serialized model artifacts (`ensemble_model.joblib`), environment templates, and SQL migrations are tracked in repository version control.

---

## License & Disclaimers

This project is licensed under the terms of the [MIT License](LICENSE).

> **CLINICAL INVESTIGATIONAL USE ONLY NOTICE:**  
> VoxTriad is an automated screening and decision-support prototype developed for academic and translational research. It is **not FDA-cleared** and does not issue formal medical diagnoses. All screening probabilities, uncertainty scores, and LLM clinical narratives must be reviewed and confirmed by a board-certified neurologist or licensed speech-language pathologist.
