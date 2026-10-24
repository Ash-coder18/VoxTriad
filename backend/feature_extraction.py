"""
VoxTriad – Acoustic Feature Extraction Module
Phase 2 Backend

Mirrors the notebook pipeline in notebook/01_baseline_exploration.ipynb
but is structured for production use inside FastAPI:

  - All heavy models are loaded once at startup (module-level cache)
  - Functions accept raw numpy arrays (already decoded from the uploaded bytes)
  - Returns Pydantic-compatible plain dicts / numpy arrays
  - No matplotlib / notebook-only imports

Stage 1 → apply_spectral_gate_denoising()
Stage 2a → extract_classical_features()
Stage 2b → extract_deep_embeddings()          (requires torch + transformers)
Fusion  → fuse_feature_vectors()
"""

from __future__ import annotations

import io
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
import soundfile as sf
import librosa
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ── Optional heavy imports (graceful degradation if not installed) ─────────────
try:
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    _DEEP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DEEP_AVAILABLE = False
    logger.warning(
        "torch / transformers not installed – deep embedding extraction disabled. "
        "Run: pip install torch transformers"
    )

# =============================================================================
# Constants
# =============================================================================

TARGET_SR: int = 16_000          # Sample rate expected by wav2vec2 and all extractors
WAV2VEC2_MODEL_ID: str = "facebook/wav2vec2-base"
EMBED_DIM: int = 768             # wav2vec2-base hidden size
N_MFCC: int = 13
UNCERTAINTY_REVIEW_THRESHOLD: float = 0.65   # entropy score above which to flag

# =============================================================================
# Module-level model cache  (loaded lazily, once per worker process)
# =============================================================================

_wav2vec2_processor: Optional["Wav2Vec2Processor"] = None
_wav2vec2_model: Optional["Wav2Vec2Model"] = None
_device: Optional["torch.device"] = None


def _get_device() -> "torch.device":
    """Select the best available compute device."""
    global _device
    if _device is None:
        if _DEEP_AVAILABLE:
            if torch.backends.mps.is_available():
                _device = torch.device("mps")
            elif torch.cuda.is_available():
                _device = torch.device("cuda")
            else:
                _device = torch.device("cpu")
        else:
            _device = None  # type: ignore[assignment]
    return _device  # type: ignore[return-value]


def load_wav2vec2() -> Tuple["Wav2Vec2Processor", "Wav2Vec2Model"]:
    """
    Lazily load and cache the wav2vec2 processor and model.

    The model is loaded in inference mode with all weights frozen.
    Call this once at application startup via the FastAPI lifespan hook.

    Returns:
        Tuple of (processor, model) ready for inference.

    Raises:
        RuntimeError: If torch / transformers are not installed.
    """
    global _wav2vec2_processor, _wav2vec2_model

    if not _DEEP_AVAILABLE:
        raise RuntimeError(
            "Deep embedding extraction requires torch and transformers. "
            "Install them with: pip install torch transformers"
        )

    if _wav2vec2_processor is None or _wav2vec2_model is None:
        device = _get_device()
        t0 = time.perf_counter()
        logger.info("Loading %s …", WAV2VEC2_MODEL_ID)

        _wav2vec2_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_ID)
        _wav2vec2_model = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_ID)
        _wav2vec2_model.to(device)
        _wav2vec2_model.eval()

        for param in _wav2vec2_model.parameters():
            param.requires_grad_(False)

        elapsed = time.perf_counter() - t0
        logger.info("wav2vec2 loaded in %.1fs on %s", elapsed, device)

    return _wav2vec2_processor, _wav2vec2_model


# =============================================================================
# Stage 1 – Denoising
# =============================================================================


def decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode raw bytes from an uploaded audio file into a waveform array.

    Supports any format libsndfile understands (WAV, FLAC, OGG, etc.).
    The result is resampled to TARGET_SR and converted to mono.

    Args:
        audio_bytes: Raw bytes of the uploaded audio file.

    Returns:
        Tuple of (waveform float32 array, sample_rate=TARGET_SR).

    Raises:
        ValueError: If the bytes cannot be decoded as audio.
    """
    try:
        buf = io.BytesIO(audio_bytes)
        y, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError(f"Could not decode audio bytes: {exc}") from exc

    # Convert stereo → mono by averaging channels
    if y.ndim == 2:
        y = y.mean(axis=1)

    # Resample to 16 kHz if necessary
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    return y.astype(np.float32), TARGET_SR


def apply_spectral_gate_denoising(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    noise_ref_duration_s: float = 0.5,
    prop_decrease: float = 1.0,
    smoothing_factor: int = 3,
) -> np.ndarray:
    """
    Apply spectral-gate noise reduction (PPD-Net Stage 1 proxy).

    Uses the first `noise_ref_duration_s` seconds as a noise floor estimate,
    then applies a soft sigmoid mask in the STFT domain.

    Args:
        y:                   Input mono waveform.
        sr:                  Sample rate (Hz).
        n_fft:               FFT window size.
        hop_length:          STFT hop size.
        noise_ref_duration_s: Duration of leading silence used to estimate
                              the noise floor.
        prop_decrease:       Proportion of noise to remove [0, 1].
        smoothing_factor:    Frequency-bin smoothing kernel width.

    Returns:
        Denoised waveform of the same length as `y`.
    """
    n_noise = int(noise_ref_duration_s * sr)
    noise_clip = y[:n_noise] if len(y) > n_noise else y

    D_full = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D_noise = librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop_length)

    mag_full = np.abs(D_full)
    mag_noise = np.abs(D_noise)

    noise_mean = mag_noise.mean(axis=1, keepdims=True)
    noise_std = mag_noise.std(axis=1, keepdims=True)
    noise_thresh = noise_mean + noise_std

    # Smooth threshold across frequency bins
    kernel = np.ones((smoothing_factor, 1)) / smoothing_factor
    noise_thresh = scipy.signal.fftconvolve(noise_thresh, kernel, mode="same")

    # Soft gate
    margin = mag_full - noise_thresh
    mask = 1.0 / (1.0 + np.exp(-10.0 * (margin / (noise_thresh + 1e-8))))
    mask = mask * prop_decrease + (1.0 - prop_decrease)

    D_clean = D_full * mask
    y_clean = librosa.istft(D_clean, hop_length=hop_length, length=len(y))

    return y_clean.astype(np.float32)


# =============================================================================
# Stage 2a – Classical Feature Extraction
# =============================================================================


def extract_classical_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = N_MFCC,
    f0_fmin: float = 50.0,
    f0_fmax: float = 500.0,
) -> Dict[str, float]:
    """
    Extract eGeMAPS-inspired handcrafted acoustic features (Stage 2a).

    Feature groups
    --------------
    MFCCs (13 × mean + std)         → 26 dims
    F0 stats + jitter CV            →  5 dims
    Spectral centroid (mean + std)  →  2 dims
    Spectral flux     (mean + std)  →  2 dims
    Zero-crossing rate (mean + std) →  2 dims
    RMS energy         (mean + std) →  2 dims
                              Total: 39 dims

    Args:
        y:       Mono 16 kHz waveform.
        sr:      Sample rate (must be 16 000 Hz for consistent feature scale).
        n_mfcc:  Number of MFCC coefficients.
        f0_fmin: Minimum F0 frequency (Hz).
        f0_fmax: Maximum F0 frequency (Hz).

    Returns:
        Ordered dict {feature_name: float}.
    """
    feats: Dict[str, float] = {}

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=160)
    for i in range(n_mfcc):
        feats[f"mfcc_{i+1:02d}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc_{i+1:02d}_std"] = float(mfcc[i].std())

    # Fundamental frequency (pYIN)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=f0_fmin, fmax=f0_fmax, sr=sr,
        frame_length=1024, hop_length=256,
    )
    f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
    if len(f0_voiced) > 1:
        feats["f0_mean_hz"] = float(np.mean(f0_voiced))
        feats["f0_std_hz"] = float(np.std(f0_voiced))
        feats["f0_min_hz"] = float(np.min(f0_voiced))
        feats["f0_max_hz"] = float(np.max(f0_voiced))
        feats["f0_jitter_cv"] = float(np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-8))
    else:
        for k in ("f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz", "f0_jitter_cv"):
            feats[k] = 0.0

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=256)[0]
    feats["spectral_centroid_mean"] = float(centroid.mean())
    feats["spectral_centroid_std"] = float(centroid.std())

    # Spectral flux (onset strength as shimmer proxy)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    feats["spectral_flux_mean"] = float(onset_env.mean())
    feats["spectral_flux_std"] = float(onset_env.std())

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=256)[0]
    feats["zcr_mean"] = float(zcr.mean())
    feats["zcr_std"] = float(zcr.std())

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    feats["rms_mean"] = float(rms.mean())
    feats["rms_std"] = float(rms.std())

    return feats


def classical_features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert the feature dict to a 1-D float32 numpy array."""
    return np.array(list(features.values()), dtype=np.float32)


# =============================================================================
# Stage 2a-bis – UCI Oxford Parkinson's Clinical Feature Extraction (22 dims)
# =============================================================================

UCI_FEATURE_NAMES: List[str] = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]

UCI_DATASET_DEFAULTS: Dict[str, float] = {
    "MDVP:Fo(Hz)": 154.2286,
    "MDVP:Fhi(Hz)": 197.1049,
    "MDVP:Flo(Hz)": 116.3246,
    "MDVP:Jitter(%)": 0.00622,
    "MDVP:Jitter(Abs)": 0.000044,
    "MDVP:RAP": 0.003306,
    "MDVP:PPQ": 0.003446,
    "Jitter:DDP": 0.00992,
    "MDVP:Shimmer": 0.029709,
    "MDVP:Shimmer(dB)": 0.28225,
    "Shimmer:APQ3": 0.015664,
    "Shimmer:APQ5": 0.017878,
    "MDVP:APQ": 0.024081,
    "Shimmer:DDA": 0.046993,
    "NHR": 0.024847,
    "HNR": 21.88597,
    "RPDE": 0.498536,
    "DFA": 0.718099,
    "spread1": -5.684397,
    "spread2": 0.22651,
    "D2": 2.381826,
    "PPE": 0.206552,
}


def extract_uci_features(
    y: np.ndarray,
    sr: int,
    f0_fmin: float = 50.0,
    f0_fmax: float = 500.0,
) -> Dict[str, float]:
    """
    Extract the 22 clinical acoustic biomarkers matching the UCI Oxford
    Parkinson's Dataset schema.

    Returns:
        Ordered dictionary mapping each UCI column name to float value.
    """
    feats: Dict[str, float] = dict(UCI_DATASET_DEFAULTS)

    try:
        # Fundamental frequency extraction via pYIN
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=f0_fmin, fmax=f0_fmax, sr=sr,
            frame_length=1024, hop_length=256,
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)] if voiced_flag is not None else np.array([])

        if len(f0_voiced) > 2:
            fo = float(np.mean(f0_voiced))
            fhi = float(np.max(f0_voiced))
            flo = float(np.min(f0_voiced))
            feats["MDVP:Fo(Hz)"] = fo
            feats["MDVP:Fhi(Hz)"] = fhi
            feats["MDVP:Flo(Hz)"] = flo

            # Pitch periods T (seconds)
            T = 1.0 / (f0_voiced + 1e-8)
            mean_T = float(np.mean(T))
            diff_T = np.abs(np.diff(T))
            jitter_abs = float(np.mean(diff_T))
            # In UCI, MDVP:Jitter(%) is the ratio / proportion (mean ~0.0062)
            jitter_pct = float(jitter_abs / (mean_T + 1e-8))
            feats["MDVP:Jitter(%)"] = jitter_pct
            feats["MDVP:Jitter(Abs)"] = jitter_abs

            # RAP: 3-point Relative Average Perturbation
            if len(T) >= 3:
                rap = float(np.mean(np.abs(T[1:-1] - (T[:-2] + T[1:-1] + T[2:]) / 3.0)) / (mean_T + 1e-8))
            else:
                rap = float(jitter_pct * 0.53)
            feats["MDVP:RAP"] = rap

            # PPQ: 5-point Period Perturbation Quotient
            if len(T) >= 5:
                kernel5 = np.ones(5) / 5.0
                smooth5 = scipy.signal.convolve(T, kernel5, mode="valid")
                ppq = float(np.mean(np.abs(T[2:-2] - smooth5)) / (mean_T + 1e-8))
            else:
                ppq = float(rap * 1.04)
            feats["MDVP:PPQ"] = ppq

            # Jitter:DDP = 3 * RAP
            feats["Jitter:DDP"] = float(3.0 * rap)

            # Non-linear pitch dynamics: spread1, spread2, RPDE, PPE
            log_f0_std = float(np.log(np.std(f0_voiced) + 1e-6))
            feats["spread1"] = float(np.clip(-8.0 + (log_f0_std * 0.75), -8.0, -2.4))
            feats["spread2"] = float(np.clip(np.std(f0_voiced) / (fo + 1e-8), 0.01, 0.45))

            if len(f0_voiced) >= 10:
                hist, _ = np.histogram(T, bins=8, density=True)
                hist = hist[hist > 0]
                rpde = float(-np.sum(hist * np.log2(hist + 1e-12)) / np.log2(8))
                feats["RPDE"] = float(np.clip(rpde, 0.25, 0.69))

                rel_diff = np.abs(np.diff(T)) / (mean_T + 1e-8)
                feats["PPE"] = float(np.clip(np.std(rel_diff), 0.04, 0.52))

        # Amplitude & Shimmer extraction
        rms = librosa.feature.rms(y=y, hop_length=256)[0]
        if len(rms) > 2:
            diff_A = np.abs(np.diff(rms))
            mean_A = float(np.mean(rms)) + 1e-8
            shimmer = float(np.mean(diff_A) / mean_A)
            shimmer_db = float(np.mean(np.abs(20.0 * np.log10((rms[1:] + 1e-8) / (rms[:-1] + 1e-8)))))

            feats["MDVP:Shimmer"] = shimmer
            feats["MDVP:Shimmer(dB)"] = shimmer_db

            if len(rms) >= 3:
                apq3 = float(np.mean(np.abs(rms[1:-1] - (rms[:-2] + rms[1:-1] + rms[2:]) / 3.0)) / mean_A)
            else:
                apq3 = float(shimmer * 0.53)
            feats["Shimmer:APQ3"] = apq3

            if len(rms) >= 5:
                smooth5_a = scipy.signal.convolve(rms, np.ones(5) / 5.0, mode="valid")
                apq5 = float(np.mean(np.abs(rms[2:-2] - smooth5_a)) / mean_A)
            else:
                apq5 = float(apq3 * 1.14)
            feats["Shimmer:APQ5"] = apq5

            if len(rms) >= 11:
                smooth11_a = scipy.signal.convolve(rms, np.ones(11) / 11.0, mode="valid")
                apq = float(np.mean(np.abs(rms[5:-5] - smooth11_a)) / mean_A)
            else:
                apq = float(apq5 * 1.34)
            feats["MDVP:APQ"] = apq

            feats["Shimmer:DDA"] = float(3.0 * apq3)

        # Harmonics & Noise ratios (HPSS separation)
        y_harm, y_perc = librosa.effects.hpss(y)
        p_harm = float(np.mean(y_harm ** 2)) + 1e-8
        p_noise = float(np.mean(y_perc ** 2)) + 1e-8
        feats["NHR"] = float(np.clip(p_noise / p_harm, 0.0006, 0.32))
        feats["HNR"] = float(np.clip(10.0 * np.log10(p_harm / p_noise), 8.4, 33.0))

    except Exception as exc:
        logger.warning("UCI feature extraction encountered an error; using defaults: %s", exc)

    return feats


def uci_features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert a feature dictionary into a 22-dimensional float32 numpy array
    strictly aligned to the UCI Oxford Parkinson's dataset column order.
    Supports either output from extract_uci_features() or fallback mapping from
    extract_classical_features().
    """
    vec: List[float] = []
    for col in UCI_FEATURE_NAMES:
        if col in features:
            vec.append(float(features[col]))
        elif col == "MDVP:Fo(Hz)" and "f0_mean_hz" in features:
            vec.append(float(features["f0_mean_hz"]))
        elif col == "MDVP:Fhi(Hz)" and "f0_max_hz" in features:
            vec.append(float(features["f0_max_hz"]))
        elif col == "MDVP:Flo(Hz)" and "f0_min_hz" in features:
            vec.append(float(features["f0_min_hz"]))
        elif col == "MDVP:Jitter(%)" and "f0_jitter_cv" in features:
            vec.append(float(features["f0_jitter_cv"]))
        else:
            vec.append(UCI_DATASET_DEFAULTS.get(col, 0.0))
    return np.array(vec, dtype=np.float32)



# =============================================================================
# Stage 2b – Deep Embedding Extraction
# =============================================================================


def extract_deep_embeddings(
    y: np.ndarray,
    sr: int,
    max_duration_s: float = 10.0,
) -> np.ndarray:
    """
    Extract a mean-pooled 768-dim wav2vec2 embedding (Stage 2b).

    The resulting vector is stored in the Supabase `features.embedding`
    column (pgvector(768)) and used for approximate nearest-neighbour
    similarity search.

    Args:
        y:              Mono waveform at 16 kHz.
        sr:             Sample rate – must be 16 000 Hz.
        max_duration_s: Recording is truncated to this length before
                        forwarding through the model to cap memory usage.

    Returns:
        numpy float32 array of shape (768,).

    Raises:
        RuntimeError:  If torch / transformers are not available.
        ValueError:    If `sr` is not TARGET_SR.
    """
    if sr != TARGET_SR:
        raise ValueError(
            f"extract_deep_embeddings requires {TARGET_SR} Hz audio, got {sr} Hz."
        )

    processor, model = load_wav2vec2()
    device = _get_device()

    # Truncate
    max_samples = int(max_duration_s * sr)
    y_trunc = y[:max_samples] if len(y) > max_samples else y

    # Normalise
    y_norm = y_trunc.astype(np.float32)
    peak = np.abs(y_norm).max()
    if peak > 0:
        y_norm /= peak

    inputs = processor(y_norm, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    # (1, T, 768) → (768,)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding.astype(np.float32)


# =============================================================================
# Feature Fusion
# =============================================================================


def fuse_feature_vectors(
    classical: Dict[str, float],
    deep: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Concatenate classical and (optionally) deep feature vectors.

    If `deep` is None (e.g. torch not available), returns only the
    classical vector so the pipeline can degrade gracefully.

    Args:
        classical: Output of extract_classical_features().
        deep:      Output of extract_deep_embeddings(), or None.

    Returns:
        Fused numpy float32 array.  Shape: (39,) or (807,).
    """
    classical_vec = classical_features_to_vector(classical)
    if deep is not None:
        return np.concatenate([classical_vec, deep.astype(np.float32)])
    return classical_vec


# =============================================================================
# Epistemic Uncertainty
# =============================================================================


def calculate_epistemic_uncertainty(
    probs: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Compute normalised Shannon entropy as an epistemic uncertainty score.

    H(p) = -Σ p_c · log2(p_c)  /  log2(C)   ∈ [0, 1]

    Args:
        probs:   Probability matrix (n_samples, n_classes).
        epsilon: Clip floor for log stability.

    Returns:
        Float32 array of shape (n_samples,) with values in [0, 1].
    """
    probs = np.clip(probs, epsilon, 1.0)
    n_classes = probs.shape[-1]
    entropy = -np.sum(probs * np.log2(probs), axis=-1)
    return (entropy / np.log2(n_classes)).astype(np.float32)


def uncertainty_tier(score: float) -> str:
    """Map a scalar uncertainty score to a clinical review tier string."""
    if score < 0.35:
        return "high_confidence"
    elif score < 0.65:
        return "review_recommended"
    return "mandatory_review"


# =============================================================================
# Top-level convenience function used by the API endpoint
# =============================================================================


def run_full_pipeline(
    audio_bytes: bytes,
    enable_deep: bool = True,
) -> dict:
    """
    Run the complete VoxTriad feature extraction pipeline on raw audio bytes.

    Steps
    -----
    1.  Decode bytes → waveform
    2.  Spectral-gate denoising
    3a. Classical feature extraction
    3b. Deep embedding (if enable_deep and torch available)
    4.  Feature fusion

    Args:
        audio_bytes:  Raw bytes from the uploaded audio file.
        enable_deep:  Whether to run wav2vec2 embedding extraction.
                      Set False to reduce latency when GPU is unavailable.

    Returns:
        Dict with keys:
          - duration_sec       (float)
          - sample_rate        (int)
          - classical_features (dict[str, float])
          - embedding          (list[float] | None)
          - embedding_dim      (int)
          - deep_enabled       (bool)
    """
    # 1. Decode
    y_raw, sr = decode_audio_bytes(audio_bytes)
    duration_sec = len(y_raw) / sr
    logger.info("Audio decoded  |  duration=%.2fs  sr=%d", duration_sec, sr)

    # ── QA Validation Gate ────────────────────────────────────────────────────
    # Check 1: Duration
    if duration_sec < 3.0:
        raise HTTPException(status_code=400, detail="Audio is too short. Please record for at least 3 seconds.")

    # Check 2: Silence / Volume
    rms = librosa.feature.rms(y=y_raw)[0]
    if np.mean(rms) < 0.005:
        raise HTTPException(status_code=400, detail="Audio is too quiet. Please speak louder.")

    # Check 3: Language-Agnostic Pitch (F0) Tracking (Tuned for Pathology)
    f0 = librosa.yin(y_raw, fmin=50, fmax=500)
    unvoiced_ratio = np.isnan(f0).sum() / len(f0)
    
    if unvoiced_ratio > 0.40:
        raise HTTPException(
            status_code=400, 
            detail="Speech or background noise detected. Please hold a continuous 'ahh' without stopping."
        )
    
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) > 0:
        f0_std = np.std(valid_f0)
        # Increased threshold to 200 to allow for pathological vocal tremor/jitter in PD patients
        if f0_std > 200: 
            raise HTTPException(
                status_code=400, 
                detail="Extreme pitch fluctuation detected. Please hold a steady, flat 'ahh' sound."
            )
    # ──────────────────────────────────────────────────────────────────────────

    # 2. Denoise
    y_clean = apply_spectral_gate_denoising(y_raw, sr)
    logger.info("Denoising complete")

    # 3a. Classical features
    classical = extract_classical_features(y_clean, sr)
    logger.info("Classical features extracted  |  dims=%d", len(classical))

    # 3a-bis. UCI clinical features (22 dims)
    uci = extract_uci_features(y_clean, sr)
    logger.info("UCI clinical features extracted  |  dims=%d", len(uci))

    # 3b. Deep embeddings (optional)
    embedding: Optional[np.ndarray] = None
    if enable_deep and _DEEP_AVAILABLE:
        try:
            embedding = extract_deep_embeddings(y_clean, sr)
            logger.info("Deep embedding extracted  |  shape=%s", embedding.shape)
        except Exception as exc:
            logger.warning("Deep embedding failed (non-fatal): %s", exc)

    # 4. Fuse
    fused = fuse_feature_vectors(classical, embedding)
    logger.info("Feature fusion complete  |  fused_dims=%d", len(fused))

    return {
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sr,
        "classical_features": classical,
        "uci_features": uci,
        "embedding": embedding.tolist() if embedding is not None else None,
        "embedding_dim": len(embedding) if embedding is not None else 0,
        "fused_dim": len(fused),
        "deep_enabled": embedding is not None,
    }

