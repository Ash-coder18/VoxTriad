"""
VoxTriad – Clinical LLM Summarisation Module
Phase 4 Backend

Integrates GroqCloud (llama-3.3-70b-versatile) to synthesise a concise,
clinician-facing summary that maps raw acoustic features to the UPDRS-III
speech subscore domain.

The system prompt is intentionally conservative and HIPAA-aware:
  • No diagnostic conclusion is stated (only "screening indicators")
  • Uncertainty flags explicitly advise re-recording
  • Output is framed as supporting clinician judgement, not replacing it

Public API
----------
  generate_updrs_summary(risk_score, uncertainty_tier, features_dict)
      → ClinicalSummaryResult dataclass
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ── Load .env BEFORE any SDK import or flag evaluation ───────────────────────
# This ensures GROQ_API_KEY is populated even when Python imports this module
# before main.py's own load_dotenv() call has run (e.g. during uvicorn reload).
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── Groq client (lazy initialisation) ────────────────────────────────────────

try:
    from groq import Groq
    _GROQ_SDK_AVAILABLE = True
except ImportError:
    _GROQ_SDK_AVAILABLE = False
    logger.warning(
        "groq package not installed – LLM summarisation disabled. "
        "Install with: pip install groq"
    )

# Whether the LLM backend is usable (SDK present + API key set)
# Evaluated AFTER load_dotenv() so the key is guaranteed to be present.
_GENAI_AVAILABLE: bool = _GROQ_SDK_AVAILABLE and bool(os.getenv("GROQ_API_KEY"))

# Model to use – overridable via GROQ_MODEL env var.
# Default: qwen/qwen3.8-27b (confirmed available on this API key).
# If you get a 404 model_not_found error, run: python -c "from groq import Groq; [print(m.id) for m in Groq().models.list().data]"
_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# Module-level Groq client singleton
_groq_client: Optional[Any] = None

# Backward-compat alias so any code that references _gemini_client still works
_gemini_client: Optional[Any] = None


def _get_groq_client() -> Any:
    """
    Lazily initialise and cache the Groq client.

    Reads GROQ_API_KEY from the environment.  Should be called once
    at application startup (via load_llm) so the first request doesn't
    pay the initialisation penalty.

    Returns:
        groq.Groq instance configured with the API key.

    Raises:
        RuntimeError: If the groq package is not installed or the API key
                      is missing from the environment.
    """
    global _groq_client, _gemini_client

    if not _GROQ_SDK_AVAILABLE:
        raise RuntimeError(
            "groq package is not installed. "
            "Run: pip install groq"
        )

    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set in the environment. "
                "Add it to backend/.env  (see .env.example)."
            )

        _groq_client = Groq(api_key=api_key)
        _gemini_client = _groq_client  # backward-compat alias
        logger.info("Groq client initialised  |  model=%s", _GROQ_MODEL)

    return _groq_client


def load_llm() -> bool:
    """
    Pre-load the Groq client singleton (call from FastAPI lifespan).

    Returns:
        True if client loaded successfully, False otherwise.
    """
    try:
        _get_groq_client()
        return True
    except RuntimeError as exc:
        logger.warning("Groq pre-load failed (non-fatal): %s", exc)
        return False


# Backward-compat alias so existing main.py `load_gemini` call still works
load_gemini = load_llm


# =============================================================================
# System Prompt (neurologist persona)
# =============================================================================

_SYSTEM_PROMPT = """You are a board-certified neurologist specialising in movement disorders, \
assisting with the interpretation of acoustic speech biomarkers for Parkinson's disease (PD) screening.

Your role is strictly SUPPORTIVE: you are helping clinicians interpret automated screening results, \
NOT making a diagnosis. Always remind clinicians that your output is a screening indicator only \
and must be confirmed through comprehensive clinical evaluation.

Output format rules:
- Write 3–4 sentences of plain clinical language suitable for a neurologist or GP.
- Map the acoustic features explicitly to the UPDRS-III speech subscore domains \
  (dysphonia, hypophonia, dysarthria, dysprosody).
- Do NOT use technical jargon like "MFCC" or "spectral flux" – translate these into \
  clinical correlates (e.g., "vocal tremor frequency", "reduced phonatory volume", \
  "irregular pitch variation").
- If uncertainty is high, start your summary with: \
  "[Data Quality Warning]: The recording quality was insufficient for reliable analysis.\
   A re-recording under quiet conditions is strongly advised before any clinical action."
- End every summary with: \
  "This is an automated screening indicator only. Clinical confirmation by a movement disorder \
   specialist is mandatory before any diagnostic or treatment decision."
- Keep the total output under 200 words.
- Write in third person (e.g., "The patient's voice…", "Findings suggest…").
"""

# =============================================================================
# Feature-to-Clinical Translation Helpers
# =============================================================================

_RISK_LABELS = {
    (0.0, 0.35): ("low", "within normal limits"),
    (0.35, 0.65): ("moderate", "borderline range requiring clinical attention"),
    (0.65, 1.01): ("high", "consistent with dysphonic markers associated with PD"),
}

_UNCERTAINTY_LABELS = {
    "high_confidence":      "high confidence",
    "low":                  "high confidence",
    "review_recommended":   "moderate confidence -- clinical review advised",
    "mandatory_review":     "LOW confidence -- re-recording strongly advised",
}


def _risk_label(score: Optional[float]) -> tuple[str, str]:
    """Return (short_label, clinical_phrase) for a given risk score."""
    if score is None:
        return "unknown", "unavailable (model not yet trained)"
    for (lo, hi), labels in _RISK_LABELS.items():
        if lo <= score < hi:
            return labels
    return "high", "consistent with dysphonic markers associated with PD"


def _format_feature_summary(features: Dict[str, float]) -> str:
    """
    Convert the raw classical feature dict into a clinical-language bullet list
    for injection into the LLM prompt.

    Maps internal feature keys to their clinical UPDRS-III correlates.
    """
    lines: list[str] = []

    def _get(k: str, default: float = 0.0) -> float:
        return float(features.get(k, default))

    f0_mean   = _get("f0_mean_hz")
    f0_std    = _get("f0_std_hz")
    f0_jitter = _get("f0_jitter_cv")
    flux_mean = _get("spectral_flux_mean")
    rms_mean  = _get("rms_mean")
    zcr_mean  = _get("zcr_mean")

    if f0_mean > 0:
        lines.append(f"- Fundamental frequency (F0): mean {f0_mean:.1f} Hz  "
                     f"(clinical correlate: vocal pitch centre)")
        lines.append(f"- F0 standard deviation: {f0_std:.2f} Hz  "
                     f"(clinical correlate: pitch variability / dysprosody)")
        lines.append(f"- Jitter coefficient of variation: {f0_jitter:.4f}  "
                     f"(clinical correlate: cycle-to-cycle pitch irregularity)")
    else:
        lines.append("- F0: voicing could not be reliably detected  "
                     "(clinical correlate: severe hypophonia / aphonia)")

    lines.append(f"- Spectral flux: {flux_mean:.4f}  "
                 f"(clinical correlate: amplitude modulation / shimmer proxy)")
    lines.append(f"- RMS energy: {rms_mean:.4f}  "
                 f"(clinical correlate: overall phonatory volume / hypophonia)")
    lines.append(f"- Zero-crossing rate: {zcr_mean:.4f}  "
                 f"(clinical correlate: breathiness / vocal noise)")

    # Include MFCC summary (first 3 coefficients are most clinically relevant)
    mfcc_means = [v for k, v in sorted(features.items())
                  if k.startswith("mfcc_") and k.endswith("_mean")]
    if mfcc_means:
        lines.append(f"- MFCC-1 (spectral envelope): {mfcc_means[0]:.2f}  "
                     f"(clinical correlate: vocal tract resonance profile)")
        if len(mfcc_means) > 2:
            lines.append(f"- MFCC-2/3 modulation: {mfcc_means[1]:.2f} / {mfcc_means[2]:.2f}  "
                         f"(clinical correlate: articulatory precision / dysarthria)")

    return "\n".join(lines)


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class ClinicalSummaryResult:
    """Structured result returned by generate_updrs_summary()."""

    summary_text: str
    risk_label: str
    risk_score: Optional[float]
    uncertainty_tier: str
    model_used: str
    llm_available: bool
    fallback_used: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_text":     self.summary_text,
            "risk_label":       self.risk_label,
            "risk_score":       self.risk_score,
            "uncertainty_tier": self.uncertainty_tier,
            "model_used":       self.model_used,
            "llm_available":    self.llm_available,
            "fallback_used":    self.fallback_used,
            "error":            self.error,
        }


# =============================================================================
# Primary public function
# =============================================================================

def generate_updrs_summary(
    risk_score: Optional[float],
    uncertainty_tier: str,
    features_dict: Dict[str, float],
    recording_duration_s: float = 0.0,
    task_type: str = "sustained_vowel",
) -> ClinicalSummaryResult:
    """
    Generate a plain-language clinical summary mapping acoustic findings to
    UPDRS-III speech subscores using the Groq LLM (llama-3.3-70b-versatile).

    The ``uncertainty_tier`` is always passed directly into the LLM prompt so
    Llama 3 can handle data-quality warnings dynamically — there is NO early
    return or short-circuit based on the tier.  The rule-based fallback is only
    used when the Groq API itself is unavailable (missing key or network error).

    If the Groq API is unavailable (missing key or import error), a
    deterministic rule-based fallback summary is returned instead, with
    ``fallback_used=True`` in the result so the frontend can indicate this.

    Args:
        risk_score:           Ensemble-predicted P(Parkinson's) ∈ [0, 1] or None.
        uncertainty_tier:     One of 'high_confidence', 'review_recommended',
                              'mandatory_review'.
        features_dict:        Classical features dict from extract_classical_features().
        recording_duration_s: Duration of the recording in seconds.
        task_type:            Vocal task type (for contextualisation).

    Returns:
        ClinicalSummaryResult with the generated text and metadata.
    """
    r_label, r_phrase = _risk_label(risk_score)
    unc_label = _UNCERTAINTY_LABELS.get(uncertainty_tier, uncertainty_tier)
    feature_summary = _format_feature_summary(features_dict)

    risk_score_str = f"{risk_score:.3f}" if risk_score is not None else "N/A"

    # ── Build LLM prompt ──────────────────────────────────────────────────────
    prompt = f"""
Acoustic screening result for a {task_type.replace('_', ' ')} task
(duration: {recording_duration_s:.1f} s).

Automated ensemble risk score: {risk_score_str}
Risk category: {r_label} -- {r_phrase}
Analysis confidence: {unc_label}

Acoustic feature profile (clinical correlates):
{feature_summary}

Please write a 3–4 sentence clinical summary suitable for a neurologist, mapping these \
acoustic features to the relevant UPDRS-III speech subscore domains. \
{"Since data quality confidence is LOW, begin with the re-recording advisory."
 if uncertainty_tier == "mandatory_review" else ""}
""".strip()

    # ── Attempt Groq API call ─────────────────────────────────────────────────
    if not _GENAI_AVAILABLE:
        return _rule_based_fallback(
            risk_score, r_label, r_phrase, uncertainty_tier, unc_label,
            features_dict, error="GROQ_API_KEY not set or groq package missing",
        )

    try:
        client = _get_groq_client()

        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            top_p=0.85,
            max_tokens=512,
        )

        summary_text = response.choices[0].message.content.strip()

        logger.info(
            "Groq summary generated  |  model=%s  risk=%s  tier=%s  words=%d",
            _GROQ_MODEL,
            risk_score_str,
            uncertainty_tier,
            len(summary_text.split()),
        )

        return ClinicalSummaryResult(
            summary_text=summary_text,
            risk_label=r_label,
            risk_score=risk_score,
            uncertainty_tier=uncertainty_tier,
            model_used=_GROQ_MODEL,
            llm_available=True,
            fallback_used=False,
        )

    except RuntimeError as exc:
        # Groq unavailable (no key / not installed) – use deterministic fallback
        print(f"[GROQ ERROR] {type(exc).__name__}: {exc}")
        logger.warning("Groq unavailable, using rule-based fallback: %s", exc)
        return _rule_based_fallback(
            risk_score, r_label, r_phrase, uncertainty_tier, unc_label,
            features_dict, error=str(exc),
        )

    except Exception as exc:
        # API error (rate limit, network, content-filter, etc.)
        print(f"[GROQ ERROR] {type(exc).__name__}: {exc}")
        logger.error("Groq API error: %s", exc, exc_info=True)
        return _rule_based_fallback(
            risk_score, r_label, r_phrase, uncertainty_tier, unc_label,
            features_dict, error=f"Groq API error: {type(exc).__name__}: {exc}",
        )


# =============================================================================
# Rule-based fallback (no LLM dependency)
# =============================================================================

def _rule_based_fallback(
    risk_score: Optional[float],
    r_label: str,
    r_phrase: str,
    uncertainty_tier: str,
    unc_label: str,
    features: Dict[str, float],
    error: Optional[str] = None,
) -> ClinicalSummaryResult:
    """
    Generate a deterministic clinical summary from feature thresholds when
    the Groq API is unavailable.

    The output mirrors the structure of the LLM-generated text but is
    produced without any external API call, guaranteeing 100% availability.
    """
    lines: list[str] = []

    if uncertainty_tier == "mandatory_review":
        lines.append(
            "[Data Quality Warning]: The recording quality was insufficient for "
            "reliable analysis. A re-recording under quiet conditions is strongly "
            "advised before any clinical action."
        )

    f0_mean   = features.get("f0_mean_hz", 0.0)
    f0_std    = features.get("f0_std_hz", 0.0)
    f0_jitter = features.get("f0_jitter_cv", 0.0)
    rms_mean  = features.get("rms_mean", 0.0)
    flux_mean = features.get("spectral_flux_mean", 0.0)

    # Risk intro
    risk_pct = f"{risk_score * 100:.0f}%" if risk_score is not None else "unavailable"
    lines.append(
        f"Automated ensemble analysis yields a PD screening risk score of {risk_pct} "
        f"({r_phrase}), with analysis confidence rated as {unc_label}."
    )

    # F0 / jitter → UPDRS-III dysphonia
    if f0_mean > 0:
        jitter_desc = (
            "mildly elevated cycle-to-cycle pitch irregularity" if f0_jitter > 0.05
            else "normal pitch regularity"
        )
        f0_desc = (
            "reduced fundamental frequency with widened pitch range" if f0_std > 10
            else "stable fundamental frequency"
        )
        lines.append(
            f"Phonatory analysis reveals {f0_desc} (F0: {f0_mean:.0f} Hz +/- {f0_std:.1f} Hz) "
            f"and {jitter_desc} (jitter CV: {f0_jitter:.3f}), "
            "which may correspond to dysphonic features on the UPDRS-III speech subscore."
        )
    else:
        lines.append(
            "Voicing could not be reliably detected, suggesting significant hypophonia "
            "that may correspond to a UPDRS-III speech subscore of >= 2."
        )

    # RMS / flux → hypophonia / shimmer
    rms_desc  = "reduced phonatory volume (hypophonia)" if rms_mean < 0.02 else "adequate phonatory volume"
    flux_desc = "elevated amplitude modulation" if flux_mean > 0.5 else "stable amplitude envelope"
    lines.append(
        f"Acoustic energy analysis indicates {rms_desc} "
        f"with {flux_desc}, consistent with the shimmer-related domain of UPDRS-III speech assessment."
    )

    lines.append(
        "This is an automated screening indicator only. Clinical confirmation by a "
        "movement disorder specialist is mandatory before any diagnostic or treatment decision."
    )

    return ClinicalSummaryResult(
        summary_text="\n\n".join(lines),
        risk_label=r_label,
        risk_score=risk_score,
        uncertainty_tier=uncertainty_tier,
        model_used="rule-based-fallback",
        llm_available=False,
        fallback_used=True,
        error=error,
    )

def generate_clinical_summary(transcript: str) -> str:
    """Generates a concise linguistic clinical summary using Groq LLaMA."""
    if not _GENAI_AVAILABLE:
        return "Automated analysis unavailable due to missing Groq API configuration."

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert neurologist. Read the patient's reading transcript. Analyze the text strictly for speech disfluencies like palilalia (repeated words like 'is, is, is'), hesitations ('uh', 'um'), and fragmented sentences. Write a concise, 2-sentence clinical observation describing these patterns and noting if they are consistent with hypokinetic dysarthria. Do not output any numbers or metrics."
                    )
                },
                {
                    "role": "user",
                    "content": f"Transcript: '{transcript}'"
                }
            ],
            temperature=0.3,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("NLP | Groq LLaMA summary generation failed: %s", e)
        return "Failed to generate AI clinical summary."

def generate_vision_summary(expressivity_variance: float, blink_rate: float, smile_symmetry: float) -> str:
    """Generates a concise vision clinical summary using Groq LLaMA."""
    if not _GENAI_AVAILABLE:
        return "Automated analysis unavailable due to missing Groq API configuration."

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert neurologist. You are analyzing facial biomarkers for signs of Parkinsonian hypomimia (facial masking). "
                        "The patient was instructed to maintain a neutral expression for 10 seconds. Therefore, low expressivity variance is expected and normal. "
                        "Only highlight hypomimia if the variance is exceptionally near-zero and accompanied by an abnormally low blink rate (under 12 bpm). "
                        "Based on the provided metrics (expressivity variance, blink rate, smile symmetry), write a concise, 2-sentence qualitative clinical observation. "
                        "Do NOT regurgitate the raw numbers in your output."
                    )
                },
                {
                    "role": "user",
                    "content": f"Metrics:\nExpressivity Variance: {expressivity_variance:.4f}\nBlink Rate: {blink_rate:.1f} bpm\nSmile Symmetry: {smile_symmetry:.2f}"
                }
            ],
            temperature=0.3,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Vision | Groq LLaMA summary generation failed: %s", e)
        return "Failed to generate AI clinical summary."
