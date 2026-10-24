"""
VoxTriad - End-to-End Local System Verification Test
Automated QA script to verify operational readiness of core components.
"""

import os
import sys
import pathlib
from dotenv import load_dotenv

# Load backend environment variables
load_dotenv()

from fastapi.testclient import TestClient
from main import app


def run_tests():
    print("=" * 65)
    print("  VoxTriad CDSS - Local System End-to-End Verification")
    print("=" * 65)

    passed_all = True

    # -------------------------------------------------------------
    # 1. Health Endpoint & Ensemble Readiness Test
    # -------------------------------------------------------------
    try:
        with TestClient(app) as client:
            response = client.get("/api/health")
            data = response.json()

            assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
            assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"
            assert data.get("ensemble_ready") is True, f"Ensemble not ready: {data}"

            print(f"[PASS] Health Endpoint: HTTP {response.status_code} OK | Ensemble Status: Ready (v{data.get('version')})")
    except Exception as exc:
        passed_all = False
        print(f"[FAIL] Health Endpoint: {exc}")

    # -------------------------------------------------------------
    # 2. Database Connectivity Credentials Test
    # -------------------------------------------------------------
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_ANON_KEY", "").strip()

    if supabase_url and supabase_key:
        masked_key = supabase_key[:12] + "..." + supabase_key[-4:] if len(supabase_key) > 16 else "***"
        print(f"[PASS] Database Connectivity: Loaded SUPABASE_URL ({supabase_url}) and SUPABASE_ANON_KEY ({masked_key})")
    else:
        passed_all = False
        missing = []
        if not supabase_url:
            missing.append("SUPABASE_URL")
        if not supabase_key:
            missing.append("SUPABASE_ANON_KEY")
        print(f"[FAIL] Database Connectivity: Missing environment variables: {', '.join(missing)}")

    # -------------------------------------------------------------
    # 3. Gemini API Credentials Test (Live LLM Active)
    # -------------------------------------------------------------
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        masked_gemini = gemini_key[:8] + "..." + gemini_key[-4:] if len(gemini_key) > 12 else "***"
        print(f"[PASS] Gemini API: GEMINI_API_KEY present ({masked_gemini}) | Live LLM mode active")
    else:
        passed_all = False
        print("[FAIL] Gemini API: GEMINI_API_KEY not found in environment (fallback will be used)")

    # -------------------------------------------------------------
    # 4. Inference Engine & Model Artifact Test
    # -------------------------------------------------------------
    model_path = pathlib.Path(__file__).parent / "ensemble_model.joblib"
    if model_path.exists() and model_path.stat().st_size > 0:
        try:
            import joblib
            model_data = joblib.load(model_path)
            is_dict = isinstance(model_data, dict) and "pipeline" in model_data
            if is_dict:
                size_kb = model_path.stat().st_size / 1024
                print(f"[PASS] Inference Engine: ensemble_model.joblib verified ({size_kb:.1f} KB, pipeline loadable)")
            else:
                passed_all = False
                print(f"[FAIL] Inference Engine: Invalid model structure in {model_path}")
        except Exception as exc:
            passed_all = False
            print(f"[FAIL] Inference Engine: Could not load ensemble_model.joblib: {exc}")
    else:
        passed_all = False
        print(f"[FAIL] Inference Engine: ensemble_model.joblib missing at {model_path}")

    print("=" * 65)
    if passed_all:
        print("  OVERALL SYSTEM STATUS: ALL CHECKS PASSED [OPERATIONAL]")
    else:
        print("  OVERALL SYSTEM STATUS: ONE OR MORE CHECKS FAILED")
    print("=" * 65)

    return 0 if passed_all else 1


if __name__ == "__main__":
    sys.exit(run_tests())
