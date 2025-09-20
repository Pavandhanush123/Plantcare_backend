# moderation_server.py
import os
import io
import json
import logging
from typing import Any, Dict
from flask import Flask, request, jsonify, abort
import requests

LOG = logging.getLogger("moderation_server")
logging.basicConfig(level=logging.INFO)

HF_TOKEN = "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"
MOD_API_KEY = "b1f7c2d3a9e84f6b8c2d4ea19b3f6c7a9d8e2f3b4c5a6d7e8f9a0b1c2d3e4f5"
# HF router endpoint — change if you use another model
HF_API_URL = os.environ.get(
    "HF_API_URL",
    "https://router.huggingface.co/hf-inference/models/LukeJacob2023/nsfw-image-detector",
)

if not HF_TOKEN:
    LOG.warning("HF_TOKEN not set — moderation calls will fail if HF_API_URL is used.")

app = Flask(__name__)

def require_api_key():
    """Return 401 if MOD_API_KEY is configured and request does not provide the correct bearer token."""
    if not MOD_API_KEY:
        return  # no server-side key configured -> allow (not recommended for production)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        abort(401, description="Missing Authorization header")
    token = auth.split(" ", 1)[1].strip()
    if token != MOD_API_KEY:
        abort(401, description="Invalid API key")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "moderation_enabled": bool(HF_TOKEN)})

@app.route("/moderate", methods=["POST"])
def moderate():
    """
    POST /moderate
      - Headers: Authorization: Bearer <MOD_API_KEY>  (if MOD_API_KEY set)
      - Form: file=@image.jpg
    Response:
      { allowed: bool, reason: str, raw: <model response> }
    """
    require_api_key()

    if "file" not in request.files:
        return jsonify({"error": "Missing file field (multipart/form-data)"}), 400

    f = request.files["file"]
    content = f.read()
    if not content:
        return jsonify({"error": "Empty file"}), 400

    # forward to Hugging Face router
    if not HF_TOKEN:
        # If HF not configured, treat as unmoderated allowed (but you can change to block)
        return jsonify({"allowed": True, "reason": "Moderation disabled (HF_TOKEN not configured)", "raw": None})

    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/octet-stream"}
        resp = requests.post(HF_API_URL, headers=headers, data=content, timeout=25)
        resp.raise_for_status()
    except requests.RequestException as e:
        LOG.exception("Failed to call HF inference")
        return jsonify({"allowed": False, "reason": f"HuggingFace request failed: {e}", "raw": None}), 502

    try:
        model_output = resp.json()
    except Exception:
        model_output = resp.text

    interpreted = interpret_hf_response(model_output)
    # Return both interpreted decision and raw model output for debugging
    return jsonify({"allowed": interpreted.allowed, "reason": interpreted.message, "raw": model_output})

# Simple container for moderation result
class ModerationResult:
    def __init__(self, allowed: bool, message: str):
        self.allowed = allowed
        self.message = message

def interpret_hf_response(raw: Any) -> ModerationResult:
    """
    Heuristics to interpret different HF model response shapes.
    Block if:
      - model returns top label 'nsfw'/'porn'/'adult' with high score (>= 0.6)
      - model returns a dict of class probabilities where any adult-like class >= 0.6
    Otherwise allow.
    """
    # If raw is string, we can't interpret -> allow but warn
    if raw is None:
        return ModerationResult(True, "No model output")

    # If HF router returns a list with single dict (common)
    if isinstance(raw, list) and raw:
        raw = raw[0]

    # Common simple object: { "label": "nsfw", "score": 0.97 }
    try:
        if isinstance(raw, dict):
            # case 1: direct label + score
            label = raw.get("label") or raw.get("category") or None
            score = raw.get("score") or raw.get("probability") or None
            if label and isinstance(label, str) and score is not None:
                label_l = label.lower()
                try:
                    score_f = float(score)
                except Exception:
                    score_f = 0.0
                if label_l in ("nsfw", "porn", "adult", "risky") and score_f >= 0.6:
                    return ModerationResult(False, f"Blocked by model label='{label}' score={score_f:.2f}")
                return ModerationResult(True, f"Allowed by label='{label}' score={score_f:.2f}")

            # case 2: probability map like {"porn":0.8,"neutral":0.1,...}
            # pick max probability
            numeric_map = {k: v for k, v in raw.items() if isinstance(v, (int, float))}
            if numeric_map:
                top_k, top_v = max(numeric_map.items(), key=lambda kv: kv[1])
                top_k_l = str(top_k).lower()
                try:
                    top_v_f = float(top_v)
                except Exception:
                    top_v_f = 0.0
                if top_k_l in ("porn", "nsfw", "adult", "sexual") and top_v_f >= 0.6:
                    return ModerationResult(False, f"Blocked: predicted {top_k}={top_v_f:.2f}")
                return ModerationResult(True, f"Allowed: predicted {top_k}={top_v_f:.2f}")

            # case 3: HF outputs deeper nested shapes; try to search for label/score keys
            # flatten dict strings
            text = json.dumps(raw).lower()
            if "nsfw" in text or "porn" in text:
                # if mentions nsfw but no scores — block conservatively
                return ModerationResult(False, "Blocked conservatively (nsfw token present in model response)")
    except Exception as e:
        LOG.exception("interpretation failed: %s", e)
        return ModerationResult(True, "Unable to interpret model response (allowed)")

    # fallback: allow and include raw shape note
    return ModerationResult(True, "Allowed (model output non-conclusive)")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
