# moderation_server.py
import os
import json
import logging
from typing import Tuple, Any
import requests
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moderation_server")

app = FastAPI(title="Image Moderation Proxy")

# Simple CORS: adjust origin in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment config
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
SERVER_API_KEY = os.environ.get("MODERATION_API_KEY", "").strip()
HF_MODEL = os.environ.get("HF_MODEL", "LukeJacob2023/nsfw-image-detector")
HF_ENDPOINT = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.60"))  # block if NSFW score >= threshold

if not HF_TOKEN:
    logger.warning("HF_TOKEN is not set; HF moderation requests will fail (use for test only).")

if not SERVER_API_KEY:
    logger.warning("MODERATION_API_KEY is not set; server will still accept requests that don't provide an API key. Set it for production!")

def hf_call_image(bytes_data: bytes) -> Any:
    """Call Hugging Face inference for an image; returns decoded JSON."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/octet-stream"}
    # Some HF models want raw bytes with Content-Type image/jpeg, others accept octet-stream.
    # We'll pass bytes as a POST body.
    res = requests.post(HF_ENDPOINT, headers=headers, data=bytes_data, timeout=30)
    try:
        body = res.json()
    except Exception:
        body = {"raw_text": res.text}
    if res.status_code != 200:
        raise Exception(f"HF API {res.status_code}: {body}")
    return body

def parse_hf_response(resp: Any) -> Tuple[float, str]:
    """
    Very small heuristic to extract 'NSFW' confidence and a label from common HF model responses.
    Returns (nsfw_score, label). nsfw_score in [0.0,1.0]. If no info found, returns (0.0, 'unknown').
    """
    try:
        # Case 1: list of {label,score}
        if isinstance(resp, list) and len(resp) > 0:
            for item in resp:
                lbl = str(item.get("label", "")).lower()
                sc = float(item.get("score", item.get("confidence", 0.0) or 0.0))
                # If label suggests NSFW or porn, return score
                if any(k in lbl for k in ("nsfw", "porn", "sexual", "explicit", "racy", "sexy", "pornography")):
                    return sc, item.get("label", "")
            # fallback: pick highest score item
            best = max(resp, key=lambda it: float(it.get("score", 0.0)))
            return float(best.get("score", 0.0)), best.get("label", "")
        # Case 2: dict with keys
        if isinstance(resp, dict):
            # models sometimes return {'label': 'NSFW', 'score': 0.95}
            if "label" in resp and "score" in resp:
                return float(resp["score"]), str(resp["label"])
            # some return {'nsfw_score': 0.98}
            for k in ("nsfw_score", "porn_score", "probabilities", "predictions"):
                if k in resp:
                    v = resp[k]
                    if isinstance(v, (int, float)):
                        return float(v), k
                    if isinstance(v, dict):
                        # try to find 'nsfw' key inside
                        if "nsfw" in v:
                            return float(v["nsfw"]), "nsfw"
            # unknown dict structure: flatten numbers > 0.5 as candidate
            def find_numeric(d):
                if isinstance(d, dict):
                    for kk, vv in d.items():
                        if isinstance(vv, (int, float)):
                            if vv > 0:
                                return float(vv), kk
                        if isinstance(vv, dict):
                            ret = find_numeric(vv)
                            if ret:
                                return ret
                return None
            found = find_numeric(resp)
            if found:
                return found
    except Exception as e:
        logger.exception("parse_hf_response error: %s", e)
    return 0.0, "unknown"

@app.post("/moderate")
async def moderate_endpoint(
    request: Request,
    file: UploadFile = File(...),
    authorization: str | None = Header(None),
):
    # Basic API key check
    if SERVER_API_KEY:
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header required")
        # Accept either 'Bearer <key>' or raw key
        token = authorization.strip()
        if token.lower().startswith("bearer "):
            token = token.split(" ", 1)[1].strip()
        if token != SERVER_API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # call HF
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Server not configured with HF_TOKEN")

    try:
        hf_resp = hf_call_image(content)
    except Exception as e:
        logger.exception("HF call failed")
        raise HTTPException(status_code=502, detail=f"Hugging Face inference error: {e}")

    # parse response
    nsfw_score, label = parse_hf_response(hf_resp)
    logger.info("HF parsed: label=%s score=%.3f", label, nsfw_score)

    blocked = nsfw_score >= SCORE_THRESHOLD
    reason = ""
    if blocked:
        reason = f"Blocked: detected '{label}' with score {nsfw_score:.2f} (threshold {SCORE_THRESHOLD})"
    else:
        reason = f"Passed: detected '{label}' score {nsfw_score:.2f}"

    return JSONResponse({
        "allowed": not blocked,
        "reason": reason,
        "score": nsfw_score,
        "label": label,
        "hf_raw": hf_resp
    })
