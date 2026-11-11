"""AADI server backed by a local Ollama audio model (default: gemma4:e4b).

Same API as aadi_server.py (/health, /decide, /v1/systemone), but instead of
running Gemma 4 on CUDA it asks Ollama for exactly one output token with
logprobs, and reads the scores of the option letters from top_logprobs.

Differences from the CUDA version:
  * Ollama only exposes the top 20 next-token logprobs, so a letter that is
    not in the top 20 gets a floor value (reported in "missing_labels").
  * Optional position debiasing: with debias=1 the options are rotated
    through every position and the per-option probabilities are averaged
    (costs one call per option).
  * The model can be switched at runtime: GET /models lists the local
    audio-capable Ollama models, POST /model {"model": "..."} selects one.

Run from the repository folder (needs Ollama running and `ollama pull gemma4:e4b`):
    uv run --with fastapi --with 'uvicorn[standard]' \
        uvicorn aadi_ollama_server:app --host 127.0.0.1 --port 8765
Then open http://127.0.0.1:8765/live for the live test page.
Full instructions: OLLAMA.md.

Env vars: OLLAMA_URL (default http://127.0.0.1:11434),
          AADI_OLLAMA_MODEL (default gemma4:e4b),
          AADI_KEEP_ALIVE (default 60m),
          AADI_DEBIAS (default 0).
"""

import asyncio
import base64
import io
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from aadi_systemone_api import create_router

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = "gemma4:e4b"
MODEL_ID = os.environ.get("AADI_OLLAMA_MODEL", DEFAULT_MODEL)  # current model
DEFAULT_DEBIAS = os.environ.get("AADI_DEBIAS", "0") == "1"
# Keep the model in memory between decisions (Ollama unloads after 5 min).
KEEP_ALIVE = os.environ.get("AADI_KEEP_ALIVE", "60m")
LETTERS = "ABCDEFGH"
SYSTEM_PROMPT = (
    "You are a multiple-choice classifier. "
    "You only ever output one capital letter."
)
LIVE_PAGE = Path(__file__).with_name("live.html")


def _ollama(path: str, body: dict | None = None, timeout: float = 120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        OLLAMA_URL + path, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(502, f"Ollama error {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        raise HTTPException(503, f"Ollama unreachable: {exc.reason}")


_capabilities: dict[str, list[str]] = {}


def _model_capabilities(model: str) -> list[str]:
    if model not in _capabilities:
        info = _ollama("/api/show", {"model": model}, timeout=10)
        _capabilities[model] = info.get("capabilities", [])
    return _capabilities[model]


def _usable(caps: list[str]) -> bool:
    """Needs audio input and text generation (rules out embedding models)."""
    return "audio" in caps and "completion" in caps


def _supports_thinking(model: str) -> bool:
    return "thinking" in _model_capabilities(model)


# ------------------------------------------------------------------ warm-up
_warm = {"model": MODEL_ID, "status": "warming", "seconds": None,
         "error": None}
_warm_lock = threading.Lock()


def _silent_wav(seconds: float = 1.0) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return out.getvalue()


def _warm_up(model: str):
    """Load the model and its audio encoder before the first real request."""
    started = time.perf_counter()
    try:
        _letter_logprobs(
            base64.b64encode(_silent_wav()).decode(), "Is there speech?",
            ["Yes", "No"], model=model,
        )
        result = {"status": "ready",
                  "seconds": round(time.perf_counter() - started, 1),
                  "error": None}
        print(f"AADI: {model} warm in {result['seconds']} s", flush=True)
    except Exception as exc:  # server still runs; /health shows the problem
        detail = getattr(exc, "detail", None) or str(exc)
        result = {"status": "error", "seconds": None, "error": detail}
        print(f"AADI: warm-up of {model} failed: {detail}", flush=True)
    with _warm_lock:
        if _warm["model"] == model:  # ignore if another switch happened since
            _warm.update(result)


def _start_warm_up(model: str):
    with _warm_lock:
        _warm.update(model=model, status="warming", seconds=None, error=None)
    threading.Thread(target=_warm_up, args=(model,), daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _start_warm_up(MODEL_ID)
    yield


app = FastAPI(title="AADI-Ollama", lifespan=lifespan)


# ------------------------------------------------------------------ routes
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/live")


@app.get("/live", include_in_schema=False)
def live_page():
    """Browser UI: define decisions, then run them live on a file or mic."""
    return FileResponse(LIVE_PAGE)


@app.get("/health")
def health():
    caps = _model_capabilities(MODEL_ID)
    with _warm_lock:
        warm = dict(_warm)
    if not _usable(caps):
        status = "model_has_no_audio"
    elif warm["status"] == "error":
        status = "error"
    else:
        status = warm["status"]  # "warming" or "ready"
    return {
        "status": status,
        "model": MODEL_ID,
        "device": "ollama",
        "capabilities": caps,
        "warm_up_seconds": warm["seconds"],
        "error": warm["error"],
    }


@app.get("/models")
def list_models():
    """Local Ollama models that take audio and generate text. Cloud models
    are left out: they do not take audio or return logprobs."""
    tags = _ollama("/api/tags", timeout=10).get("models", [])
    models = []
    for m in tags:
        name = m.get("name") or m.get("model")
        if not name or m.get("remote_host") or name.endswith("-cloud") \
                or name.endswith(":cloud"):
            continue
        try:
            caps = _model_capabilities(name)
        except HTTPException:
            continue
        if _usable(caps):
            models.append({
                "name": name,
                "size_gb": round(m.get("size", 0) / 1e9, 1),
                "parameters": m.get("details", {}).get("parameter_size"),
                "quantization": m.get("details", {}).get("quantization_level"),
            })
    models.sort(key=lambda x: x["name"])
    return {"current": MODEL_ID, "default": DEFAULT_MODEL, "models": models}


class ModelChoice(BaseModel):
    model: str


@app.post("/model")
def set_model(choice: ModelChoice):
    """Switch the model used for every following decision, and warm it up."""
    global MODEL_ID
    name = choice.model.strip()
    if not name:
        raise HTTPException(400, "Model name is required")
    _capabilities.pop(name, None)  # re-read, e.g. after a re-pull
    if not _usable(_model_capabilities(name)):
        raise HTTPException(
            400, f"{name} cannot be used: it needs audio input and text output"
        )
    MODEL_ID = name
    _start_warm_up(name)
    return {"model": MODEL_ID, "status": "warming"}


def _validate_wav(audio_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as f:
            if (
                f.getframerate() != 16000
                or f.getnchannels() != 1
                or f.getsampwidth() != 2
                or f.getcomptype() != "NONE"
            ):
                raise ValueError("Expected 16 kHz, mono, PCM 16-bit WAV")
            duration = f.getnframes() / 16000
    except (wave.Error, EOFError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    if duration <= 0 or duration > 30:
        raise HTTPException(
            400, "Audio duration must be between 0 and 30 seconds"
        )
    return duration


def _letter_logprobs(audio_b64: str, question: str, choices: list[str],
                     model: str | None = None):
    """One Ollama call -> (logprob per letter, missing letters, ms)."""
    model = model or MODEL_ID
    letters = LETTERS[: len(choices)]
    prompt = (
        "Listen carefully to the audio.\n"
        + question + "\n"
        + "\n".join(f"{k}. {v}" for k, v in zip(letters, choices))
        + f"\nAnswer with a single letter ({', '.join(letters)}) "
        "and nothing else."
    )
    body = {
        "model": model,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 20,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_predict": 1, "temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt, "images": [audio_b64]},
        ],
    }
    if _supports_thinking(model):
        # Otherwise the first token is reasoning, not the answer letter.
        body["think"] = False
    started = time.perf_counter()
    result = _ollama("/api/chat", body)
    ms = (time.perf_counter() - started) * 1000

    lp = result.get("logprobs") or []
    if not lp:
        raise HTTPException(502, "Ollama returned no logprobs")
    top = lp[0].get("top_logprobs", [])

    # Merge variants like "B" and " B" with logsumexp.
    found: dict[str, float] = {}
    for t in top:
        key = t["token"].strip()
        if key in letters:
            prev = found.get(key)
            v = t["logprob"]
            found[key] = v if prev is None else max(prev, v) + math.log1p(
                math.exp(-abs(prev - v))
            )

    floor = min((t["logprob"] for t in top), default=-30.0) - 1.0
    missing = [k for k in letters if k not in found]
    scores = [found.get(k, floor) for k in letters]
    return scores, missing, ms


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def run_decision(audio_bytes: bytes, question=None, choices=None,
                 debias: bool | None = None):
    start_total = time.perf_counter()
    if not question or not choices:
        raise HTTPException(400, "question and choices are required")
    if len(choices) > len(LETTERS):
        raise HTTPException(400, "At most 8 choices")
    debias = DEFAULT_DEBIAS if debias is None else debias
    model = MODEL_ID  # same model for every call of this decision

    duration = _validate_wav(audio_bytes)
    audio_b64 = base64.b64encode(audio_bytes).decode()
    n = len(choices)

    rotations = range(n) if debias else range(1)
    per_option = [0.0] * n
    label_mass = 0.0
    forward_ms = 0.0
    missing_all = set()

    for r in rotations:
        order = [(i + r) % n for i in range(n)]  # position -> option index
        scores, missing, ms = _letter_logprobs(
            audio_b64, question, [choices[j] for j in order], model=model
        )
        forward_ms += ms
        label_mass += sum(
            math.exp(s) for k, s in zip(LETTERS, scores) if k not in missing
        )
        for pos, p in enumerate(_softmax(scores)):
            per_option[order[pos]] += p
        missing_all.update(LETTERS[order[LETTERS.index(m)]] for m in missing)

    k = len(rotations)
    probs = [p / k for p in per_option]
    probabilities = {
        LETTERS[i]: round(p, 6) for i, p in enumerate(probs)
    }
    decision = max(probabilities, key=probabilities.get)

    return {
        "model": f"AADI-Ollama ({model})",
        "decision": decision,
        "label": choices[LETTERS.index(decision)],
        "probabilities": probabilities,
        "label_mass": round(label_mass / k, 6),
        "missing_labels": sorted(missing_all),
        "debiased": debias,
        "calls": k,
        "audio_duration_sec": round(duration, 3),
        "forward_ms": round(forward_ms, 2),
        "total_ms": round((time.perf_counter() - start_total) * 1000, 2),
    }


@app.post("/decide")
async def decide(request: Request):
    audio_bytes = await request.body()
    if not audio_bytes or len(audio_bytes) > 5_000_000:
        raise HTTPException(400, "Empty or oversized WAV")

    question = unquote(request.headers.get("x-aadi-question") or "")
    options = unquote(request.headers.get("x-aadi-options") or "")
    debias_hdr = request.headers.get("x-aadi-debias")
    debias = None if debias_hdr is None else debias_hdr == "1"

    if not question or not options:
        raise HTTPException(
            400, "Both x-aadi-question and x-aadi-options are required"
        )
    choices = [x.strip() for x in options.split("|")]
    if (
        not 2 <= len(choices) <= 8
        or any(not x or len(x) > 100 for x in choices)
        or len(question) > 500
    ):
        raise HTTPException(400, "Invalid question or choices")

    return await asyncio.to_thread(
        run_decision, audio_bytes, question, choices, debias
    )


# /v1/systemone: reuse the original router. Its model-name check expects
# "AADI-E4B", so clients keep sending that name.
app.include_router(create_router(run_decision))
