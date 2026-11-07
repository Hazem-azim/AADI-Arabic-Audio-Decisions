import asyncio
import io
import threading
import time
import wave
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

MODEL_PATH = "/model"
MODEL_ID = "google/gemma-4-E4B-it"
LABELS = {
    "A": "1 speaker",
    "B": "3 speakers",
    "C": "5 speakers",
    "D": "7 speakers",
}

QUESTION = (
    "Listen carefully to the Arabic conversation.\n"
    "How many distinct speakers can you hear in this audio?\n"
    "A. 1 speaker\n"
    "B. 3 speakers\n"
    "C. 5 speakers\n"
    "D. 7 speakers\n"
    "Respond with only A, B, C, or D."
)

gpu_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AADI: Loading processor...", flush=True)

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True
    )

    print("AADI: Loading E4B model...", flush=True)

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
    )

    model = model.to("cuda").eval()

    option_ids = []
    for label in LABELS:
        ids = processor.tokenizer.encode(
            label, add_special_tokens=False
        )
        if len(ids) != 1:
            raise RuntimeError(f"Invalid option token: {label}")
        option_ids.append(ids[0])

    app.state.processor = processor
    app.state.model = model
    app.state.option_ids = option_ids

    print("AADI-E4B READY — Model resident on GPU", flush=True)

    yield


app = FastAPI(title="AADI-E4B", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ready",
        "model": MODEL_ID,
        "device": "cuda:0",
    }


def run_decision(audio_bytes: bytes, question=None, choices=None):
    start_total = time.perf_counter()

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as f:
            if (
                f.getframerate() != 16000
                or f.getnchannels() != 1
                or f.getsampwidth() != 2
                or f.getcomptype() != "NONE"
            ):
                raise ValueError(
                    "Expected 16 kHz, mono, PCM 16-bit WAV"
                )

            samples = np.frombuffer(
                f.readframes(f.getnframes()),
                dtype="<i2",
            ).astype(np.float32) / 32768.0

    except (wave.Error, EOFError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    duration = len(samples) / 16000

    if duration <= 0 or duration > 30:
        raise HTTPException(
            status_code=400,
            detail="Audio duration must be between 0 and 30 seconds",
        )

    labels = (
        LABELS if choices is None
        else dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", choices))
    )

    prompt = QUESTION if choices is None else (
        "Listen carefully to the audio.\n"
        + question + "\n"
        + "\n".join(f"{k}. {v}" for k, v in labels.items())
        + "\nRespond with only the option letter."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": samples},
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = app.state.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda", dtype=torch.bfloat16)

    with gpu_lock:
        torch.cuda.synchronize()
        start_forward = time.perf_counter()

        with torch.inference_mode():
            outputs = app.state.model(**inputs)
            logits = outputs.logits[0, -1, :].float()

        torch.cuda.synchronize()
        forward_ms = (
            time.perf_counter() - start_forward
        ) * 1000

        ids = []
        for label in labels:
            token_ids = app.state.processor.tokenizer.encode(
                label, add_special_tokens=False
            )
            if len(token_ids) != 1:
                raise ValueError(f"Invalid option token: {label}")
            ids.append(token_ids[0])
        selected = logits[ids]

        probs = torch.softmax(selected, dim=0)
        label_mass = torch.softmax(
            logits, dim=-1
        )[ids].sum().item()

        probabilities = {
            label: round(probs[i].item(), 6)
            for i, label in enumerate(labels)
        }

    decision = max(probabilities, key=probabilities.get)

    return {
        "model": "AADI-E4B",
        "decision": decision,
        "label": labels[decision],
        "probabilities": probabilities,
        "label_mass": round(label_mass, 6),
        "audio_duration_sec": round(duration, 3),
        "forward_ms": round(forward_ms, 2),
        "total_ms": round(
            (time.perf_counter() - start_total) * 1000, 2
        ),
    }


@app.post("/decide")
async def decide(request: Request):
    audio_bytes = await request.body()

    if not audio_bytes or len(audio_bytes) > 5000000:
        raise HTTPException(
            status_code=400,
            detail="Empty or oversized WAV",
        )

    from urllib.parse import unquote
    question = unquote(request.headers.get("x-aadi-question") or "")
    options = unquote(request.headers.get("x-aadi-options") or "")

    if bool(question) != bool(options):
        raise HTTPException(
            status_code=400,
            detail="Both question and options headers are required",
        )

    choices = None
    if options:
        choices = [x.strip() for x in options.split("|")]
        if (
            not 2 <= len(choices) <= 8
            or any(not x or len(x) > 100 for x in choices)
            or len(question) > 500
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid question or choices",
            )

    return await asyncio.to_thread(
        run_decision, audio_bytes, question, choices
    )


from aadi_systemone_api import create_router

app.include_router(create_router(run_decision))
