# AADI — Installation and Deployment

**AADI (Arabic Audio Decision Intelligence)** is a Genarabia research prototype powered by **Google Gemma 4 E4B-IT**. It performs native audio decision-making by reading the model's next-token logits; it does **not** fine-tune Gemma 4, run a separate speech-to-text model, or generate output tokens.

Foundation model: [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it). The foundation model and its license belong to Google; the weights are **not** included here. Review and accept any applicable Hugging Face model access terms before downloading.

## 1. Requirements

- Linux machine with an NVIDIA GPU capable of running Gemma 4 E4B-IT in BF16, sufficient available GPU memory, and working NVIDIA drivers/CUDA.
- Python 3.12 recommended; a working CUDA-enabled PyTorch installation is required.
- Reference tested environment: NVIDIA GB10 ARM64, PyTorch `2.13.0+cu130`, Transformers `5.19.0`, one Uvicorn worker.
- Files from this repository: `aadi_server.py`, `aadi_systemone_api.py`, `requirements.txt` and optionally `example_client.py`.
- This is a **research deployment**, not a hardened public API.

GPU/PyTorch installation is platform-dependent. Install a PyTorch build compatible with **your GPU, operating system and CUDA drivers** first, following the official [PyTorch installation guide](https://pytorch.org/get-started/locally/). Do not indiscriminately install the GB10 ARM64 CUDA build on a different GPU architecture.

## 2. Prepare the Python runtime

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install a compatible CUDA-enabled torch build for your machine FIRST.
# Verify that your existing PyTorch installation can see the GPU:
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'

pip install -r requirements.txt
pip install huggingface_hub
```

**Expected:** `torch.cuda.is_available()` returns `True` before you start AADI. If it returns `False`, install/fix CUDA-enabled PyTorch before proceeding. The `requirements.txt` in this repository intentionally does not choose a platform-specific PyTorch wheel.

## 3. Download Gemma 4 E4B-IT separately

Visit the [Google Gemma 4 E4B-IT model page](https://huggingface.co/google/gemma-4-E4B-it) and follow its access and license instructions. Then:

```bash
hf auth login
hf download google/gemma-4-E4B-it --local-dir ./models/gemma-4-E4B-it
```

The model can be large. **Do not commit model weights**, HF authentication tokens, or downloaded recordings to GitHub. The server expects to find the model files under `/model` (see the next section).

## 4. Run the API locally

The checked-in server uses absolute filesystem paths:

- `/model` — downloaded Google Gemma 4 E4B-IT files
- `/server` — AADI Python source files
- `127.0.0.1:8765` — HTTP API on the local host

### Docker reference command (for an already prepared CUDA image)

Our GB10 experiment used a **custom local Docker image** called `gemma4-vllm-audio:latest`. **That image is not published by this repository.** The following command is a reference for users who already have a compatible Python/CUDA image with PyTorch, Transformers 5.19.0, FastAPI, NumPy and Uvicorn installed; replace `gemma4-vllm-audio:latest` with the name of *your* prepared image if different.

From the cloned repository root, after downloading the model into `models/gemma-4-E4B-it`:

```bash
docker run --rm --gpus all \
  --name aadi-e4b \
  -p 127.0.0.1:8765:8765 \
  -v "$(pwd)/models/gemma-4-E4B-it:/model:ro" \
  -v "$(pwd):/server:ro" \
  --entrypoint python3 \
  gemma4-vllm-audio:latest \
  -u -m uvicorn aadi_server:app \
  --app-dir /server --host 0.0.0.0 --port 8765 --workers 1
```

The model is loaded once at server startup. The container uses an explicit localhost-only port binding; **do not expose this development API directly to the public internet**.

### Non-Docker alternative

For a Linux host with a working CUDA Python environment, make the same paths available (for example, using your own container or appropriately configured directories), then run:

```bash
python -m uvicorn aadi_server:app --host 127.0.0.1 --port 8765 --workers 1
```

**Note:** As currently released, `aadi_server.py` hardcodes `/model`. On a non-Docker host either provide that path, or change `MODEL_PATH` in your local copy to point to your downloaded model directory. The optional `/showcase` route was excluded from the initial public code release, so the server does not require the private showcase HTML file.

## 5. Confirm the server works

```bash
curl http://127.0.0.1:8765/health
```

Expected fields include `status`, `model`, and `device` (for example `"status":"ready"`).

### Input audio constraints

- WAV container; **16,000 Hz**, **mono**, **16-bit signed PCM**; uncompressed.
- Duration: greater than 0 and **no more than 30 seconds**.
- Raw WAV request body or Base64-encoded WAV through the SystemOne endpoint.
- Maximum WAV payload size for the endpoints: 5,000,000 bytes.

To convert a source recording when necessary:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le -t 30 input_16k.wav
```

## 6. Run an Arabic multiple-choice decision

### `/decide` — raw WAV with encoded question/options headers

Use the accompanying Python script, which safely URL-encodes Arabic headers and sends the WAV using Python's standard library:

```bash
python example_client.py input_16k.wav
```

This example asks **«هما اتفقوا يتقابلوا إمتى؟»** with five Arabic choices. Its output contains:

- `decision`: selected letter (`A`, `B`, ...)
- `label`: selected option text
- `probabilities`: softmax probabilities over supplied option letters
- `label_mass`: probability mass of the candidate letters relative to the complete vocabulary
- `forward_ms`, `total_ms`: timing diagnostics

### `/v1/systemone` — structured batch of decisions

This endpoint accepts one Base64-encoded WAV and up to **8 questions**. Supported types:

- `noul`: binary Yes/No decision; no `criteria` required.
- `choice`: dictionary of 2–8 named options, optionally mapped to descriptions.
- `score`: ordered list of 2–8 textual levels.

Example request structure (replace `BASE64_WAV_HERE` with an encoded valid WAV):

```json
{
  "model": "AADI-E4B",
  "state": {"audio_base64": "BASE64_WAV_HERE"},
  "questions": {
    "meeting_agreed": {
      "type": "noul",
      "instructions": "هل اتفق المتحدثون على موعد للقاء؟"
    },
    "meeting_time": {
      "type": "choice",
      "instructions": "هما اتفقوا يتقابلوا إمتى؟",
      "criteria": {
        "الأحد الساعة 3": null,
        "الخميس الساعة 2": null,
        "الثلاثاء الساعة واحدة": null,
        "ما اتفقوش يتقابلوا": null
      }
    },
    "meeting_clarity": {
      "type": "score",
      "instructions": "ما مدى وضوح موعد اللقاء المذكور في المقطع؟",
      "criteria": [
        "لم يُذكر موعد لقاء",
        "ذُكر اليوم أو الساعة فقط",
        "ذُكر اليوم والساعة بوضوح"
      ]
    }
  }
}
```

For `noul`, the API returns a Boolean `noul` and its probability; for `choice`, the selected option; for `score`, a zero-based score index. Every answer also includes probability and latency diagnostics.

## 7. Decision mechanism and limitations

AADI passes audio and question/options through **Gemma 4 E4B-IT** in a **single forward pass**. It gathers the logits at the final input position for single-token candidate labels `A`, `B`, ..., applies a softmax over **only** those candidates, and returns the highest-scoring option. It invokes **no text generation** and uses **no external ASR pipeline**.

**Important limitations:**

- Choice-normalized probabilities are **not calibrated accuracy estimates**.
- `label_mass` helps diagnose how much of the full next-token distribution is assigned to the candidate letters, but does not establish correctness.
- Letter-based classification may be sensitive to prompt wording, answer order, tokenization, and audio content.
- Multiple questions currently trigger separate forward passes, even though the model remains loaded.
- Prototype measurements on GB10 (~208 ms warm per decision in early testing) are not general performance or accuracy guarantees.
- Model outputs need independent human-labeled evaluation before high-stakes deployment.
- The FastAPI app has no authentication, request quota, or production abuse controls.

## 8. Publication scope

This GitHub repository publishes Genarabia's **wrapper/inference implementation**, not Google's model or modified weights. The experimental audio/video showcase is not included in this initial source release. The repository currently has **no explicit license for its own source code**; readers should not assume permission to redistribute or commercially reuse it merely because it is public.

Developed by **Genarabia** — [genarabia.ai](https://genarabia.ai).
