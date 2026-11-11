# AADI on Ollama — local Mac setup and live test page

`aadi_ollama_server.py` runs AADI through a local [Ollama](https://ollama.com) audio model instead of Gemma 4 on CUDA. It needs no NVIDIA GPU and runs on an Apple Silicon Mac. It keeps the API of `aadi_server.py` (`/health`, `/decide`, `/v1/systemone`), so `example_client.py` works unchanged. It also adds a browser page for defining decisions and testing them live on a video, an audio file or the microphone.

## How a decision works

For each question, the server sends the WAV and the lettered options to Ollama's `/api/chat`. It asks for **one** output token with `top_logprobs`, then reads the scores of the option letters (`A`, `B`, …) and softmaxes them over the options. This is the same letter-logit method as the CUDA server. Thinking is switched off automatically for models that support it, so the first token is the answer, not reasoning.

## Files

| File | What it is |
|---|---|
| `aadi_ollama_server.py` | FastAPI server that talks to Ollama. Start this. |
| `live.html` | The live test page. The server serves it at `/live`; you don't open the file directly. |
| `aadi_systemone_api.py` | Existing repo file, reused unchanged for `/v1/systemone`. Must stay in the same folder. |
| `example_client.py` | Existing repo file. Sends one WAV to `/decide`; works against this server too. |

Nothing else from the repo is needed: no PyTorch, Transformers, CUDA or Docker, and no `requirements.txt` install.

## Quick start

**1. Install Ollama and get the model** (once). Install Ollama from <https://ollama.com/download> and make sure it is running (the menu-bar app, or `ollama serve`). Then:

```bash
ollama pull gemma4:e4b
ollama show gemma4:e4b      # "Capabilities" must include audio
```

**2. Start the AADI server** from the repository folder, so that `aadi_systemone_api.py` and `live.html` are found.

With [uv](https://docs.astral.sh/uv/) (no install step; dependencies are fetched on first run):

```bash
cd AADI-Arabic-Audio-Decisions
uv run --with fastapi --with 'uvicorn[standard]' \
  uvicorn aadi_ollama_server:app --host 127.0.0.1 --port 8765
```

Or with a plain virtual environment:

```bash
cd AADI-Arabic-Audio-Decisions
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi 'uvicorn[standard]'
uvicorn aadi_ollama_server:app --host 127.0.0.1 --port 8765
```

The server is ready when the log shows:

```
AADI: gemma4:e4b warm in 5.9 s
```

It loads the model in the background at startup (a few seconds the first time) and asks Ollama to keep it in memory, so the first decision isn't slowed by a model load. Stop the server with `Ctrl+C`.

**3. Use it.**

- **Browser:** open <http://127.0.0.1:8765/live> (see [Live test page](#live-test-page-livehtml) below).
- **Check status:** `curl http://127.0.0.1:8765/health` should return `"status":"ready"` once the model is loaded.
- **From Python:** `python example_client.py input_16k.wav` (a 16 kHz mono 16-bit WAV, ≤ 30 s; see `SETUP.md` for converting with ffmpeg).

**Another model or port:** set the variables in front of the start command, or switch the model later from the page header:

```bash
AADI_OLLAMA_MODEL=lfm3-audio:latest uv run --with fastapi --with 'uvicorn[standard]' \
  uvicorn aadi_ollama_server:app --host 127.0.0.1 --port 9000
```

**Troubleshooting**

| You see | Fix |
|---|---|
| `Ollama unreachable` | Start Ollama, or set `OLLAMA_URL` if it runs elsewhere. |
| `Could not import module "aadi_ollama_server"` | Run the command from the repository folder. |
| Badge says *Model has no audio input* | Pick an audio model in the header, or `ollama pull gemma4:e4b`. |
| `model ... not found` in the warm-up log | The model isn't pulled: `ollama pull <name>`. |
| `address already in use` | Another server is on port 8765: stop it, or use `--port 9000`. |

`gemma4:e4b` is the default model, the same model the CUDA server uses (Gemma 4 E4B-IT).

### Settings

| Env var | Default | Meaning |
|---|---|---|
| `AADI_OLLAMA_MODEL` | `gemma4:e4b` | Model used at startup |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server |
| `AADI_KEEP_ALIVE` | `60m` | How long Ollama keeps the model loaded between requests |
| `AADI_DEBIAS` | `0` | `1` = rotate the options through every position and average (one call per option) |

## Endpoints added on top of the original API

- `GET /models` lists the local Ollama models that accept audio **and** generate text. Embedding models and cloud models are left out.
- `POST /model` with `{"model": "lfm3-audio:latest"}` switches the model for all following decisions and loads it in the background. It returns 400 for a model that can't be used.
- `GET /health` also reports `status: "warming" | "ready" | "error" | "model_has_no_audio"`, plus `warm_up_seconds`.
- `GET /live` serves the test page.
- On `/decide`, the header `x-aadi-debias: 1` turns on debiasing for that request. Responses also include `missing_labels` (see Limitations).

## Live test page (`live.html`)

1. **Decisions.** Start from an example (talk show commentary, customer service call, meeting and scheduling, sports commentary, news topic) or write your own. Up to 8 questions, each one of:
   - **Multiple choice:** 2–8 options, one per line. You can add a hint for the model after `::`, e.g. `تطبيل :: مدح مبالغ فيه للحكومة`.
   - **Yes / No.**
   - **Score:** 2–8 ordered levels, lowest first.

   Questions are saved in the browser and can be exported or imported as JSON.
2. **Live test.** Load a video or audio file (the browser extracts the audio and resamples it to 16 kHz mono) or use the microphone. It uses a sliding window: every *N* seconds (default 2) the last *M* seconds of audio (default 8) are scored.
   - Each question is its own request, so each card updates as soon as its answer arrives.
   - The first decision waits for 3 s of audio, because a 1 s clip is mostly a guess.
   - Only one update runs at a time, and the next one always takes the newest audio, so it never builds a backlog.
   - **Stopping the microphone** cancels any update still running and immediately decides on the audio up to the click. By default this final decision hears the **whole recording** (up to 30 s), so it has the full context; a final on 18 s of audio takes ~3 s. Untick the option to judge only the last window instead (~1.5 s). Final decisions are marked "Final" and outlined in the timeline.
   - Updates are driven by the audio itself, not the page's animation loop, so they keep running when the tab is in the background.
   - Cards show every option's probability, the winner, its confidence and a colour timeline of earlier decisions.
   - Results can be downloaded as JSON.
3. **Model picker** in the header: switch between the local audio models. A switch stops any running test, and the badge shows "Loading model…" until the new model is ready.

## Measured on an Apple Silicon Mac

**Speed (`gemma4:e4b`):** Ollama reuses the processed audio across questions on the same window. With a 10 s window, the first question takes ~0.6 s and each further one ~0.15 s. With 5 questions, one update takes ~1.5 s and decisions trail the audio by ~1.5 s.

**Accuracy** on real speech: 12 native Egyptian-speaker recordings from [FLEURS](https://huggingface.co/datasets/google/fleurs) `ar_eg` (CC-BY 4.0).

| Model | Topic (8 shuffled options) | Detail questions | Notes |
|---|---|---|---|
| `gemma4:e4b` | 11/12 | 6/6 | Accurate transcription; picks the right letter at ~1.00 confidence |
| `lfm3-audio` | — | — | Hears Arabic, but often answers with option text instead of a letter, so letter scores are unreliable |
| `lfm3-omni` | — | — | Same letter problem; consistently wrong on synthetic test clips |

These are small sanity checks, not a benchmark.

## Limitations

- **Only 20 scores per call.** Ollama returns just the top 20 next-token logprobs. An option letter outside them gets a floor score and is listed in `missing_labels`.
- **No cloud models.** Ollama cloud models (e.g. `gemma4:31b-cloud`) neither accept audio nor return logprobs.
- **Confidence isn't calibrated.** One wrong FLEURS answer had 0.98 confidence.
- **Silence and dialect are weak spots.** On silence the model still tends to say speech is present, though with lower confidence. Dialect identification was unreliable.
- **One model for everyone.** The selected model is global to the server, not per client.
- **Microphone mode** was tested with a recording fed in as the microphone, not with a live microphone.
