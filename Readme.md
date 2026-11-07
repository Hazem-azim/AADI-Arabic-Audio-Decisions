# AADI — Arabic Audio Decision Intelligence

### Native Arabic Audio Understanding Through Zero-Token Decision Making

**AADI** is an experimental AI framework developed by **Genarabia** for making fast, structured decisions directly from spoken Arabic, without speech-to-text transcription or autoregressive answer generation.

Instead of asking a multimodal language model to generate a textual response, AADI formulates audio comprehension as a **System-1 decision problem**: selecting the most likely answer from a predefined set of alternatives using model logits.

## Key Features

- **Native audio processing:** Analyze spoken Arabic directly, without a separate ASR pipeline.
- **Zero-token generation:** Decisions are obtained from a forward pass, without generating response tokens.
- **Arabic multiple-choice questions:** Ask questions and supply two or more possible answers.
- **Decision types:** Yes/No, multiple-choice classification, and ordinal scoring.
- **Probability distributions:** Return normalized scores for every candidate answer.
- **Low-latency inference:** Designed for GPU-based, real-time audio decision applications.

## Architecture

```text
Arabic Audio (16 kHz, mono)
             |
             v
     Native Audio Encoder
             |
             v
     Gemma 4 E4B-IT
             |
   Audio + Arabic Question
       + Answer Choices
             |
             v
      Single Forward Pass
             |
             v
     Next-Token Letter Logits
             |
             v
   Softmax Over Candidate Tokens
             |
             v
    Structured Decision Output
```

AADI uses **Gemma 4 E4B-IT** as its initial foundation model. Its decision engine extracts next-token logits associated with answer labels rather than requesting generated text.

The resulting probabilities are normalized across the provided choices. They should not be interpreted as independently calibrated confidence scores.

## Example — Arabic Audio Comprehension

**Input:** An Egyptian Arabic conversation.

**Question:**

> هما اتفقوا يتقابلوا إمتى؟

**Answer choices:**

- الأحد الساعة 3
- الخميس الساعة 2
- الثلاثاء الساعة واحدة
- الخميس الساعة 6
- ما اتفقوش يتقابلوا

**Decision:**

`الخميس الساعة 2`

No intermediate transcript or generated answer is required.

## Initial Experimental Results

Our early experiments on an NVIDIA GB10 GPU demonstrated:

- Correct identification of Arabic speech.
- Extraction of names and conversational facts.
- Meeting-agreement and meeting-time decisions.
- Warm inference latency averaging approximately **208 ms per decision** in initial Gemma 4 E4B tests.

These are preliminary results from a small exploratory test set, not a comprehensive accuracy benchmark.

## Technology Stack

- Python and PyTorch
- Google Gemma 4 E4B-IT
- Hugging Face Transformers
- NVIDIA CUDA
- FastAPI for structured inference endpoints
- Docker for reproducible deployment

## Project Status

**Research prototype — public documentation and source-code release in progress.**

The project currently includes an experimental inference engine, structured System-1 decision API, and an Arabic audio demonstration interface.

Implementation files, setup instructions, and reproducible benchmarks will be added to this repository.

Model weights are not distributed with the repository and remain subject to their original licensing terms.

## Research Directions

- Arabic audio comprehension benchmarking.
- Comparison of generative-logit and trained decision-head architectures.
- Multimodal audio–text embedding similarity.
- Decision calibration and uncertainty estimation.
- High-throughput, low-latency System-1 audio intelligence.

---

**Developed by [Genarabia](https://genarabia.ai)**

*Arabic-first AI research and engineering.*

**AADI — Listen. Decide. No Generation.**
