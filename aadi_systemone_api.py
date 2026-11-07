import asyncio
import base64
import binascii
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal


class AudioState(BaseModel):
    audio_base64: str


class DecisionQuestion(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: str
    criteria: dict[str, str | None] | list[str] | None = None


class SystemOneRequest(BaseModel):
    model: str = "AADI-E4B"
    state: AudioState
    questions: dict[str, DecisionQuestion]


def create_router(run_decision):
    router = APIRouter()

    @router.post("/v1/systemone")
    async def system_one(payload: SystemOneRequest):
        started = time.perf_counter()

        if payload.model != "AADI-E4B":
            raise HTTPException(400, "Unsupported model")

        if not 1 <= len(payload.questions) <= 8:
            raise HTTPException(400, "Expected 1 to 8 questions")

        encoded = payload.state.audio_base64

        if len(encoded) > 7_000_000:
            raise HTTPException(413, "Audio too large")

        try:
            audio = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(400, "Invalid base64 audio")

        if not audio or len(audio) > 5_000_000:
            raise HTTPException(400, "Invalid audio size")

        answers = {}
        duration = None

        for question_id, q in payload.questions.items():
            if not question_id or len(question_id) > 100:
                raise HTTPException(400, "Invalid question ID")

            if not 1 <= len(q.instructions.strip()) <= 500:
                raise HTTPException(400, "Invalid instructions")

            if q.type == "noul":
                if q.criteria is not None:
                    raise HTTPException(
                        400, "Noul does not require criteria"
                    )
                choices = ["Yes", "No"]
                values = ["Yes", "No"]

            elif q.type == "choice":
                if not isinstance(q.criteria, dict):
                    raise HTTPException(
                        400, "Choice requires a criteria dictionary"
                    )

                values = list(q.criteria.keys())

                if not 2 <= len(values) <= 8:
                    raise HTTPException(
                        400, "Choice requires 2 to 8 options"
                    )

                choices = [
                    f"{key}: {description}"
                    if description else key
                    for key, description in q.criteria.items()
                ]

            else:
                if not isinstance(q.criteria, list):
                    raise HTTPException(
                        400, "Score requires an ordered criteria list"
                    )

                values = list(range(len(q.criteria)))

                if not 2 <= len(values) <= 8:
                    raise HTTPException(
                        400, "Score requires 2 to 8 levels"
                    )

                choices = [
                    f"Score {i}: {description}"
                    for i, description in enumerate(q.criteria)
                ]

            if any(
                not choice.strip() or len(choice) > 100
                for choice in choices
            ):
                raise HTTPException(400, "Invalid option text")

            result = await asyncio.to_thread(
                run_decision,
                audio,
                q.instructions,
                choices,
            )

            if duration is None:
                duration = result["audio_duration_sec"]

            letter_probs = result["probabilities"]
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            probabilities = {
                str(value): letter_probs[letters[i]]
                for i, value in enumerate(values)
            }

            selected = letters.index(result["decision"])
            selected_value = values[selected]
            confidence = letter_probs[result["decision"]]

            answer = {
                "type": q.type,
                "confidence": confidence,
                "probabilities": probabilities,
                "label_mass": result["label_mass"],
                "forward_ms": result["forward_ms"],
            }

            if q.type == "noul":
                answer["noul"] = selected == 0
                answer["probability"] = letter_probs["A"]

            elif q.type == "choice":
                answer["choice"] = selected_value

            else:
                answer["score"] = selected_value

            answers[question_id] = answer

        return {
            "model": "AADI-E4B",
            "answers": answers,
            "usage": {
                "audio_duration_sec": duration,
                "questions": len(answers),
                "output_tokens": 0,
            },
            "latency_ms": round(
                (time.perf_counter() - started) * 1000, 2
            ),
        }

    return router
