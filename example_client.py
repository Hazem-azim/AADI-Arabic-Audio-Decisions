#!/usr/bin/env python3
"""Send one Arabic multiple-choice audio decision to the local AADI /decide API.

Usage:
    python example_client.py input_16k.wav

The file must be 16-kHz, mono, 16-bit PCM WAV, 30 seconds or shorter.
This script uses Python's standard library (no requests package needed).
"""

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:8765/decide"
QUESTION = "هما اتفقوا يتقابلوا إمتى؟"
OPTIONS = [
    "الأحد الساعة 3",
    "الخميس الساعة 2",
    "الثلاثاء الساعة واحدة",
    "الخميس الساعة 6",
    "ما اتفقوش يتقابلوا",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python example_client.py input_16k.wav")
        return 2

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"No such audio file: {path}")
        return 2

    data = path.read_bytes()
    request = Request(
        URL,
        data=data,
        headers={
            "Content-Type": "audio/wav",
            "x-aadi-question": quote(QUESTION, safe=""),
            "x-aadi-options": quote("|".join(OPTIONS), safe=""),
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
        return 1
    except URLError as exc:
        print(f"Connection failed: {exc}")
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
