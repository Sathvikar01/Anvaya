"""Live smoke test for the Gnani integration: Timbre TTS -> WAV -> Prisma STT round trip.

Usage:
    python scripts/gnani_smoke.py [--language en-IN] [--text "..."] [--out results/gnani_smoke.wav]

Exits 0 with a "skipped" message when GNANI_API_KEY is not set, so it is safe in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hh_goa_rag.stt.gnani import GnaniSTT  # noqa: E402
from hh_goa_rag.tts.gnani import GnaniTTS, GnaniTTSConfig  # noqa: E402

SAMPLE_TEXT = {
    "en-IN": "Goa is famous for its beaches and its spice plantations.",
    "hi-IN": "गोवा अपने समुद्र तटों के लिए प्रसिद्ध है।",
    "kn-IN": "ಗೋವಾ ತನ್ನ ಕಡಲತೀರಗಳಿಗೆ ಪ್ರಸಿದ್ಧವಾಗಿದೆ.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="en-IN")
    parser.add_argument("--text", default=None)
    parser.add_argument("--voice", default=None, help="Timbre voice (default: GNANI_TTS_VOICE)")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "gnani_smoke.wav")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    api_key = os.getenv("GNANI_API_KEY", "").strip()
    if not api_key:
        print("skipped: GNANI_API_KEY is not set (add it to .env or the environment)")
        return 0

    text = args.text or SAMPLE_TEXT.get(args.language, SAMPLE_TEXT["en-IN"])
    voice = args.voice or os.getenv("GNANI_TTS_VOICE", "Kaveri")
    tts = GnaniTTS(api_key, config=GnaniTTSConfig(voice=voice, sample_rate_hz=16_000))
    spoken = tts.synthesize(text, language_code=args.language)
    print("TTS:", json.dumps(spoken.to_dict(), ensure_ascii=False))
    if spoken.status != "ok":
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(spoken.audio)

    heard = GnaniSTT(api_key).transcribe_rest(args.out, language_code=args.language)
    report = heard.to_dict()
    report.pop("events", None)
    print("STT:", json.dumps(report, ensure_ascii=False))
    print(f"input : {text}\nheard : {heard.transcript}\naudio : {args.out}")
    return 0 if heard.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
