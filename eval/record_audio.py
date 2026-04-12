"""Interactively record real human evaluation speech as 16 kHz mono PCM WAV files."""

from __future__ import annotations

import argparse
import json
import os
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
VARIANT_INSTRUCTIONS = {
    "normal": "Speak naturally at your usual pace.",
    "fast": "Speak clearly but faster than your usual conversational pace.",
    "pauses": "Use one or two natural pauses within the question.",
    "mild_noise": "Record with realistic mild background noise; keep speech intelligible.",
    "accent": "Use your natural regional accent; do not imitate another speaker.",
    "difficult_wording": "Read the difficult wording naturally without simplifying it.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def detect_speech_end_ms(samples: np.ndarray, threshold_dbfs: float = -42.0) -> float | None:
    """Estimate the final voiced frame for latency accounting; never alters the recording."""
    mono = np.asarray(samples, dtype=np.float64).reshape(-1)
    frame_samples = round(SAMPLE_RATE * 0.02)
    last_voiced: int | None = None
    for start in range(0, len(mono), frame_samples):
        frame = mono[start : start + frame_samples]
        if not len(frame):
            continue
        rms = float(np.sqrt(np.mean(np.square(frame))))
        dbfs = 20 * np.log10(max(rms, 1.0) / 32768.0)
        if dbfs >= threshold_dbfs:
            last_voiced = min(start + len(frame), len(mono))
    return last_voiced / SAMPLE_RATE * 1000 if last_voiced is not None else None


def write_wav_atomic(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.wav")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(np.dtype(DTYPE).itemsize)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(np.asarray(samples, dtype=DTYPE).tobytes())
    os.replace(temporary, path)


def record_sample(
    row: dict[str, Any],
    *,
    duration_s: float,
    device: str | int | None,
    speaker_id: str,
    accent: str | None,
    noise: str | None,
) -> dict[str, Any]:
    variant = str(row.get("variant", "normal"))
    if variant not in VARIANT_INSTRUCTIONS:
        raise ValueError(f"Unknown recording variant: {variant}")
    print(f"\nSample: {row['sample_id']} ({row['category']}, {variant})")
    print(f"Instruction: {VARIANT_INSTRUCTIONS[variant]}")
    print(f"Read exactly: {row['reference_text']}")
    input("Press Enter when ready. A three-second countdown will begin...")
    for count in (3, 2, 1):
        print(count, flush=True)
        time.sleep(1)
    print(f"RECORDING for {duration_s:g} seconds", flush=True)
    frames = round(duration_s * SAMPLE_RATE)
    samples = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        device=device,
        blocking=True,
    )
    print("Recording complete.")
    speech_end_ms = detect_speech_end_ms(samples)
    if speech_end_ms is None:
        raise RuntimeError("No speech detected above -42 dBFS; manifest was not updated")
    peak = int(np.max(np.abs(np.asarray(samples, dtype=np.int32))))
    if peak >= 32767:
        print("Warning: clipping detected; consider recording this sample again.")
    output = Path(str(row["audio_path"]))
    write_wav_atomic(output, samples)
    updated = dict(row)
    updated.update(
        {
            "status": "ready",
            "speaker_id": speaker_id,
            "accent": accent,
            "noise": noise,
            "sample_rate_hz": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_width_bytes": 2,
            "duration_ms": frames / SAMPLE_RATE * 1000,
            "speech_end_ms": speech_end_ms,
            "peak_pcm16": peak,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("eval/stt_manifest.jsonl"))
    parser.add_argument("--sample-id", action="append", help="Record only these planned samples")
    parser.add_argument("--all-pending", action="store_true")
    parser.add_argument("--speaker-id")
    parser.add_argument("--accent")
    parser.add_argument("--noise")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--device", help="sounddevice input index or exact device name")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    if not args.speaker_id:
        parser.error("--speaker-id is required for recording")
    if not args.sample_id and not args.all_pending:
        parser.error("choose --sample-id (repeatable) or --all-pending")
    if not 1 <= args.duration_s <= 30:
        parser.error("--duration-s must be between 1 and 30 seconds")
    device: str | int | None = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    rows = read_jsonl(args.manifest)
    wanted = set(args.sample_id or [])
    selected = [
        row
        for row in rows
        if (args.all_pending and row.get("status") == "pending")
        or (wanted and row.get("sample_id") in wanted)
    ]
    if not selected:
        raise SystemExit("No matching pending samples")
    indexes = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    for row in selected:
        path = Path(str(row["audio_path"]))
        if (row.get("status") == "ready" or path.exists()) and not args.overwrite:
            print(f"Skipping existing sample {row['sample_id']}; use --overwrite to replace")
            continue
        updated = record_sample(
            row,
            duration_s=args.duration_s,
            device=device,
            speaker_id=args.speaker_id,
            accent=args.accent or row.get("accent"),
            noise=args.noise or row.get("noise"),
        )
        rows[indexes[str(row["sample_id"])]] = updated
        write_jsonl_atomic(args.manifest, rows)
        print(f"Saved {updated['audio_path']} and updated the manifest atomically.")


if __name__ == "__main__":
    main()
