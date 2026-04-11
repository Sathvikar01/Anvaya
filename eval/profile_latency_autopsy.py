"""Profile frozen retrieval and Sarvam STT using additive measurement code."""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import io
import json
import os
import statistics
import time
import wave
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI

from hh_goa_rag.guardrails.retrieval import evidence_sufficiency
from hh_goa_rag.harness import VoiceRAGHarness
from hh_goa_rag.latency import latency_summary
from hh_goa_rag.stt.sarvam import _wav_chunks

AUTOPSY_CSV = Path("results/latency_autopsy.csv")
STT_CSV = Path("results/stt_latency_ablation.csv")
RAW_DIR = Path("cache/latency")
STT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"


class Trace:
    def __init__(self, started_ns: int) -> None:
        self.started_ns = started_ns
        self.events: dict[str, float] = {}

    def __call__(self, name: str, _info: dict[str, Any]) -> None:
        self.events[name] = (time.perf_counter_ns() - self.started_ns) / 1e6

    def duration(self, start: str, end: str) -> float | None:
        if start not in self.events or end not in self.events:
            return None
        return max(0.0, self.events[end] - self.events[start])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(
    component: str,
    stage: str,
    values: list[float | None],
    *,
    configuration: str,
    notes: str = "",
) -> dict[str, Any]:
    metrics = latency_summary(values)
    return {
        "measurement_label": "ABLATION",
        "component": component,
        "stage": stage,
        "configuration": configuration,
        "samples": metrics.pop("n"),
        **metrics,
        "notes": notes,
    }


def profile_retrieval(repeats: int) -> list[dict[str, Any]]:
    cases = _read_jsonl(Path("cache/generation/gold_contexts_top10.jsonl"))
    init_started = time.perf_counter_ns()
    harness = VoiceRAGHarness.from_frozen_artifacts(include_stt=False)
    init_ms = (time.perf_counter_ns() - init_started) / 1e6
    observations: list[dict[str, Any]] = []
    try:
        observation_index = 0
        for repeat in range(repeats):
            for case in cases:
                started = time.perf_counter_ns()
                vectors, _ = harness.embedder.encode_queries([case["question"]])
                embedding_ms = (time.perf_counter_ns() - started) / 1e6
                started = time.perf_counter_ns()
                contexts = harness.retriever.retrieve(vectors[0])
                search_ms = (time.perf_counter_ns() - started) / 1e6
                started = time.perf_counter_ns()
                evidence_sufficiency(contexts)
                guardrail_ms = (time.perf_counter_ns() - started) / 1e6
                observations.append(
                    {
                        "case_id": case["case_id"],
                        "repeat": repeat,
                        "temperature": "cold" if observation_index == 0 else "warm",
                        "embedding_ms": embedding_ms,
                        "search_ms": search_ms,
                        "evidence_guardrail_ms": guardrail_ms,
                    }
                )
                observation_index += 1
    finally:
        harness.close()
    _write_jsonl(RAW_DIR / "retrieval_observations.jsonl", observations)
    _write_jsonl(
        RAW_DIR / "retrieval_startup.jsonl",
        [{"configuration": "cold_process", "startup_ms": init_ms}],
    )
    rows = [
        _summary_row(
            "startup",
            "embedding_model_and_index_initialization",
            [init_ms],
            configuration="cold_process",
            notes="startup-only; excluded from warmed request latency",
        )
    ]
    for temperature in ("cold", "warm"):
        selected = [row for row in observations if row["temperature"] == temperature]
        for field, stage in (
            ("embedding_ms", "query_embedding"),
            ("search_ms", "faiss_search"),
            ("evidence_guardrail_ms", "evidence_guardrail"),
        ):
            rows.append(
                _summary_row(
                    "retrieval",
                    stage,
                    [row[field] for row in selected],
                    configuration=temperature,
                )
            )
    return rows


def _wav_pcm(path: Path) -> tuple[list[tuple[bytes, float]], float]:
    with wave.open(str(path), "rb") as source:
        if (
            source.getframerate() != 16_000
            or source.getnchannels() != 1
            or source.getsampwidth() != 2
        ):
            raise ValueError("STT ablation audio must be 16 kHz mono PCM16 WAV")
        duration_ms = source.getnframes() / source.getframerate() * 1000
    return _wav_chunks(path, 64), duration_ms


def _silence_wav(duration_ms: float = 64) -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(bytes(round(16_000 * duration_ms / 1000) * 2))
    return target.getvalue()


def _rest_once(
    api_key: str, audio_path: Path, client: httpx.Client | None
) -> dict[str, Any]:
    own_client = client is None
    setup_started = time.perf_counter_ns()
    active = client or httpx.Client(timeout=30)
    client_setup_ms = (time.perf_counter_ns() - setup_started) / 1e6
    started = time.perf_counter_ns()
    trace = Trace(started)
    status = "error"
    try:
        with audio_path.open("rb") as audio:
            response = active.post(
                STT_ENDPOINT,
                headers={"api-subscription-key": api_key},
                data={"model": "saaras:v3", "mode": "transcribe", "language_code": "en-IN"},
                files={"file": (audio_path.name, audio, "audio/wav")},
                extensions={"trace": trace},
            )
        response.raise_for_status()
        parse_started = time.perf_counter_ns()
        body = response.json()
        parse_ms = (time.perf_counter_ns() - parse_started) / 1e6
        status = "ok" if str(body.get("transcript") or "").strip() else "error"
        error_code = None if status == "ok" else "empty_transcript"
    except Exception as error:
        parse_ms = None
        error_code = type(error).__name__
    finally:
        total_ms = (time.perf_counter_ns() - started) / 1e6
        if own_client:
            active.close()
    return {
        "configuration": "rest_persistent" if client else "rest_new_connection",
        "status": status,
        "error_code": error_code,
        "connection_setup_ms": trace.duration(
            "connection.connect_tcp.started", "connection.start_tls.complete"
        ),
        "audio_send_ms": trace.duration(
            "http11.send_request_body.started", "http11.send_request_body.complete"
        ),
        "server_processing_ms": trace.duration(
            "http11.send_request_body.complete", "http11.receive_response_headers.complete"
        ),
        "response_parse_ms": parse_ms,
        "client_setup_ms": client_setup_ms,
        "total_ms": total_ms,
        "speech_duration_ms": _wav_pcm(audio_path)[1],
        "eos_to_final_ms": total_ms,
    }


async def _stream_turn(
    socket: Any,
    messages: asyncio.Queue[tuple[int, Any]],
    chunks: list[tuple[bytes, float]],
    *,
    duration_ms: float,
    pace: bool,
    flush: bool,
    append_silence_ms: int,
) -> dict[str, Any]:
    send_started = time.perf_counter_ns()
    target_elapsed_s = 0.0
    for chunk, chunk_duration_ms in chunks:
        await socket.transcribe(
            audio=base64.b64encode(chunk).decode("ascii"),
            encoding="audio/wav",
            sample_rate=16_000,
        )
        if pace:
            target_elapsed_s += chunk_duration_ms / 1000
            actual_elapsed_s = (time.perf_counter_ns() - send_started) / 1e9
            await asyncio.sleep(max(0.0, target_elapsed_s - actual_elapsed_s))
    eos_ns = time.perf_counter_ns()
    silence_chunks = round(append_silence_ms / 64)
    silence = _silence_wav()
    for _ in range(silence_chunks):
        await socket.transcribe(
            audio=base64.b64encode(silence).decode("ascii"),
            encoding="audio/wav",
            sample_rate=16_000,
        )
        if pace:
            await asyncio.sleep(0.064)
    audio_sent_ns = time.perf_counter_ns()
    if flush:
        await socket.flush()
    first_transcript_ns = None
    transcript_events: list[tuple[int, str]] = []
    vad_end_ns = None
    deadline = time.perf_counter() + 8
    idle_timeouts = 0
    while time.perf_counter() < deadline:
        try:
            received_ns, message = await asyncio.wait_for(messages.get(), timeout=0.35)
        except TimeoutError:
            idle_timeouts += 1
            if transcript_events and idle_timeouts >= 2:
                break
            continue
        idle_timeouts = 0
        if isinstance(message, Exception):
            raise message
        message_type = str(getattr(message, "type", ""))
        data = getattr(message, "data", None)
        if message_type == "events":
            signal = f"{getattr(data, 'signal_type', '')} {getattr(data, 'event_type', '')}".lower()
            if "end" in signal and received_ns >= eos_ns:
                vad_end_ns = received_ns
        if message_type != "data":
            continue
        text = str(getattr(data, "transcript", "") or "").strip()
        is_final = getattr(data, "is_final", True)
        if text and first_transcript_ns is None:
            first_transcript_ns = received_ns
        if text and is_final is not False:
            transcript_events.append((received_ns, text))
    if not transcript_events:
        raise TimeoutError("No final transcript before timeout")
    final_ns = transcript_events[-1][0]
    transcript = " ".join(text for _, text in transcript_events)
    return {
        "status": "ok",
        "audio_send_ms": (audio_sent_ns - send_started) / 1e6,
        "speech_duration_ms": duration_ms,
        "vad_detection_ms": ((vad_end_ns - eos_ns) / 1e6 if vad_end_ns else None),
        "eos_to_final_ms": max(0.0, (final_ns - eos_ns) / 1e6),
        "server_processing_ms": max(0.0, (final_ns - audio_sent_ns) / 1e6),
        "time_to_first_transcript_ms": (
            (first_transcript_ns - send_started) / 1e6 if first_transcript_ns else None
        ),
        "transcript_chars": len(transcript),
    }


async def _stream_session(
    api_key: str,
    audio_path: Path,
    *,
    configuration: str,
    repeats: int,
    persistent: bool,
    pace: bool,
    preopen: bool,
    flush: bool,
    append_silence_ms: int = 0,
    vad: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    chunks, duration_ms = _wav_pcm(audio_path)
    observations: list[dict[str, Any]] = []
    sessions = 1 if persistent else repeats
    turns_per_session = repeats if persistent else 1
    for _ in range(sessions):
        client = AsyncSarvamAI(api_subscription_key=api_key, timeout=30)
        connect_started = time.perf_counter_ns()
        parameters: dict[str, Any] = {
            "language_code": "en-IN",
            "model": "saaras:v3",
            "mode": "transcribe",
            "sample_rate": "16000",
            "high_vad_sensitivity": "true",
            "vad_signals": "true",
            "flush_signal": "true" if flush else "false",
            "input_audio_codec": "wav",
        }
        parameters.update(vad or {})
        connect = client.speech_to_text_streaming.connect(**parameters)
        try:
            async with connect as socket:
                connection_ms = (time.perf_counter_ns() - connect_started) / 1e6
                if preopen:
                    await asyncio.sleep(0.2)
                messages: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()

                async def receive(
                    queue: asyncio.Queue[tuple[int, Any]] = messages,
                ) -> None:
                    try:
                        while True:
                            message = await socket.recv()
                            await queue.put((time.perf_counter_ns(), message))
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        await queue.put((time.perf_counter_ns(), error))

                receiver = asyncio.create_task(receive())
                try:
                    for turn in range(turns_per_session):
                        started = time.perf_counter_ns()
                        result = await _stream_turn(
                            socket,
                            messages,
                            chunks,
                            duration_ms=duration_ms,
                            pace=pace,
                            flush=flush,
                            append_silence_ms=append_silence_ms,
                        )
                        result.update(
                            {
                                "configuration": configuration,
                                "connection_setup_ms": connection_ms if turn == 0 else 0.0,
                                "connection_preopened": preopen,
                                "persistent_connection": persistent,
                                "total_ms": (time.perf_counter_ns() - started) / 1e6,
                            }
                        )
                        observations.append(result)
                finally:
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)
        except Exception as error:
            observations.append(
                {
                    "configuration": configuration,
                    "status": "error",
                    "error_code": type(error).__name__,
                    "connection_setup_ms": (time.perf_counter_ns() - connect_started) / 1e6,
                    "speech_duration_ms": duration_ms,
                }
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                maybe = close()
                if asyncio.iscoroutine(maybe):
                    await maybe
    return observations


async def _all_streaming(
    api_key: str, audio_path: Path, repeats: int
) -> list[dict[str, Any]]:
    configurations = [
        {
            "configuration": "websocket_buffered_preopened",
            "persistent": False,
            "pace": False,
            "preopen": True,
            "flush": True,
        },
        {
            "configuration": "websocket_live_stream_preopened",
            "persistent": False,
            "pace": True,
            "preopen": True,
            "flush": True,
        },
        {
            "configuration": "websocket_live_stream_persistent",
            "persistent": True,
            "pace": True,
            "preopen": True,
            "flush": True,
        },
        {
            "configuration": "vad_current",
            "persistent": False,
            "pace": True,
            "preopen": True,
            "flush": False,
            "append_silence_ms": 1600,
        },
        {
            "configuration": "vad_aggressive",
            "persistent": False,
            "pace": True,
            "preopen": True,
            "flush": False,
            "append_silence_ms": 1200,
            "vad": {
                "negative_speech_threshold": "0.50",
                "negative_frames_count": "12",
                "negative_frames_window": "18",
            },
        },
        {
            "configuration": "vad_minimum_practical",
            "persistent": False,
            "pace": True,
            "preopen": True,
            "flush": False,
            "append_silence_ms": 800,
            "vad": {
                "negative_speech_threshold": "0.55",
                "negative_frames_count": "6",
                "negative_frames_window": "10",
            },
        },
    ]
    observations: list[dict[str, Any]] = []
    for configuration in configurations:
        observations.extend(
            await _stream_session(
                api_key,
                audio_path,
                repeats=repeats,
                **configuration,
            )
        )
    return observations


def profile_stt(
    audio_path: Path, repeats: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    load_dotenv(".env", override=False)
    api_key = os.getenv("SARVAM_API_KEY", "")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is required")
    observations = [_rest_once(api_key, audio_path, None) for _ in range(repeats)]
    with httpx.Client(timeout=30) as client:
        persistent = [_rest_once(api_key, audio_path, client) for _ in range(repeats)]
        for index, observation in enumerate(persistent):
            observation["configuration"] = (
                "rest_persistent_cold" if index == 0 else "rest_persistent_warm"
            )
        observations.extend(persistent)
    observations.extend(asyncio.run(_all_streaming(api_key, audio_path, repeats)))
    _write_jsonl(RAW_DIR / "stt_observations.jsonl", observations)
    rows: list[dict[str, Any]] = []
    for configuration in dict.fromkeys(row["configuration"] for row in observations):
        selected = [row for row in observations if row["configuration"] == configuration]
        successes = [row for row in selected if row["status"] == "ok"]
        result: dict[str, Any] = {
            "measurement_label": "ABLATION (pre-existing real-human smoke clip)",
            "configuration": configuration,
            "samples": len(selected),
            "successes": len(successes),
            "failures": len(selected) - len(successes),
            "failure_rate": 1 - len(successes) / len(selected),
            "speech_duration_ms": _mean(successes, "speech_duration_ms"),
            "speech_duration_excluded_from_avoidable_latency": True,
        }
        for field, prefix in (
            ("client_setup_ms", "client_setup"),
            ("connection_setup_ms", "connection"),
            ("audio_send_ms", "audio_send"),
            ("vad_detection_ms", "vad_detection"),
            ("server_processing_ms", "server_processing"),
            ("eos_to_final_ms", "eos_to_final"),
            ("total_ms", "wall_clock"),
        ):
            summary = latency_summary(row.get(field) for row in successes)
            result.update({f"{prefix}_{key}": value for key, value in summary.items()})
        rows.append(result)
    _write_csv(STT_CSV, rows)
    autopsy = []
    for row in rows:
        configuration = str(row["configuration"])
        for prefix, stage in (
            ("connection", "connection_setup"),
            ("audio_send", "audio_upload_or_send"),
            ("vad_detection", "vad_endpoint_detection"),
            ("server_processing", "server_processing"),
            ("eos_to_final", "eos_to_final_transcript"),
        ):
            autopsy.append(
                {
                    "measurement_label": row["measurement_label"],
                    "component": "stt",
                    "stage": stage,
                    "configuration": configuration,
                    "samples": row.get(f"{prefix}_n"),
                    "p50_ms": row.get(f"{prefix}_p50_ms"),
                    "p70_ms": row.get(f"{prefix}_p70_ms"),
                    "p95_ms": row.get(f"{prefix}_p95_ms"),
                    "p100_ms": row.get(f"{prefix}_p100_ms"),
                    "mean_ms": row.get(f"{prefix}_mean_ms"),
                }
            )
    return rows, autopsy


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("retrieval", "stt", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--audio", type=Path, default=Path("cache/stt_smoke/audio3_en_16k.wav")
    )
    args = parser.parse_args()
    autopsy: list[dict[str, Any]] = []
    if args.phase in {"retrieval", "all"}:
        autopsy.extend(profile_retrieval(args.repeats))
    if args.phase in {"stt", "all"}:
        _, stt_autopsy = profile_stt(args.audio, args.repeats)
        autopsy.extend(stt_autopsy)
    _write_csv(AUTOPSY_CSV, autopsy)


if __name__ == "__main__":
    main()
