#!/usr/bin/env python3
"""
OpenAI-compatible Speech-to-Text server using Qwen3-ASR-1.7B.

Architecture:
    Client → POST /v1/audio/transcriptions (port 11114)

Endpoints:
    POST /v1/audio/transcriptions   — OpenAI-compatible transcription
    GET  /health                    — liveness probe

Usage:
    python3 qwen_asr_server.py
    python3 qwen_asr_server.py --port 11114 --model Qwen/Qwen3-ASR-1.7B

Supported audio formats: wav, mp3, ogg, flac, m4a, webm
"""

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

try:
    import av
except ImportError:
    av = None

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "asr_server.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("qwen_asr_server")

DEFAULT_MODEL = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
DEFAULT_PORT = int(os.environ.get("QWEN_ASR_PORT", "11114"))
DEFAULT_HOST = os.environ.get("QWEN_ASR_HOST", "0.0.0.0")
DEFAULT_LANGUAGE = os.environ.get("QWEN_ASR_LANGUAGE", None)  # None = auto-detect

_model = None  # lazy-loaded Qwen3ASRModel


def _pcm_to_float32(audio: np.ndarray) -> np.ndarray:
    """Normalize decoded PCM data into float32 in the [-1, 1] range."""
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32)
    if np.issubdtype(audio.dtype, np.signedinteger):
        info = np.iinfo(audio.dtype)
        peak = float(max(abs(info.min), info.max))
        return np.clip(audio.astype(np.float32) / peak, -1.0, 1.0)
    if np.issubdtype(audio.dtype, np.unsignedinteger):
        info = np.iinfo(audio.dtype)
        midpoint = (info.max + 1) / 2.0
        return np.clip((audio.astype(np.float32) - midpoint) / midpoint, -1.0, 1.0)
    raise TypeError(f"Unsupported decoded audio dtype: {audio.dtype}")


def _decode_audio_with_soundfile(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as buffer:
        audio, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _decode_audio_with_av(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    if av is None:
        raise RuntimeError("PyAV is not installed")

    with av.open(io.BytesIO(audio_bytes), mode="r") as container:
        stream = next((candidate for candidate in container.streams if candidate.type == "audio"), None)
        if stream is None:
            raise ValueError("No audio stream found in upload")

        chunks: list[np.ndarray] = []
        sample_rate: int | None = None
        for frame in container.decode(stream):
            sample_rate = int(
                frame.sample_rate
                or getattr(stream.codec_context, "sample_rate", 0)
                or getattr(stream, "rate", 0)
            )
            chunk = np.asarray(frame.to_ndarray())
            if chunk.size == 0:
                continue
            chunks.append(chunk)

    if not chunks or not sample_rate:
        raise ValueError("Decoded audio contained no PCM frames")

    concat_axis = 0 if chunks[0].ndim == 1 else 1
    audio = np.concatenate(chunks, axis=concat_axis)
    return _pcm_to_float32(audio), sample_rate


def _decode_audio_bytes(audio_bytes: bytes, filename: str) -> tuple[np.ndarray, int]:
    try:
        return _decode_audio_with_soundfile(audio_bytes)
    except Exception as soundfile_error:
        if av is None:
            raise RuntimeError(
                f"Unsupported or unreadable audio format for {filename!r}; install PyAV for WebM/Opus uploads."
            ) from soundfile_error

        try:
            audio, sample_rate = _decode_audio_with_av(audio_bytes)
            logger.info("Decoded %s with PyAV fallback at %d Hz", filename, sample_rate)
            return audio, sample_rate
        except Exception as av_error:
            raise RuntimeError(
                f"Unsupported or unreadable audio format for {filename!r}."
            ) from av_error


def _load_model(model_id: str):
    """Load Qwen3ASRModel once and cache it."""
    global _model
    if _model is not None:
        return _model

    logger.info("Loading ASR model: %s", model_id)
    t0 = time.time()

    from qwen_asr import Qwen3ASRModel

    _model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_inference_batch_size=8,
        max_new_tokens=512,
    )

    logger.info("ASR model loaded in %.1fs", time.time() - t0)
    return _model


def _transcribe(audio_bytes: bytes, filename: str, language: str | None) -> dict:
    """Run ASR on raw audio bytes. Returns dict with text and metadata."""
    model = _load_model(DEFAULT_MODEL)
    audio, sample_rate = _decode_audio_bytes(audio_bytes, filename)

    t0 = time.time()
    results = model.transcribe(audio=(audio, sample_rate), language=language)
    elapsed = time.time() - t0

    result = results[0]
    text = result.text.strip()
    detected_language = result.language or "unknown"

    logger.info(
        "Transcribed %s → %d chars in %.2fs  (lang=%s)",
        filename, len(text), elapsed, detected_language,
    )
    return {"text": text, "language": detected_language, "elapsed": elapsed}


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Qwen3-ASR Server", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok", "model": DEFAULT_MODEL}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=DEFAULT_MODEL),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: str = Form(default="segment"),
):
    """
    OpenAI-compatible transcription endpoint.
    https://platform.openai.com/docs/api-reference/audio/createTranscription
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    lang = language or DEFAULT_LANGUAGE

    try:
        result = _transcribe(audio_bytes, file.filename, lang)
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if response_format == "text":
        return result["text"]

    if response_format == "verbose_json":
        return JSONResponse({
            "task": "transcribe",
            "language": result["language"],
            "duration": result["elapsed"],
            "text": result["text"],
        })

    # Default: json (OpenAI-compatible minimal response)
    return JSONResponse({"text": result["text"]})


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    global DEFAULT_MODEL
    parser = argparse.ArgumentParser(description="Qwen3-ASR OpenAI-compatible server")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--preload", action="store_true",
                        help="Load model at startup instead of on first request")
    args = parser.parse_args()

    DEFAULT_MODEL = args.model

    if args.preload:
        _load_model(args.model)

    logger.info("🎙️  Qwen3-ASR server starting on http://%s:%d", args.host, args.port)
    logger.info("   Model: %s", args.model)
    logger.info("   Endpoint: POST /v1/audio/transcriptions")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
