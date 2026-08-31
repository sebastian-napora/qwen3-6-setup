import io
import wave
from types import SimpleNamespace

import numpy as np

import qwen_asr_server


def _wav_bytes(sample_rate: int = 8000, frame_count: int = 80) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.zeros(frame_count, dtype=np.int16).tobytes())
    return payload.getvalue()


def test_decode_audio_bytes_reads_wav_from_memory():
    audio, sample_rate = qwen_asr_server._decode_audio_bytes(_wav_bytes(), "clip.wav")

    assert sample_rate == 8000
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.shape == (80,)


def test_decode_audio_bytes_falls_back_to_av(monkeypatch):
    expected_audio = np.zeros((1, 16), dtype=np.float32)

    def fail_soundfile(_: bytes) -> tuple[np.ndarray, int]:
        raise RuntimeError("soundfile failed")

    monkeypatch.setattr(qwen_asr_server, "_decode_audio_with_soundfile", fail_soundfile)
    monkeypatch.setattr(qwen_asr_server, "av", object())
    monkeypatch.setattr(
        qwen_asr_server,
        "_decode_audio_with_av",
        lambda _: (expected_audio, 16000),
    )

    audio, sample_rate = qwen_asr_server._decode_audio_bytes(b"webm-bytes", "recording.webm")

    assert sample_rate == 16000
    np.testing.assert_array_equal(audio, expected_audio)


def test_transcribe_uses_decoded_audio_tuple(monkeypatch):
    expected_audio = np.zeros(32, dtype=np.float32)
    captured: dict[str, object] = {}

    class FakeModel:
        def transcribe(self, *, audio, language):
            captured["audio"] = audio
            captured["language"] = language
            return [SimpleNamespace(text=" hello ", language="pl")]

    monkeypatch.setattr(qwen_asr_server, "_load_model", lambda _: FakeModel())
    monkeypatch.setattr(
        qwen_asr_server,
        "_decode_audio_bytes",
        lambda _audio_bytes, _filename: (expected_audio, 16000),
    )

    result = qwen_asr_server._transcribe(b"blob", "recording.webm", "Polish")

    assert captured["language"] == "Polish"
    assert isinstance(captured["audio"], tuple)
    np.testing.assert_array_equal(captured["audio"][0], expected_audio)
    assert captured["audio"][1] == 16000
    assert result["text"] == "hello"
    assert result["language"] == "pl"
