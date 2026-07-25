from __future__ import annotations

import wave
from pathlib import Path

DISCORD_SAMPLE_RATE = 48_000
DISCORD_CHANNELS = 2
DISCORD_SAMPLE_WIDTH = 2  # signed 16-bit PCM
BYTES_PER_SECOND = DISCORD_SAMPLE_RATE * DISCORD_CHANNELS * DISCORD_SAMPLE_WIDTH


def pcm_duration_seconds(pcm: bytes) -> float:
    return len(pcm) / BYTES_PER_SECOND if pcm else 0.0


def trim_pcm(pcm: bytes, max_seconds: int) -> bytes:
    max_bytes = int(BYTES_PER_SECOND * max_seconds)
    if len(pcm) <= max_bytes:
        return pcm
    return pcm[-max_bytes:]


def write_pcm_wav(path: Path, pcm: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(DISCORD_CHANNELS)
        wav.setsampwidth(DISCORD_SAMPLE_WIDTH)
        wav.setframerate(DISCORD_SAMPLE_RATE)
        wav.writeframes(pcm)
    return path
