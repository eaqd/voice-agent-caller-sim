"""The callers' voices: Kokoro text-to-speech on the CPU, free. Each scenario picks a voice (bf_ = British woman,
bm_ = British man, af_/am_ = American) and a speed."""
import os
import wave
from functools import lru_cache

import numpy as np

SR = 24000  # Kokoro's sample rate
KOKORO_DIR = os.getenv("KOKORO_DIR", os.path.join(os.path.expanduser("~"), ".cache", "kokoro"))


@lru_cache(maxsize=1)
def _kokoro():
    from kokoro_onnx import Kokoro
    return Kokoro(os.path.join(KOKORO_DIR, "kokoro-v1.0.onnx"), os.path.join(KOKORO_DIR, "voices-v1.0.bin"))


def speak(text: str, voice: str = "bf_alice", speed: float = 1.0) -> np.ndarray:
    samples, _ = _kokoro().create(text, voice=voice, speed=speed, lang="en-gb" if voice.startswith("b") else "en-us")
    return np.asarray(samples, dtype=np.float32)


def save_wav(path, samples: np.ndarray, rate: int = SR):
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
