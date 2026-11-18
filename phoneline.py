"""What a phone line does to a voice, in plain numpy: 8 kHz sampling, only 300-3400 Hz gets through, and background
noise. Real callers sound like this, not like a studio recording, so a voice agent should be tested this way."""
import numpy as np


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(x) == 0:
        return x.astype(np.float32)
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


# No numpy.fft on purpose: Windows Smart App Control blocked numpy 2.5's FFT library on the PC this was built on.
# A plain filter (a convolution) does the same job and runs anywhere.

def _lowpass(cutoff: float, rate: int, taps: int = 129) -> np.ndarray:
    """A windowed-sinc filter that lets through everything below `cutoff` Hz."""
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff / rate * n) * np.hamming(taps)
    return h / h.sum()


def band_limit(x: np.ndarray, rate: int, low: float = 300, high: float = 3400) -> np.ndarray:
    """Keep only the frequencies a phone line carries (300-3400 Hz)."""
    if len(x) == 0:
        return x
    band = _lowpass(high, rate) - _lowpass(low, rate)
    return np.convolve(x, band, mode="same").astype(np.float32)


def pink_noise(n: int, seed: int = 0) -> np.ndarray:
    """Noise with more low rumble than hiss, like a busy room or a car: white noise smoothed over windows of
    1, 2, 4 ... 128 samples and added up gives roughly equal energy per octave (a 1/f, "pink" spectrum)."""
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for k in range(8):
        width = 2 ** k
        white = np.concatenate([np.zeros(1), np.cumsum(rng.standard_normal(n + width))])
        noise += (white[width:width + n] - white[:n]) / np.sqrt(width)  # a moving average of `width` samples
    return (noise / (np.std(noise) + 1e-9)).astype(np.float32)


def rms(x: np.ndarray) -> float:
    loud = x[np.abs(x) > 0.02]  # the speech, not the pauses between words
    return float(np.sqrt(np.mean((loud if len(loud) else x) ** 2))) if len(x) else 0.0


def add_noise(x: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Background noise snr_db below the voice (12 dB is a noisy street; 30 dB is nearly quiet)."""
    if len(x) == 0:
        return x
    noise = pink_noise(len(x), seed) * rms(x) / (10 ** (snr_db / 20))
    out = x + noise
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.95 if peak > 0.95 else out).astype(np.float32)


def degrade(x: np.ndarray, rate: int, condition: dict, seed: int = 0) -> tuple[np.ndarray, int]:
    """Apply one condition from scenarios.yaml. Returns (samples, sample_rate)."""
    out_rate = condition.get("rate", rate)
    if out_rate < rate and len(x):  # filter BEFORE dropping samples, or high sounds fold down into the voice band
        x = np.convolve(x, _lowpass(min(3400, 0.45 * out_rate), rate), mode="same").astype(np.float32)
    y = resample(x, rate, out_rate)
    if "rate" in condition:
        y = band_limit(y, out_rate)
    if "noise_db" in condition:
        y = add_noise(y, condition["noise_db"], seed)
    return y, out_rate
