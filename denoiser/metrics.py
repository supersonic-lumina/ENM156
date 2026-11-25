"""
The composite quality measures that speech-enhancement papers on VoiceBank-DEMAND
report next to PESQ: CSIG (signal distortion), CBAK (background intrusion) and
COVL (overall quality), each a predicted 1..5 rating, and SSNR (segmental SNR, dB).

CSIG, CBAK and COVL combine PESQ with three simpler measures, through the
regression formulas of Hu & Loizou, "Evaluation of objective quality measures
for speech enhancement", IEEE TASLP 16(1), 2008:
  LLR   log-likelihood ratio of the two signals' LPC models (lower is better)
  WSS   weighted spectral slope distance over 25 critical bands (lower is better)
  SSNR  segmental SNR, each frame clipped to [-10, 35] dB
Frames, windows and band tables follow Loizou's reference implementation
(composite.m, "Speech Enhancement: Theory and Practice", 2007) as used by the
published results; LLR and WSS average the best 95% of the frames.
"""
import numpy as np

ALPHA = 0.95  # LLR and WSS: fraction of frames kept (the best ones)
SSNR_RANGE = (-10.0, 35.0)

# Critical bands of the WSS measure: centre frequencies and bandwidths (Hz)
CENT_FREQ = np.array([50.0, 120.0, 190.0, 260.0, 330.0, 400.0, 470.0, 540.0, 617.372, 703.378,
                      798.717, 904.128, 1020.38, 1148.30, 1288.72, 1442.54, 1610.70, 1794.16,
                      1993.93, 2211.08, 2446.71, 2701.97, 2978.04, 3276.17, 3597.63])
BANDWIDTH = np.array([70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 77.3724, 86.0056, 95.3398,
                      105.411, 116.256, 127.914, 140.423, 153.823, 168.154, 183.457, 199.776,
                      217.153, 235.631, 255.255, 276.072, 298.126, 321.465, 346.136])
WSS_KMAX, WSS_KLOCMAX = 20.0, 1.0


def frames(x: np.ndarray, sr: int) -> np.ndarray:
    """x cut into 30 ms Hann-windowed frames every 7.5 ms: (n_frames, frame_length)."""
    length = round(30 * sr / 1000)
    hop = length // 4
    n = int(len(x) / hop - length / hop)
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, length + 1) / (length + 1)))
    index = np.arange(n)[:, None] * hop + np.arange(length)
    return x[index] * window


def ssnr(clean: np.ndarray, est: np.ndarray, sr: int) -> float:
    """Segmental SNR (dB): the mean over frames of each frame's SNR, clipped to SSNR_RANGE."""
    c, e = frames(clean, sr), frames(est, sr)
    eps = np.finfo(np.float64).eps
    snr = 10 * np.log10(np.sum(c ** 2, axis=1) / (np.sum((c - e) ** 2, axis=1) + eps) + eps)
    return float(np.mean(np.clip(snr, *SSNR_RANGE)))


def lpc(f: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: autocorrelation r (n, order + 1) and LPC polynomial a (n, order + 1),
    a[:, 0] = 1, by Levinson-Durbin."""
    length = f.shape[1]
    r = np.stack([np.sum(f[:, :length - k] * f[:, k:], axis=1) for k in range(order + 1)], axis=1)
    coef = np.zeros((len(f), order))
    err = r[:, 0].copy()
    silent = err == 0  # an all-zero frame: no prediction (a = 1, 0, 0, ...)
    err[silent] = 1.0
    for i in range(order):
        k = (r[:, i + 1] - np.sum(coef[:, :i] * r[:, i:0:-1], axis=1)) / err
        k[silent] = 0.0
        coef[:, :i] = coef[:, :i] - k[:, None] * coef[:, i - 1::-1] if i else coef[:, :i]
        coef[:, i] = k
        err = (1 - k ** 2) * err
    return r, np.concatenate([np.ones((len(f), 1)), -coef], axis=1)


def llr(clean: np.ndarray, est: np.ndarray, sr: int) -> float:
    """Log-likelihood ratio: how much worse the estimate's LPC model predicts the clean
    signal than the clean signal's own model does; mean of the best ALPHA of frames."""
    order = 10 if sr < 10000 else 16
    r_clean, a_clean = lpc(frames(clean, sr), order)
    _, a_est = lpc(frames(est, sr), order)
    toeplitz = r_clean[:, np.abs(np.arange(order + 1)[:, None] - np.arange(order + 1))]
    num = np.einsum("ni,nij,nj->n", a_est, toeplitz, a_est)
    den = np.einsum("ni,nij,nj->n", a_clean, toeplitz, a_clean)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.log(num / den)
    return best_mean(dist)


def band_energies(f: np.ndarray, sr: int) -> np.ndarray:
    """Per frame, the energy (dB) in each critical band: (n_frames, 25)."""
    n_fft = 2 ** int(np.ceil(np.log2(2 * f.shape[1])))
    half = n_fft // 2
    max_freq = sr / 2
    min_factor = np.exp(-30.0 / (2.0 * 2.303))  # -30 dB
    centre = CENT_FREQ / max_freq * half
    width = BANDWIDTH / max_freq * half
    norm = np.log(BANDWIDTH[0]) - np.log(BANDWIDTH)
    bins = np.arange(half)
    filters = np.exp(-11 * ((bins - np.floor(centre)[:, None]) / width[:, None]) ** 2 + norm[:, None])
    filters *= filters > min_factor
    power = np.abs(np.fft.fft(f, n_fft, axis=1)[:, :half]) ** 2
    return 10 * np.log10(np.maximum(power @ filters.T, 1e-10))


def nearest_peaks(energy: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """For each band below the top one, the energy of the nearest spectral peak: searched
    upwards where the slope rises, downwards where it falls (as composite.m does)."""
    n, bands = slope.shape
    rows = np.arange(n)
    up = np.empty((n, bands + 1), dtype=int)  # first band >= j whose slope doesn't rise
    up[:, bands] = bands
    for j in range(bands - 1, -1, -1):
        up[:, j] = np.where(slope[:, j] > 0, up[:, j + 1], j)
    down = np.empty((n, bands), dtype=int)  # last band <= j whose slope rises, -1 if none
    for j in range(bands):
        down[:, j] = np.where(slope[:, j] > 0, j, down[:, j - 1] if j else -1)
    peaks = np.empty((n, bands))
    for i in range(bands):
        peaks[:, i] = np.where(slope[:, i] > 0, energy[rows, up[:, i] - 1], energy[rows, down[:, i] + 1])
    return peaks


def wss(clean: np.ndarray, est: np.ndarray, sr: int) -> float:
    """Weighted spectral slope distance: the difference of the two signals' band-to-band
    spectral slopes, weighted towards spectral peaks; mean of the best ALPHA of frames."""
    weights, slopes = [], []
    for x in (clean, est):
        energy = band_energies(frames(x, sr), sr)
        slope = energy[:, 1:] - energy[:, :-1]
        peaks = nearest_peaks(energy, slope)
        level = energy[:, :-1]
        top = energy.max(axis=1, keepdims=True)
        weights.append(WSS_KMAX / (WSS_KMAX + top - level) * WSS_KLOCMAX / (WSS_KLOCMAX + peaks - level))
        slopes.append(slope)
    w = (weights[0] + weights[1]) / 2
    return best_mean(np.sum(w * (slopes[0] - slopes[1]) ** 2, axis=1) / np.sum(w, axis=1))


def best_mean(dist: np.ndarray) -> float:
    """Mean of the lowest ALPHA fraction of per-frame distances."""
    return float(np.mean(np.sort(dist)[:round(len(dist) * ALPHA)]))


def composite(clean: np.ndarray, est: np.ndarray, pesq: float, sr: int = 16000) -> dict[str, float]:
    """CSIG, CBAK, COVL (each 1..5) and SSNR (dB) of `est` against `clean`, given their
    wide-band PESQ (nan PESQ gives nan CSIG, CBAK and COVL)."""
    length = min(len(clean), len(est))
    clean, est = np.asarray(clean[:length], np.float64), np.asarray(est[:length], np.float64)
    l, w, s = llr(clean, est, sr), wss(clean, est, sr), ssnr(clean, est, sr)
    rate = lambda v: float(np.clip(v, 1.0, 5.0))  # noqa: E731
    return {
        "csig": rate(3.093 - 1.029 * l + 0.603 * pesq - 0.009 * w),
        "cbak": rate(1.634 + 0.478 * pesq - 0.007 * w + 0.063 * s),
        "covl": rate(1.594 + 0.805 * pesq - 0.512 * l - 0.007 * w),
        "ssnr": s,
    }
