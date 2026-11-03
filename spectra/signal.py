"""Signal processing of a 1-D IR spectrum: resampling, smoothing, baseline, peaks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks, peak_widths, savgol_filter

from .ir_table import ARTEFACTS


@dataclass
class Peak:
    wavenumber: float
    height: float          # normalised absorbance 0..1
    prominence: float
    width: float           # FWHM in cm-1
    intensity: str         # s / m / w
    broad: bool
    artefact: str = ""

    def label(self) -> str:
        return f"{self.wavenumber:.0f}"


@dataclass
class Spectrum:
    """Spectrum on a uniform, decreasing wavenumber grid, signal = absorbance-like (peaks up)."""
    x: np.ndarray
    y: np.ndarray

    @property
    def step(self) -> float:
        return abs(self.x[1] - self.x[0])


def to_absorbance(y: np.ndarray, mode: str) -> np.ndarray:
    """Convert raw normalised curve to absorbance-like signal (peaks pointing up).

    ``mode`` is ``"T"`` (transmittance, peaks point down), ``"A"`` (absorbance) or ``"auto"``.
    """
    y = np.asarray(y, float)
    if mode == "auto":
        mode = guess_mode(y)
    if mode == "T":
        t = (y - np.nanmin(y)) / max(np.nanmax(y) - np.nanmin(y), 1e-12)
        t = np.clip(t, 1e-3, 1.0)
        return -np.log10(0.02 + 0.98 * t)   # -log T, with small floor for saturated bands
    return y


def guess_mode(y: np.ndarray) -> str:
    """Transmittance spectra have their baseline near the top, absorbance near the bottom."""
    y = np.asarray(y, float)
    lo, hi = np.nanpercentile(y, 2), np.nanpercentile(y, 98)
    med = np.nanmedian(y)
    rel = (med - lo) / max(hi - lo, 1e-12)
    return "T" if rel > 0.5 else "A"


def resample(x: np.ndarray, y: np.ndarray, step: float = 1.0) -> Spectrum:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    # remove duplicates
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    grid = np.arange(np.ceil(x.min()), np.floor(x.max()) + step / 2, step)
    yi = np.interp(grid, x, y)
    return Spectrum(grid[::-1].copy(), yi[::-1].copy())


def smooth(y: np.ndarray, window: int = 9, order: int = 3) -> np.ndarray:
    window = int(window)
    if window < 5:
        return y.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) - (1 - len(y) % 2))
    return savgol_filter(y, window, min(order, window - 2))


def als_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers & Boelens, 2005)."""
    n = len(y)
    d = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    dd = lam * d.dot(d.transpose())
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        wm = sparse.spdiags(w, 0, n, n)
        z = spsolve(sparse.csc_matrix(wm + dd), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z)


def preprocess(spec: Spectrum, smooth_window: int = 9, baseline: bool = True) -> Spectrum:
    y = smooth(spec.y, smooth_window)
    if baseline:
        y = y - als_baseline(y, lam=1e7 * (spec.step ** 2) if spec.step < 1 else 1e7)
    # clip small negatives and normalise; ignore the CO2 artefact for the maximum
    mask = np.ones_like(y, bool)
    for lo, hi, _ in ARTEFACTS:
        mask &= ~((spec.x >= lo) & (spec.x <= hi))
    ymax = np.max(y[mask]) if mask.any() else np.max(y)
    y = np.clip(y, 0, None) / max(ymax, 1e-12)
    return Spectrum(spec.x, y)


def intensity_class(h: float) -> str:
    if h >= 0.55:
        return "s"
    if h >= 0.2:
        return "m"
    return "w"


def find_bands(spec: Spectrum, prominence: float = 0.03, min_distance_cm: float = 8.0,
               broad_fwhm: float = 120.0) -> list[Peak]:
    """Find absorption maxima in a preprocessed (normalised, baseline corrected) spectrum."""
    y = spec.y
    step = spec.step
    distance = max(1, int(round(min_distance_cm / step)))
    idx, props = find_peaks(y, prominence=prominence, distance=distance)
    if len(idx) == 0:
        return []
    widths = peak_widths(y, idx, rel_height=0.5)[0] * step
    peaks: list[Peak] = []
    for i, prom, w in zip(idx, props["prominences"], widths):
        wn = float(spec.x[i])
        h = float(y[i])
        art = ""
        for lo, hi, text in ARTEFACTS:
            if lo <= wn <= hi:
                art = text
        broad = bool(w >= broad_fwhm or (wn > 2400 and w >= 0.6 * broad_fwhm and h > 0.1))
        peaks.append(Peak(wn, h, float(prom), float(w), intensity_class(h), broad, art))
    peaks.sort(key=lambda p: -p.wavenumber)
    return peaks
