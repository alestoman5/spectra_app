"""Synthetic spectra for tests: sum of Lorentzian bands."""
import numpy as np

from spectra.references import REFERENCES


def lorentz_spectrum(bands, x=None):
    """bands: iterable of (center, height, fwhm). Returns x (decreasing) and absorbance."""
    if x is None:
        x = np.arange(4000, 399, -1.0)
    y = np.zeros_like(x)
    for c, h, w in bands:
        y += h * (0.5 * w) ** 2 / ((x - c) ** 2 + (0.5 * w) ** 2)
    return x, y


def reference_bands(name, broad_width=300):
    ref = next(r for r in REFERENCES if r.name == name)
    out = []
    for w, s in ref.peaks:
        h = {"s": 1.0, "m": 0.35, "w": 0.12}[s]
        fw = broad_width if (ref.broad_oh and w >= 2900) else 10
        if ref.broad_oh and w >= 2900:
            h = 0.8
        out.append((w, h, fw))
    return out
