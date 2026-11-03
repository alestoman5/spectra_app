"""High level helpers joining digitisation, signal processing and interpretation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import digitize as dg
from .interpret import Interpretation, interpret
from .signal import Peak, Spectrum, find_bands, guess_mode, preprocess, resample, to_absorbance


@dataclass
class Settings:
    y_mode: str = "auto"          # auto / T / A
    color: str = "auto"           # auto / dark / color
    left_value: float = 4000.0    # wavenumber at the left end of the x-axis
    tick_step: float = 500.0      # wavenumber difference between labelled ticks
    right_value: float | None = None   # if set, calibrate from frame edges instead of labels
    smooth_window: int = 9
    prominence: float = 0.03
    baseline: bool = True


@dataclass
class Result:
    raw: Spectrum            # absorbance-like, before baseline correction
    processed: Spectrum
    peaks: list[Peak]
    interpretation: Interpretation
    y_mode: str
    info: str = ""
    area: dg.PlotArea | None = None
    trace: dg.Trace | None = None
    plot_img: np.ndarray | None = None


def analyse_xy(x: np.ndarray, y: np.ndarray, settings: Settings, info: str = "") -> Result:
    mode = settings.y_mode if settings.y_mode != "auto" else guess_mode(y)
    yn = (y - np.nanmin(y)) / max(np.nanmax(y) - np.nanmin(y), 1e-12)
    spec = resample(x, to_absorbance(yn, mode), 1.0)
    return _finish(spec, settings, mode, info)


def _finish(spec: Spectrum, settings: Settings, mode: str, info: str,
            x_range: tuple[float, float] | None = None, **kw) -> Result:
    proc = preprocess(spec, settings.smooth_window, settings.baseline)
    peaks = find_bands(proc, settings.prominence)
    interp = interpret(peaks, x_range or (float(spec.x.min()), float(spec.x.max())))
    return Result(spec, proc, peaks, interp, mode, info, **kw)


def analyse_image(rgb: np.ndarray, settings: Settings, area: dg.PlotArea | None = None,
                  calibration: dg.XCalibration | None = None) -> Result:
    if area is None:
        area = dg.detect_plot_area(rgb)
    plot = dg.rectify(rgb, area)
    trace = dg.extract_curve(plot, settings.color)
    if calibration is None:
        calibration, info = dg.auto_x_calibration(
            rgb, area, settings.left_value, settings.tick_step, settings.right_value)
    else:
        info = "ruční kalibrace osy x"
    wn = calibration(trace.px)
    mid = 1.0 - 0.5 * (trace.top + trace.bottom) / (trace.height - 1)
    mode = settings.y_mode if settings.y_mode != "auto" else guess_mode(mid)
    y = trace.values(mode)
    spec = resample(wn, to_absorbance(y, mode) if mode == "T" else y, 1.0)
    info = f"Osa x: {info}. Osa y: {'transmitance (%T)' if mode == 'T' else 'absorbance'}" + \
           (" – určeno automaticky." if settings.y_mode == "auto" else ".")
    coverage = len(trace.px) / trace.width
    if coverage < 0.85:
        info += (f" POZOR: křivka byla nalezena jen v {coverage * 100:.0f} % šířky grafu – "
                 "zkontrolujte modrou stopu v obrázku, případně upravte rohy nebo barvu křivky.")
    ends = sorted(float(v) for v in calibration(np.array([0, trace.width - 1])))
    return _finish(spec, settings, mode, info, x_range=(ends[0], ends[1]),
                   area=area, trace=trace, plot_img=plot)


def load_xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column text/CSV spectrum (wavenumber, intensity)."""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.replace(";", " ").replace(",", " ").replace("\\t", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if len(rows) < 10:
        raise ValueError("Soubor neobsahuje dvousloupcová číselná data (vlnočet, intenzita).")
    a = np.array(rows)
    return a[:, 0], a[:, 1]
