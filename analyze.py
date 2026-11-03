#!/usr/bin/env python3
"""Command line interface: analyse an IR spectrum image (or x,y data file) and print a report.

    python analyze.py spektrum.jpg [--mode T|A|auto] [--left 4000] [--step 500]
"""
import argparse
import os
import sys

from spectra.digitize import load_image
from spectra.interpret import report_text
from spectra.pipeline import Settings, analyse_image, analyse_xy, load_xy


def main(argv=None):
    ap = argparse.ArgumentParser(description="Interpretace IR spektra z obrázku nebo dat.")
    ap.add_argument("file")
    ap.add_argument("--mode", default="auto", choices=["auto", "T", "A"],
                    help="osa y: transmitance (T), absorbance (A) nebo auto")
    ap.add_argument("--color", default="auto", choices=["auto", "dark", "color", "blue"])
    ap.add_argument("--left", type=float, default=4000, help="vlnočet na levém konci osy x")
    ap.add_argument("--step", type=float, default=500, help="krok mezi popisky osy x")
    ap.add_argument("--right", type=float, default=None,
                    help="vlnočet na pravém konci osy x (vypne kalibraci z popisků)")
    ap.add_argument("--prominence", type=float, default=0.03)
    ap.add_argument("--smooth", type=int, default=9)
    a = ap.parse_args(argv)
    st = Settings(y_mode=a.mode, color=a.color, left_value=a.left, tick_step=a.step,
                  right_value=a.right, prominence=a.prominence, smooth_window=a.smooth)
    ext = os.path.splitext(a.file)[1].lower()
    if ext in (".csv", ".txt", ".dat", ".xy", ".dpt"):
        x, y = load_xy(a.file)
        res = analyse_xy(x, y, st, "data ze souboru")
    else:
        res = analyse_image(load_image(a.file), st)
    print(res.info)
    print()
    print(report_text(res.interpretation))


if __name__ == "__main__":
    sys.exit(main())
