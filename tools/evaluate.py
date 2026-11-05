#!/usr/bin/env python3
"""Evaluate the interpretation on a labelled set of spectrum images.

    python tools/evaluate.py [local_data/presentation] [-v]

The folder must contain ``labels.json`` written by ``extract_presentation_spectra.py``.
Prints the expected vs. predicted main class for every spectrum and the overall accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectra.digitize import load_image  # noqa: E402
from spectra.interpret import report_text  # noqa: E402
from spectra.pipeline import Settings, analyse_image  # noqa: E402

# classes that are acceptable alternatives (e.g. phenol is also an alcohol)
EQUIVALENT = {("phenol", "alcohol"), ("alcohol", "phenol")}


def evaluate(folder: str, verbose: bool = False) -> list[dict]:
    with open(os.path.join(folder, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    rows = []
    for lab in labels:
        st = Settings()
        for k in ("left_value", "right_value", "tick_step", "y_mode"):
            if k in lab:
                setattr(st, k, lab[k])
        try:
            res = analyse_image(load_image(os.path.join(folder, lab["file"])), st)
            pred, skel, summary = res.interpretation.main_class, res.interpretation.skeleton, \
                res.interpretation.summary
        except Exception as e:  # noqa: BLE001
            res, pred, skel, summary = None, "error", "", str(e)
        ok = pred == lab["class"] or (pred, lab["class"]) in EQUIVALENT
        skel_ok = skel == lab["skeleton"] or (lab["skeleton"] == "aliphatic" and skel == "")
        rows.append({**lab, "pred": pred, "pred_skeleton": skel, "ok": ok, "skeleton_ok": skel_ok,
                     "summary": summary})
        if verbose and res is not None:
            print("=" * 80, "\n", lab["file"], "\n", res.info)
            print(report_text(res.interpretation))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="local_data/presentation")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    rows = evaluate(a.folder, a.verbose)
    for r in rows:
        mark = "OK " if r["ok"] else "XX "
        print(f"{mark}{r['compound']:24s} očekáváno {r['class']:9s} → {r['pred']:9s} "
              f"skelet {r['pred_skeleton'] or '-':10s}{'' if r['skeleton_ok'] else ' (!)'}")
    n = len(rows)
    acc = sum(r["ok"] for r in rows) / n if n else 0
    sk = sum(r["skeleton_ok"] for r in rows) / n if n else 0
    print(f"\nTřída správně: {sum(r['ok'] for r in rows)}/{n} ({acc * 100:.0f} %), "
          f"skelet správně: {sum(r['skeleton_ok'] for r in rows)}/{n} ({sk * 100:.0f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
