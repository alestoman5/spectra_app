"""Calibration test on the sample spectra of the CACH II presentation and the lab protocol.

The images are not part of the repository. Create them locally with::

    python tools/extract_presentation_spectra.py Prezentace.pdf --protocol protokol_ir.pdf

The test is skipped when ``local_data/presentation/labels.json`` does not exist.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "local_data", "presentation")
sys.path.insert(0, os.path.join(ROOT, "tools"))

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "labels.json")),
                                reason="ukázková spektra nejsou lokálně k dispozici")


def test_presentation_accuracy():
    from evaluate import evaluate
    rows = evaluate(DATA)
    wrong = [(r["compound"], r["class"], r["pred"]) for r in rows if not r["ok"]]
    acc = 1 - len(wrong) / len(rows)
    assert acc >= 0.9, wrong
