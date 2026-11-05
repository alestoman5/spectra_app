#!/usr/bin/env python3
"""Extract the labelled sample spectra from the CACH II presentation "Interpretace
infračervených spekter" (and optionally the LACH I IR protocol photos) into a local folder.

The images are NOT part of the repository – only this table of where they are and what they
show. Usage::

    python tools/extract_presentation_spectra.py Prezentace.pdf [--protocol protokol_ir.pdf]
        [--out local_data/presentation]

Requires ``pdfimages`` (poppler-utils).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

# (page, image number from `pdfimages -list`, crop (y0, y1) as fraction of height or None,
#  file name, compound, expected main class, expected skeleton[, settings overrides])
# main class ids: alkane, alkene, alkyne, arene, alcohol, phenol, aldehyde, ketone, ester,
#                 acid, ether, nitrile, amine, nitro
PRESENTATION = [
    (6, 43, None, "hexan-3-on", "hexan-3-on", "ketone", "aliphatic"),
    (6, 47, None, "hexan", "hexan", "alkane", "aliphatic"),
    (12, 112, None, "oktan", "oktan", "alkane", "aliphatic"),
    (13, 122, None, "hept-1-en", "hept-1-en", "alkene", "unsaturated"),
    (14, 130, None, "pent-1-yn", "pent-1-yn", "alkyne", "aliphatic"),
    (14, 132, None, "okt-4-yn", "okt-4-yn", "alkane", "aliphatic"),   # symmetric: C≡C invisible
    (15, 140, None, "toluen", "toluen", "arene", "aromatic"),
    (16, 149, None, "butan-1-ol", "butan-1-ol", "alcohol", "aliphatic"),
    (16, 151, None, "fenol", "fenol", "phenol", "aromatic"),
    (17, 159, None, "heptanal", "heptanal", "aldehyde", "aliphatic"),
    (18, 167, None, "butan-2-on", "butan-2-on", "ketone", "aliphatic"),
    (18, 169, None, "cyklohexanon", "cyklohexanon", "ketone", "aliphatic"),
    (19, 177, None, "ethylacetat", "ethyl-acetát", "ester", "aliphatic"),
    (19, 179, None, "ethylbenzoat", "ethyl-benzoát", "ester", "aromatic"),
    (20, 187, None, "kyselina_butanova", "kyselina butanová", "acid", "aliphatic"),
    # the two upper panels share the x axis of the bottom one (3980 → 393 cm-1)
    (21, 195, (0.0, 0.335), "nonan-2-on", "nonan-2-on", "ketone", "aliphatic",
     {"left_value": 3980, "right_value": 393}),
    (21, 195, (0.32, 0.655), "nonan-1-ol", "nonan-1-ol", "alcohol", "aliphatic",
     {"left_value": 3980, "right_value": 393}),
    (21, 195, (0.64, 1.0), "nonan-2-ol", "nonan-2-ol", "alcohol", "aliphatic"),
    (22, 202, None, "ethoxybenzen", "ethoxybenzen", "ether", "aromatic"),
    (22, 204, None, "diethylether", "diethylether", "ether", "aliphatic"),
    (23, 212, None, "acetonitril", "acetonitril", "nitrile", "aliphatic"),
    (24, 220, None, "butanamin", "butanamin", "amine", "aliphatic"),
    (25, 228, None, "nitrobenzen", "nitrobenzen", "nitro", "aromatic"),
    (25, 230, None, "nitromethan", "nitromethan", "nitro", "aliphatic"),
]

# protocol photos (pdfimages numbering of protokol_ir.pdf, pages 6–8)
PROTOCOL = [
    (6, None, "protokol_polystyren", "polystyren", "arene", "aromatic"),
    (7, 0, "protokol_vz8_ATR", "vzorek 8 (alkan)", "alkane", "aliphatic"),
    (7, 1, "protokol_vz8_T", "vzorek 8 (alkan)", "alkane", "aliphatic"),
    (8, 0, "protokol_vz14_ATR", "vzorek 14 (acetát)", "ester", "aliphatic"),
    (8, 1, "protokol_vz14_T", "vzorek 14 (acetát)", "ester", "aliphatic"),
]


def _extract_all(pdf: str, tmp: str) -> str:
    prefix = os.path.join(tmp, "img")
    subprocess.run(["pdfimages", "-png", "-p", pdf, prefix], check=True)
    return prefix


def _compose(prefix: str, page: int, num: int) -> Image.Image:
    """Image with its soft mask composited over white (slides store spectra with alpha)."""
    rgb = Image.open(f"{prefix}-{page:03d}-{num:03d}.png").convert("RGB")
    mask_path = f"{prefix}-{page:03d}-{num + 1:03d}.png"
    if os.path.exists(mask_path):
        m = Image.open(mask_path)
        if m.mode == "L" and m.size == rgb.size:
            a = np.asarray(m, float)[..., None] / 255.0
            arr = np.asarray(rgb, float) * a + 255.0 * (1 - a)
            rgb = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return rgb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("presentation")
    ap.add_argument("--protocol", help="protokol_ir.pdf with the lab spectrum photos")
    ap.add_argument("--out", default="local_data/presentation")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    labels = []
    with tempfile.TemporaryDirectory() as tmp:
        prefix = _extract_all(a.presentation, tmp)
        for page, num, crop, fn, name, cls, skel, *extra in PRESENTATION:
            img = _compose(prefix, page, num)
            if crop:
                h = img.size[1]
                img = img.crop((0, int(crop[0] * h), img.size[0], int(crop[1] * h)))
            img.save(os.path.join(a.out, fn + ".png"))
            labels.append({"file": fn + ".png", "compound": name, "class": cls, "skeleton": skel,
                           "source": f"prezentace, snímek {page}", **(extra[0] if extra else {})})
    if a.protocol:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "p")
            subprocess.run(["pdfimages", "-png", "-p", "-f", "6", "-l", "8", a.protocol, prefix],
                           check=True)
            files = sorted(os.listdir(tmp))
            for page, idx, fn, name, cls, skel in PROTOCOL:
                cand = [f for f in files if f.startswith(f"p-{page:03d}-")]
                cand.sort()
                src = cand[0 if idx is None else idx]
                Image.open(os.path.join(tmp, src)).convert("RGB").save(os.path.join(a.out, fn + ".png"))
                labels.append({"file": fn + ".png", "compound": name, "class": cls,
                               "skeleton": skel, "source": "protokol"})
    with open(os.path.join(a.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    print(f"{len(labels)} spekter uloženo do {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
