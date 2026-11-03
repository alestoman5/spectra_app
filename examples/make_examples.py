"""Generate synthetic example spectrum images (OMNIC-like printouts) for trying the app.

    python examples/make_examples.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from tests.test_digitize import render  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = [
    ("polystyren", "T", "k", 0.0, "polystyren_T.png"),
    ("ethyl-acetát", "A", "tab:red", 0.0, "ethylacetat_A.png"),
    ("kyselina benzoová", "T", "k", 0.8, "kyselina_benzoova_T.png"),
    ("benzaldehyd", "T", "tab:blue", 0.0, "benzaldehyd_T.png"),
]

if __name__ == "__main__":
    for name, mode, color, rot, fn in EXAMPLES:
        Image.fromarray(render(name, mode, color, rot)).save(os.path.join(HERE, fn))
        print("saved", fn)
