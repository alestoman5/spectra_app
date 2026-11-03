"""End-to-end test: render a synthetic spectrum as an image, digitise it and interpret it."""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from spectra.pipeline import Settings, analyse_image  # noqa: E402
from tests.synthetic import lorentz_spectrum, reference_bands  # noqa: E402


def render(name, mode="T", color="k", rotate=0.0):
    x, a = lorentz_spectrum(reference_bands(name))
    a = a / a.max() * 1.2
    fig = plt.figure(figsize=(10, 6), dpi=110)
    ax = fig.add_axes([0.1, 0.12, 0.85, 0.8])
    if mode == "T":
        ax.plot(x, 100 * 10 ** (-a), color=color, lw=0.8)
        ax.set_ylabel("%Transmittance")
    else:
        ax.plot(x, a, color=color, lw=0.8)
        ax.set_ylabel("Absorbance")
    ax.set_xlim(4000, 400)
    ax.set_xticks(np.arange(4000, 399, -500))
    ax.set_xlabel("Wavenumbers (cm-1)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    img = Image.open(buf).convert("RGB")
    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    return np.asarray(img)


@pytest.mark.parametrize("mode,color,rotate", [("T", "k", 0), ("A", "tab:red", 0), ("T", "k", 1.0)])
def test_polystyrene_image(mode, color, rotate):
    res = analyse_image(render("polystyren", mode, color, rotate), Settings())
    assert res.y_mode == mode
    wns = np.array([p.wavenumber for p in res.peaks])
    for true in (3026, 2923, 1601, 1493, 1452, 757, 700):
        assert np.min(np.abs(wns - true)) <= 10, (true, wns)
    st = {r.group.id: r.status for r in res.interpretation.groups}
    assert st["aromatic"] == "potvrzeno"
    assert res.interpretation.references[0][0] == "polystyren"


def test_ester_image():
    res = analyse_image(render("ethyl-acetát"), Settings())
    st = {r.group.id: r.status for r in res.interpretation.groups}
    assert st["ester"] == "potvrzeno"


@pytest.mark.parametrize("name,mode,color,group", [
    ("kyselina benzoová", "T", "k", "acid"),
    ("benzaldehyd", "T", "tab:blue", "aldehyde"),
    ("aceton", "A", "tab:green", "ketone"),
])
def test_other_images(name, mode, color, group):
    res = analyse_image(render(name, mode, color, 0.5), Settings())
    st = {r.group.id: r.status for r in res.interpretation.groups}
    assert st[group] == "potvrzeno", res.interpretation.summary
