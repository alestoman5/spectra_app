import numpy as np
import pytest

from spectra.interpret import interpret, report_text
from spectra.signal import Spectrum, find_bands, preprocess
from tests.synthetic import lorentz_spectrum, reference_bands


def analyse(bands):
    x, y = lorentz_spectrum(bands)
    spec = preprocess(Spectrum(x, y + 0.002 * np.random.default_rng(0).normal(size=len(x))),
                      smooth_window=7)
    peaks = find_bands(spec, prominence=0.03)
    return interpret(peaks, (400, 4000), spectrum=spec)


def status(interp, gid):
    return next(r.status for r in interp.groups if r.group.id == gid)


def test_alkane_long_chain():
    it = analyse(reference_bands("parafínový olej / dlouhý n-alkan"))
    assert it.main_class == "alkane"
    assert status(it, "ch2") == "potvrzeno"
    assert status(it, "arene") != "potvrzeno"
    assert "alkan" in it.summary and "(CH2)n" in it.summary


def test_ethyl_acetate():
    it = analyse(reference_bands("ethyl-acetát"))
    assert it.main_class == "ester"
    assert status(it, "ester") == "potvrzeno"
    assert status(it, "ketone") == "vyloučeno"
    assert "acetát" in it.summary
    assert it.references[0][0] == "ethyl-acetát"


def test_polystyrene():
    it = analyse(reference_bands("polystyren"))
    assert it.main_class == "arene"
    assert it.substitution and it.substitution[0].startswith("mono")
    assert it.references[0][0] == "polystyren"


def test_acetic_acid():
    it = analyse(reference_bands("kyselina octová", broad_width=500))
    assert it.main_class == "acid"
    assert "kyselina" in it.summary


@pytest.mark.parametrize("name,cls", [
    ("aceton", "ketone"), ("cyklohexanon", "ketone"), ("benzaldehyd", "aldehyde"),
    ("butanal", "aldehyde"), ("ethanol", "alcohol"), ("butan-1-ol", "alcohol"),
    ("acetonitril", "nitrile"), ("benzonitril", "nitrile"), ("nitrobenzen", "nitro"),
    ("toluen", "arene"), ("diethylether", "ether"), ("anisol", "ether"),
    ("acetanhydrid", "anhydride"), ("methyl-benzoát", "ester"), ("hex-1-en", "alkene"),
    ("fenylacetylen", "alkyne"), ("n-hexan", "alkane"), ("anilin", "amine"),
])
def test_main_classes(name, cls):
    it = analyse(reference_bands(name))
    assert it.main_class == cls, (name, it.steps)


def test_report_contains_steps():
    it = analyse(reference_bands("ethyl-acetát"))
    txt = report_text(it)
    assert "Postup (rozhodovací schéma)" in txt and "karbonyl ANO" in txt
