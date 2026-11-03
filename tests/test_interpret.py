import numpy as np
import pytest

from spectra.interpret import interpret
from spectra.signal import Spectrum, find_bands, preprocess
from tests.synthetic import lorentz_spectrum, reference_bands


def analyse(bands):
    x, y = lorentz_spectrum(bands)
    spec = preprocess(Spectrum(x, y + 0.002 * np.random.default_rng(0).normal(size=len(x))),
                      smooth_window=7)
    peaks = find_bands(spec, prominence=0.03)
    return interpret(peaks, (400, 4000))


def status(interp, gid):
    return next(r.status for r in interp.groups if r.group.id == gid)


def test_alkane_long_chain():
    it = analyse(reference_bands("parafínový olej / dlouhý n-alkan"))
    assert status(it, "ch2") == "potvrzeno"
    assert status(it, "ch3") == "potvrzeno"
    assert status(it, "ch2n") == "potvrzeno"
    assert status(it, "aromatic") != "potvrzeno"
    assert status(it, "ester") != "potvrzeno"
    assert "alkan" in it.summary


def test_ethyl_acetate():
    it = analyse(reference_bands("ethyl-acetát"))
    assert status(it, "ester") == "potvrzeno"
    assert status(it, "ketone") != "potvrzeno"
    assert "acetát" in it.summary
    assert it.references[0][0] == "ethyl-acetát"


def test_polystyrene():
    it = analyse(reference_bands("polystyren"))
    assert status(it, "aromatic") == "potvrzeno"
    assert status(it, "ch2") == "potvrzeno"
    assert it.substitution and it.substitution[0].startswith("mono")
    assert it.references[0][0] == "polystyren"


def test_acetic_acid():
    it = analyse(reference_bands("kyselina octová", broad_width=500))
    assert status(it, "acid") == "potvrzeno"
    assert status(it, "ester") != "potvrzeno"
    assert "kyselina" in it.summary


@pytest.mark.parametrize("name,group", [
    ("aceton", "ketone"), ("benzaldehyd", "aldehyde"), ("ethanol", "alcohol"),
    ("acetonitril", "nitrile"), ("nitrobenzen", "nitro"), ("toluen", "aromatic"),
    ("diethylether", "ether"), ("acetanhydrid", "anhydride"),
])
def test_main_groups(name, group):
    it = analyse(reference_bands(name))
    assert status(it, group) == "potvrzeno", (name, [(r.group.id, r.status, r.reason)
                                                     for r in it.groups[:6]])
