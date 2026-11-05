"""Interpretation of an IR spectrum following the CACH II presentation
*Interpretace infračervených spekter* (ÚACh VŠCHT Praha).

Procedure (slides "Postup interpretace", "Rozhodovací schema", "Interpretace – rychlokuchařka"):

1. The skeleton first: bands 3100–3000 cm⁻¹ → arene / alkene, bands 3000–2850 cm⁻¹ → alkane.
2. A strong band 1800–1650 cm⁻¹ (C=O)?
   * yes → broad O–H 3300–2500 reaching below 3000 → carboxylic acid;
     otherwise two strong C–O bands 1310–1050 → ester;
     otherwise Fermi doublet 2830–2650 → aldehyde; otherwise → ketone (by exclusion).
   * no  → broad band 3550–3230 → alcohol / phenol (type from C–O 1260–1020);
     N–H 3550–3250 → amine (2 bands primary, 1 secondary);
     sharp band 2260–2100 → nitrile / alkyne; two strong bands 1570–1320 → nitro compound;
     strong C–O 1150–1050 without O–H → ether.
3. For every class found, look up *all* of its characteristic bands (missing band → the class is
   not present).
4. When no significant band is left unexplained, the interpretation is complete.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ir_table import AROMATIC_SUBSTITUTION, CLASS_NAMES, GROUP_BY_ID, GROUPS, Band, Group
from .references import REFERENCES
from .signal import Peak, Spectrum

TOL = 12.0  # cm-1 tolerance added to table ranges (resolution + digitisation error)

# relative intensity thresholds (height of a band relative to the strongest band)
STRONG = 0.35        # "silný / velmi intenzivní pás"
STRONG_CO = 0.25     # C=O: in esters the C–O–C band is often stronger than C=O itself
MEDIUM = 0.12
WEAK = 0.04

INTENSITY_FACTOR = {
    "s": {"s": 1.0, "m": 0.75, "w": 0.4},
    "m": {"s": 1.0, "m": 1.0, "w": 0.7},
    "w": {"s": 0.9, "m": 1.0, "w": 1.0},
    "v": {"s": 1.0, "m": 1.0, "w": 0.9},
}


@dataclass
class BandMatch:
    band: Band
    peak: Peak | None
    factor: float


@dataclass
class GroupResult:
    group: Group
    score: float                       # 0..1 – fraction of characteristic bands found
    matches: list[BandMatch]
    status: str = "nepřítomno"         # potvrzeno / možné / vyloučeno / nepřítomno
    reason: str = ""

    @property
    def missing_keys(self) -> list[Band]:
        return [m.band for m in self.matches if m.band.key and m.peak is None]

    @property
    def missing(self) -> list[Band]:
        return [m.band for m in self.matches if m.peak is None]


@dataclass
class Interpretation:
    peaks: list[Peak]
    groups: list[GroupResult]
    main_class: str = "unknown"        # decision-tree result (see ir_table.CLASS_NAMES)
    skeleton: str = ""                 # aliphatic / aromatic / unsaturated / ""
    steps: list[str] = field(default_factory=list)
    assignments: dict[float, list[str]] = field(default_factory=dict)
    substitution: list[str] = field(default_factory=list)
    branching: str = ""
    summary: str = ""
    notes: list[str] = field(default_factory=list)
    unexplained: list[Peak] = field(default_factory=list)
    references: list[tuple[str, float, str]] = field(default_factory=list)

    @property
    def confirmed(self) -> list[GroupResult]:
        return [g for g in self.groups if g.status == "potvrzeno"]

    @property
    def possible(self) -> list[GroupResult]:
        return [g for g in self.groups if g.status == "možné"]


# --------------------------------------------------------------------------- helpers
def _usable(peaks: list[Peak]) -> list[Peak]:
    return [p for p in peaks if not p.artefact]


def _in(peaks: list[Peak], lo: float, hi: float, min_h: float = 0.0, tol: float = 0.0) -> list[Peak]:
    return [p for p in peaks if lo - tol <= p.wavenumber <= hi + tol and p.height >= min_h]


def _max(peaks: list[Peak]) -> Peak | None:
    return max(peaks, key=lambda p: p.height) if peaks else None


def _fmt(p: Peak | None) -> str:
    return f"{p.wavenumber:.0f}" if p is not None else "–"


class _Features:
    """Region features of the processed (baseline corrected, normalised) spectrum."""

    def __init__(self, spectrum: Spectrum | None):
        self.s = spectrum

    def mean(self, lo: float, hi: float) -> float:
        if self.s is None:
            return 0.0
        m = (self.s.x >= lo) & (self.s.x <= hi)
        return float(np.mean(self.s.y[m])) if m.any() else 0.0

    def covered(self, lo: float, hi: float) -> bool:
        if self.s is None:
            return True
        return self.s.x.min() <= lo and self.s.x.max() >= hi


def _best_peak(band: Band, peaks: list[Peak], used: set[float]) -> tuple[Peak | None, float]:
    best, best_f = None, 0.0
    for p in peaks:
        if p.wavenumber in used or not band.lo - TOL <= p.wavenumber <= band.hi + TOL:
            continue
        f = INTENSITY_FACTOR.get(band.intensity, INTENSITY_FACTOR["m"])[p.intensity]
        if band.broad and not p.broad:
            f *= 0.5
        if not band.lo <= p.wavenumber <= band.hi:
            f *= 0.85
        if f > best_f:
            best, best_f = p, f
    return best, best_f


def score_group(group: Group, peaks: list[Peak], x_range: tuple[float, float]) -> GroupResult:
    """Look up all characteristic bands of a group ("dohledat všechny pásy")."""
    matches, total, got = [], 0.0, 0.0
    used: set[float] = set()
    for band in sorted(group.bands, key=lambda b: (not b.key, -b.weight)):
        p, f = _best_peak(band, peaks, used)
        if p is not None:
            used.add(p.wavenumber)
        matches.append(BandMatch(band, p, f))
        if band.hi < x_range[0] or band.lo > x_range[1]:
            continue  # outside the measured range – neither for nor against
        total += band.weight
        got += band.weight * f
    matches.sort(key=lambda m: -m.band.hi)
    return GroupResult(group, got / total if total else 0.0, matches)


# --------------------------------------------------------------------------- decision tree
def interpret(peaks: list[Peak], x_range: tuple[float, float] | None = None,
              spectrum: Spectrum | None = None) -> Interpretation:
    usable = _usable(peaks)
    xmin, xmax = x_range if x_range else (400.0, 4000.0)
    feat = _Features(spectrum)
    res = {g.id: score_group(g, usable, (xmin, xmax)) for g in GROUPS}
    it = Interpretation(peaks=peaks, groups=[])
    steps = it.steps
    found: dict[str, str] = {}          # group id -> reason (decided present)
    rejected: dict[str, str] = {}       # group id -> reason (decided absent)

    # ---------------------------------------------------------------- 1. skeleton
    sp3 = _in(usable, 2840, 3000, MEDIUM * 0.6)
    sp2 = [p for p in _in(usable, 3000, 3110, WEAK) if not p.broad]
    ring = _in(usable, 1575, 1625, WEAK, tol=8)
    ring2 = _in(usable, 1470, 1525, WEAK, tol=5)
    overt = _in(usable, 1650, 2000, 0.0)
    oop = _in(usable, 670, 900, MEDIUM)
    cc_alkene = [p for p in _in(usable, 1620, 1680, WEAK, tol=5) if not p.broad]
    gamma_alkene = _in(usable, 880, 1010, MEDIUM)

    # arene: =C–H above 3000 + ring ν(C=C) (1600 may be missing for centrosymmetric rings)
    # + at least one more aromatic feature (second ring band, overtone pattern, strong γ(CH))
    evidence = sum([bool(ring), bool(ring2), len(overt) >= 2, bool(oop)])
    arene = bool(sp2) and (bool(ring) or bool(ring2)) and evidence >= 2 and \
        (len(sp2) >= 2 or len(overt) >= 2 or bool(oop))
    alkene = bool(sp2) and bool(cc_alkene) and bool(gamma_alkene) and not arene
    if arene:
        found["arene"] = (f"pásy ν(C–H) nad 3000 ({', '.join(_fmt(p) for p in sp2)}) + "
                          f"skeletální ν(C=C) {', '.join(_fmt(p) for p in ring + ring2)}")
        steps.append(f"Pásy 3100–3000 cm⁻¹ ({', '.join(_fmt(p) for p in sp2)}) a ν(C=C) kruhu "
                      f"({', '.join(_fmt(p) for p in ring + ring2)}) → aromatický skelet (aren).")
    elif alkene:
        found["alkene"] = (f"ν(=C–H) {_fmt(sp2[0])}, ν(C=C) {_fmt(_max(cc_alkene))}, "
                           f"γ(=C–H) {_fmt(_max(gamma_alkene))}")
        steps.append(f"Pás nad 3000 cm⁻¹ ({_fmt(sp2[0])}) + ν(C=C) {_fmt(_max(cc_alkene))} + "
                     f"γ(=C–H) {_fmt(_max(gamma_alkene))} → alken.")
    elif sp2:
        steps.append(f"Slabé pásy nad 3000 cm⁻¹ ({', '.join(_fmt(p) for p in sp2)}) bez dalších "
                     "pásů arenu/alkenu – nenasycený skelet nepotvrzen.")
    else:
        steps.append("Žádné pásy 3100–3000 cm⁻¹ → není aren ani alken s =C–H.")
        rejected["arene"] = "chybí ν(C–H) 3100–3000 cm⁻¹"
        rejected["alkene"] = "chybí ν(=C–H) 3100–3000 cm⁻¹"
    if sp3:
        steps.append(f"Pásy 3000–2850 cm⁻¹ ({', '.join(_fmt(p) for p in sp3[:4])}) → alifatické "
                     "C–H (CH3/CH2).")
    else:
        steps.append("Žádné pásy 3000–2850 cm⁻¹ → bez alifatických C–H.")

    # ---------------------------------------------------------------- 2. carbonyl?
    strong_co_band = _max(_in(usable, 1050, 1310, 0.6, tol=8))
    co = [p for p in _in(usable, 1650, 1850, STRONG_CO, tol=8)
          if p.wavenumber >= 1650 and (not p.broad or p.height > 0.6) and p.width < 120
          # a medium band only counts as C=O when a very strong C–O band accompanies it
          # (esters: "C–O–C obvykle intenzivnější než ν(C=O)"); otherwise it is an overtone
          and (p.height >= 0.5 or strong_co_band is not None)]
    co_peak = _max(co)
    main = "unknown"

    # O–H/N–H region features
    # carboxylic O–H: very broad absorption 3300–2500 reaching below 3000 cm-1 – either seen as
    # elevated absorbance down to ~2600, or as one very broad band centred near 3000
    acid_band = [p for p in _in(usable, 2850, 3200, 0.15) if p.width >= 200]
    oh_acid = (feat.mean(2550, 2700) >= 0.07 and feat.mean(2950, 3250) >= 0.18) or bool(acid_band)
    oh_peaks = [p for p in _in(usable, 3150, 3600, MEDIUM * 0.8)
                if p.broad or p.width >= 90]
    oh_region = feat.mean(3250, 3450)
    nh = [p for p in _in(usable, 3250, 3550, WEAK) if p.width < 220]
    # sharp ≡C–H of a terminal alkyne (3340–3260) must not be mistaken for N–H
    ch_alkyne = [p for p in _in(usable, 3260, 3340, 0.06, tol=5) if p.width < 70 and
                 feat.mean(p.wavenumber + 70, p.wavenumber + 130) < 0.35 * p.height and
                 feat.mean(p.wavenumber - 130, p.wavenumber - 70) < 0.35 * p.height and
                 (_in(usable, 2100, 2160, 0.0, tol=8) or _in(usable, 575, 730, MEDIUM))]

    if co_peak is not None:
        steps.append(f"Silný pás {_fmt(co_peak)} cm⁻¹ v oblasti 1800–1650 → karbonyl ANO.")
        cos = sorted(co, key=lambda p: -p.wavenumber)
        anh = [(a, b) for a in cos for b in cos if 40 <= a.wavenumber - b.wavenumber <= 80
               and a.wavenumber >= 1780]
        c_o_as = _in(usable, 1160, 1310, 0.0, tol=8)
        c_o_s = _in(usable, 1030, 1160, 0.0, tol=8)
        as_best, s_best = _max(c_o_as), _max(c_o_s)
        ester_ok = (as_best is not None and s_best is not None and
                    as_best.height >= max(0.45, 0.6 * co_peak.height) and
                    s_best.height >= max(0.2, 0.25 * co_peak.height))
        fermi1 = _in(usable, 2790, 2840, WEAK * 0.5)
        fermi2 = _in(usable, 2650, 2760, WEAK * 0.5)
        fermi = bool(fermi2) and (bool(fermi1) or feat.mean(2690, 2760) > 0.03)
        amide_nh = [p for p in _in(usable, 3150, 3500, MEDIUM) if p.width < 250]
        if oh_acid:
            main = "acid"
            steps.append("Široký pás O–H 3300–2500 zasahující pod 3000 cm⁻¹ → karboxylová kyselina.")
            for g in ("ester", "aldehyde", "ketone"):
                rejected[g] = "kyselina má přednost (široký O–H 3300–2500)"
        elif anh:
            main = "anhydride"
            steps.append(f"Dva pásy C=O {_fmt(anh[0][0])} a {_fmt(anh[0][1])} cm⁻¹ (odstup "
                         "~60) → anhydrid.")
        elif ester_ok:
            main = "ester"
            steps.append(f"Široký O–H chybí; dva silné pásy C–O {_fmt(as_best)} a {_fmt(s_best)} "
                         "cm⁻¹ → ester.")
            rejected["acid"] = "chybí široký pás O–H 3300–2500"
            rejected["aldehyde"] = rejected["ketone"] = "pásy C–O odpovídají esteru"
        elif fermi:
            main = "aldehyde"
            steps.append(f"Bez širokého O–H a bez dvou silných C–O; Fermiho dublet "
                         f"{', '.join(_fmt(p) for p in fermi1 + fermi2)} cm⁻¹ → aldehyd.")
            rejected["acid"] = "chybí široký pás O–H 3300–2500"
            rejected["ester"] = "chybí dva silné pásy C–O 1310–1050"
            rejected["ketone"] = "Fermiho dublet → aldehyd"
        elif amide_nh and co_peak.wavenumber <= 1700 and _in(usable, 1510, 1650, MEDIUM):
            main = "amide"
            steps.append(f"C=O {_fmt(co_peak)} + N–H {_fmt(amide_nh[0])} + amid II → amid "
                         "(mimo schéma).")
        else:
            main = "ketone"
            steps.append("Bez širokého O–H, bez dvou silných C–O, bez Fermiho dubletu → keton "
                         "(vylučovací metodou).")
            rejected["acid"] = "chybí široký pás O–H 3300–2500"
            rejected["ester"] = "chybí dva silné pásy C–O 1310–1050"
            rejected["aldehyde"] = "chybí Fermiho dublet 2830–2650"
        found[{"acid": "acid", "ester": "ester", "aldehyde": "aldehyde", "ketone": "ketone",
               "amide": "amide", "anhydride": "anhydride"}[main]] = steps[-1]
    else:
        strongest_co = _max(_in(usable, 1650, 1850, 0.0))
        steps.append("Silný pás 1800–1650 cm⁻¹ chybí → karbonyl NE" +
                     (f" (nejsilnější pás v oblasti {_fmt(strongest_co)} jen "
                      f"{strongest_co.height * 100:.0f} %)." if strongest_co else "."))
        for g in ("acid", "ester", "aldehyde", "ketone", "amide", "anhydride"):
            rejected[g] = "chybí silný pás C=O 1800–1650 cm⁻¹"
        # ---- alcohol / phenol
        oh_ok = bool(oh_peaks) and (oh_region >= 0.1 or _max(oh_peaks).height >= 0.2)
        nh = [p for p in nh if p not in ch_alkyne]
        amine_pair = [(a, b) for a in nh for b in nh if 50 <= a.wavenumber - b.wavenumber <= 140]
        if ch_alkyne:
            main = "alkyne"
            tb = _max(_in(usable, 2100, 2160, 0.0, tol=8))
            steps.append(f"Ostrý úzký pás ≡C–H {_fmt(ch_alkyne[0])} cm⁻¹" +
                         (f" + ν(C≡C) {_fmt(tb)}" if tb else "") +
                         (f" + δ(≡C–H) {_fmt(_max(_in(usable, 575, 730, MEDIUM)))}"
                          if _in(usable, 575, 730, MEDIUM) else "") + " → terminální alkyn.")
            found["alkyne"] = steps[-1]
            rejected["alcohol"] = "pás 3300 je úzký ≡C–H, ne široký O–H"
        elif amine_pair and not (oh_ok and _max(oh_peaks).width > 250):
            main = "amine"
            a, b = amine_pair[0]
            steps.append(f"Dva užší pásy N–H {_fmt(a)} a {_fmt(b)} cm⁻¹ → primární amin.")
            found["amine1"] = steps[-1]
        elif oh_ok:
            ohp = _max(oh_peaks)
            main = "alcohol"
            steps.append(f"Široký pás {_fmt(ohp)} cm⁻¹ (šířka {ohp.width:.0f} cm⁻¹) v oblasti "
                         "3550–3230 → O–H (alkohol/fenol).")
            found["alcohol"] = steps[-1]
            c_o = _max(_in(usable, 1020, 1260, 0.0, tol=8))
            if c_o is not None:
                w = c_o.wavenumber
                if "arene" in found and w >= 1180:
                    main = "phenol"
                    kind = "fenol (Ar–OH, 1260–1180)"
                elif w >= 1125:
                    kind = "terciární alkohol (1205–1125)"
                elif w >= 1085:
                    kind = "sekundární alkohol (1125–1085)"
                else:
                    kind = "primární alkohol (1085–1030)"
                if "arene" in found and w >= 1180:
                    pass
                steps.append(f"Nejsilnější ν(C–O) {w:.0f} cm⁻¹ → {kind}.")
                it.notes.append(f"ν(C–O) {w:.0f} cm⁻¹ → {kind}.")
        elif len(nh) == 1 and nh[0].height >= WEAK and not nh[0].broad:
            main = "amine"
            steps.append(f"Jeden užší pás N–H {_fmt(nh[0])} cm⁻¹ → sekundární amin.")
            found["amine2"] = steps[-1]
        else:
            steps.append("Široký pás 3550–3230 ani pásy N–H nenalezeny → není alkohol ani "
                         "primární/sekundární amin.")
            rejected["alcohol"] = "chybí široký pás O–H 3550–3230"
            rejected["amine1"] = rejected["amine2"] = "chybí pásy N–H 3550–3250"
        # ---- triple bonds
        if main == "unknown":
            triple = [p for p in _in(usable, 2100, 2270, WEAK) if p.width < 60]
            if ch_alkyne and triple:
                main = "alkyne"
                steps.append(f"Ostrý ≡C–H {_fmt(ch_alkyne[0])} + ν(C≡C) {_fmt(triple[0])} cm⁻¹ "
                             "→ terminální alkyn.")
                found["alkyne"] = steps[-1]
            elif triple and _max(triple).wavenumber >= 2200 and _max(triple).height >= 0.08:
                main = "nitrile"
                steps.append(f"Ostrý pás {_fmt(_max(triple))} cm⁻¹ (2270–2200) bez ≡C–H → nitril.")
                found["nitrile"] = steps[-1]
            elif triple:
                main = "alkyne"
                steps.append(f"Slabý pás {_fmt(_max(triple))} cm⁻¹ (2260–2190) → disubstituovaný "
                             "alkyn.")
                found["alkyne"] = steps[-1]
            else:
                rejected["nitrile"] = rejected["alkyne"] = "chybí ostrý pás 2260–2100 cm⁻¹"
        # ---- nitro
        if main == "unknown":
            # (gas-phase spectra in the NIST WebBook have νas(NO2) up to ~1600)
            n1 = _max([p for p in _in(usable, 1485, 1610, STRONG, tol=8) if p.wavenumber >= 1485])
            n2 = _max(_in(usable, 1320, 1405, STRONG * 0.8))
            if n2 is not None and "arene" in found and n2.height < 0.45:
                n2 = None   # ring ν(C=C) ~1500 + a medium band – not two strong NO2 bands
            if n1 is not None and n2 is not None:
                main = "nitro"
                steps.append(f"Dva silné pásy {_fmt(n1)} a {_fmt(n2)} cm⁻¹ (1570–1320) → "
                             "nitrosloučenina.")
                found["nitro"] = steps[-1]
            else:
                rejected["nitro"] = "chybí dva silné pásy 1570–1320 cm⁻¹"
        # ---- ether
        if main == "unknown":
            eth = _max(_in(usable, 1020, 1155, STRONG + 0.1, tol=8))
            eth_ar = _max(_in(usable, 1210, 1310, STRONG + 0.1, tol=8))
            eth_ar2 = _max(_in(usable, 1020, 1120, MEDIUM, tol=8))
            aryl = "arene" in found and eth_ar is not None and eth_ar.height >= 0.6 and \
                eth_ar2 is not None
            if (eth is not None and eth.height >= 0.55) or aryl:
                main = "ether"
                txt = f"{_fmt(eth_ar)} + {_fmt(eth_ar2)} (Ar–O–R)" if aryl else _fmt(eth)
                steps.append(f"Silný pás C–O {txt} cm⁻¹ bez O–H a bez C=O → ether.")
                found["ether"] = steps[-1]
            else:
                rejected["ether"] = "chybí silný pás C–O 1150–1050 cm⁻¹"
        if main == "unknown":
            main = "arene" if "arene" in found else "alkene" if "alkene" in found else \
                "alkane" if sp3 else "unknown"
            steps.append(f"Žádná další funkční skupina → {CLASS_NAMES[main]}.")
    if co_peak is not None:
        for g in ("alcohol", "amine1", "amine2", "ether", "nitro", "nitrile"):
            rejected.setdefault(g, "karbonylová sloučenina – hledá se ve větvi C=O")
        if not oh_acid and main != "amide":
            rejected["alcohol"] = "karbonyl bez širokého O–H"

    it.main_class = main
    it.skeleton = "aromatic" if "arene" in found else "unsaturated" if "alkene" in found \
        else "aliphatic" if sp3 else ""
    if sp3:
        found.setdefault("ch2", "pásy 3000–2850")
        found.setdefault("ch3", "pásy 3000–2850")

    # ---------------------------------------------------------------- 3. confirmation
    for gid, r in res.items():
        if gid in found:
            # verify all characteristic bands; missing bands are reported
            miss = [b for b in r.missing_keys if xmin <= b.center <= xmax]
            if gid in ("ch2", "ch3"):
                r.status = "potvrzeno" if not miss else ("možné" if r.score >= 0.4 else "nepřítomno")
                r.reason = ("chybí " + ", ".join(f"{b.vib} ({b.range_str()})" for b in miss)) \
                    if miss else ""
            else:
                r.status = "potvrzeno" if r.score >= 0.5 and len(miss) <= 1 else "možné"
                r.reason = found[gid] if not miss else \
                    "nenalezeno: " + ", ".join(f"{b.vib} ({b.range_str()})" for b in miss)
        elif gid in rejected:
            r.status = "vyloučeno"
            r.reason = rejected[gid]
        elif gid == "water":
            wat = feat.mean(3250, 3500) > 0.08 and _in(usable, 1600, 1660, WEAK) and \
                main not in ("alcohol", "phenol", "acid", "amine", "amide")
            r.status = "možné" if wat else "nepřítomno"
            r.reason = "široký pás ~3400 + ~1630 (vlhkost)" if wat else ""
        else:
            r.status = "nepřítomno"
            r.reason = ("chybí " + ", ".join(f"{b.vib} ({b.range_str()})" for b in r.missing_keys)) \
                if r.missing_keys else "neprokázáno rozhodovacím schématem"
    order = {"potvrzeno": 0, "možné": 1, "vyloučeno": 2, "nepřítomno": 3}
    it.groups = sorted(res.values(), key=lambda r: (order[r.status], -r.score))

    _details(it, usable, res, main)
    _assign_peaks(it)
    it.unexplained = [p for p in usable if p.height >= STRONG and p.wavenumber > 1300
                      and not it.assignments.get(p.wavenumber)]
    if it.unexplained:
        it.notes.append("Nevysvětlené silné pásy v oblasti charakteristických vibrací: " +
                        ", ".join(_fmt(p) for p in it.unexplained) + " cm⁻¹ – zkontrolujte.")
    else:
        steps.append("Všechny významné pásy nad 1300 cm⁻¹ jsou přiřazeny → interpretace je "
                     "úplná.")
    it.references = match_references(peaks, xmin, xmax, main_class=main)
    it.summary = _summary(it, res)
    return it


# --------------------------------------------------------------------------- details
def _details(it: Interpretation, peaks: list[Peak], res: dict[str, GroupResult], main: str) -> None:
    notes = it.notes
    if main in ("ester", "ketone", "aldehyde", "acid"):
        c = _max(_in(peaks, 1650, 1850, STRONG, tol=8))
        if c is not None:
            if main == "ester":
                if c.wavenumber >= 1750:
                    notes.append(f"ν(C=O) {c.wavenumber:.0f} cm⁻¹ → spíše fenyl-/vinyl-ester.")
                elif c.wavenumber >= 1730:
                    notes.append(f"ν(C=O) {c.wavenumber:.0f} cm⁻¹ → nasycený alifatický ester.")
                else:
                    notes.append(f"ν(C=O) {c.wavenumber:.0f} cm⁻¹ → konjugovaný (aromatický nebo "
                                 "α,β-nenasycený) ester.")
                ac = _in(peaks, 1355, 1385, MEDIUM)
                if ac and _in(peaks, 1210, 1265, STRONG):
                    notes.append("δs(CH3) ~1370 + νas(C–O–C) ~1240 cm⁻¹ → acetát (ester kyseliny "
                                 "octové).")
                ar = _in(peaks, 1250, 1310, STRONG) and _in(peaks, 1100, 1150, MEDIUM)
                if ar and "arene" in [r.group.id for r in it.confirmed]:
                    notes.append("C–O–C 1310–1250 + 1150–1100 → ester aromatické kyseliny "
                                 "(Ar–CO–O–R).")
            elif c.wavenumber < 1700 and main in ("ketone", "aldehyde"):
                notes.append(f"ν(C=O) {c.wavenumber:.0f} cm⁻¹ (nízký vlnočet) → konjugovaný / "
                             "aromatický karbonyl.")
            else:
                notes.append(f"ν(C=O) {c.wavenumber:.0f} cm⁻¹.")
    # CH3 umbrella vibration: singlet / doublet
    if res["ch3"].status in ("potvrzeno", "možné") and main != "ester":
        d = sorted(_in(peaks, 1355, 1395, WEAK), key=lambda p: -p.wavenumber)
        if len(d) >= 2 and abs(d[0].wavenumber - d[1].wavenumber) <= 25:
            hi_p, lo_p = d[0], d[1]
            if lo_p.height / max(hi_p.height, 1e-9) > 1.6:
                it.branching = (f"Dublet δs(CH3) {hi_p.wavenumber:.0f}/{lo_p.wavenumber:.0f} cm⁻¹, "
                                "složka s nižším vlnočtem ~2× intenzivnější → terc-butyl –C(CH3)3.")
            else:
                it.branching = (f"Dublet δs(CH3) {hi_p.wavenumber:.0f}/{lo_p.wavenumber:.0f} cm⁻¹ "
                                "→ rozvětvení (isopropyl / gem-dimethyl).")
        elif d:
            it.branching = (f"δs(CH3) {d[0].wavenumber:.0f} cm⁻¹ je singlet → CH3–C– / CH3–O–, "
                            "bez rozvětvení >C(CH3)2.")
    # CH2 rocking → long chain
    if res["ch2"].status == "potvrzeno" and "arene" not in [r.group.id for r in it.confirmed]:
        rock = _in(peaks, 715, 730, WEAK * 0.7)
        if rock:
            notes.append(f"ρ(CH2) {rock[0].wavenumber:.0f} cm⁻¹ → řetězec –(CH2)n–, n ≥ 4.")
    # aromatic substitution
    if res["arene"].status in ("potvrzeno", "možné"):
        for name, ranges in AROMATIC_SUBSTITUTION:
            hits = [_in(peaks, lo - 5, hi + 5, MEDIUM) for hi, lo in ranges]
            if all(hits):
                it.substitution.append(name)
                for h in hits:
                    p = _max(h)
                    it.assignments.setdefault(p.wavenumber, []).append(f"γ(CH) – {name}")
                break


def _assign_peaks(it: Interpretation) -> None:
    for r in it.groups:
        if r.status not in ("potvrzeno", "možné"):
            continue
        for m in r.matches:
            if m.peak is None:
                continue
            tag = f"{m.band.vib} – {r.group.formula}" + (" (?)" if r.status == "možné" else "")
            targets = [m.peak]
            if r.status == "potvrzeno" and m.band.hi - m.band.lo <= 400:
                targets += [p for p in it.peaks if m.band.lo <= p.wavenumber <= m.band.hi
                            and p is not m.peak and not p.artefact
                            and not it.assignments.get(p.wavenumber)]
            for pk in targets:
                lst = it.assignments.setdefault(pk.wavenumber, [])
                if tag not in lst:
                    lst.append(tag)
    for p in it.peaks:
        if p.artefact:
            it.assignments[p.wavenumber] = [p.artefact]


def _summary(it: Interpretation, res: dict[str, GroupResult]) -> str:
    main = it.main_class
    if main == "unknown":
        return "Ve spektru nebyly nalezeny charakteristické pásy – zkontrolujte kalibraci a ořez."
    name = CLASS_NAMES[main]
    aromatic = it.skeleton == "aromatic"
    aliph = res["ch2"].status == "potvrzeno" or res["ch3"].status == "potvrzeno"
    if main == "alkane":
        name = "nasycený alifatický uhlovodík (alkan)"
        if any("–(CH2)n–" in n for n in it.notes):
            name += " s dlouhým řetězcem –(CH2)n–"
    elif main == "arene":
        name = "aromatický uhlovodík" + (" s alifatickou částí (–CH2–/–CH3)" if aliph else "")
    elif main == "alkene":
        name = "alken (nenasycený uhlovodík)"
    elif main in ("phenol",):
        name = "fenol (aromatický alkohol)"
    else:
        fem = name.split()[-1].endswith("a")
        if aromatic:
            name = ("aromatická " if fem else "aromatický ") + name
        elif aliph:
            name = ("alifatická " if fem else "alifatický ") + name
    if main == "ester" and any("acetát" in n for n in it.notes):
        name += " – acetát (ester kyseliny octové)"
    if main == "alcohol":
        kind = next((n.split("→ ")[1].rstrip(".") for n in it.notes if n.startswith("ν(C–O)")), "")
        if kind:
            name += f" – {kind}"
    s = f"Pravděpodobně: {name}."
    if it.substitution:
        s += f" Substituce jádra: {it.substitution[0]}."
    if it.references and it.references[0][1] >= 0.45:
        s += f" Nejpodobnější referenční látka: {it.references[0][0]} " \
             f"(shoda {it.references[0][1] * 100:.0f} %)."
    return s


# --------------------------------------------------------------------------- references
def match_references(peaks: list[Peak], xmin: float = 400, xmax: float = 4000,
                     tol: float = 15.0, top: int = 5, main_class: str | None = None
                     ) -> list[tuple[str, float, str]]:
    """Compare the band list with the built-in library; only compounds of the class found by the
    decision tree are considered (if given)."""
    meas = [p for p in _usable(peaks) if p.height >= 0.08]
    strong = [p for p in meas if p.height >= 0.25 and not p.broad]
    if not meas:
        return []
    out = []
    for ref in REFERENCES:
        if main_class and main_class != "unknown" and ref.cls != main_class:
            continue
        ref_peaks = [(w, s) for w, s in ref.peaks if xmin + 5 <= w <= xmax - 5]
        if len(ref_peaks) < 2:
            continue
        wt = {"s": 2.0, "m": 1.0, "w": 0.5}
        tot = sum(wt[s] for _, s in ref_peaks)
        hit = 0.0
        for w, s in ref_peaks:
            t = tol * (4 if ref.broad_oh and w > 2900 else 1)
            if any(abs(p.wavenumber - w) <= t for p in meas):
                hit += wt[s]
        recall = hit / tot
        explained = sum(1 for p in strong if any(abs(p.wavenumber - w) <= tol * 1.5
                                                 for w, _ in ref.peaks))
        precision = explained / len(strong) if strong else 1.0
        out.append((ref.name, recall * (0.35 + 0.65 * precision), ref.formula))
    out.sort(key=lambda t: -t[1])
    return out[:top]


# --------------------------------------------------------------------------- report
def report_text(interp: Interpretation) -> str:
    lines = ["VÝSLEDEK INTERPRETACE IR SPEKTRA", "=" * 34, "", interp.summary, ""]
    lines.append("Postup (rozhodovací schéma):")
    for i, s in enumerate(interp.steps, 1):
        lines.append(f"  {i}. {s}")
    lines.append("")
    lines.append("Nalezené pásy:")
    lines.append(f"{'vlnočet':>9} {'int.':>6}  přiřazení")
    for p in interp.peaks:
        ass = "; ".join(interp.assignments.get(p.wavenumber, [])) or "–"
        inten = p.intensity + (", br" if p.broad else "")
        lines.append(f"{p.wavenumber:9.0f} {inten:>6}  {ass}")
    lines.append("")
    lines.append("Funkční skupiny (dohledání všech charakteristických pásů):")
    for r in interp.groups:
        if r.status in ("potvrzeno", "možné"):
            found = ", ".join(f"{m.peak.wavenumber:.0f}" for m in r.matches if m.peak)
            miss = ", ".join(b.range_str() for b in r.missing)
            lines.append(f"  [{r.status}] {r.group.name} ({r.group.formula}) – nalezeno: "
                         f"{found or '–'}" + (f"; chybí: {miss}" if miss else ""))
    lines.append("")
    lines.append("Vyloučené skupiny:")
    for r in interp.groups:
        if r.status == "vyloučeno":
            lines.append(f"  {r.group.name}: {r.reason}")
    if interp.branching or interp.notes:
        lines.append("")
        lines.append("Komentář:")
        for n in interp.notes:
            lines.append("  • " + n)
        if interp.branching:
            lines.append("  • " + interp.branching)
    if interp.references:
        lines.append("")
        lines.append("Podobnost s referenčními látkami stejné třídy (orientační knihovna):")
        for name, sc, formula in interp.references:
            lines.append(f"  {sc * 100:5.0f} %  {name} ({formula})")
    lines.append("")
    lines.append("Pozn.: Automatická interpretace podle prezentace „Interpretace infračervených "
                 "spekter“ (ÚACh VŠCHT) – výsledek je třeba ověřit.")
    return "\n".join(lines)
