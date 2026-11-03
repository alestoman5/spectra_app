"""Interpretation of detected bands: functional groups, compound class, reference matching."""
from __future__ import annotations

from dataclasses import dataclass, field

from .ir_table import (AROMATIC_SUBSTITUTION, CARBONYL_GROUPS, GROUP_BY_ID, GROUPS, Band,
                       Group)
from .references import REFERENCES
from .signal import Peak

TOL = 12.0  # cm-1 tolerance added to the table ranges (resolution + digitisation error)

INTENSITY_FACTOR = {
    # expected -> {found: factor}
    "s": {"s": 1.0, "m": 0.75, "w": 0.35},
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
    score: float                       # 0..1
    matches: list[BandMatch]
    status: str = "nepřítomno"         # potvrzeno / možné / vyloučeno / nepřítomno
    reason: str = ""

    @property
    def missing_keys(self) -> list[Band]:
        return [m.band for m in self.matches if m.band.key and m.peak is None]


@dataclass
class Interpretation:
    peaks: list[Peak]
    groups: list[GroupResult]
    assignments: dict[float, list[str]] = field(default_factory=dict)
    substitution: list[str] = field(default_factory=list)
    branching: str = ""
    summary: str = ""
    notes: list[str] = field(default_factory=list)
    references: list[tuple[str, float, str]] = field(default_factory=list)

    @property
    def confirmed(self) -> list[GroupResult]:
        return [g for g in self.groups if g.status == "potvrzeno"]

    @property
    def possible(self) -> list[GroupResult]:
        return [g for g in self.groups if g.status == "možné"]


def _usable(peaks: list[Peak]) -> list[Peak]:
    return [p for p in peaks if not p.artefact]


def _best_peak(band: Band, peaks: list[Peak], used: set[float] = frozenset()
               ) -> tuple[Peak | None, float]:
    best, best_f = None, 0.0
    for p in peaks:
        if p.wavenumber in used:
            continue
        if band.lo - TOL <= p.wavenumber <= band.hi + TOL:
            f = INTENSITY_FACTOR.get(band.intensity, INTENSITY_FACTOR["m"])[p.intensity]
            if band.strict and "wms".index(p.intensity) < "wms".index(
                    band.intensity if band.intensity in "wms" else "w"):
                continue
            if band.min_width and p.width < band.min_width:
                continue
            if band.broad and not p.broad:
                f *= 0.35
            if band.sharp and p.broad:
                f *= 0.25
            # slight penalty when only inside the tolerance margin
            if not band.lo <= p.wavenumber <= band.hi:
                f *= 0.85
            if f > best_f:
                best, best_f = p, f
    return best, best_f


def score_group(group: Group, peaks: list[Peak]) -> GroupResult:
    matches, total, got = [], 0.0, 0.0
    used: set[float] = set()
    # key bands first so that each peak explains at most one band of the group
    for band in sorted(group.bands, key=lambda b: (not b.key, -b.weight)):
        p, f = _best_peak(band, peaks, used)
        # a key band found only with a very poor factor (e.g. sharp band where broad expected)
        # counts as missing
        if band.key and p is not None and f < 0.4:
            p, f = None, 0.0
        if p is not None:
            used.add(p.wavenumber)
        matches.append(BandMatch(band, p, f))
        total += band.weight
        got += band.weight * f
    score = got / total if total else 0.0
    matches.sort(key=lambda m: -m.band.hi)
    return GroupResult(group, score, matches)


def _has(peaks: list[Peak], lo: float, hi: float, min_h: float = 0.0) -> list[Peak]:
    return [p for p in peaks if lo <= p.wavenumber <= hi and p.height >= min_h]


def interpret(peaks: list[Peak], x_range: tuple[float, float] | None = None) -> Interpretation:
    usable = _usable(peaks)
    xmin, xmax = x_range if x_range else (400.0, 4000.0)
    results = {g.id: score_group(g, usable) for g in GROUPS}

    # 1) status from key bands + score
    for r in results.values():
        keys_out_of_range = [b for b in r.missing_keys if b.hi < xmin or b.lo > xmax]
        missing = [b for b in r.missing_keys if b not in keys_out_of_range]
        if missing:
            r.status = "nepřítomno"
            r.reason = "chybí pás " + ", ".join(f"{b.vib} ({b.range_str()} cm⁻¹)" for b in missing)
        elif r.score >= 0.55:
            r.status = "potvrzeno"
        elif r.score >= 0.35:
            r.status = "možné"
        else:
            r.status = "nepřítomno"
            r.reason = "nízká shoda doplňkových pásů"

    # 2) requirements
    for r in results.values():
        for req in r.group.requires:
            if results[req].status not in ("potvrzeno", "možné") and r.status != "nepřítomno":
                r.status = "nepřítomno"
                r.reason = f"chybí skupina {GROUP_BY_ID[req].name}"

    # 3) exclusions – process groups from the most specific / highest priority
    order = sorted(results.values(), key=lambda r: (-r.group.priority, -r.score))
    for r in order:
        if r.status == "nepřítomno":
            continue
        for other_id in r.group.excluded_by:
            o = results[other_id]
            if o.status == "potvrzeno" or (o.status == "možné" and r.status == "možné"
                                            and o.score >= r.score):
                r.status = "vyloučeno"
                r.reason = f"pásy lépe vysvětluje skupina: {o.group.name}"
                break

    # 4) only one main carbonyl class – keep the best scoring one, demote others
    carb = [results[c] for c in CARBONYL_GROUPS if results[c].status == "potvrzeno"]
    if len(carb) > 1:
        carb.sort(key=lambda r: (-(r.score + 0.002 * r.group.priority)))
        for r in carb[1:]:
            r.status = "možné"
            r.reason = f"alternativa k: {carb[0].group.name}"

    interp = Interpretation(peaks=peaks,
                            groups=sorted(results.values(),
                                          key=lambda r: ({"potvrzeno": 0, "možné": 1,
                                                          "vyloučeno": 2, "nepřítomno": 3}[r.status],
                                                         -r.score)))
    _assign_peaks(interp)
    _special_rules(interp, usable, results, xmin, xmax)
    interp.references = match_references(peaks, xmin, xmax)
    interp.summary = _summary(interp, results)
    return interp


def _assign_peaks(interp: Interpretation) -> None:
    for r in interp.groups:
        if r.status not in ("potvrzeno", "možné"):
            continue
        for m in r.matches:
            if m.peak is None:
                continue
            tag = f"{m.band.vib} – {r.group.formula}"
            if r.status == "možné":
                tag += " (?)"
            targets = [m.peak]
            if r.status == "potvrzeno" and m.band.hi - m.band.lo <= 400:
                # several bands of one type (e.g. aromatic C–H, overtone pattern)
                targets += [p for p in interp.peaks if m.band.lo <= p.wavenumber <= m.band.hi
                            and p is not m.peak and not p.artefact
                            and not interp.assignments.get(p.wavenumber)]
            for pk in targets:
                lst = interp.assignments.setdefault(pk.wavenumber, [])
                if tag not in lst:
                    lst.append(tag)
    for p in interp.peaks:
        if p.artefact:
            interp.assignments[p.wavenumber] = [p.artefact]


def _special_rules(interp: Interpretation, peaks: list[Peak], res: dict[str, GroupResult],
                   xmin: float, xmax: float) -> None:
    notes = interp.notes
    above = _has(peaks, 3000, 3110, 0.03)
    below = _has(peaks, 2840, 2999, 0.05)
    if below and not above:
        notes.append("Pásy ν(C–H) pouze pod 3000 cm⁻¹ → nasycený (alifatický) skelet, "
                     "aromát a alken s =C–H lze vyloučit.")
    elif above and below:
        notes.append("Pásy ν(C–H) nad i pod 3000 cm⁻¹ → nenasycené/aromatické =C–H "
                     "i alifatické C–H.")
    elif above:
        notes.append("Pásy ν(C–H) pouze nad 3000 cm⁻¹ → jen nenasycené/aromatické C–H.")
    co = _has(peaks, 1650, 1850, 0.3)
    if not co:
        notes.append("Chybí intenzivní pás v oblasti 1850–1650 cm⁻¹ → karbonylová skupina "
                     "(C=O) není přítomna.")
    else:
        strongest = max(co, key=lambda p: p.height)
        notes.append(f"Intenzivní pás {strongest.wavenumber:.0f} cm⁻¹ → karbonylová skupina C=O.")
    oh = [p for p in _has(peaks, 3100, 3650, 0.1) if p.broad]
    if not oh and xmax >= 3500:
        notes.append("Chybí široký pás 3650–3100 cm⁻¹ → skupiny O–H (alkohol, voda) "
                     "nejsou výrazně zastoupeny.")

    def assign(pk: Peak, text: str) -> None:
        lst = interp.assignments.setdefault(pk.wavenumber, [])
        if text not in lst:
            lst.append(text)

    # CH3 doublet → branching (Tabulka I, text u 1390–1335 cm-1)
    if res["ch3"].status == "potvrzeno" and res["ester"].status != "potvrzeno":
        d = sorted(_has(peaks, 1355, 1395, 0.03), key=lambda p: -p.wavenumber)
        if len(d) >= 2 and abs(d[0].wavenumber - d[1].wavenumber) <= 25:
            hi_p, lo_p = d[0], d[1]
            ratio = lo_p.height / max(hi_p.height, 1e-9)
            if ratio > 1.6:
                interp.branching = (f"Dublet δs(CH3) {hi_p.wavenumber:.0f}/{lo_p.wavenumber:.0f} cm⁻¹, "
                                    "složka s nižším vlnočtem ~2× intenzivnější → –C(CH3)3 (terc-butyl).")
            else:
                interp.branching = (f"Dublet δs(CH3) {hi_p.wavenumber:.0f}/{lo_p.wavenumber:.0f} cm⁻¹ "
                                    "→ rozvětvení >C(CH3)2 (isopropyl / gem-dimethyl).")
        elif d:
            interp.branching = (f"δs(CH3) {d[0].wavenumber:.0f} cm⁻¹ je singlet → skupina CH3–C– "
                                "nebo CH3–O–, rozvětvení >C(CH3)2 / –C(CH3)3 vyloučeno.")

    if res["aromatic"].status in ("potvrzeno", "možné"):
        for name, ranges in AROMATIC_SUBSTITUTION:
            found = [_has(peaks, lo - 5, hi + 5, 0.15) for hi, lo in ranges]
            if all(found):
                interp.substitution.append(name)
                for f in found:
                    assign(max(f, key=lambda p: p.height), f"γ(CH) – {name}")
                break  # table is ordered from the most specific pattern

    if res["ester"].status == "potvrzeno":
        ac = _has(peaks, 1355, 1385, 0.15)
        if ac and _has(peaks, 1210, 1265, 0.3):
            for pk in ac:
                assign(pk, "δs(CH3) – CH3–(C=O)–O– (acetát)")
            notes.append("Pás δs(CH3) ~1370 cm⁻¹ u CH3–(C=O)–O– spolu s νas(C–O–C) ~1240 cm⁻¹ "
                         "→ ester kyseliny octové (acetát).")
        c = max(_has(peaks, 1700, 1800, 0.3), key=lambda p: p.height, default=None)
        if c is not None:
            if c.wavenumber >= 1750:
                notes.append("ν(C=O) ≥ 1750 cm⁻¹ → spíše fenyl- nebo vinyl-ester.")
            elif c.wavenumber >= 1732:
                notes.append("ν(C=O) 1750–1730 cm⁻¹ → nasycený alifatický ester.")
            else:
                notes.append("ν(C=O) pod 1730 cm⁻¹ → konjugovaný (aromatický nebo α,β-nenasycený) ester.")
    if res["ketone"].status == "potvrzeno":
        c = max(_has(peaks, 1650, 1760, 0.3), key=lambda p: p.height, default=None)
        if c is not None:
            notes.append("ν(C=O) %.0f cm⁻¹ → %s." % (
                c.wavenumber, "nasycený / cyklický keton" if c.wavenumber >= 1705
                else "aryl-keton nebo α,β-nenasycený keton"))
    if res["alcohol"].status == "potvrzeno":
        co = max(_has(peaks, 1000, 1260, 0.25), key=lambda p: p.height, default=None)
        if co is not None:
            w = co.wavenumber
            if w >= 1180 and res["aromatic"].status == "potvrzeno":
                kind = "fenol (Ar–OH)"
            elif w >= 1150:
                kind = "terciární alkohol (R3C–OH)"
            elif w > 1090:
                kind = "sekundární alkohol (R2CH–OH)"
            else:
                kind = "primární alkohol (RCH2–OH)"
            notes.append(f"ν(C–O) {w:.0f} cm⁻¹ → {kind}.")


ORDER_NAMES = {
    "acid": "karboxylová kyselina", "anhydride": "anhydrid karboxylové kyseliny",
    "carboxylate": "sůl karboxylové kyseliny (karboxylát)", "amide1": "primární amid",
    "amide2": "sekundární amid", "amide3": "terciární amid", "aldehyde": "aldehyd",
    "ester": "ester", "ketone": "keton", "alcohol": "alkohol", "amine1": "primární amin",
    "amine2": "sekundární amin", "nitro": "nitrosloučenina", "isocyanate": "isokyanát",
    "nitrile": "nitril", "ether": "ether", "thiol": "thiol", "alkyne_t": "terminální alkyn",
    "alkyne_i": "alkyn", "alkene": "alken", "aromatic": "aromatický uhlovodík",
    "carbonate": "uhličitan", "sulfate": "síran", "nitrate": "dusičnan",
    "phosphate": "fosforečnan / křemičitan", "ch2": "alkan", "ch3": "alkan",
}


def _summary(interp: Interpretation, res: dict[str, GroupResult]) -> str:
    conf = [r for r in interp.confirmed if r.group.id != "water"]
    if not conf:
        if interp.possible:
            return ("Žádná funkční skupina nebyla jednoznačně potvrzena; možné jsou: " +
                    ", ".join(r.group.name for r in interp.possible) + ".")
        return "Ve spektru nebyly nalezeny charakteristické pásy – zkontrolujte kalibraci a ořez."
    main = max(conf, key=lambda r: r.group.priority)
    aromatic = res["aromatic"].status == "potvrzeno"
    aliph = res["ch2"].status == "potvrzeno" or res["ch3"].status == "potvrzeno"
    name = ORDER_NAMES.get(main.group.id, main.group.name)
    if main.group.id in ("ch2", "ch3"):
        name = "nasycený alifatický uhlovodík (alkan)"
        if res["ch2n"].status == "potvrzeno":
            name += " s nerozvětveným řetězcem –(CH2)n–, n > 3"
    elif main.group.id == "aromatic":
        name = "aromatický uhlovodík"
        if aliph:
            name += " s alifatickou částí (–CH2–/–CH3)"
    elif main.group.id in ("carbonate", "sulfate", "nitrate", "phosphate"):
        name = "anorganická látka – " + name
    else:
        fem = name.split()[-1].endswith("a") or name.startswith("sůl")
        if name.startswith("sůl"):
            if aromatic:
                name += " – aromatická"
        elif aromatic:
            name = ("aromatická " if fem else "aromatický ") + name
        elif aliph:
            name = ("alifatická " if fem else "alifatický ") + name
    if main.group.id == "ester" and any("acetát" in n for n in interp.notes):
        name += " – acetát (ester kyseliny octové)"
    extra = [r.group.name for r in conf if r is not main and r.group.id not in
             ("ch2", "ch3", "ch2n", "aromatic")]
    s = f"Pravděpodobně: {name}."
    if interp.substitution:
        s += f" Typ substituce jádra: {interp.substitution[0]}."
    if extra:
        s += " Další skupiny: " + ", ".join(extra) + "."
    if interp.references:
        top = interp.references[0]
        if top[1] >= 0.5:
            s += f" Nejpodobnější referenční látka: {top[0]} (shoda {top[1]*100:.0f} %)."
    return s


def match_references(peaks: list[Peak], xmin: float = 400, xmax: float = 4000,
                     tol: float = 15.0, top: int = 5) -> list[tuple[str, float, str]]:
    """Compare the band list with a small built-in library of typical compounds."""
    meas = [p for p in _usable(peaks) if p.height >= 0.08]
    strong = [p for p in meas if p.height >= 0.25 and not p.broad]
    if not meas:
        return []
    out = []
    for ref in REFERENCES:
        ref_peaks = [(w, s) for w, s in ref.peaks if xmin + 5 <= w <= xmax - 5]
        if len(ref_peaks) < 3:
            continue
        tot = sum(2.0 if s == "s" else 1.0 for _, s in ref_peaks)
        hit = 0.0
        for w, s in ref_peaks:
            t = tol * (4 if w > 2500 and ref.broad_oh and w > 2900 else 1)
            if any(abs(p.wavenumber - w) <= t for p in meas):
                hit += 2.0 if s == "s" else 1.0
        recall = hit / tot
        explained = sum(1 for p in strong if any(abs(p.wavenumber - w) <= tol * 1.5
                                                 for w, _ in ref.peaks))
        precision = explained / len(strong) if strong else 1.0
        score = recall * (0.35 + 0.65 * precision)
        out.append((ref.name, score, ref.formula))
    out.sort(key=lambda t: -t[1])
    return out[:top]


def report_text(interp: Interpretation) -> str:
    lines = ["VÝSLEDEK INTERPRETACE IR SPEKTRA", "=" * 34, "", interp.summary, ""]
    lines.append("Nalezené pásy:")
    lines.append(f"{'vlnočet':>9} {'int.':>5}  přiřazení")
    for p in interp.peaks:
        ass = "; ".join(interp.assignments.get(p.wavenumber, [])) or "–"
        inten = p.intensity + (", br" if p.broad else "")
        lines.append(f"{p.wavenumber:9.0f} {inten:>5}  {ass}")
    lines.append("")
    lines.append("Funkční skupiny:")
    for r in interp.groups:
        if r.status in ("potvrzeno", "možné"):
            found = ", ".join(f"{m.peak.wavenumber:.0f}" for m in r.matches if m.peak)
            lines.append(f"  [{r.status}] {r.group.name} ({r.group.formula}), shoda "
                         f"{r.score*100:.0f} % – pásy: {found}"
                         + (f" – {r.reason}" if r.reason else ""))
    lines.append("")
    lines.append("Vyloučené skupiny:")
    for r in interp.groups:
        if r.status == "vyloučeno" or (r.status == "nepřítomno" and r.group.id in ORDER_NAMES
                                         and r.group.id not in ("carbonate", "sulfate",
                                                                "nitrate", "phosphate")):
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
        lines.append("Podobnost s referenčními látkami (orientační knihovna):")
        for name, sc, formula in interp.references:
            lines.append(f"  {sc*100:5.0f} %  {name} ({formula})")
    lines.append("")
    lines.append("Pozn.: Automatická interpretace podle Tabulky I – výsledek je třeba ověřit; "
                 "přesnou identifikaci látky poskytne až porovnání s knihovnou spekter.")
    return "\n".join(lines)
