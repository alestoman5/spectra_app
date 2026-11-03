"""Knowledge base of characteristic IR bands.

Based on "Tabulka I. - Vlnočty charakteristických vibrací vybraných funkčních skupin"
from the VŠCHT LACH I guide *Infračervená spektroskopie* (P. Kania).

Every functional group is described by a list of bands. A band marked ``key``
must be found for the group to be confirmed (it corresponds to the first
characteristic band in the table and the most important "další charakteristický
pás"). Other bands only increase the confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Band:
    lo: float               # lower wavenumber limit (cm-1)
    hi: float               # upper wavenumber limit (cm-1)
    vib: str                # vibration label, e.g. "ν(C=O)"
    intensity: str = "m"    # expected intensity: s / m / w / v (variable)
    key: bool = False       # required for confirmation
    broad: bool = False     # band is expected to be broad
    sharp: bool = False     # band is expected to be narrow (penalise broad matches)
    min_width: float = 0.0  # minimal FWHM (cm-1) of the matching peak
    strict: bool = False    # the found peak must be at least as intense as expected
    weight: float = 1.0
    note: str = ""

    @property
    def center(self) -> float:
        return 0.5 * (self.lo + self.hi)

    def range_str(self) -> str:
        return f"{self.hi:.0f}–{self.lo:.0f}"


@dataclass(frozen=True)
class Group:
    id: str
    name: str                       # Czech name shown in the report
    formula: str                    # structural fragment
    bands: tuple[Band, ...]
    excluded_by: tuple[str, ...] = ()   # ids of groups which explain the same bands better
    requires: tuple[str, ...] = ()      # ids of groups which must be confirmed as well
    priority: int = 50                  # used to name the compound class (higher = main group)
    adjective: str = ""                 # used in the summary sentence
    note: str = ""


def B(hi, lo, vib, intensity="m", key=False, broad=False, weight=1.0, note="", sharp=False,
      min_width=0.0, strict=False):
    """Helper taking the range in the table order (high–low)."""
    return Band(lo=min(lo, hi), hi=max(lo, hi), vib=vib, intensity=intensity,
                key=key, broad=broad, weight=weight, note=note, sharp=sharp,
                min_width=min_width, strict=strict)


CARBONYL_GROUPS = ("ester", "acid", "aldehyde", "ketone", "amide1", "amide2", "amide3",
                   "anhydride", "carboxylate")

GROUPS: tuple[Group, ...] = (
    # ------------------------------------------------------------------ alkany
    Group("ch3", "methylová skupina", "–CH3", (
        B(2975, 2950, "νas(CH3)", "m", key=True, weight=2),
        B(2885, 2860, "νs(CH3)", "m"),
        B(1475, 1435, "δas(CH3)", "m"),
        B(1395, 1365, "δs(CH3) deštníková", "m", key=True, weight=1.5),
    ), priority=5, adjective="alifatický"),
    Group("ch2", "methylenová skupina", "–CH2–", (
        B(2940, 2915, "νas(CH2)", "m", key=True, weight=2),
        B(2870, 2840, "νs(CH2)", "m", key=True),
        B(1480, 1440, "δ(CH2) nůžková", "m"),
    ), priority=4, adjective="alifatický"),
    Group("ch2n", "dlouhý řetězec –(CH2)n–, n > 3", "–(CH2)n–", (
        B(728, 716, "ρ(CH2) kyvadlová", "w", key=True, weight=2),
        B(2940, 2915, "νas(CH2)", "m", key=True),
    ), requires=("ch2",), excluded_by=("aromatic",), priority=3,
        note="U aromátů se pás překrývá s γ(CH) jádra, proto se v jejich přítomnosti nehodnotí."),
    # ------------------------------------------------------------------ aromat
    Group("aromatic", "aromatické jádro", "Ar", (
        B(3105, 3000, "ν(=C–H) Ar", "w", key=True, weight=2),
        B(2000, 1650, "kombinační a svrchní pásy γ(CH)", "w", weight=1.0),
        B(1625, 1575, "ν(C=C) Ar", "v", key=True, weight=2),
        B(1525, 1470, "ν(C=C) Ar", "v", weight=1.5),
        B(900, 670, "γ(CH) mimorovinná", "s", weight=1.5),
    ), priority=20, adjective="aromatický"),
    # ------------------------------------------------------------------ alken
    Group("alkene", "alken (C=C)", "–C=C–", (
        B(3100, 3010, "ν(=C–H)", "m", key=True, weight=1.5),
        B(1680, 1620, "ν(C=C) izolovaná", "m", key=True, weight=2),
        B(1000, 880, "γ(=C–H)", "s", weight=1.5),
        B(1420, 1405, "δ(=CH2) nůžková", "w", weight=0.5),
    ), excluded_by=("aromatic",), priority=15, adjective="nenasycený"),
    # ------------------------------------------------------------------ alkyn
    Group("alkyne_t", "terminální alkyn", "–C≡C–H", (
        B(3340, 3270, "ν(≡C–H) ostrý pás", "s", key=True, weight=2, sharp=True),
        B(2150, 2100, "ν(C≡C)", "w", key=True, weight=2),
        B(730, 575, "δ(≡C–H)", "s"),
    ), priority=25, adjective="nenasycený"),
    Group("alkyne_i", "vnitřní alkyn", "R–C≡C–R'", (
        B(2260, 2190, "ν(C≡C) (u symetrických molekul chybí)", "w", key=True, weight=2),
    ), excluded_by=("nitrile", "isocyanate", "alkyne_t"), priority=24, adjective="nenasycený"),
    # ------------------------------------------------------------------ N, S, trojné
    Group("nitrile", "nitril", "–C≡N", (
        B(2260, 2200, "ν(C≡N) úzký pás", "m", key=True, weight=3),
    ), priority=55),
    Group("isocyanate", "isokyanát", "–N=C=O", (
        B(2300, 2250, "νas(NCO)", "s", key=True, weight=3),
        B(1460, 1340, "νs(NCO)", "m"),
    ), priority=56),
    Group("thiol", "thiol", "–SH", (
        B(2600, 2520, "ν(S–H)", "w", key=True, weight=2),
        B(750, 585, "ν(C–S)", "w"),
    ), excluded_by=("acid",), priority=35),
    Group("nitro", "nitrosloučenina", "–NO2", (
        B(1570, 1485, "νas(NO2)", "s", key=True, weight=2),
        B(1385, 1315, "νs(NO2)", "s", key=True, weight=2),
        B(1000, 850, "ν(C–N)", "w"),
    ), priority=60),
    # ------------------------------------------------------------------ O–H
    Group("alcohol", "alkohol / fenol", "–OH", (
        B(3550, 3200, "ν(O–H) intermolekulární H-vazba", "s", key=True, broad=True, weight=3),
        B(1440, 1260, "δ(C–O–H) rovinná", "m", weight=0.7),
        B(1260, 1000, "ν(C–O)", "s", key=True, weight=2),
    ), excluded_by=("acid", "amine1", "amide1", "water"), priority=70, adjective=""),
    Group("water", "voda (vlhkost / krystalová voda)", "H2O", (
        B(3600, 3100, "ν(H2O)", "m", key=True, broad=True, weight=2),
        B(1650, 1600, "δ(H2O)", "w", key=True, weight=2),
    ), priority=1, note="Často jen nečistota (vlhkost v KBr / vzorku)."),
    Group("acid", "karboxylová kyselina", "–COOH", (
        B(3300, 2500, "ν(O–H) dimer, velmi široký", "m", key=True, broad=True, weight=3,
          min_width=150),
        B(2700, 2500, "kombinační pásy dimeru", "w", weight=1),
        B(1740, 1660, "ν(C=O)", "s", key=True, weight=3),
        B(1440, 1395, "ν(C–O)+δ(O–H)", "w", weight=0.7),
        B(1320, 1210, "ν(C–O)", "s", weight=1.5),
        B(970, 875, "γ(O–H···O) dimeru", "m", weight=1.5),
    ), priority=100, adjective=""),
    # ------------------------------------------------------------------ amines / amides
    Group("amine1", "primární amin", "–NH2", (
        B(3550, 3330, "νas(NH2)", "m", key=True, weight=2, sharp=True),
        B(3450, 3250, "νs(NH2)", "m", key=True, weight=2, sharp=True),
        B(1650, 1580, "δ(NH2) nůžková", "m", weight=1.5),
        B(1360, 1020, "ν(C–N)", "m"),
        B(895, 650, "ω(NH2) vějířová", "m", broad=True, weight=0.7),
    ), excluded_by=("amide1",), priority=65),
    Group("amine2", "sekundární amin", "–NH–", (
        B(3500, 3300, "ν(N–H) jeden pás", "w", key=True, weight=2, sharp=True),
        B(1580, 1490, "δ(N–H)", "w"),
        B(1360, 1130, "ν(C–N)", "m"),
        B(750, 700, "ω(N–H)", "s", broad=True),
    ), excluded_by=("amine1", "amide2", "alcohol", "acid", "amide1", "ester", "ketone", "aldehyde"),
        priority=64,
        note="Slabý pás ~3450 cm⁻¹ u karbonylových sloučenin je 1. svrchní přechod C=O, ne N–H."),
    Group("amide1", "primární amid", "–CO–NH2", (
        B(3375, 3320, "νas(NH2) asociované", "m", key=True, weight=1.5, sharp=True),
        B(3205, 3155, "νs(NH2) asociované", "m", key=True, weight=1.5, sharp=True),
        B(1690, 1650, "ν(C=O) amid I", "s", key=True, weight=3),
        B(1650, 1590, "δ(NH2) amid II", "m", weight=1.5),
        B(1420, 1400, "ν(C–N) amid III", "m"),
    ), priority=90),
    Group("amide2", "sekundární amid", "–CO–NH–", (
        B(3370, 3270, "ν(N–H)", "m", key=True, weight=2, sharp=True),
        B(1700, 1630, "ν(C=O) amid I", "s", key=True, weight=3),
        B(1570, 1510, "δ(N–H) amid II", "s", key=True, weight=2),
        B(1305, 1200, "ν(C–N) amid III", "m"),
    ), excluded_by=("amide1",), priority=89),
    Group("amide3", "terciární amid", "–CO–N<", (
        B(1670, 1630, "ν(C=O)", "s", key=True, weight=3),
        B(870, 700, "νas(C–N–C)", "w", weight=0.5),
        B(620, 570, "δ(N–C=O)", "s", weight=1),
    ), excluded_by=("amide1", "amide2", "acid", "ketone", "alkene", "aromatic", "carboxylate"),
        priority=88),
    # ------------------------------------------------------------------ carbonyl compounds
    Group("aldehyde", "aldehyd", "–CHO", (
        B(2900, 2800, "ν(C–H) Fermiho dublet", "w", weight=1),
        B(2745, 2650, "ν(C–H) Fermiho dublet (obv. 2720)", "w", key=True, weight=2.5),
        B(1740, 1685, "ν(C=O)", "s", key=True, weight=3),
        B(1450, 1325, "δ(C–H)", "m", weight=0.7),
        B(975, 780, "γ(C–H)", "w", weight=0.5),
    ), excluded_by=("acid",), priority=85),
    Group("ester", "ester", "–CO–O–", (
        B(3470, 3420, "2ν(C=O) 1. svrchní přechod", "w", weight=0.5),
        B(1800, 1705, "ν(C=O)", "s", key=True, weight=3),
        B(1310, 1160, "νas(C–O–C)", "s", key=True, weight=2.5),
        B(1160, 1030, "νs(C–O–C)", "s", key=True, weight=1.5),
    ), excluded_by=("acid", "anhydride"), priority=80),
    Group("ketone", "keton", ">C=O", (
        B(3500, 3260, "2ν(C=O) 1. svrchní přechod", "w", weight=0.3),
        B(1745, 1660, "ν(C=O)", "s", key=True, weight=3),
        B(1325, 1075, "ν(C–C–C)", "m", weight=1),
    ), excluded_by=("ester", "acid", "aldehyde", "anhydride", "amide1", "amide2", "amide3"),
        priority=75),
    Group("anhydride", "anhydrid karboxylové kyseliny", "–CO–O–CO–", (
        B(1850, 1780, "νas(C=O)", "s", key=True, weight=3),
        B(1790, 1740, "νs(C=O) (odstup ~60 cm-1)", "s", key=True, weight=3),
        B(1300, 900, "ν(C–O–C)", "s"),
    ), priority=95),
    Group("carboxylate", "karboxylát (sůl karboxylové kyseliny)", "–COO⁻", (
        B(1695, 1540, "νas(COO⁻)", "s", key=True, weight=3),
        B(1440, 1335, "νs(COO⁻)", "s", key=True, broad=True, weight=2),
    ), excluded_by=("acid", "ester", "ketone", "aldehyde", "amide1", "amide2", "nitro", "aromatic"),
        priority=92),
    # ------------------------------------------------------------------ ethers
    Group("ether", "ether", "–O–", (
        B(1310, 1020, "νas(C–O–C) nejintenzivnější pás", "s", key=True, weight=3, strict=True),
        B(1140, 900, "νs(C–O–C)", "s", weight=1),
    ), excluded_by=("ester", "acid", "alcohol", "anhydride", "carboxylate", "aldehyde", "ketone",
                    "amide1", "amide2", "amide3", "nitro"), priority=40,
        note="Hodnotí se jen bez C=O a O–H – jinak pás C–O patří jiné skupině."),
    # ------------------------------------------------------------------ inorganic (orientační)
    Group("carbonate", "uhličitan (anorg., orientačně)", "CO3²⁻", (
        B(1500, 1380, "νas(CO3) velmi silný, široký", "s", key=True, broad=True, weight=3),
        B(890, 850, "γ(CO3)", "m", key=True, weight=2),
        B(720, 690, "δ(CO3)", "w"),
    ), excluded_by=("ch2", "ch3", "aromatic", "nitro"), priority=30),
    Group("sulfate", "síran (anorg., orientačně)", "SO4²⁻", (
        B(1200, 1040, "ν3(SO4) velmi silný, široký", "s", key=True, broad=True, weight=3),
        B(680, 580, "ν4(SO4)", "m", key=True, weight=2),
    ), excluded_by=("ch2", "ch3", "aromatic", "alcohol", "ester", "ether"), priority=30),
    Group("nitrate", "dusičnan (anorg., orientačně)", "NO3⁻", (
        B(1450, 1320, "ν3(NO3) velmi silný, široký", "s", key=True, broad=True, weight=3),
        B(840, 800, "ν2(NO3) ostrý", "w", key=True, weight=2),
    ), excluded_by=("ch2", "ch3", "aromatic", "carbonate", "nitro"), priority=30),
    Group("phosphate", "fosforečnan / křemičitan (anorg., orientačně)", "PO4³⁻ / SiO4⁴⁻", (
        B(1150, 950, "ν3 velmi silný, široký", "s", key=True, broad=True, weight=3),
        B(620, 440, "ν4", "m", weight=1.5),
    ), excluded_by=("ch2", "ch3", "aromatic", "alcohol", "ester", "ether", "sulfate"), priority=29),
)

GROUP_BY_ID = {g.id: g for g in GROUPS}

# Substitution patterns of the benzene ring (γ(CH) out-of-plane bands, Tabulka I).
AROMATIC_SUBSTITUTION = (
    # name, required ranges (all must have a strong/medium band)
    ("monosubstituovaný benzen (5 sousedních H)", ((770, 730), (710, 685))),
    ("1,2-disubstituovaný (ortho) benzen (4 H)", ((790, 735),)),
    ("1,3-disubstituovaný (meta) benzen", ((820, 765), (710, 680), (960, 860))),
    ("1,4-disubstituovaný (para) benzen (2 sousední H)", ((860, 790),)),
)

# Artefacts that should not be interpreted.
ARTEFACTS = (
    (2390, 2280, "CO2 z atmosféry (dublet 2360/2335 cm-1) – artefakt pozadí"),
)
