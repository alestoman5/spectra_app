"""Knowledge base of characteristic IR bands.

Band tables follow the CACH II presentation *Interpretace infračervených spekter*
(ÚACh VŠCHT Praha), slides "Alkany" … "Nitrosloučeniny", which are themselves based on
"Tabulka I" of the LACH I guide. Every class lists its characteristic bands; ``key`` bands are
the ones the decision scheme (slide "Rozhodovací schema") and the "rychlokuchařka" use to decide
whether the class is present.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    lo: float               # lower wavenumber limit (cm-1)
    hi: float               # upper wavenumber limit (cm-1)
    vib: str                # vibration label, e.g. "ν(C=O)"
    intensity: str = "m"    # expected intensity: s / m / w / v (variable)
    key: bool = False       # decisive band of the class
    broad: bool = False     # expected to be broad
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
    name: str            # Czech name shown in the report
    formula: str         # structural fragment
    bands: tuple[Band, ...]
    note: str = ""


def B(hi, lo, vib, intensity="m", key=False, broad=False, weight=1.0, note=""):
    """Helper taking the range in the order used on the slides (high–low)."""
    return Band(lo=min(lo, hi), hi=max(lo, hi), vib=vib, intensity=intensity, key=key,
                broad=broad, weight=weight, note=note)


GROUPS: tuple[Group, ...] = (
    # ------------------------------------------------------------ slide "Alkany"
    Group("ch3", "methylová skupina", "–CH3", (
        B(2995, 2950, "νas(CH3)", "s", key=True, weight=2),
        B(2895, 2840, "νs(CH3)", "m"),
        B(1475, 1435, "δas(CH3)", "m"),
        B(1395, 1335, "δs(CH3) deštníková", "m", key=True, weight=1.5,
          note="dublet → rozvětvení (isopropyl/terc-butyl)"),
    )),
    Group("ch2", "methylenová skupina", "–CH2–", (
        B(2940, 2915, "νas(CH2)", "s", key=True, weight=2),
        B(2870, 2840, "νs(CH2)", "m", key=True),
        B(1480, 1440, "δ(CH2) nůžková", "m"),
        B(785, 720, "ρ(CH2) kolébavá", "w", note="intenzivnější s delším řetězcem"),
    )),
    # ------------------------------------------------------------ slide "Alkeny"
    Group("alkene", "alken", ">C=C<", (
        B(3100, 3010, "ν(=C–H) (jeden pás)", "m", key=True, weight=1.5),
        B(1985, 1800, "2×δ(CH) overton", "w", weight=0.5),
        B(1680, 1620, "ν(C=C)", "m", key=True, weight=2),
        B(1425, 1405, "δ(=CH2)", "w", weight=0.5),
        B(1010, 880, "γ(=C–H) mimorovinná", "s", key=True, weight=1.5),
    ), note="terminální: 1010–940 + 980–885 (s); trans 980–955 (s); cis 730–650 (s)"),
    # ------------------------------------------------------------ slide "Alkyny"
    Group("alkyne", "alkyn", "–C≡C–", (
        B(3340, 3260, "ν(≡C–H) terminální alkyn", "s", key=True, weight=2),
        B(2260, 2100, "ν(C≡C) (u symetrických molekul chybí)", "w", key=True, weight=2),
        B(1375, 1225, "2×ω(C–H)", "w", weight=0.5),
        B(730, 575, "δ(≡C–H), typicky 630", "s", weight=1.5),
    )),
    # ------------------------------------------------------------ slide "Areny"
    Group("arene", "aromatické jádro (aren)", "Ar", (
        B(3105, 3000, "ν(C–H) Ar – skupina pásů", "m", key=True, weight=2),
        B(2000, 1650, "overtony + kombinační přechody", "w", weight=1),
        B(1625, 1575, "ν(C=C) Ar", "v", key=True, weight=1.5),
        B(1525, 1470, "ν(C=C) Ar", "v", key=True, weight=1.5),
        B(1470, 1430, "ν(C=C) Ar", "v", weight=0.5),
        B(900, 670, "γ(CH) – charakterizuje substituci", "s", weight=1.5),
    )),
    # ------------------------------------------------------------ slide "Alkoholy"
    Group("alcohol", "alkohol / fenol", "–OH", (
        B(3550, 3230, "ν(O–H) široký pás", "s", key=True, broad=True, weight=3),
        B(1440, 1260, "δ(COH) (často překrytý)", "w", weight=0.5),
        B(1260, 1020, "ν(C–O) – určuje typ alkoholu", "s", key=True, weight=2),
    ), note="Ar-OH 1260–1180, 3° 1205–1125, 2° 1125–1085, 1° 1085–1030 cm⁻¹"),
    # ------------------------------------------------------------ slide "Aldehydy"
    Group("aldehyde", "aldehyd", "–CHO", (
        B(3470, 3320, "2×ν(C=O) overton", "w", weight=0.3),
        B(2830, 2800, "ν(C–H) Fermiho dublet", "m", key=True, weight=1.5),
        B(2745, 2650, "ν(C–H) Fermiho dublet", "m", key=True, weight=2),
        B(1740, 1685, "ν(C=O)", "s", key=True, weight=3),
        B(1450, 1325, "δ(C–H) CHO", "m", weight=0.7),
        B(975, 780, "γ(C–H)", "w", weight=0.3),
    )),
    # ------------------------------------------------------------ slide "Ketony"
    Group("ketone", "keton", ">C=O", (
        B(3500, 3260, "2×ν(C=O) overton", "w", weight=0.3),
        B(1745, 1660, "ν(C=O)", "s", key=True, weight=3),
        B(1325, 1075, "ν(CC) / νas(CCC)", "m", weight=1),
    ), note="určuje se vylučovací metodou (nejde o kyselinu, ester ani aldehyd)"),
    # ------------------------------------------------------------ slide "Estery"
    Group("ester", "ester", "–CO–O–", (
        B(3470, 3420, "2×ν(C=O) overton (~3450)", "w", weight=0.3),
        B(1800, 1715, "ν(C=O)", "s", key=True, weight=3),
        B(1310, 1185, "νas(C–O–C) – velmi silný, obvykle silnější než C=O", "s", key=True,
          weight=2.5),
        B(1160, 1050, "νs(C–O–C)", "s", key=True, weight=1.5),
    ), note="R–CO–O–R': 1275–1185 + 1160–1050; Ar–CO–O–R: 1310–1250 + 1150–1100"),
    # ------------------------------------------------------------ slide "Karboxylové kyseliny"
    Group("acid", "karboxylová kyselina", "–COOH", (
        B(3300, 2500, "ν(O–H) velmi široký, zasahuje pod 3000", "m", key=True, broad=True,
          weight=3),
        B(2700, 2500, "kombinační přechody", "m", weight=1),
        B(1740, 1660, "ν(C=O)", "s", key=True, weight=3),
        B(1440, 1395, "ν(C–O) + δ(O–H)", "w", weight=0.5),
        B(1320, 1210, "ν(C–O)", "s", weight=1.5),
        B(970, 875, "γ(O–H) mimorovinná", "s", weight=1.5),
    )),
    # ------------------------------------------------------------ slide "Ethery"
    Group("ether", "ether", "–O–", (
        B(1310, 1210, "νas(C–O–C) Ar–O–R", "s", weight=1.5),
        B(1155, 1020, "νas/νs(C–O–C) (R–O–R' 1155–1060, Ar–O–R 1120–1020)", "s", key=True,
          weight=2.5),
    ), note="jen bez C=O a bez O–H"),
    # ------------------------------------------------------------ slide "Nitrily"
    Group("nitrile", "nitril", "–C≡N", (
        B(2270, 2200, "ν(C≡N) kyanidová jehla", "m", key=True, weight=3),
    )),
    # ------------------------------------------------------------ slide "Aminy"
    Group("amine1", "primární amin", "–NH2", (
        B(3550, 3330, "νas(NH2)", "m", key=True, weight=2),
        B(3450, 3250, "νs(NH2)", "m", key=True, weight=2),
        B(1650, 1580, "δ(NH2) nůžková", "m", weight=1.5),
        B(1240, 1020, "ν(C–N) (Ar: 1360–1250)", "m", weight=0.5),
        B(895, 650, "ω(NH2) vějířová", "m", broad=True, weight=1),
    ), note="primární amin – 2 pásy ~100 cm⁻¹ od sebe"),
    Group("amine2", "sekundární amin", "–NH–", (
        B(3500, 3300, "ν(N–H) jeden pás", "w", key=True, weight=2),
        B(1240, 1020, "ν(C–N)", "m", weight=0.5),
        B(895, 650, "ω(N–H) vějířová", "m", broad=True, weight=1),
    )),
    # ------------------------------------------------------------ slide "Nitrosloučeniny"
    Group("nitro", "nitrosloučenina", "–NO2", (
        B(1570, 1485, "νas(NO2)", "s", key=True, weight=2),
        B(1385, 1320, "νs(NO2)", "s", key=True, weight=2),
    ), note="k prokázání stačí dva výše uvedené pásy"),
    # ------------------------------------------------------------ outside the scheme
    Group("amide", "amid", "–CO–NH–", (
        B(3500, 3150, "ν(N–H)", "m", key=True, weight=1.5),
        B(1700, 1630, "ν(C=O) amid I", "s", key=True, weight=3),
        B(1650, 1510, "δ(N–H) amid II", "m", weight=1.5),
    ), note="mimo rozhodovací schéma"),
    Group("anhydride", "anhydrid karboxylové kyseliny", "–CO–O–CO–", (
        B(1850, 1780, "νas(C=O)", "s", key=True, weight=3),
        B(1790, 1740, "νs(C=O) (odstup ~60 cm⁻¹)", "s", key=True, weight=3),
    ), note="mimo rozhodovací schéma"),
    Group("water", "voda (vlhkost)", "H2O", (
        B(3600, 3100, "ν(H2O)", "m", key=True, broad=True, weight=2),
        B(1650, 1600, "δ(H2O)", "w", key=True, weight=2),
    ), note="často jen nečistota (vlhkost v KBr / vzorku)"),
)

GROUP_BY_ID = {g.id: g for g in GROUPS}

# γ(CH) substitution patterns of the benzene ring (slide "Areny"); ordered from the most
# specific pattern.
AROMATIC_SUBSTITUTION = (
    ("monosubstituovaný benzen", ((820, 720), (710, 670))),
    ("meta-disubstituovaný benzen", ((820, 765), (710, 680), (880, 830))),
    ("ortho-disubstituovaný benzen", ((790, 720),)),
    ("para-disubstituovaný benzen", ((860, 780),)),
)

# Artefacts that should not be interpreted.
ARTEFACTS = (
    (2390, 2280, "CO2 z atmosféry (dublet 2360/2335 cm-1) – artefakt pozadí"),
)

# main classes of the decision tree → human readable Czech names
CLASS_NAMES = {
    "acid": "karboxylová kyselina", "ester": "ester", "aldehyde": "aldehyd", "ketone": "keton",
    "amide": "amid", "anhydride": "anhydrid karboxylové kyseliny",
    "alcohol": "alkohol", "phenol": "fenol", "amine": "amin", "nitrile": "nitril",
    "alkyne": "alkyn", "nitro": "nitrosloučenina", "ether": "ether",
    "arene": "aromatický uhlovodík (aren)", "alkene": "alken", "alkane": "alkan",
    "unknown": "neurčeno",
}
