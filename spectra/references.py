"""Small library of approximate band positions of common laboratory compounds.

Values are typical literature maxima (liquid film / ATR, ±10 cm-1) and serve only as an
orientation aid – the main output of the app is the functional-group analysis.
Intensity flag: "s" = strong, "w" = weak, no suffix = medium.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    name: str
    formula: str
    peaks: tuple[tuple[float, str], ...]
    broad_oh: bool = False
    cls: str = "other"     # main class as returned by the decision tree (interpret.main_class)


def R(name, formula, spec, broad_oh=False):
    peaks = []
    for tok in spec.split():
        s = tok[-1] if tok[-1] in "sw" else "m"
        peaks.append((float(tok.rstrip("sw")), s))
    return Reference(name, formula, tuple(peaks), broad_oh, CLASS_OF.get(name, "other"))


CLASS_OF = {
    "polystyren": "arene", "n-hexan": "alkane", "parafínový olej / dlouhý n-alkan": "alkane",
    "cyklohexan": "alkane", "isooktan (2,2,4-trimethylpentan)": "alkane", "benzen": "arene",
    "toluen": "arene", "o-xylen": "arene", "p-xylen": "arene", "styren": "arene",
    "methanol": "alcohol", "ethanol": "alcohol", "propan-2-ol": "alcohol", "butan-1-ol": "alcohol",
    "fenol": "phenol", "aceton": "ketone", "butan-2-on": "ketone", "cyklohexanon": "ketone",
    "acetofenon": "ketone", "ethyl-acetát": "ester", "butyl-acetát": "ester",
    "methyl-benzoát": "ester", "methyl-methakrylát": "ester", "kyselina octová": "acid",
    "kyselina benzoová": "acid", "benzaldehyd": "aldehyde", "butanal": "aldehyde",
    "diethylether": "ether", "tetrahydrofuran": "ether", "anisol": "ether",
    "acetonitril": "nitrile", "benzonitril": "nitrile", "nitrobenzen": "nitro",
    "nitromethan": "nitro", "anilin": "amine", "N,N-dimethylformamid": "amide",
    "acetamid": "amide", "acetanhydrid": "anhydride", "hex-1-en": "alkene",
    "fenylacetylen": "alkyne",
    # compounds from the CACH II presentation
    "hexan-3-on": "ketone", "oktan": "alkane", "hept-1-en": "alkene", "pent-1-yn": "alkyne",
    "okt-4-yn": "alkane", "heptanal": "aldehyde", "ethyl-benzoát": "ester",
    "kyselina butanová": "acid", "nonan-1-ol": "alcohol", "nonan-2-ol": "alcohol",
    "nonan-2-on": "ketone", "ethoxybenzen": "ether", "butanamin": "amine",
}


REFERENCES = (
    R("polystyren", "(C8H8)n", "3082 3060 3026s 2923s 2849 1943 1871 1802 1601 1583 1493s 1452s 1028 757s 700s 540"),
    R("n-hexan", "C6H14", "2958s 2925s 2873 2859 1467 1379 727"),
    R("parafínový olej / dlouhý n-alkan", "CnH2n+2", "2954 2922s 2853s 1465 1377 721"),
    R("cyklohexan", "C6H12", "2927s 2852s 1449 1260 903 863"),
    R("isooktan (2,2,4-trimethylpentan)", "C8H18", "2955s 2905 2870 1468 1393 1366 1245 1202"),
    R("benzen", "C6H6", "3091 3071 3036 1960 1815 1478s 1036 674s"),
    R("toluen", "C7H8", "3087 3062 3028 2920 2873 1605 1496s 1460 1082 1030 729s 694s"),
    R("o-xylen", "C8H10", "3017 2943 1606 1495s 1466 1053 742s"),
    R("p-xylen", "C8H10", "3020 2920 1518s 1450 795s"),
    R("styren", "C8H8", "3083 3060 3027 1630 1600 1495 1450 990s 907s 776s 698s"),
    R("methanol", "CH3OH", "3330s 2945 2833 1450 1030s", True),
    R("ethanol", "C2H5OH", "3340s 2974s 2928 2881 1381 1330 1088s 1050s 880", True),
    R("propan-2-ol", "C3H7OH", "3350s 2972s 1380 1370 1160 1130s 951 817", True),
    R("butan-1-ol", "C4H9OH", "3330s 2959s 2933 2874 1465 1380 1073 1030s", True),
    R("fenol", "C6H5OH", "3350s 3040 1595s 1500s 1475 1360 1230s 810 755s 690s", True),
    R("aceton", "C3H6O", "3005w 2965w 1715s 1420 1362s 1222s 1092w 530"),
    R("butan-2-on", "C4H8O", "2980 1717s 1460 1365 1172s 945w 760w"),
    R("cyklohexanon", "C6H10O", "2938s 2862 1715s 1449 1312 1220 1118"),
    R("acetofenon", "C8H8O", "3062 1685s 1599 1449 1360 1266s 955 761s 691s"),
    R("ethyl-acetát", "C4H8O2", "2984 2940 1742s 1447 1373s 1243s 1048s 938 848"),
    R("butyl-acetát", "C6H12O2", "2962 2875 1741s 1466 1366s 1243s 1030s"),
    R("methyl-benzoát", "C8H8O2", "3064 2953 1724s 1602 1452 1435 1279s 1112s 711s"),
    R("methyl-methakrylát", "C5H8O2", "2995 2952 1724s 1638 1440 1324 1300 1166s 943 815"),
    R("kyselina octová", "CH3COOH", "3000s 1712s 1413 1294s 936 624", True),
    R("kyselina benzoová", "C6H5COOH", "3000s 2670 2560 1688s 1602 1454 1292s 935 708s", True),
    R("benzaldehyd", "C7H6O", "3063 2820 2737 1703s 1597 1584 1456 1311 1204s 827 746s 688s"),
    R("butanal", "C4H8O", "2965s 2878 2818 2720 1728s 1460 1390 1120"),
    R("diethylether", "C4H10O", "2977s 2935 2869 1445 1384 1123s 1060"),
    R("tetrahydrofuran", "C4H8O", "2976s 2861s 1460 1068s 909"),
    R("anisol", "C7H8O", "3003 2955 2836 1600 1498s 1247s 1041 753s 691"),
    R("acetonitril", "CH3CN", "3000 2943 2253s 1445 1376 1041 918"),
    R("benzonitril", "C7H5N", "3068 2229s 1599 1491 1447 758s 688s"),
    R("nitrobenzen", "C6H5NO2", "3080 1523s 1479 1347s 852 795 705s 681"),
    R("nitromethan", "CH3NO2", "1560s 1380s 918"),
    R("anilin", "C6H5NH2", "3433 3356 3215 1619s 1600s 1500s 1277s 753s 693s"),
    R("triethylamin", "C6H15N", "2970s 2934 2799 1462 1380 1210 1070"),
    R("N,N-dimethylformamid", "C3H7NO", "2930 2860 1675s 1500 1438 1388 1255 1092 659"),
    R("acetamid", "C2H5NO", "3350s 3180s 1670s 1620 1400 1150"),
    R("chloroform", "CHCl3", "3019 1216s 758s 669"),
    R("dichlormethan", "CH2Cl2", "3054 2987 1265s 739s 704s"),
    R("acetanhydrid", "C4H6O3", "2940 1825s 1754s 1371 1223 1125s 995"),
    R("hex-1-en", "C6H12", "3080 2960 2927s 1642 1460 993 910s"),
    R("fenylacetylen", "C8H6", "3292s 3060 2110 1489 1443 756s 691s 613"),
    R("dimethylsulfoxid", "C2H6OS", "2996 2913 1436 1407 1312 1043s 953 698"),
    R("uhličitan vápenatý", "CaCO3", "1420s 875s 712"),
    R("síran sodný", "Na2SO4", "1110s 617s"),
    R("dusičnan draselný", "KNO3", "1380s 825"),
    R("oxid křemičitý", "SiO2", "1080s 800 460s"),
    # ---- compounds shown in the CACH II presentation (approximate maxima)
    R("hexan-3-on", "C6H12O", "2970s 2940 2880 1717s 1460 1378 1110"),
    R("oktan", "C8H18", "2958s 2925s 2856 1467 1379 723"),
    R("hept-1-en", "C7H14", "3080 2960s 2927s 2858 1825w 1642 1460 993s 910s"),
    R("pent-1-yn", "C5H8", "3312s 2967 2120w 1462 1380 632s"),
    R("okt-4-yn", "C8H14", "2962s 2934s 2874 1462 1380"),
    R("heptanal", "C7H14O", "2930s 2860 2820 2715 1730s 1465 1410 1380 1120"),
    R("ethyl-benzoát", "C9H10O2", "3065w 2982 1720s 1603 1452 1368 1275s 1110s 1027 711s"),
    R("kyselina butanová", "C4H8O2", "3000s 2970 2670 1712s 1415 1285 1220 935", True),
    R("nonan-1-ol", "C9H20O", "3330s 2926s 2856s 1466 1378 1058s 722", True),
    R("nonan-2-ol", "C9H20O", "3340s 2960 2927s 2857 1462 1374 1118s", True),
    R("nonan-2-on", "C9H18O", "2928s 2858 1717s 1460 1415 1361 1164"),
    R("ethoxybenzen", "C8H10O", "3063 2980 1600s 1497s 1245s 1048s 753s 691"),
    R("butanamin", "C4H11N", "3370 3290 2960s 2930s 2873 1605 1465 1075 800s"),
)
