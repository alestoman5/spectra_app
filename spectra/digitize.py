"""Digitisation of a spectrum from an image (photo, scan or screenshot of a printout).

Pipeline
--------
1. ``detect_plot_area`` – find the L-shaped pair of axes (left y-axis + bottom x-axis) with a
   probabilistic Hough transform and build the quadrilateral of the plot area. Small
   perspective distortions of photographs are removed by a homography.
2. ``detect_x_ticks`` – find tick marks under the x-axis for automatic x calibration.
3. ``extract_curve`` – separate the curve from background, grid, labels and handwritten notes
   (colour or dark-ink mask, small connected components removed) and trace it column by
   column.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_DIM = 2000


@dataclass
class PlotArea:
    """Corners of the plot area in image pixel coordinates: TL, TR, BR, BL."""
    corners: np.ndarray            # shape (4, 2) float32
    auto: bool = True

    def homography(self, width: int, height: int) -> np.ndarray:
        dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        return cv2.getPerspectiveTransform(np.float32(self.corners), dst)

    def size(self) -> tuple[int, int]:
        c = self.corners
        w = int(round(max(np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[3]))))
        h = int(round(max(np.linalg.norm(c[3] - c[0]), np.linalg.norm(c[2] - c[1]))))
        return max(w, 10), max(h, 10)


@dataclass
class Trace:
    """Curve traced in the rectified plot area (pixel units, y downwards)."""
    px: np.ndarray        # column index
    top: np.ndarray       # upper edge of the curve stroke in the column
    bottom: np.ndarray    # lower edge of the curve stroke
    width: int
    height: int
    mask: np.ndarray | None = field(default=None, repr=False)

    def values(self, mode: str) -> np.ndarray:
        """Normalised 0..1 signal (1 = top of plot). For transmittance the troughs (bottom edge)
        carry the information, for absorbance the maxima (top edge)."""
        y = self.bottom if mode == "T" else self.top
        return 1.0 - y / (self.height - 1)


def load_image(path: str) -> np.ndarray:
    """Load an image as RGB uint8, downscaled so that the longer side is <= MAX_DIM."""
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Obrázek nelze načíst: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    s = MAX_DIM / max(rgb.shape[:2])
    if s < 1:
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return rgb


# --------------------------------------------------------------------------- masks
def _dark_mask(rgb: np.ndarray, block: int = 31, c: int = 12) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                              block | 1, c)
    return m > 0


def _saturation(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]


def _neutral_dark_mask(rgb: np.ndarray) -> np.ndarray:
    """Dark, low-saturation pixels (black/grey printed lines and text)."""
    return _dark_mask(rgb) & (_saturation(rgb) < 110)


# --------------------------------------------------------------------------- plot area
def _segments(mask: np.ndarray, horizontal: bool, min_frac: float) -> list[tuple]:
    h, w = mask.shape
    m = mask.astype(np.uint8) * 255
    k = max(15, (w if horizontal else h) // 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1) if horizontal else (1, k))
    lines = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    min_len = int(min_frac * (w if horizontal else h))
    segs = cv2.HoughLinesP(lines, 1, np.pi / 720, threshold=max(50, min_len // 3),
                           minLineLength=min_len, maxLineGap=max(10, min_len // 20))
    out = []
    if segs is None:
        return out
    for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(int):
        dx, dy = x2 - x1, y2 - y1
        ang = abs(np.degrees(np.arctan2(dy, dx))) % 180
        if horizontal and min(ang, 180 - ang) < 6:
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            out.append((x1, y1, x2, y2))
        elif not horizontal and abs(ang - 90) < 6:
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            out.append((x1, y1, x2, y2))   # top point first
    return _merge_collinear(out, horizontal)


def _merge_collinear(segs: list[tuple], horizontal: bool, tol: float = 6) -> list[tuple]:
    """Merge Hough segments lying on the same line."""
    segs = sorted(segs, key=lambda s: -(abs(s[2] - s[0]) + abs(s[3] - s[1])))
    merged: list[list] = []
    for s in segs:
        for m in merged:
            if horizontal:
                ym = np.interp(s[0], [m[0], m[2]], [m[1], m[3]]) if m[2] != m[0] else m[1]
                if abs(ym - s[1]) < tol and abs(np.interp(s[2], [m[0], m[2]], [m[1], m[3]]) - s[3]) < tol:
                    if s[0] < m[0]:
                        m[0], m[1] = s[0], s[1]
                    if s[2] > m[2]:
                        m[2], m[3] = s[2], s[3]
                    break
            else:
                xm = np.interp(s[1], [m[1], m[3]], [m[0], m[2]]) if m[3] != m[1] else m[0]
                if abs(xm - s[0]) < tol and abs(np.interp(s[3], [m[1], m[3]], [m[0], m[2]]) - s[2]) < tol:
                    if s[1] < m[1]:
                        m[0], m[1] = s[0], s[1]
                    if s[3] > m[3]:
                        m[2], m[3] = s[2], s[3]
                    break
        else:
            merged.append(list(s))
    return [tuple(m) for m in merged]


def _intersection(h, v) -> np.ndarray:
    (x1, y1, x2, y2), (x3, y3, x4, y4) = h, v
    a = np.array([[x2 - x1, -(x4 - x3)], [y2 - y1, -(y4 - y3)]], float)
    b = np.array([x3 - x1, y3 - y1], float)
    try:
        t, _ = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.array([x3, y2], float)
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])


def _extend(mask: np.ndarray, p_from: np.ndarray, p_to: np.ndarray, max_gap: int) -> np.ndarray:
    """Walk along the line p_from→p_to beyond p_to while dark pixels continue."""
    h, w = mask.shape
    d = (p_to - p_from) / max(np.linalg.norm(p_to - p_from), 1e-9)
    n = np.array([-d[1], d[0]])
    best = p_to.copy()
    gap = 0
    p = p_to.copy()
    while gap <= max_gap:
        p = p + d
        x, y = int(round(p[0])), int(round(p[1]))
        if not (0 <= x < w and 0 <= y < h):
            break
        hit = False
        for o in (-2, -1, 0, 1, 2):
            xx, yy = int(round(p[0] + o * n[0])), int(round(p[1] + o * n[1]))
            if 0 <= xx < w and 0 <= yy < h and mask[yy, xx]:
                hit = True
                break
        if hit:
            best, gap = p.copy(), 0
        else:
            gap += 1
    return best


def _tick_count(mask: np.ndarray, p0: np.ndarray, p1: np.ndarray, outward: np.ndarray) -> int:
    """Number of short tick marks on the outer side of an axis line p0→p1."""
    h, w = mask.shape
    L = int(np.linalg.norm(p1 - p0))
    if L < 10:
        return 0
    d = (p1 - p0) / L
    prof = np.zeros(L, bool)
    for t in range(L):
        q = p0 + d * t
        cnt = 0
        for o in range(3, 8):
            x, y = int(round(q[0] + outward[0] * o)), int(round(q[1] + outward[1] * o))
            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                cnt += 1
        prof[t] = cnt >= 3
    runs = np.split(np.flatnonzero(prof), np.flatnonzero(np.diff(np.flatnonzero(prof)) > 1) + 1)
    return sum(1 for r in runs if 0 < len(r) <= 6)


def _find_top_right(mask, hs, vs, tl, br, bl, hseg=None, vseg=None):
    """Top-right corner: intersection of the top and right frame lines if the plot is boxed,
    otherwise complete the parallelogram (affine approximation)."""
    h, w = mask.shape
    tol_x, tol_y = 0.04 * w, 0.04 * h
    def dist(p, s):
        a, b = np.array(s[:2], float), np.array(s[2:], float)
        d = b - a
        return abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / max(np.linalg.norm(d), 1e-9)

    # top frame line passes through TL and reaches far to the right;
    # right frame line passes through BR and reaches high up
    top = [s for s in hs if dist(tl, s) < 0.015 * h and s[2] > br[0] - tol_x
           and s[0] < tl[0] + tol_x]
    right = [s for s in vs if dist(br, s) < 0.015 * w and s[1] < tl[1] + tol_y
             and s[3] > br[1] - tol_y]
    gap = max(8, int(0.02 * max(h, w)))
    if top and right:
        t = max(top, key=lambda s: s[2] - s[0])
        r = max(right, key=lambda s: s[3] - s[1])
        p = _intersection(t, r)
        guess = tl + (br - bl)
        if np.linalg.norm(p - guess) < 0.05 * max(h, w):
            if hseg is not None and vseg is not None:
                # closed box: all corners as precise line intersections
                return _intersection(t, vseg), p, _intersection(hseg, r)
            return tl, p, br
    if right:
        r = max(right, key=lambda s: s[3] - s[1])
        top_end = _extend(mask, np.array(r[2:], float), np.array(r[:2], float), gap)
        if abs(top_end[1] - tl[1]) < 0.015 * h:
            return tl, top_end, (_intersection(hseg, r) if hseg is not None else br)
    if top:
        t = max(top, key=lambda s: s[2] - s[0])
        right_end = _extend(mask, np.array(t[:2], float), np.array(t[2:], float), gap)
        if abs(right_end[0] - br[0]) < 0.015 * w:
            return (_intersection(t, vseg) if vseg is not None else tl), right_end, br
    return tl, tl + (br - bl), br


def detect_plot_area(rgb: np.ndarray) -> PlotArea:
    """Find the plot area from the left y-axis and bottom x-axis (L-shaped corner)."""
    h, w = rgb.shape[:2]
    mask = _neutral_dark_mask(rgb)
    hs = _segments(mask, True, 0.35)
    vs = _segments(mask, False, 0.30)
    best, best_key = None, None
    for hseg in hs:
        hl = np.hypot(hseg[2] - hseg[0], hseg[3] - hseg[1])
        for vseg in vs:
            vl = np.hypot(vseg[2] - vseg[0], vseg[3] - vseg[1])
            p = _intersection(hseg, vseg)
            # corner must be near the left end of the horizontal and bottom end of the vertical
            dl = np.hypot(*(p - np.array(hseg[:2])))
            db = np.hypot(*(p - np.array(vseg[2:])))
            if dl > 0.06 * w or db > 0.06 * h:
                continue
            area = hl * vl
            if area < 0.12 * w * h:
                continue
            gap = max(8, int(0.02 * max(h, w)))
            br = _extend(mask, p, np.array(hseg[2:], float), gap)
            tl = _extend(mask, p, np.array(vseg[:2], float), gap)
            if np.linalg.norm(br - p) < 0.35 * w or np.linalg.norm(tl - p) < 0.3 * h:
                continue
            # axes carry tick marks on their outer side (frames do not)
            ticks = (_tick_count(mask, p, br, np.array([0.0, 1.0])) +
                     _tick_count(mask, tl, p, np.array([-1.0, 0.0])))
            key = (ticks, -np.linalg.norm(br - p) * np.linalg.norm(tl - p))
            if best_key is None or key > best_key:
                best, best_key = (p, br, tl, hseg, vseg), key
    if best is None:
        m = 0.03
        corners = np.float32([[w * m, h * m], [w * (1 - m), h * m],
                              [w * (1 - m), h * (1 - m)], [w * m, h * (1 - m)]])
        return PlotArea(corners, auto=False)
    bl, br, tl, hseg, vseg = best
    tl, tr, br = _find_top_right(mask, hs, vs, tl, br, bl, hseg, vseg)
    corners = np.float32([tl, tr, br, bl])
    return PlotArea(corners, auto=True)


def rectify(rgb: np.ndarray, area: PlotArea) -> np.ndarray:
    w, h = area.size()
    hm = area.homography(w, h)
    return cv2.warpPerspective(rgb, hm, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def map_points(points: np.ndarray, area: PlotArea) -> np.ndarray:
    """Map image points to rectified plot coordinates."""
    w, h = area.size()
    hm = area.homography(w, h)
    pts = np.float32(points).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, hm).reshape(-1, 2)


# --------------------------------------------------------------------------- ticks
def _below_axis_strip(rgb: np.ndarray, area: PlotArea, frac: float = 0.07) -> np.ndarray:
    """Rectified strip of the image directly below the x-axis (same column scale as plot)."""
    w, h = area.size()
    pad = max(20, int(frac * h))
    c = np.float32(area.corners).copy()
    down = (c[3] - c[0]) / max(np.linalg.norm(c[3] - c[0]), 1) * pad
    strip_area = PlotArea(np.float32([c[3], c[2], c[2] + down, c[3] + down]))
    hm = strip_area.homography(w, pad)
    return cv2.warpPerspective(rgb, hm, (w, pad), borderMode=cv2.BORDER_REPLICATE)


def detect_x_labels(rgb: np.ndarray, area: PlotArea) -> np.ndarray:
    """Centres (rectified plot x pixels) of the numeric labels below the x-axis.

    The tick labels (4000, 3500, ...) are centred under the major ticks, they are much easier
    to find than the tiny tick marks themselves.
    """
    w, h = area.size()
    strip = _below_axis_strip(rgb, area)
    m = _neutral_dark_mask(strip, ).astype(np.uint8)
    sh = m.shape[0]
    # drop the axis line / ticks at the top of the strip: skip rows until text starts
    rows = m.mean(axis=1)
    top = 0
    while top < sh // 2 and (rows[top] > 0.25 or top < 3):
        top += 1
    m[:top] = 0
    # keep only the first text line (labels), not the axis title below it
    rows = m.mean(axis=1)
    text_rows = np.flatnonzero(rows > 0.01)
    if text_rows.size == 0:
        return np.array([])
    segs = np.split(text_rows, np.flatnonzero(np.diff(text_rows) > 2) + 1)
    min_h = max(5, int(0.008 * h))
    segs = [g for g in segs if len(g) >= min_h]
    if not segs:
        return np.array([])
    r0, r1 = segs[0][0], segs[0][-1]
    m[r1 + 1:] = 0
    m[:r0] = 0
    lh = max(4, r1 - r0 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(lh * 0.9)), 1))
    words = cv2.dilate(m, kernel)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(words, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, cw, ch, area_px = stats[i]
        if x <= 1 or x + cw >= w - 2:
            continue      # label cut by the strip border – its centre would be biased
        if lh * 1.2 <= cw <= lh * 6 and ch >= lh * 0.5:
            out.append(x + cw / 2.0)
    return np.sort(np.array(out))


def detect_x_ticks(rgb: np.ndarray, area: PlotArea) -> np.ndarray:
    """Kept for API compatibility – returns label centres (= major tick positions)."""
    return detect_x_labels(rgb, area)


def regular_ticks(ticks: np.ndarray, width: int) -> tuple[float, float] | None:
    """Find a regular comb in tick positions. Returns (first_tick_px, spacing_px) or None."""
    ticks = np.sort(np.asarray(ticks, float))
    if len(ticks) < 3:
        return None
    best = None
    for i in range(len(ticks)):
        for j in range(i + 1, min(i + 3, len(ticks))):
            spacing = ticks[j] - ticks[i]
            if spacing < width / 30:
                continue
            k = np.round((ticks - ticks[i]) / spacing)
            ok = np.abs(ticks - (ticks[i] + k * spacing)) < max(3, 0.06 * spacing)
            if ok.sum() < 3 or len(np.unique(k[ok])) < ok.sum():
                continue
            fit = np.polyfit(k[ok], ticks[ok], 1)
            resid = np.abs(ticks[ok] - np.polyval(fit, k[ok])).mean()
            cand = (int(ok.sum()), -resid, fit)
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best is None:
        return None
    spacing, t0 = best[2]
    first = t0 - np.floor((t0 + 0.4 * spacing) / spacing) * spacing
    return float(first), float(spacing)


# --------------------------------------------------------------------------- curve
def _clean_curve_mask(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.uint8)
    h, w = m.shape
    # remove border (axes) region and long straight horizontal / vertical lines (grid)
    b = max(3, int(0.006 * max(h, w)))
    m[:b, :] = 0
    m[:, :b] = 0
    m[-b:, :] = 0
    m[:, -b:] = 0
    # Straight horizontal pieces (>= w/6) are frame/grid lines when they (possibly slightly
    # tilted) chain into a line spanning most of the width, or when they lie at the frame.
    # Flat parts of the spectrum itself (e.g. 100 %T baseline) are shorter.
    hs = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 6, 1)))
    chain = cv2.dilate(hs, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 25), 7)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(chain, connectivity=8)
    long_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] >= 0.6 * w]
    hl = (np.isin(lab, long_ids) & (hs > 0)).astype(np.uint8)
    band = max(4, int(0.03 * h))
    hl[:band] |= hs[:band]
    hl[-band:] |= hs[-band:]
    hl = cv2.dilate(hl, np.ones((3, 1), np.uint8))
    vl = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.97))))
    m = m & ~hl & ~vl
    # connected components: the curve is large; text, labels and specks are small
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if cw >= 0.04 * w or ch >= 0.18 * h or (cw >= 0.015 * w and area > 0.004 * w * 3):
            keep[i] = True
    return keep[lab]


def curve_mask(plot: np.ndarray, color: str = "auto") -> tuple[np.ndarray, str]:
    """Binary mask of the spectrum line in the rectified plot image.

    ``color``: ``"auto"``, ``"dark"`` (black/grey line), ``"color"`` (coloured, non-blue line,
    e.g. the red OMNIC trace) or ``"blue"``. Handwritten blue notes are removed unless the
    blue line is the curve itself.
    """
    hsv = cv2.cvtColor(plot, cv2.COLOR_RGB2HSV)
    hue, sat = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    dark = _dark_mask(plot, block=41, c=10)
    blue = (hue >= 95) & (hue <= 135) & (sat > 35)
    masks = {}
    paper_sat = float(np.percentile(sat[~dark], 90)) if (~dark).any() else 10.0
    thr = max(22.0, 2.0 * paper_sat + 5.0)
    if color in ("auto", "color"):
        colored = (sat > thr) & (hsv[..., 2] < 250) & ~blue & (dark | (sat > 1.6 * thr))
        hh = hue[colored]
        if hh.size:
            hist = np.bincount(hh, minlength=180).astype(float)
            hist = np.convolve(np.r_[hist[-10:], hist, hist[:10]], np.ones(9), "same")[10:-10]
            hpk = int(np.argmax(hist))
            dh = np.minimum(np.abs(hue - hpk), 180 - np.abs(hue - hpk))
            colored = colored & (dh < 14)
        masks["color"] = _clean_curve_mask(colored)
    if color in ("auto", "dark"):
        masks["dark"] = _clean_curve_mask(dark & (sat < 110) & ~blue)
    if color in ("auto", "blue"):
        # blue curve (handwritten notes are usually blue too, so only used when it is the
        # dominant line spanning the plot)
        masks["blue"] = _clean_curve_mask((hue >= 95) & (hue <= 135) & (sat > thr) &
                                          (dark | (sat > 1.6 * thr)))
    if color != "auto":
        return masks[color], color
    cover = {k: (m.any(axis=0)).mean() for k, m in masks.items()}
    for k in ("color", "dark", "blue"):
        if cover[k] >= 0.6:
            return masks[k], k
    best = max(cover, key=cover.get)
    return masks[best], best


def extract_curve(plot: np.ndarray, color: str = "auto") -> Trace:
    """Trace the curve column by column, following continuity between columns."""
    mask, _ = curve_mask(plot, color)
    h, w = mask.shape
    gap = max(3, int(0.012 * h))   # bridge small breaks in thin, faint lines
    tops = np.full(w, np.nan)
    bots = np.full(w, np.nan)
    runs_per_col = []
    for x in range(w):
        ys = np.flatnonzero(mask[:, x])
        if ys.size == 0:
            runs_per_col.append([])
            continue
        splits = np.split(ys, np.flatnonzero(np.diff(ys) > gap) + 1)
        runs_per_col.append([(r[0], r[-1], len(r)) for r in splits])
    # seed: middle column of the widest connected component (the main curve)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        raise ValueError("Křivku spektra se nepodařilo najít – upravte ořez nebo barvu křivky.")
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_WIDTH]))
    mx, mw = stats[main, cv2.CC_STAT_LEFT], stats[main, cv2.CC_STAT_WIDTH]
    seed = None
    for x in list(range(mx + mw // 2, mx + mw)) + list(range(mx + mw // 2, mx - 1, -1)):
        ys = np.flatnonzero(lab[:, x] == main)
        if ys.size:
            seed = x
            break
    max_jump = 0.2 * h
    max_lost = max(10, int(0.08 * w))

    def track(xs, prev):
        lost = 0
        for x in xs:
            runs = runs_per_col[x]
            cand = []
            for r in runs:
                if r[0] - 2 <= prev[1] and r[1] + 2 >= prev[0]:
                    d = 0
                else:
                    d = min(abs(r[0] - prev[1]), abs(r[1] - prev[0]))
                if d <= max_jump:
                    cand.append((d, -r[2], r))
            if not cand and lost >= 2:
                # re-acquire the curve after a break, but only on wide components
                # (short isolated strokes are labels, frames or notes)
                for r in runs:
                    comp = lab[r[0], x]
                    if comp and stats[comp, cv2.CC_STAT_WIDTH] >= 0.03 * w:
                        d = min(abs(r[0] - prev[1]), abs(r[1] - prev[0]))
                        cand.append((d, -r[2], r))
            if not cand:
                lost += 1
                if lost > max_lost:
                    break
                continue
            lost = 0
            r = min(cand)[2]
            tops[x], bots[x] = r[0], r[1]
            prev = (r[0], r[1])

    ys = np.flatnonzero(mask[:, seed])
    r0 = next(r for r in runs_per_col[seed] if r[0] <= np.flatnonzero(lab[:, seed] == main)[0] <= r[1])
    track(range(seed, w), (r0[0], r0[1]))
    track(range(seed, -1, -1), (r0[0], r0[1]))
    valid = np.isfinite(tops)
    if valid.sum() < 2:
        raise ValueError("Křivku spektra se nepodařilo najít – upravte ořez nebo barvu křivky.")
    xs = np.arange(w)
    tops = np.interp(xs, xs[valid], tops[valid])
    bots = np.interp(xs, xs[valid], bots[valid])
    first, last = np.flatnonzero(valid)[[0, -1]]
    return Trace(xs[first:last + 1], tops[first:last + 1], bots[first:last + 1], w, h, mask)


# --------------------------------------------------------------------------- calibration
@dataclass
class XCalibration:
    """Linear map from rectified pixel column to wavenumber: wn = a * px + b."""
    a: float
    b: float

    @classmethod
    def from_points(cls, p1: float, w1: float, p2: float, w2: float) -> "XCalibration":
        a = (w2 - w1) / (p2 - p1)
        return cls(a, w1 - a * p1)

    def __call__(self, px):
        return self.a * np.asarray(px, float) + self.b


def auto_x_calibration(rgb: np.ndarray, area: PlotArea, left_value: float = 4000,
                       tick_step: float = 500, right_value: float | None = None
                       ) -> tuple[XCalibration, str]:
    """Calibrate x from tick marks (left axis = ``left_value``, every major tick = ``tick_step``).

    Falls back to the frame edges (left_value .. right_value) if ticks are not found.
    """
    w, _ = area.size()
    ticks = detect_x_ticks(rgb, area)
    reg = regular_ticks(ticks, w)
    if reg is not None and right_value is None:
        first, spacing = reg
        # the label nearest to the left axis is the left value
        p0 = first if abs(first) < 0.4 * spacing else 0.0
        cal = XCalibration.from_points(p0, left_value, p0 + spacing, left_value - tick_step)
        return cal, (f"automaticky z popisků osy x (rozteč {spacing:.0f} px = {tick_step:g} cm⁻¹, "
                     f"rozsah {cal(0):.0f}–{cal(w - 1):.0f} cm⁻¹)")
    rv = 400.0 if right_value is None else right_value
    return XCalibration.from_points(0, left_value, w - 1, rv), \
        f"z okrajů grafu ({left_value:g} → {rv:g} cm⁻¹)"


def digitize(rgb: np.ndarray, area: PlotArea | None = None, color: str = "auto",
             calibration: XCalibration | None = None, **cal_kw):
    """Full image → (wavenumbers, trace, calibration, info) pipeline."""
    if area is None:
        area = detect_plot_area(rgb)
    plot = rectify(rgb, area)
    trace = extract_curve(plot, color)
    info = ""
    if calibration is None:
        calibration, info = auto_x_calibration(rgb, area, **cal_kw)
    wn = calibration(trace.px)
    return wn, trace, area, calibration, info, plot
