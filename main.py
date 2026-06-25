"""Classical periodic seamless-tiling for regular floor textures.

Given a photo of a regular floor (stone grid, parquet, wood planks) this module
produces a texture that tiles at any repeat count (2x2, 5x5, 10x10, ...) with:

  * no visible seam,
  * uniform plank/tile size across every tile->tile join,
  * no cut-off planks, and
  * quality identical to the input (the output is made of REAL input pixels).

The idea is purely classical (no GPU, deterministic):

  1. locate the joints (grout lines / plank grooves) with an edge map + Otsu, and
     crop the tiling boundary exactly onto a joint (falling back to an FFT/auto-
     correlation period crop where there is no clean joint ladder). This is what
     guarantees uniform planks / no cut boards.
  2. heal each axis' wrap seam adaptively (`_heal_seam`): a short cosine feather
     where the seam sits on a continuous joint, or a min-cut (image-quilting)
     boundary where it does not (staggered plank-ends), so the seam follows the
     joints/grain instead of cutting across a board.
  3. for a diagonal (herringbone) lattice, whose joints run at +/-45 deg, deskew
     the weave onto the axes (`detect_herringbone`) and crop an exact integer
     number of square periods so the chevrons interlock across the seam.

Every step only moves real input pixels (bar a sub-degree deskew of the
herringbone), so output quality equals the input.
See README.md for the method comparison and rationale.
"""

from __future__ import annotations

import os
from typing import Optional
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, sobel, rotate


def load_rgb(path: str) -> np.ndarray:
    """Load an image as an HxWx3 uint8 array (drops alpha)."""
    return np.asarray(Image.open(path).convert("RGB"))


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude of a high-passed grayscale image.

    The high-pass removes slow brightness variation (which otherwise swamps the
    autocorrelation), leaving the thin grout lines / plank grooves that actually
    carry the periodic structure.
    """
    hp = gray - gaussian_filter(gray, 6)
    return np.hypot(sobel(hp, axis=0), sobel(hp, axis=1))


def _axis_period(
        profile: np.ndarray,
        min_lag: int,
        max_lag: int,
        prominence: float = 0.05
) -> Optional[int]:
    """Lag of the most prominent autocorrelation peak in [min_lag, max_lag].

    `profile` is a 1D edge projection. We return the *most prominent* peak (the
    dominant repeat), which is always a true period of the pattern; smaller
    fundamentals would be exact divisors and tile identically, so the dominant
    peak is the safe, robust choice.
    """
    sig = profile - profile.mean()
    ac = np.correlate(sig, sig, mode="full")[sig.size - 1:]
    ac /= ac[0]
    seg = ac[min_lag:max_lag + 1]
    if seg.size < 3:
        return None
    peaks, props = find_peaks(seg, prominence=prominence)
    if peaks.size == 0:
        return None
    best = peaks[int(np.argmax(props["prominences"]))]
    return int(min_lag + best)


def detect_period(
        arr: np.ndarray,
        min_frac: float = 0.04,
        max_frac: float = 0.6
) -> tuple[Optional[int], Optional[int]]:
    """Detect the dominant (px, py) period of a regular texture.

    Projects an edge map onto each axis (integrating thin grout/groove lines
    over the whole image) and finds the dominant repeat by 1D autocorrelation.
    Returns None for an axis with no clear periodicity (e.g. randomly staggered
    plank ends), which the caller heals as a single blended seam.
    """
    gray = arr.astype(np.float64).mean(axis=2)
    edge = _edge_map(gray)
    h, w = gray.shape
    px = _axis_period(edge.sum(axis=0), max(8, int(min_frac * w)),
                      int(max_frac * w))
    py = _axis_period(edge.sum(axis=1), max(8, int(min_frac * h)),
                      int(max_frac * h))
    return px, py


def _refine_size(
        arr: np.ndarray,
        origin: int,
        target: int,
        axis: int,
        band: int = 8
) -> int:
    """Nudge a crop size near `target` so the pattern closes exactly on itself.

    The detected period is integer-rounded, so k*period accumulates sub-pixel
    drift that would misalign grout lines across the seam. We search a small
    window around `target` for the size whose start band best matches the band
    one period later, absorbing the rounding error.
    """
    h, w = arr.shape[:2]
    limit = (w if axis == 1 else h) - origin - band
    r = max(2, target // 8)
    best, best_cost = target, None
    a = (arr[:, origin:origin + band] if axis == 1
         else arr[origin:origin + band, :]).astype(np.float64)
    for size in range(target - r, target + r + 1):
        if size < band or size + band > limit:
            continue
        b = (arr[:, origin + size:origin + size + band] if axis == 1
             else arr[origin + size:origin + size + band, :]).astype(np.float64)
        cost = float(np.abs(a - b).mean())
        if best_cost is None or cost < best_cost:
            best, best_cost = size, cost
    return best


def _cosine_ramp(m: int) -> np.ndarray:
    """Smooth 0->1 weight ramp of length m (cosine / smoothstep)."""
    return 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, m))


def _wrap_blend(
        arr: np.ndarray,
        x0: int,
        y0: int,
        wt: int,
        ht: int,
        m: int
) -> np.ndarray:
    """Crop a (wt+m) x (ht+m) region and feather-blend the m-px wrap overlap.

    The two overlapping bands are one period/joint-interval apart, i.e. the same
    phase, so the cosine blend only averages residual lighting/noise and leaves
    the geometry intact. Returns the healed `wt x ht` uint8 texture.
    """
    src = arr[y0:y0 + ht + m, x0:x0 + wt + m].astype(np.float64)
    wx = _cosine_ramp(m)[None, :, None]
    src[:, :m] = src[:, :m] * wx + src[:, wt:wt + m] * (1.0 - wx)
    out = src[:, :wt].copy()
    wy = _cosine_ramp(m)[:, None, None]
    out[:m] = out[:m] * wy + out[ht:ht + m] * (1.0 - wy)
    return np.clip(out[:ht], 0, 255).astype(np.uint8)


def _period_span(
        arr: np.ndarray,
        p: Optional[int],
        axis: int,
        m: int
) -> tuple[int, int]:
    """Centred (origin, size) for the largest integer number of periods on `axis`.

    For an axis with no period (`p is None`) returns a single near-full span. The
    size is refined to self-close, absorbing integer-period rounding.
    """
    dim = arr.shape[1] if axis == 1 else arr.shape[0]
    size = max(1, (dim - m) // p) * p if p else (dim - m)
    origin = max(0, (dim - (size + m)) // 2)
    if p:
        size = _refine_size(arr, origin, size, axis)
    return origin, size


def _mincut_path(
        A: np.ndarray,
        B: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Least-cost connected boundary across an overlap band (Efros-Freeman).

    `A` is the tile's own band and `B` its continuation one period away, both
    shape `(overlap, length, 3)` with the seam running along axis 1. We build the
    per-pixel SSD surface and dynamic-program the cheapest 8-connected path that
    crosses every along-seam position, so the stitch can route the seam through
    matching grain / staggered butt-joints instead of cutting straight across a
    board. Returns the cut row per position plus the straight-band and along-path
    mean costs (used to decide whether the cut beats a plain feather).
    """
    e = ((A - B) ** 2).mean(axis=2)                     # (overlap, length)
    n, length = e.shape
    cum = e.copy()
    back = np.zeros((n, length), np.int32)
    for c in range(1, length):
        prev = cum[:, c - 1]
        up = np.r_[prev[1:], np.inf]                    # neighbour at row+1
        dn = np.r_[np.inf, prev[:-1]]                   # neighbour at row-1
        choices = np.stack([dn, prev, up])              # 0:row-1 1:row 2:row+1
        k = np.argmin(choices, axis=0)
        cum[:, c] += np.choose(k, choices)
        back[:, c] = k - 1
    path = np.empty(length, np.int32)
    path[-1] = int(np.argmin(cum[:, -1]))
    for c in range(length - 1, 0, -1):
        path[c - 1] = min(max(path[c] + back[path[c], c], 0), n - 1)
    return path, float(e.mean()), float(e[path, np.arange(length)].mean())


def _stitch_band(
        A: np.ndarray,
        B: np.ndarray,
        path: np.ndarray,
        feath: float = 2.0
) -> np.ndarray:
    """Stitch two bands along a cut `path`: `B` before the cut, `A` after.

    A few-px cosine-like ramp across the path hides the join and any residual
    lighting step. Mirrors `_wrap_blend`'s phase convention (the band's leading
    edge becomes the continuation `B`, ramping to the tile's own content `A`).
    """
    n = A.shape[0]
    rr = np.arange(n)[:, None]
    wgt = np.clip((rr - path[None, :]) / (2.0 * feath) + 0.5, 0.0, 1.0)[..., None]
    return B * (1.0 - wgt) + A * wgt


def _axis_decision(
        arr: np.ndarray,
        x0: int,
        y0: int,
        wt: int,
        ht: int,
        axis: int,
        ratio_thresh: float = 0.38,
        min_overlap: int = 16
) -> tuple[bool, int, Optional[np.ndarray]]:
    """Decide feather vs min-cut for one crop axis (and return the cut path).

    Compares the min-cut path cost to the straight band-average over a wide
    overlap. A *continuous* joint (grout line / plank groove) lets the cut follow
    it and beat the straight average by a large margin -> the straight feather
    already hides in that groove, so keep it. *Staggered* plank-ends have no
    continuous joint: the cut only modestly beats the average, but it weaves
    through the butt-joints instead of cutting boards -> use min-cut. The
    rms-ratio cleanly separates the two regimes; min-cut is also safe where it is
    not strictly needed (it only ever moves real pixels along a low-error path).
    """
    h, w = arr.shape[:2]
    if axis == 1:                                       # vertical seam (x)
        ov = min(w - (x0 + wt), wt // 4)
        if ov < min_overlap:
            return False, ov, None
        band = arr[y0:y0 + ht, x0:x0 + wt + ov].astype(np.float64)
        A = band[:, :ov].transpose(1, 0, 2)
        B = band[:, wt:wt + ov].transpose(1, 0, 2)
    else:                                               # horizontal seam (y)
        ov = min(h - (y0 + ht), ht // 4)
        if ov < min_overlap:
            return False, ov, None
        band = arr[y0:y0 + ht + ov, x0:x0 + wt].astype(np.float64)
        A = band[:ov]
        B = band[ht:ht + ov]
    path, straight_ms, cut_ms = _mincut_path(A, B)
    use_mincut = np.sqrt(cut_ms) > ratio_thresh * np.sqrt(straight_ms)
    return bool(use_mincut), ov, (path if use_mincut else None)


def _heal_seam(
        arr: np.ndarray,
        x0: int,
        y0: int,
        wt: int,
        ht: int,
        m: int
) -> np.ndarray:
    """Crop `wt x ht` and heal both wrap seams, per axis adaptively.

    Each axis is healed with a min-cut boundary when its seam does not sit on a
    continuous joint (staggered plank-ends), otherwise with the plain cosine
    feather. When both axes feather, this is byte-identical to `_wrap_blend`
    (grids / grooves / `main` are untouched). x is handled before y so the 2-D
    corner is resolved sequentially, exactly as `_wrap_blend` does.
    """
    use_y, ovy_w, path_y = _axis_decision(arr, x0, y0, wt, ht, axis=0)
    ovy = ovy_w if use_y else m
    # x is healed across the full (ht + ovy) height, so evaluate its cut there.
    use_x, ovx_w, path_x = _axis_decision(arr, x0, y0, wt, ht + ovy, axis=1)
    ovx = ovx_w if use_x else m

    if not use_x and not use_y:
        return _wrap_blend(arr, x0, y0, wt, ht, m)

    src = arr[y0:y0 + ht + ovy, x0:x0 + wt + ovx].astype(np.float64)
    if use_x:
        a = src[:, :ovx].transpose(1, 0, 2)
        b = src[:, wt:wt + ovx].transpose(1, 0, 2)
        src[:, :ovx] = _stitch_band(a, b, path_x).transpose(1, 0, 2)
    else:
        wx = _cosine_ramp(m)[None, :, None]
        src[:, :m] = src[:, :m] * wx + src[:, wt:wt + m] * (1.0 - wx)

    out = src[:, :wt].copy()
    if use_y:
        out[:ovy] = _stitch_band(out[:ovy], out[ht:ht + ovy], path_y)
    else:
        wy = _cosine_ramp(m)[:, None, None]
        out[:m] = out[:m] * wy + out[ht:ht + m] * (1.0 - wy)
    return np.clip(out[:ht], 0, 255).astype(np.uint8)


def _otsu_threshold(v: np.ndarray, nbins: int = 256) -> float:
    """Otsu's threshold (in [0, 1]) of values already normalized to [0, 1].

    Picks the cut that maximises between-class variance of the value histogram,
    i.e. best separates the high "joint" samples from the low "field" samples.
    """
    hist, _ = np.histogram(v, bins=nbins, range=(0.0, 1.0))
    p = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (np.arange(nbins) + 0.5) / nbins)
    denom = omega * (1.0 - omega)
    sigma_b2 = np.where(denom > 0, (mu[-1] * omega - mu) ** 2 / (denom + 1e-12), 0.0)
    return (int(np.argmax(sigma_b2)) + 0.5) / nbins


def _run_centroids(mask: np.ndarray) -> list[int]:
    """Centroid index of each maximal run of True values in a 1D boolean mask."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    centers, run = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b == a + 1:
            run.append(b)
        else:
            centers.append(int(np.mean(run)))
            run = [b]
    centers.append(int(np.mean(run)))
    return centers


def _merge_close(centers: np.ndarray, min_sep: float) -> np.ndarray:
    """Collapse runs of centers closer than `min_sep` into their centroid.

    A single line produces two edges; smoothing leaves residual doubles. Merging
    yields one coordinate per physical joint.
    """
    groups = [[centers[0]]]
    for c in centers[1:]:
        if c - groups[-1][-1] < min_sep:
            groups[-1].append(c)
        else:
            groups.append([c])
    return np.array([int(np.mean(g)) for g in groups])


def _longest_regular_chain(coords: np.ndarray, s: float, tol: float) -> np.ndarray:
    """Longest run of coords whose consecutive spacing is within tol of `s`.

    Keeps only evenly-spaced joints (so the tiles end up uniform) and drops
    spurious detections — e.g. stone-internal edges in a tile grid — that break
    the rhythm.
    """
    lo, hi = s * (1.0 - tol), s * (1.0 + tol)
    best, cur = [coords[0]], [coords[0]]
    for a, b in zip(coords, coords[1:]):
        if lo <= (b - a) <= hi:
            cur.append(b)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [b]
    if len(cur) > len(best):
        best = cur
    return np.array(best)


def _detect_regular(
        arr: np.ndarray,
        axis: int,
        spacing_tol: float = 0.15,
        min_chain: int = 3,
        min_spacing_frac: float = 0.04,
        min_span_frac: float = 0.2
) -> Optional[np.ndarray]:
    """Strict path: the longest evenly-spaced joint ladder (grids, uniform planks).

    Uses the isotropic edge map with no projection smoothing, so thin grout lines
    stay sharp, and filters spurious peaks by requiring a regular rhythm.
    """
    gray = arr.astype(np.float64).mean(axis=2)
    dim = gray.shape[1] if axis == 1 else gray.shape[0]
    proj = _edge_map(gray).sum(axis=0 if axis == 1 else 1)
    v = (proj - proj.min()) / (np.ptp(proj) + 1e-9)
    centers = np.array(_run_centroids(v > _otsu_threshold(v)))
    if centers.size < 2:
        return None
    merged = _merge_close(centers, max(3.0, 0.3 * np.median(np.diff(centers))))
    if merged.size < 2:
        return None
    chain = _longest_regular_chain(merged,
                                   float(np.median(np.diff(merged))),
                                   spacing_tol)
    if (chain.size < min_chain
            or np.median(np.diff(chain)) < min_spacing_frac * dim
            or (chain[-1] - chain[0]) < min_span_frac * dim):
        return None
    return chain


def _detect_relaxed(
        arr: np.ndarray,
        axis: int,
        min_count: int = 3,
        min_spacing_frac: float = 0.05,
        min_span_frac: float = 0.4
) -> Optional[np.ndarray]:
    """Relaxed path for irregular planks (e.g. `03`): smoothed, no rhythm needed.

    A high-passed edge projection is **smoothed** before Otsu so high-frequency
    wood grain is merged and only the real, full-length grooves survive. Spacing
    is NOT required to be uniform (plank widths vary) — the crop just runs between
    the outermost grooves — but the result is rejected if there are too few
    grooves, the spacing is implausibly dense, or coverage is poor.
    """
    gray = arr.astype(np.float64).mean(axis=2)
    dim = gray.shape[1] if axis == 1 else gray.shape[0]
    hp = gray - gaussian_filter(gray, 25)
    edge = np.hypot(sobel(hp, axis=0), sobel(hp, axis=1))
    proj = gaussian_filter(edge.sum(axis=0 if axis == 1 else 1).astype(np.float64),
                           max(3, dim // 300))
    v = (proj - proj.min()) / (np.ptp(proj) + 1e-9)
    centers = np.array(_run_centroids(v > _otsu_threshold(v)))
    if centers.size < min_count:
        return None
    merged = _merge_close(centers, max(3.0, 0.3 * np.median(np.diff(centers))))
    if (merged.size < min_count
            or np.median(np.diff(merged)) < min_spacing_frac * dim
            or (merged[-1] - merged[0]) < min_span_frac * dim):
        return None
    return merged


def detect_joints(arr: np.ndarray, axis: int) -> Optional[np.ndarray]:
    """Locate grout/groove joints along an axis via Otsu.

    Tries the strict evenly-spaced detector first (grids / uniform planks); if
    that finds nothing, falls back to the relaxed smoothed detector for irregular
    planks. Returns sorted joint coordinates, or None (caller uses a period crop).
    """
    chain = _detect_regular(arr, axis)
    return chain if chain is not None else _detect_relaxed(arr, axis)


def _joint_span(joints: Optional[np.ndarray], dim: int, m: int) -> Optional[tuple[int, int]]:
    """(origin, size) spanning the joint ladder, leaving m px for the overlap.

    origin and origin+size both sit on a joint (an integer number of joint
    intervals apart), so tiling lands the seam exactly inside a groove.
    """
    if joints is None or joints.size < 2:
        return None
    j0 = int(joints[0])
    cand = joints[joints + m <= dim]
    if cand.size < 1:
        return None
    j_last = int(cand[-1])
    return (j0, j_last - j0) if (j_last - j0) > m else None


def detect_diagonal_period(
        arr: np.ndarray,
        cv_max: float = 0.15,
        min_count: int = 5
) -> Optional[int]:
    """Square tiling period of a diagonal-lattice pattern (herringbone) via Otsu.

    A herringbone's joints run at +/-45 deg, so no axis projection finds them.
    We instead sum the edge map along both diagonals, Otsu-detect the joint ladder
    there, and -- if it is highly regular (low coefficient of variation) -- return
    the square tiling period, which is twice the diagonal joint spacing. Returns
    None for non-diagonal patterns (grids, planks), so they keep the axis-aligned
    path.
    """
    gray = arr.astype(np.float64).mean(axis=2)
    h, w = gray.shape
    hp = gray - gaussian_filter(gray, 15)
    edge = np.hypot(sobel(hp, axis=0), sobel(hp, axis=1))
    ii, jj = np.indices((h, w))
    best = None
    for idx in (ii + jj, jj - ii + (h - 1)):           # the two diagonals
        n = h + w - 1
        ds = np.zeros(n)
        np.add.at(ds, idx.ravel(), edge.ravel())
        cnt = np.zeros(n)
        np.add.at(cnt, idx.ravel(), 1.0)
        ds /= np.maximum(cnt, 1)                        # mean per diagonal (debias)
        p = gaussian_filter(ds, 7)
        v = (p - p.min()) / (np.ptp(p) + 1e-9)
        c = np.array(_run_centroids(v > _otsu_threshold(v)))
        if c.size < min_count:
            continue
        d = np.diff(c)
        cv = float(d.std() / d.mean())
        if cv < cv_max and (best is None or c.size > best[1]):
            best = (float(np.median(d)), c.size)
    return int(round(2 * best[0])) if best else None


def _rot(a: np.ndarray, angle: float) -> np.ndarray:
    """Rotate `a` (gray 2-D or RGB 3-D) by `angle` deg about its centre.

    One convention for both the angle search (gray) and the final deskew (RGB),
    so the angle that maximises periodicity also deskews in the right direction.
    """
    return rotate(a, angle, axes=(0, 1), reshape=False,
                  order=3 if a.ndim == 3 else 1, mode="reflect")


def detect_herringbone(arr: np.ndarray) -> Optional[tuple[float, int]]:
    """Detect a herringbone and return its (deskew_angle_deg, axis_period_px).

    A herringbone is a *diagonal* lattice (planks at +/-45 deg), so no axis crop
    tiles it; and because real floors are photographed a fraction of a degree off
    square, it has no small rectangular period at all. We gate on the diagonal
    Otsu ladder (`detect_diagonal_period`), then find the small rotation that makes
    the weave axis-periodic -- the angle maximising the strength of the dominant
    x-autocorrelation peak -- and read that peak's lag as the square period.
    Returns None for grids / planks (they keep the axis-aligned joint path).
    """
    p0 = detect_diagonal_period(arr)
    if p0 is None:
        return None
    gray = arr.astype(np.float64).mean(axis=2)
    h, w = gray.shape
    half = min(700, h // 2 - 10, w // 2 - 10)
    cen = gray[h // 2 - half:h // 2 + half, w // 2 - half:w // 2 + half]
    lo, hi = max(8, int(p0 * 0.33)), int(p0 * 0.66)     # axis period ~ p0 / 2
    trim = max(8, int(half * 0.12))

    def axis_peak(g: np.ndarray) -> tuple[int, float]:
        e = g - gaussian_filter(g, 6)
        e = e - e.mean()
        f = np.fft.rfft(e, axis=1)
        ac = np.fft.irfft(f * np.conj(f), axis=1).mean(axis=0)
        ac /= ac[0] + 1e-9
        k = int(np.argmax(ac[lo:hi]))
        return lo + k, float(ac[lo:hi][k])

    strength = lambda angle: axis_peak(_rot(cen, angle)[trim:-trim, trim:-trim])[1]
    coarse = max(np.arange(-1.5, 1.51, 0.1), key=strength)
    angle = float(max(np.arange(coarse - 0.12, coarse + 0.121, 0.03), key=strength))
    period, _ = axis_peak(_rot(cen, angle)[trim:-trim, trim:-trim])
    return angle, int(period)


def _herringbone_seamless(arr: np.ndarray, angle: float, P: int) -> np.ndarray:
    """Deskew the herringbone and crop an exact integer number of square periods.

    Rotating the weave onto the axes makes a real rectangular period exist, so the
    boards align across the seam by construction. We drop the rotation border, crop
    the largest centred `k*P` square, and let `_heal_seam` route the residual grain
    seam along the diagonal grooves.
    """
    rot = _rot(arr, angle)
    h, w = rot.shape[:2]
    b = int(np.ceil(max(h, w) * np.sin(np.radians(abs(angle))))) + 2
    rot = rot[b:h - b, b:w - b]
    h, w = rot.shape[:2]
    ov = max(8, P // 3)
    k = max(1, (min(h, w) - 2 * ov) // P)
    side = k * P
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    m = max(4, min(ov, w - x0 - side, h - y0 - side, P - 1))
    return _heal_seam(rot, x0, y0, side, side, m)


def make_seamless_joints(
        arr: np.ndarray,
        overlap: Optional[int] = None
) -> tuple[np.ndarray, dict]:
    """Seamless crop with boundaries placed on Otsu-detected joints.

    Each axis crops between detected joints (seam hidden in the groove); an axis
    with no clean joint ladder falls back to the period crop. Returns the uint8
    texture and a dict recording which strategy each axis used.
    """
    h, w = arr.shape[:2]

    # Diagonal lattice (herringbone): joints run at +/-45 deg, so no axis crop
    # tiles it. Deskew the weave onto the axes, then crop an exact integer number
    # of square periods so the chevrons interlock. (detect_herringbone returns None
    # for grids / planks, which keep the axis-aligned joint path below.)
    hb = detect_herringbone(arr)
    if hb is not None:
        return _herringbone_seamless(arr, *hb), {"x": "herringbone", "y": "herringbone"}

    jx = detect_joints(arr, axis=1)
    jy = detect_joints(arr, axis=0)

    spac = [np.median(np.diff(j)) for j in (jx, jy) if j is not None]
    ref = int(min(spac)) if spac else min(h, w)
    if overlap is None:
        overlap = max(8, ref // 12)      # small: the seam already sits in a groove
    m = max(4, min(overlap, h // 4, w // 4))

    sx, sy = _joint_span(jx, w, m), _joint_span(jy, h, m)
    px = py = None
    if sx is None or sy is None:
        px, py = detect_period(arr)
    x0, wt = sx if sx else _period_span(arr, px, 1, m)
    y0, ht = sy if sy else _period_span(arr, py, 0, m)

    m = max(4, min(m, w - x0 - wt, h - y0 - ht))
    used = {"x": "joints" if sx else "period",
            "y": "joints" if sy else "period"}
    return _heal_seam(arr, x0, y0, wt, ht, m), used


def tile_image(arr: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Tile `arr` nx times across and ny times down."""
    return np.tile(arr, (ny, nx, 1))


def process_image(
        path: str,
        out_dir: str,
        name: Optional[str] = None,
        tilings: tuple[int, ...] = (3,),
        preview_max: int = 1800
) -> dict:
    """Run the Otsu joint-detection pipeline for one image and write the results.

    Crops the seam onto a detected grout/groove joint (per-axis fallback to the
    period crop where there is no clean joint ladder). Writes `<name>_seamless.png`
    and a `<name>_tiled_NxN.png` for each N in `tilings`. Returns a small info dict.
    """
    name = name or os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)

    arr = load_rgb(path)
    seamless, axes = make_seamless_joints(arr)
    detail = {"method": "joints", "axes": axes}
    seamless_path = os.path.join(out_dir, f"{name}_seamless.png")
    Image.fromarray(seamless).save(seamless_path)

    outputs = {"seamless": seamless_path}
    for n in tilings:
        tiled = Image.fromarray(tile_image(seamless, n, n))
        # Downscale the tiling to a preview size; it exists to verify seams, and
        # # full-res NxN of a 2000px texture would be hundreds of MB.
        # if max(tiled.size) > preview_max:
        #     s = preview_max / max(tiled.size)
        #     tiled = tiled.resize((round(tiled.width * s), round(tiled.height * s)),
        #                          Image.LANCZOS)
        tiled_path = os.path.join(out_dir, f"{name}_tiled_{n}x{n}.png")
        tiled.save(tiled_path)
        outputs[f"tiled_{n}x{n}"] = tiled_path

    return {
        "name": name,
        "input_size": (arr.shape[1], arr.shape[0]),
        "seamless_size": (seamless.shape[1], seamless.shape[0]),
        "outputs": outputs,
        **detail,
    }


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.join(here, "images")
    out_dir = os.path.join(here, "results")
    for fname in ("main.png", "01.png", "02.png", "03.png"):
        info = process_image(
            os.path.join(assets, fname),
            out_dir,
            tilings=(3,)
        )
        extra = info.get("axes", info["method"])
        print(f"{info['name']:>6}: {extra} "
              f"input={info['input_size']} -> seamless={info['seamless_size']}")


if __name__ == "__main__":
    main()
