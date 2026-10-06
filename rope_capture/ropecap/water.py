"""Water, applied to the rendered picture with each pixel's distance:

    I = J * exp(-beta * d) + B * (1 - exp(-beta * d))

(the underwater image formation model OceanSim uses, BSD-3; Akkaynak &
Treibitz): the colour J of whatever is there fades with distance d into the
water's own colour B (backscatter), each colour channel at its own rate. The
visibility (capture.gd's fog_depth_end - begin, 2-30 m) is where only 2% of the
contrast is left. On top: the ROV lamp lighting up the water in front of it,
and specks of suspended particles.
"""
import numpy as np

CONTRAST_AT_VISIBILITY = 0.02


def coefficients(water):
    """Per-channel attenuation (1/m) from the water's tint and visibility:
    the channels the tint has least of fade fastest (red, in blue water)."""
    tint = np.array(water["tint"], float)
    visibility = max(water["fog_end"] - water["fog_begin"], 0.5)
    mean = -np.log(CONTRAST_AT_VISIBILITY) / visibility
    m = max(tint.mean(), 0.05)
    w = np.clip(1.0 + 0.6 * (m - tint) / m, 0.6, 1.8)
    return mean * w / w.mean(), mean


def clean_distance(dist):
    d = np.asarray(dist, np.float32).copy()
    bad = ~np.isfinite(d) | (d <= 0) | (d > 1e5)  # nothing there: open water
    d[bad] = np.inf
    return d


def transmission(dist, water):
    """How much of an object's own contrast survives the water to the camera (mean of channels)."""
    _, mean = coefficients(water)
    d = np.maximum(clean_distance(dist) - water["fog_begin"], 0.0)
    return np.exp(-mean * d)


_cones = {}


def _cone(h, w, cx, cy, fx, fy):
    """Where the lamp's 45 deg beam is, seen from the camera next to it (1 inside, fading out)."""
    key = (h, w, cx, cy, fx, fy)
    if key not in _cones:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ang = np.degrees(np.arctan(np.hypot((xx - cx) / fx, (yy - cy) / fy)))
        _cones[key] = np.clip((45.0 - ang) / 20.0, 0.0, 1.0).astype(np.float32)
    return _cones[key]


def underwater(rgb, dist, water, k, rng=None):
    """rgb: (H, W, 3) uint8 as rendered; dist: (H, W) m (inf where open water)."""
    rng = rng or np.random.default_rng(water.get("seed", 0))
    h, w = dist.shape
    J = rgb[..., :3].astype(np.float32) / 255.0
    B = np.array(water["tint"], np.float32)
    beta, mean = coefficients(water)
    d = np.maximum(clean_distance(dist) - water["fog_begin"], 0.0)
    T = np.exp(-d[..., None] * beta[None, None, :].astype(np.float32))
    out = J * T + B[None, None, :] * (1.0 - T)
    # the lamp lights the water just in front of the camera (backscatter in its beam)
    if water["lamp"] > 0:
        cone = _cone(h, w, k["cx"], k["cy"], k["fx"], k["fy"])
        haze = 0.06 * (water["lamp"] / 4.0) * (1.0 - np.exp(-mean * np.minimum(d, 8.0))) * cone
        out += haze[..., None] * (0.5 + 0.5 * np.clip(B * 1.6, 0, 1))[None, None, :]
    # suspended particles: small soft specks in front of things
    if water.get("particles", 0) > 0:
        import cv2

        n = int(water["particles"] * h * w / 6000)
        layer = np.zeros((h, w), np.float32)
        xs = rng.integers(0, w, n)
        ys = rng.integers(0, h, n)
        dist_s = np.exp(rng.uniform(np.log(0.15), np.log(4.0), n))
        bright = rng.uniform(0.05, 0.35, n) * (0.3 + 0.7 * min(water["lamp"] / 4.0 + water["ambient"] / 2.0, 1.0))
        for x, y, ds, b in zip(xs, ys, dist_s, bright):
            if d[y, x] > ds:
                r = max(1, int(round(0.003 * k["fx"] / ds)))
                cv2.circle(layer, (int(x), int(y)), r, float(b * np.exp(-mean * ds)), -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 1.0)
        out += layer[..., None]
    return (np.clip(out, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
