"""The worlds the rope is put in, as plain data: the floor's shape and look,
pilings, a quay wall, rocks, an optional wreck or MarineGym's seabed. A new
layout every scene (every new rope), so the network sees many floors and
backgrounds rather than many pictures of one. capture.py builds them in
Isaac Sim; the rope's physics uses the same floor and pilings.

The water surface is at z = 0; the floor is around z = -depth.
"""
import numpy as np

FLOOR_STYLES = ["sand", "rippled_sand", "mud", "gravel", "rock", "shell_sand", "silt_weed", "pool_tiles"]
WORLDS = ["seabed", "harbour", "marinegym", "wreck"]


class Floor:
    """A gently uneven floor: -depth plus a few long, low waves."""

    def __init__(self, depth, rng, bumpiness):
        self.depth = depth
        k = rng.uniform(0.05, 0.6, size=(4, 2)) * rng.choice([-1, 1], size=(4, 2))
        self.waves = [(kx, ky, rng.uniform(0, 2 * np.pi), a) for (kx, ky), a in
                      zip(k, rng.uniform(0.0, bumpiness, size=4) / 4.0)]

    def __call__(self, x, y):
        z = np.full(np.broadcast(x, y).shape, -self.depth, float)
        for kx, ky, ph, a in self.waves:
            z = z + a * np.sin(kx * np.asarray(x) + ky * np.asarray(y) + ph)
        return z


def make_world(kind, rng, wreck_path=None):
    """A new layout of `kind` with the rope's anchor point near (0, 0)."""
    w = {"kind": kind, "pilings": [], "walls": [], "rocks": [], "wreck": None, "marinegym": False}
    if kind == "harbour":
        depth = rng.uniform(4.0, 9.0)
    elif kind == "marinegym":
        depth = 15.0  # its sand is 15 m below its water surface
    else:
        depth = rng.uniform(4.0, 15.0)
    style = "pool_tiles" if kind == "seabed" and rng.random() < 0.08 else \
        rng.choice([s for s in FLOOR_STYLES if s != "pool_tiles"])
    if kind == "marinegym":
        style = "marinegym"
    w["floor_style"] = str(style)
    w["floor"] = Floor(depth, rng, 0.0 if style in ("pool_tiles", "marinegym") else rng.uniform(0.0, 0.6))
    w["depth"] = depth
    w["anchor"] = np.array([0.0, 0.0])
    if kind == "harbour":
        # rows of pilings; the rope hangs among them, so they hide parts of it
        spacing = rng.uniform(1.8, 4.0)
        radius = rng.uniform(0.12, 0.35)
        rows = int(rng.integers(1, 4))
        heading = rng.uniform(0, np.pi)
        along = np.array([np.cos(heading), np.sin(heading)])
        across = np.array([-along[1], along[0]])
        offset = rng.uniform(0.4, 1.6) * rng.choice([-1, 1])  # rope between/near a row
        square = rng.random() < 0.3
        material = str(rng.choice(["concrete", "timber", "steel"]))
        for r in range(rows):
            for k in range(-4, 5):
                c = (k + rng.uniform(-0.05, 0.05)) * spacing * along + (offset + r * spacing) * across
                w["pilings"].append({"x": float(c[0]), "y": float(c[1]), "radius": float(radius * rng.uniform(0.9, 1.1)),
                                     "square": bool(square), "material": material})
        if rng.random() < 0.7:  # a quay wall behind the pilings
            d = offset + rows * spacing + rng.uniform(0.5, 3.0)
            w["walls"].append({"centre": (d * across).tolist(), "normal": (-across).tolist(),
                               "width": 30.0, "material": str(rng.choice(["concrete", "stone", "sheet_pile"]))})
    if kind in ("seabed", "harbour", "wreck"):
        n = int(rng.integers(0, 40 if kind == "seabed" else 15))
        for _ in range(n):
            r = rng.uniform(0.0, 14.0) ** 1.0
            a = rng.uniform(0, 2 * np.pi)
            x, y = r * np.cos(a), r * np.sin(a)
            if np.hypot(x, y) < 0.8:
                continue
            size = float(np.exp(rng.uniform(np.log(0.08), np.log(1.0))))
            w["rocks"].append({"x": float(x), "y": float(y), "size": size, "seed": int(rng.integers(1 << 30)),
                               "tone": float(rng.uniform(0.15, 0.6))})
    if kind == "wreck" and wreck_path:
        a = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(2.0, 6.0)
        w["wreck"] = {"path": wreck_path, "x": float(dist * np.cos(a)), "y": float(dist * np.sin(a)),
                      "yaw_deg": float(rng.uniform(0, 360))}
    if kind == "marinegym":
        w["marinegym"] = True
    w["obstacles"] = [(p["x"], p["y"], p["radius"]) for p in w["pilings"]]
    return w


def clear_of_obstacles(world, xy, margin):
    """Is the point at least `margin` m outside every piling (and in front of the wall)?"""
    for (x, y, r) in world["obstacles"]:
        if np.hypot(xy[0] - x, xy[1] - y) < r + margin:
            return False
    for wall in world["walls"]:
        if np.dot(np.asarray(xy) - wall["centre"], wall["normal"]) < margin:
            return False
    return True
