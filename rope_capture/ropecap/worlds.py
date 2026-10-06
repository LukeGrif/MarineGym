"""The worlds the rope is put in, as plain data. With --world mixed (the
default) every scene (every new rope) is a different kind of place, so the
network sees many backgrounds rather than many pictures of one:

  seabed      open floor (sand, ripples, mud, gravel, rock, shells, silt and
              weed), some rocks, maybe kelp or debris
  reef        rocky: boulders and outcrops, weed and kelp
  open_water  deep: the floor far below or out of sight, little around
  harbour     a few pilings (1-6), sometimes a quay wall
  quay        a quay or harbour wall behind the rope, no pilings
  pool        a test tank: tiled or painted floor and walls
  marinegym   MarineGym's own EmptyMarine seabed (needs internet for its materials)
  wreck       a seabed plus a wreck model (--wreck)

capture.py builds them in Isaac Sim; the rope's physics uses the same floor
and pilings. The water surface is at z = 0, the floor around z = -depth.
"""
import numpy as np

FLOOR_STYLES = ["sand", "rippled_sand", "mud", "gravel", "rock", "shell_sand", "silt_weed"]
WORLDS = ["mixed", "seabed", "reef", "open_water", "harbour", "quay", "pool", "marinegym", "wreck"]
# how often each kind of place comes up in a mixed run
MIX = {"seabed": 0.30, "reef": 0.17, "open_water": 0.15, "harbour": 0.12, "quay": 0.12, "pool": 0.08, "wreck": 0.06}


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


def scatter(rng, n, r_min, r_max):
    """n points round the rope, r_min-r_max m from it."""
    a = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2, n))
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def add_rocks(w, rng, n, r_min, r_max, size_min, size_max):
    for x, y in scatter(rng, n, r_min, r_max):
        w["rocks"].append({"x": float(x), "y": float(y),
                           "size": float(np.exp(rng.uniform(np.log(size_min), np.log(size_max)))),
                           "seed": int(rng.integers(1 << 30)), "tone": float(rng.uniform(0.15, 0.6))})


def add_kelp(w, rng, n, r_min, r_max):
    for x, y in scatter(rng, n, r_min, r_max):
        w["kelp"].append({"x": float(x), "y": float(y), "height": float(rng.uniform(0.4, 3.5)),
                          "width": float(rng.uniform(0.03, 0.15)), "seed": int(rng.integers(1 << 30)),
                          "color": [float(c) for c in np.array([0.25, 0.3, 0.08]) * rng.uniform(0.4, 1.3) + rng.uniform(-0.05, 0.05, 3)]})


def add_debris(w, rng, n, r_min, r_max):
    """Things lying about: mooring blocks, pipes, logs, a tyre."""
    for x, y in scatter(rng, n, r_min, r_max):
        kind = str(rng.choice(["block", "pipe", "log", "tyre"]))
        w["debris"].append({"kind": kind, "x": float(x), "y": float(y), "yaw": float(rng.uniform(0, 2 * np.pi)),
                            "size": float(rng.uniform(0.3, 1.2)), "seed": int(rng.integers(1 << 30))})


def make_world(kind, rng, wreck_path=None):
    """A new layout of `kind` (or, for "mixed", a kind chosen at random) with
    the rope's anchor point at (0, 0)."""
    if kind == "mixed":
        kinds = [k for k in MIX if k != "wreck" or wreck_path]
        p = np.array([MIX[k] for k in kinds])
        kind = str(rng.choice(kinds, p=p / p.sum()))
    w = {"kind": kind, "pilings": [], "walls": [], "rocks": [], "kelp": [], "debris": [], "wreck": None,
         "marinegym": False, "anchor": np.array([0.0, 0.0])}
    style = str(rng.choice(FLOOR_STYLES))
    bumpy = rng.uniform(0.0, 0.6)
    if kind == "harbour":
        depth = rng.uniform(4.0, 9.0)
    elif kind == "quay":
        depth = rng.uniform(3.0, 10.0)
    elif kind == "pool":
        depth, style, bumpy = rng.uniform(3.0, 6.0), "pool_tiles", 0.0
    elif kind == "open_water":
        depth, bumpy = rng.uniform(15.0, 30.0), rng.uniform(0.0, 1.5)
    elif kind == "reef":
        depth, style = rng.uniform(4.0, 14.0), str(rng.choice(["rock", "gravel", "silt_weed", "sand"]))
    elif kind == "marinegym":
        depth, style, bumpy = 15.0, "marinegym", 0.0
    else:
        depth = rng.uniform(4.0, 15.0)
    w["floor_style"] = style
    w["floor"] = Floor(depth, rng, bumpy)
    w["depth"] = depth

    if kind == "seabed" or kind == "wreck":
        add_rocks(w, rng, int(rng.choice([0, 0, 3, 8, 20])), 1.0, 14.0, 0.08, 0.8)
        if rng.random() < 0.3:
            add_kelp(w, rng, int(rng.integers(3, 25)), 1.0, 10.0)
        if rng.random() < 0.3:
            add_debris(w, rng, int(rng.integers(1, 4)), 1.5, 8.0)
    elif kind == "reef":
        add_rocks(w, rng, int(rng.integers(15, 50)), 0.8, 12.0, 0.2, 1.8)
        add_kelp(w, rng, int(rng.integers(10, 60)), 0.8, 10.0)
    elif kind == "open_water":
        if rng.random() < 0.3:
            add_rocks(w, rng, int(rng.integers(1, 6)), 2.0, 15.0, 0.3, 2.0)
    elif kind == "harbour":
        # a few pilings near the rope (1-6), not a forest of them
        n = int(rng.integers(1, 7))
        radius = rng.uniform(0.12, 0.35)
        square = bool(rng.random() < 0.3)
        material = str(rng.choice(["concrete", "timber", "steel"]))
        if rng.random() < 0.5:  # in a row
            heading = rng.uniform(0, np.pi)
            along = np.array([np.cos(heading), np.sin(heading)])
            across = np.array([-along[1], along[0]])
            spacing = rng.uniform(2.0, 5.0)
            offset = rng.uniform(0.6, 3.0) * rng.choice([-1, 1])
            first = -int(rng.integers(0, n))
            spots = [(k + first + rng.uniform(-0.1, 0.1)) * spacing * along + offset * across for k in range(n)]
        else:  # scattered
            spots = list(scatter(rng, n, 0.7, 6.0))
        for c in spots:
            w["pilings"].append({"x": float(c[0]), "y": float(c[1]), "radius": float(radius * rng.uniform(0.9, 1.1)),
                                 "square": square, "material": material})
        if rng.random() < 0.35:
            add_wall(w, rng, rng.uniform(3.0, 8.0))
        if rng.random() < 0.4:
            add_debris(w, rng, int(rng.integers(1, 3)), 1.5, 6.0)
    elif kind == "quay":
        add_wall(w, rng, rng.uniform(1.0, 5.0))
        if rng.random() < 0.4:
            add_rocks(w, rng, int(rng.integers(2, 12)), 1.0, 8.0, 0.2, 1.0)  # rubble at the foot
        if rng.random() < 0.3:
            add_debris(w, rng, int(rng.integers(1, 3)), 1.5, 6.0)
    elif kind == "pool":
        add_wall(w, rng, rng.uniform(1.5, 5.0), material="pool")
        if rng.random() < 0.6:
            add_wall(w, rng, rng.uniform(1.5, 5.0), material="pool", turn=np.pi / 2)
    if kind == "wreck" and wreck_path:
        a = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(2.0, 6.0)
        w["wreck"] = {"path": wreck_path, "x": float(dist * np.cos(a)), "y": float(dist * np.sin(a)),
                      "yaw_deg": float(rng.uniform(0, 360))}
    if kind == "marinegym":
        w["marinegym"] = True
    w["obstacles"] = [(p["x"], p["y"], p["radius"]) for p in w["pilings"]]
    return w


def add_wall(w, rng, distance, material=None, turn=None):
    """A vertical wall `distance` m from the rope, facing it."""
    if turn is not None and w["walls"]:
        n0 = np.array(w["walls"][0]["normal"])
        c, s = np.cos(turn), np.sin(turn)
        normal = np.array([c * n0[0] - s * n0[1], s * n0[0] + c * n0[1]])
    else:
        a = rng.uniform(0, 2 * np.pi)
        normal = np.array([np.cos(a), np.sin(a)])
    w["walls"].append({"centre": (-normal * distance).tolist(), "normal": normal.tolist(), "width": 40.0,
                       "material": material or str(rng.choice(["concrete", "stone", "sheet_pile"]))})


def clear_of_obstacles(world, xy, margin):
    """Is the point at least `margin` m outside every piling (and in front of the walls)?"""
    for (x, y, r) in world["obstacles"]:
        if np.hypot(xy[0] - x, xy[1] - y) < r + margin:
            return False
    for wall in world["walls"]:
        if np.dot(np.asarray(xy) - wall["centre"], wall["normal"]) < margin:
            return False
    return True
