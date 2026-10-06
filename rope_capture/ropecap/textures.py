"""Pictures painted on the rope, floor, pilings and walls (numpy, RGB uint8).

The rope's texture follows the same strands as its shape (geom.strand_radius):
u goes once round the rope, v along it, one texture covering ROPE_TILE_LAYS
turns of the lay. The floors are tileable, a new one for every scene.
"""
import numpy as np

ROPE_TILE_LAYS = 2  # lay lengths along the rope per texture


def tileable_noise(rng, size, cells):
    """Smooth noise in [0, 1], seamless when tiled (cells: grid of random values across)."""
    import cv2

    g = rng.random((cells, cells)).astype(np.float32)
    big = cv2.resize(np.tile(g, (3, 3)), (size * 3, size * 3), interpolation=cv2.INTER_CUBIC)
    return big[size:2 * size, size:2 * size]


def fbm(rng, size, base_cells=4, octaves=5, gain=0.5):
    out = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = min(base_cells * 2 ** o, size // 2)
        out += amp * tileable_noise(rng, size, cells)
        total += amp
        amp *= gain
    out /= total
    return (out - out.min()) / max(out.max() - out.min(), 1e-6)


def to_u8(img):
    return (np.clip(img, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------- rope
def rope_texture(look, width=512, height=1024):
    """The rope's colour picture: strands (or braid carriers) shaded in their
    grooves, yarns, an optional tracer/fleck and loose fibres."""
    rng = np.random.default_rng(look.get("seed", 0))
    lay = look["lay_length"]
    tile = lay * ROPE_TILE_LAYS
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    theta = 2 * np.pi * u[None, :]
    s = v[:, None] * tile
    base = np.array(look["color"], np.float32)
    tracer = None if look.get("tracer") is None else np.array(look["tracer"], np.float32)
    if look["construction"] == "3-strand":
        phase = theta - 2 * np.pi * s / lay
        w = np.mod(phase * 3 / (2 * np.pi), 1.0)  # across a strand, 0 and 1 are the grooves
        strand = np.floor(np.mod(phase, 2 * np.pi) * 3 / (2 * np.pi)).astype(int)
        lobe = np.sqrt(np.abs(np.cos(1.5 * phase)))
        # yarns twist the other way round within each strand
        yarn = 0.5 + 0.5 * np.sin(2 * np.pi * (7 * w + 3 * s / lay))
        shade = 0.45 + 0.55 * lobe
        shade *= 0.82 + 0.18 * yarn
        tint = 1.0 + 0.06 * (strand - 1)  # strands differ a little
        img = base[None, None, :] * (shade * tint)[..., None]
        if tracer is not None:
            line = (strand == 0) & (np.abs(w - 0.5) < 0.07)
            img[line] = tracer * shade[line][:, None]
    else:
        k = 2 * np.pi * s / (lay * 0.35)
        a = theta * 8 - k
        c = theta * 8 + k
        weave = 0.5 * (np.abs(np.sin(a)) + np.abs(np.sin(c)))
        top_a = (np.floor(a / np.pi) + np.floor(c / np.pi)) % 2 == 0
        carrier = np.where(top_a, np.floor(a / (2 * np.pi)), np.floor(c / (2 * np.pi)) + 8).astype(int) % 16
        along = np.where(top_a, np.mod(a, np.pi), np.mod(c, np.pi)) / np.pi
        yarn = 0.5 + 0.5 * np.cos(2 * np.pi * 4 * along)
        shade = (0.55 + 0.45 * weave) * (0.88 + 0.12 * yarn)
        img = base[None, None, :] * shade[..., None]
        if tracer is not None:
            chosen = rng.choice(16, size=int(rng.integers(1, 4)), replace=False)
            fleck = np.isin(carrier, chosen)
            img[fleck] = tracer * shade[fleck][:, None]
    noise = rng.normal(0.0, 0.04, size=img.shape[:2]).astype(np.float32)
    img *= (1.0 + noise)[..., None]
    if look.get("fuzz", 0.0) > 0:  # loose fibres: short light streaks
        n = int(400 * look["fuzz"])
        lighter = np.clip(base * 1.4 + 0.15, 0, 1)
        for _ in range(n):
            x, y = int(rng.integers(width)), int(rng.integers(height))
            length = int(rng.integers(6, 30))
            dx, dy = rng.normal(size=2)
            norm = max(np.hypot(dx, dy), 1e-6)
            for t in range(length):
                xi = (x + int(dx / norm * t)) % width
                yi = (y + int(dy / norm * t)) % height
                img[yi, xi] = img[yi, xi] * 0.4 + lighter * 0.6
    return to_u8(img[::-1])  # row 0 is the top of the picture, which a texture lookup takes as v = 1


# ----------------------------------------------------------------- floors
def floor_texture(style, rng, size=1024):
    """A tileable floor picture of the given style, with random colours."""
    import cv2

    n1 = fbm(rng, size, 4, 6)
    n2 = fbm(rng, size, 16, 4)
    hue_shift = rng.uniform(-0.08, 0.08, 3)
    if style in ("sand", "rippled_sand", "shell_sand"):
        base = np.array([0.62, 0.55, 0.40]) * rng.uniform(0.6, 1.25) + hue_shift * 0.5
        img = base * (0.7 + 0.3 * n1[..., None]) * (0.9 + 0.1 * n2[..., None])
        if style == "rippled_sand":
            yy, xx = np.mgrid[0:size, 0:size] / size
            warp = n1 * rng.uniform(0.5, 2.0)
            k = int(rng.integers(6, 18))
            rip = 0.5 + 0.5 * np.sin(2 * np.pi * (k * xx + warp))
            img *= (0.8 + 0.2 * rip[..., None])
        if style == "shell_sand":
            for _ in range(int(rng.integers(300, 1500))):
                c = rng.integers(0, size, 2)
                r = int(rng.integers(1, 5))
                col = np.clip(base * 1.5 + rng.uniform(0, 0.2), 0, 1)
                cv2.circle(img, (int(c[0]), int(c[1])), r, col.tolist(), -1)
    elif style == "mud":
        base = np.array([0.30, 0.27, 0.20]) * rng.uniform(0.5, 1.2) + hue_shift * 0.3
        img = base * (0.75 + 0.25 * n1[..., None])
    elif style == "gravel":
        img = np.zeros((size, size, 3)) + np.array([0.35, 0.33, 0.30]) * rng.uniform(0.6, 1.2)
        for _ in range(int(rng.integers(1500, 4000))):
            c = rng.integers(0, size, 2)
            ax = (int(rng.integers(4, 22)), int(rng.integers(4, 22)))
            tone = rng.uniform(0.2, 0.75)
            col = np.clip(tone + rng.uniform(-0.06, 0.06, 3), 0, 1)
            cv2.ellipse(img, (int(c[0]), int(c[1])), ax, float(rng.uniform(0, 180)), 0, 360, col.tolist(), -1)
        img *= (0.85 + 0.15 * n2[..., None])
    elif style == "rock":
        base = np.array([0.42, 0.41, 0.38]) * rng.uniform(0.5, 1.2) + hue_shift * 0.3
        img = base * (0.5 + 0.5 * n1[..., None]) * (0.8 + 0.2 * n2[..., None])
        cracks = np.abs(fbm(rng, size, 8, 3) - 0.5) < 0.012
        img[cracks] *= 0.4
    elif style == "silt_weed":
        base = np.array([0.33, 0.33, 0.22]) * rng.uniform(0.6, 1.2) + hue_shift * 0.3
        img = base * (0.7 + 0.3 * n1[..., None])
        weed = fbm(rng, size, 12, 3) > rng.uniform(0.6, 0.75)
        img[weed] = img[weed] * 0.5 + np.array([0.05, 0.18, 0.06]) * 0.5
    else:  # pool_tiles
        tiles = int(rng.integers(4, 10))
        img = np.zeros((size, size, 3)) + np.array([0.55, 0.7, 0.8]) * rng.uniform(0.7, 1.1)
        step = size // tiles
        img[::step, :] = 0.25
        img[:, ::step] = 0.25
        img = cv2.dilate(img.astype(np.float32), np.ones((1, 1))) * (0.92 + 0.08 * n2[..., None])
    return to_u8(img)


def surface_texture(material, rng, size=512):
    """Pilings, walls and rocks: concrete, timber, steel, stone, sheet piles,
    with marine growth."""
    n1 = fbm(rng, size, 4, 6)
    n2 = fbm(rng, size, 24, 3)
    if material == "timber":
        yy, xx = np.mgrid[0:size, 0:size] / size
        grain = 0.5 + 0.5 * np.sin(2 * np.pi * (30 * xx + 3 * n1))
        img = np.array([0.36, 0.26, 0.16]) * rng.uniform(0.6, 1.2) * (0.75 + 0.25 * grain[..., None])
    elif material == "steel":
        img = np.array([0.38, 0.22, 0.12]) * rng.uniform(0.6, 1.3) * (0.6 + 0.4 * n1[..., None])
    elif material == "sheet_pile":
        yy, xx = np.mgrid[0:size, 0:size] / size
        ribs = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4 * xx))
        img = np.array([0.32, 0.25, 0.18]) * rng.uniform(0.6, 1.2) * (0.7 + 0.3 * ribs[..., None]) * (0.8 + 0.2 * n1[..., None])
    elif material == "stone":
        img = np.array([0.45, 0.43, 0.40]) * rng.uniform(0.6, 1.2) * (0.7 + 0.3 * n1[..., None])
        rows = int(rng.integers(4, 9))
        step = size // rows
        img[::step, :] *= 0.35
        for r in range(rows):
            off = int(rng.integers(0, step))
            img[r * step:(r + 1) * step, off::step] *= 0.35
    else:  # concrete (and rocks)
        img = np.array([0.5, 0.5, 0.48]) * rng.uniform(0.4, 1.2) * (0.75 + 0.25 * n1[..., None]) * (0.9 + 0.1 * n2[..., None])
    growth = fbm(rng, size, 6, 4) > rng.uniform(0.45, 0.8)  # weed, mussels, slime
    colour = [np.array([0.08, 0.2, 0.08]), np.array([0.12, 0.1, 0.06]), np.array([0.25, 0.28, 0.12])][int(rng.integers(3))]
    img = img.copy()
    img[growth] = img[growth] * 0.35 + colour * 0.65 * (0.8 + 0.4 * n2[growth][:, None])
    return to_u8(img)
