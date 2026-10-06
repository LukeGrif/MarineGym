"""Geometry shared by the asset builder, the planner and the capture.

Frames:
- world: Isaac Sim's, Z up, metres. The water surface is at z = 0.
- ROV: x forward, y left, z up (BlueSim's BlueRov node: +Z forward, +Y up,
  +X left; GODOT_TO_ROV turns one into the other).
- camera (USD): looks along its -Z, +Y up, +X right. Labels use BlueSim's
  [right, up, ahead] = [x, y, -z].
"""
import re

import numpy as np

# isaac/ROV (forward, left, up) = GODOT_TO_ROV @ godot (x left, y up, z forward)
GODOT_TO_ROV = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
Z_UP = np.array([0.0, 0.0, 1.0])


def godot_transform(text):
    """'Transform( xx, xy, xz, yx, yy, yz, zx, zy, zz, ox, oy, oz )' (rows of the
    basis, then the origin, as Godot 3 writes them) -> 4x4."""
    v = [float(x) for x in re.findall(r"[-+0-9.e]+", text.split("(", 1)[1])]
    t = np.eye(4)
    t[:3, :3] = np.array(v[:9]).reshape(3, 3)
    t[:3, 3] = v[9:12]
    return t


def godot_to_rov(t):
    """A 4x4 pose in BlueSim's ROV frame -> the same pose in the ROV frame here."""
    c = np.eye(4)
    c[:3, :3] = GODOT_TO_ROV
    return c @ t @ np.linalg.inv(c)


def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


def rotation(axis, angle):
    """Rotation matrix about a unit axis (radians, right-handed)."""
    x, y, z = normalize(axis)
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def pose(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply(T, points):
    p = np.asarray(points, float)
    return p @ T[:3, :3].T + T[:3, 3]


def look_at(eye, target, up=Z_UP):
    """Camera rotation (columns: camera x=right, y=up, z=back in world) looking
    from eye at target with the given world up."""
    f = normalize(np.asarray(target, float) - np.asarray(eye, float))
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:  # looking straight up or down
        r = np.cross(f, [1.0, 0.0, 0.0])
    r = normalize(r)
    u = np.cross(r, f)
    return np.column_stack([r, u, -f])


def camera_from_rov(tilt_deg):
    """The camera's rotation in the ROV frame, tilted down tilt_deg."""
    t = np.radians(tilt_deg)
    ahead = np.array([np.cos(t), 0.0, -np.sin(t)])
    right = np.array([0.0, -1.0, 0.0])
    up = np.cross(right, ahead)
    return np.column_stack([right, up, -ahead])


def to_camera(points, cam_R, cam_t):
    """World points -> [right, up, ahead] in metres from the camera."""
    q = (np.asarray(points, float) - cam_t) @ cam_R
    return np.column_stack([q[:, 0], q[:, 1], -q[:, 2]]) if q.ndim == 2 else np.array([q[0], q[1], -q[2]])


def intrinsics(width, height, hfov_deg):
    f = (width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    return {"fx": f, "fy": f, "cx": width / 2.0, "cy": height / 2.0}


def project(cam_points, k):
    """[right, up, ahead] -> pixel (u, v), and ahead."""
    p = np.asarray(cam_points, float)
    z = p[..., 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return np.stack([k["cx"] + k["fx"] * p[..., 0] / zs, k["cy"] - k["fy"] * p[..., 1] / zs], -1), z


def bezier(p0, p1, p2, p3, steps=80):
    t = np.linspace(0.0, 1.0, steps + 1)[:, None]
    u = 1.0 - t
    return u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3


def resample(points, spacing):
    """Points along a polyline, evenly spaced (Catmull-Rom between the given
    points, so the rope's mesh bends smoothly). Returns the points and the
    distance of each along the line."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return p, np.zeros(len(p))
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    fine = []
    sub = 8
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t = np.linspace(0.0, 1.0, sub, endpoint=False)[:, None]
        fine.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                           + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    fine.append(p[-1:])
    fine = np.vstack(fine)
    seg = np.linalg.norm(np.diff(fine, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
    targets = np.linspace(0.0, s[-1], n)
    out = np.column_stack([np.interp(targets, s, fine[:, k]) for k in range(3)])
    return out, targets


def frames(points):
    """Tangents and a normal/binormal carried along the line (no twisting)."""
    p = np.asarray(points, float)
    t = normalize(np.gradient(p, axis=0))
    n = np.zeros_like(p)
    ref = Z_UP if abs(t[0] @ Z_UP) < 0.9 else np.array([1.0, 0.0, 0.0])
    n[0] = normalize(np.cross(t[0], ref))
    for i in range(1, len(p)):
        v = n[i - 1] - t[i] * (t[i] @ n[i - 1])
        n[i] = normalize(v) if np.linalg.norm(v) > 1e-9 else n[i - 1]
    b = np.cross(t, n)
    return t, n, b


def tube(points, radius, sides=10, radius_fn=None, along=None, caps=True):
    """A tube along the points. radius_fn(theta, s) -> radius per vertex (for
    strands). Each ring has sides + 1 vertices (the last repeats the first
    with u = 1, so a texture doesn't smear across the seam). Returns points
    (V, 3), normals (V, 3), triangles (F, 3) facing out, and uv (u round the
    tube 0-1, v = distance along it in m)."""
    p = np.asarray(points, float)
    t, n, b = frames(p)
    s = along if along is not None else np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    cols = sides + 1
    theta = 2 * np.pi * np.arange(cols) / sides
    dirs = np.cos(theta)[None, :, None] * n[:, None, :] + np.sin(theta)[None, :, None] * b[:, None, :]
    r = np.full((len(p), cols), radius, float) if radius_fn is None else radius_fn(theta[None, :], s[:, None])
    verts = p[:, None, :] + dirs * r[..., None]
    normals = dirs
    if radius_fn is not None:  # tilt the normals with the relief (finite differences round the tube)
        h = 2 * np.pi / sides
        dr = (radius_fn(theta[None, :] + h, s[:, None]) - radius_fn(theta[None, :] - h, s[:, None])) / (2 * h)
        side = np.cross(t[:, None, :], dirs)
        normals = normalize(dirs - side * (dr / np.maximum(r, 1e-6))[..., None])
    i = np.arange(len(p) - 1)[:, None]
    j = np.arange(sides)[None, :]
    a, bq, c, d = i * cols + j, (i + 1) * cols + j, i * cols + j + 1, (i + 1) * cols + j + 1
    faces = np.stack([np.stack([a, c, bq], -1), np.stack([c, d, bq], -1)], -2).reshape(-1, 3)
    uv = np.stack(np.broadcast_arrays((theta / (2 * np.pi))[None, :], s[:, None]), -1).reshape(-1, 2)
    verts = verts.reshape(-1, 3)
    normals = normals.reshape(-1, 3)
    if caps:  # close the two ends
        extra_v, extra_n, extra_uv, extra_f = [], [], [], []
        for ring, sign in ((0, -1.0), (len(p) - 1, 1.0)):
            centre = len(verts) + len(extra_v)
            extra_v.append(p[ring])
            extra_n.append(t[ring] * sign)
            extra_uv.append([0.5, s[ring]])
            k = ring * cols + np.arange(sides)
            tri = np.stack([np.full(sides, centre), k + 1, k], -1) if sign > 0 else np.stack([np.full(sides, centre), k, k + 1], -1)
            extra_f.append(tri)
        verts = np.vstack([verts, extra_v])
        normals = np.vstack([normals, extra_n])
        uv = np.vstack([uv, extra_uv])
        faces = np.vstack([faces] + extra_f)
    return verts, normals, faces, uv


def strand_radius(radius, construction, lay_length, rng_phase=0.0):
    """The rope's surface, so it isn't a smooth tube: three strands twisted
    round each other (3-strand, lay_length per turn) or a braided cover (a
    bumpy basket weave)."""
    def fn(theta, s):
        if construction == "3-strand":
            phase = theta - 2 * np.pi * s / lay_length + rng_phase
            # three round strands: each lobe a half-circle bump, deep grooves between
            lobe = np.abs(np.cos(1.5 * phase))
            return radius * (0.80 + 0.20 * np.sqrt(lobe))
        # braided: 16 carriers, 8 each way, over-under
        k = 8
        a = theta * k - 2 * np.pi * s / (lay_length * 0.35) * k / 8
        c = theta * k + 2 * np.pi * s / (lay_length * 0.35) * k / 8
        weave = 0.5 * (np.abs(np.sin(a)) + np.abs(np.sin(c)))
        return radius * (0.93 + 0.07 * weave)
    return fn
