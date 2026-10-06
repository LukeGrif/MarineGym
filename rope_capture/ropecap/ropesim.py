"""A physics rope: a chain of points (position based dynamics) with the same
forces as BlueSim's rope pieces (scripts/rope_segment.gd):
  - weight minus buoyancy: (rope density - water density) * volume * g
  - water drag on the flow through the water (Morison): across the rope Cd 1.2
    on the projected area, along it Cd 0.01 on the surface area
  - added mass (Ca = 1): the water it pushes aside when it accelerates
  - the current, as the flow the drag works against
held by pins (the surface buoy, the floor anchor, U ends), lying on the floor,
floating at the surface and kept out of pilings. It runs in numpy, so a
picture's physics is paused simply by not stepping it, and the same seed gives
the same rope on any machine.
"""
import numpy as np

RHO_WATER = 1000.0  # kg/m3, fresh water (as BlueSim's pool)
G = 9.81
CD_NORMAL = 1.2
CD_TANGENT = 0.01
C_ADDED_MASS = 1.0


class Rope:
    """points: (N, 3) start shape (spaced `spacing` apart along the rope);
    pins: {index: (3,) position}; floor(x, y) -> z of the floor;
    obstacles: [(x, y, radius)] vertical cylinders (pilings)."""

    def __init__(self, points, diameter, density, pins=None, floor=None, surface=0.0,
                 current=(0.0, 0.0, 0.0), obstacles=(), iterations=30, bend=0.02):
        self.x = np.array(points, float)
        self.v = np.zeros_like(self.x)
        self.n = len(self.x)
        self.rest = np.linalg.norm(np.diff(self.x, axis=0), axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(self.rest)])
        self.d = diameter
        self.r = diameter / 2.0
        self.density = density
        self.pins = {int(k): np.asarray(p, float) for k, p in (pins or {}).items()}
        self.floor = floor or (lambda x, y: np.full(np.shape(x), -1e9))
        self.surface = surface
        self.current = np.asarray(current, float)
        self.obs = np.array([tuple(o) for o in obstacles], float).reshape(-1, 3)
        self.iterations = iterations
        self.bend = bend
        seg = np.zeros(self.n)  # rope length each point stands for
        seg[:-1] += self.rest / 2
        seg[1:] += self.rest / 2
        self.seg = seg
        area = np.pi * diameter ** 2 / 4.0
        self.mass = (density + C_ADDED_MASS * RHO_WATER) * area * seg
        self.net_weight = (density - RHO_WATER) * area * seg * G  # N, + sinks
        self.inv_mass = np.ones(self.n)
        for k in self.pins:
            self.inv_mass[k] = 0.0
        self._links = []
        for start in (0, 1):
            i = np.arange(start, self.n - 1, 2)
            wsum = self.inv_mass[i] + self.inv_mass[i + 1]
            i = i[wsum > 0]
            wsum = self.inv_mass[i] + self.inv_mass[i + 1]
            self._links.append((i, self.rest[i], self.inv_mass[i][:, None], self.inv_mass[i + 1][:, None], wsum))
        self._pin_idx = np.array(sorted(self.pins), int)
        self._pin_pos = np.array([self.pins[k] for k in sorted(self.pins)]).reshape(-1, 3)
        self.time = 0.0

    def tangents(self, x):
        t = np.gradient(x, axis=0)
        n = np.linalg.norm(t, axis=1, keepdims=True)
        return t / np.where(n < 1e-9, 1.0, n)

    def forces_dv(self, dt):
        flow = self.v - self.current
        t = self.tangents(self.x)
        along = t * np.sum(t * flow, axis=1, keepdims=True)
        across = flow - along
        drag = -0.5 * RHO_WATER * CD_NORMAL * (self.d * self.seg)[:, None] * np.linalg.norm(across, axis=1, keepdims=True) * across
        drag += -0.5 * RHO_WATER * CD_TANGENT * (np.pi * self.d * self.seg)[:, None] * np.linalg.norm(along, axis=1, keepdims=True) * along
        dv = drag / self.mass[:, None] * dt
        # quadratic drag can at most stop the motion through the water in one step
        big = np.linalg.norm(dv, axis=1) > np.linalg.norm(flow, axis=1)
        dv[big] = -flow[big]
        dv[:, 2] -= self.net_weight / self.mass * dt
        return dv

    def solve_lengths(self, p):
        for i, rest, wa, wb, wsum in self._links:  # red-black Gauss-Seidel: even links, then odd
            d = p[i + 1] - p[i]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            corr = d * ((dist - rest) / (np.maximum(dist, 1e-9) * wsum))[:, None]
            p[i] += corr * wa
            p[i + 1] -= corr * wb

    def solve_bending(self, p):
        if self.bend <= 0 or self.n < 3:
            return
        # pull each point a little towards the middle of its neighbours (stiffness of a 2 inch rope)
        mid = 0.5 * (p[:-2] + p[2:])
        delta = (mid - p[1:-1]) * self.bend
        p[1:-1] += delta * self.inv_mass[1:-1, None]

    def solve_attachments(self, p):
        # long-range attachments: no point further from a pin than the rope between them
        for k, pos in self.pins.items():
            d = p - pos
            dist = np.linalg.norm(d, axis=1)
            limit = np.abs(self.arc - self.arc[k]) + 1e-4
            over = (dist > limit) & (self.inv_mass > 0)
            p[over] = pos + d[over] * (limit[over] / dist[over])[:, None]

    def solve_collisions(self, p, fz):
        free = self.inv_mass > 0  # pinned points stay where they're pinned
        below = (p[:, 2] < fz) & free
        p[below, 2] = fz[below]
        above = (p[:, 2] > self.surface - self.r) & free
        p[above, 2] = self.surface - self.r
        if len(self.obs):
            free_idx = np.nonzero(free)[0]
            d = p[free_idx, None, :2] - self.obs[None, :, :2]  # points x pilings
            dist = np.linalg.norm(d, axis=2)
            reach = self.obs[None, :, 2] + self.r
            inside = dist < reach
            if inside.any():
                pi, oi = np.nonzero(inside)
                safe = np.maximum(dist[pi, oi], 1e-9)
                p[free_idx[pi], :2] = self.obs[oi, :2] + d[pi, oi] * (reach[0, oi] / safe)[:, None]
        return below

    def step(self, dt=1.0 / 60.0):
        self.v += self.forces_dv(dt)
        self.v[self.inv_mass == 0] = 0.0
        p = self.x + self.v * dt
        fz = self.floor(p[:, 0], p[:, 1]) + self.r  # the floor under each point, once a step
        for it in range(self.iterations):
            self.solve_lengths(p)
            self.solve_bending(p)
            if it % 4 == 0 or it == self.iterations - 1:
                self.solve_attachments(p)
            p[self._pin_idx] = self._pin_pos
            on_floor = self.solve_collisions(p, fz)
        new_v = (p - self.x) / dt
        new_v[on_floor, :2] *= 0.7  # friction on the floor
        self.v = new_v
        self.x = p
        self.time += dt

    def run(self, seconds, dt=1.0 / 60.0):
        for _ in range(int(round(seconds / dt))):
            self.step(dt)

    def stretch(self):
        """Longest link over its rest length (1.0 = not stretched)."""
        return float(np.max(np.linalg.norm(np.diff(self.x, axis=0), axis=1) / self.rest))


def polyline(path, spacing):
    """Points every `spacing` m along a polyline (first and last included)."""
    path = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, path[:, k]) for k in range(3)])
