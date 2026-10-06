"""What goes in each picture, chosen exactly as BlueSim's capture does it
(bluesim/scripts/capture.gd, target_rope.gd, rope_types.gd): the same
choices and odds, in Isaac Sim's world (Z up, water surface at z = 0).

Plain numpy and a seeded generator, no Isaac Sim, so the same seed gives the
same pictures' contents, and it can be checked without a GPU (dryrun.py).
"""
import colorsys

import numpy as np

from . import geom, worlds
from .ropesim import Rope, polyline

HFOV = 80.0  # deg, BlueROV2 Low-Light HD USB Camera
PICTURES_PER_SCENE = 20  # new rope (and world layout) after this many pictures
SETTLE_TIME = 3.0  # s of physics after building a new rope
MOVE_TIME = 0.3  # s of physics between pictures
NO_ROPE_FRACTION = 0.1
TETHER_VIEW_FRACTION = 0.25
TETHER_DIAMETER = 0.012  # m, as drawn (the real one is 7.6 mm)
TETHER_PIECE = 0.145  # m, label segment length
MIN_DISTANCE = 0.25
MAX_DISTANCE = 4.0
ROV_VIEW_FRACTION = 0.6
JAWS_FROM_CAMERA = np.array([0.12, 0.0, -0.26])  # ROV frame (forward, left, up)
ROPE_DIAMETER = 0.0508
ROPE_SPACING = 0.1  # m between the rope's physics points
DT = 1.0 / 60.0

MATERIALS = [  # rope_types.gd (no "fixed" rod: never in the pictures)
    ("polypropylene", 910.0), ("dyneema", 975.0), ("nylon", 1140.0), ("polyester", 1380.0), ("leadcore", 2000.0)]
SETUPS = ["surface_floor", "hanging", "standing", "u_shape"]
LENGTHS = [5.0, 10.0, 20.0]
CURRENT_SPEEDS = [0.0, 0.1, 0.25, 0.5]
CURRENT_DIRECTIONS = [("left", 90.0), ("right", -90.0), ("ahead", 0.0), ("behind", 180.0)]
U_SPAN = 3.0
# rope_types.gd LOOKS (marine rope colours), as RGB
LOOK_COLOURS = [(0.72, 0.06, 0.05), (0.04, 0.22, 0.7), (0.95, 0.72, 0.04), (0.95, 0.38, 0.03), (0.08, 0.4, 0.14),
                (0.88, 0.88, 0.84), (0.7, 0.55, 0.33), (0.04, 0.07, 0.25), (0.9, 0.9, 0.87), (0.05, 0.2, 0.65),
                (0.04, 0.04, 0.045), (0.8, 0.05, 0.04)]
TRACERS = [(0.9, 0.9, 0.9), (0.75, 0.05, 0.05), (0.05, 0.2, 0.7), (0.95, 0.75, 0.05), (0.03, 0.03, 0.03)]
TETHER_COLOURS = [(0.05, 0.2, 0.7), (0.03, 0.03, 0.03), (0.9, 0.9, 0.88), (0.95, 0.4, 0.05), (0.1, 0.45, 0.15), (0.5, 0.5, 0.5)]


def hsv(h, s, v):
    return tuple(colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0)))


def rgb_to_hsv(c):
    return colorsys.rgb_to_hsv(*c)


class Planner:
    def __init__(self, seed=1, size=(1920, 1080), world_kind="seabed", rov_meta=None, wreck_path=None):
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.world_kind = world_kind
        self.wreck_path = wreck_path
        meta = rov_meta or {}
        self.camera_mount = np.array(meta.get("camera_position", [0.378, 0.004, 0.013]))
        self.tether_mount = np.array(meta.get("tether_position", [-0.45, 0.0, 0.05]))
        self.k = geom.intrinsics(size[0], size[1], HFOV)
        self.vfov = np.degrees(2 * np.arctan(np.tan(np.radians(HFOV / 2)) * size[1] / size[0]))
        self.scene = None
        self.index = 0

    def u(self, a=0.0, b=1.0):
        return float(self.rng.uniform(a, b))

    # ---------------------------------------------------------------- scene
    def new_scene(self):
        """A new world layout, rope (material, setup, length, current), the ROV's
        place and its tether, then SETTLE_TIME of physics."""
        rng = self.rng
        world = worlds.make_world(self.world_kind, rng, self.wreck_path)
        heading = self.u(0, 2 * np.pi)  # the ROV's forward when the rope is placed
        forward = np.array([np.cos(heading), np.sin(heading), 0.0])
        side = np.cross(geom.Z_UP, forward)  # left
        material, density = MATERIALS[int(rng.integers(len(MATERIALS)))]
        setup = SETUPS[int(rng.integers(len(SETUPS)))]
        length = LENGTHS[int(rng.integers(len(LENGTHS)))]
        speed = CURRENT_SPEEDS[int(rng.integers(len(CURRENT_SPEEDS)))]
        direction, angle = CURRENT_DIRECTIONS[int(rng.integers(len(CURRENT_DIRECTIONS)))]
        from_dir = geom.rotation(geom.Z_UP, np.radians(angle)) @ forward
        current = -from_dir * speed
        anchor = np.array([world["anchor"][0], world["anchor"][1]])
        floor = world["floor"]
        floor_z = float(floor(anchor[0], anchor[1]))
        depth = -floor_z
        top = np.array([anchor[0], anchor[1], 0.0])
        pins = {}
        if setup == "u_shape":
            used = float(np.clip(length, U_SPAN + 1.0, U_SPAN + 2.0 * (depth - 1.0)))
            drop = (used - U_SPAN) / 2.0
            left = top - side * U_SPAN / 2.0
            right = left + side * U_SPAN
            pts = polyline([left, left - [0, 0, drop], right - [0, 0, drop], right], ROPE_SPACING)
            pins = {0: pts[0], len(pts) - 1: pts[-1]}
        else:
            if setup == "surface_floor":
                used = depth
                top_z = 0.0
            elif setup == "hanging":
                used = min(length, depth)
                top_z = 0.0
            else:  # standing
                used = min(length, depth)
                top_z = floor_z + used
            top_xy = anchor.copy()
            if setup == "standing":  # never perfectly upright: a heavy rope then falls over, a light one rises
                lean = np.radians(self.u(2.0, 8.0))
                a = self.u(0, 2 * np.pi)
                top_xy = anchor + used * np.sin(lean) * np.array([np.cos(a), np.sin(a)])
            pts = polyline([[top_xy[0], top_xy[1], top_z], [anchor[0], anchor[1], top_z - used]], ROPE_SPACING)
            if setup in ("surface_floor", "hanging"):
                pins[0] = pts[0]
            if setup in ("surface_floor", "standing"):
                pins[len(pts) - 1] = pts[-1]
        rope = Rope(pts, ROPE_DIAMETER, density, pins, floor=floor, current=current, obstacles=world["obstacles"])
        # the ROV waits 3 m back from the rope, at mid depth (where BlueSim starts it),
        # its tether running up to the surface behind it
        rov_home = top - forward * 3.0
        rov_home[2] = -min(max(depth * 0.5, 1.0), depth - 0.6)
        self.scene = {
            "world": world, "rope": rope, "forward": forward, "current": current,
            "rov_home": geom.pose(self.rov_rotation(forward, 0, 0, 0), rov_home),
            "jaws_open": bool(self.rng.random() < 0.6),
            "info": {"rope": material, "density": density, "setup": setup, "length": length,
                     "rope_length_m": round(float(rope.arc[-1]), 2), "current": speed, "current_from": direction,
                     "depth_m": round(depth, 2), "world": world["kind"], "floor": world["floor_style"]},
        }
        self.scene["tether"] = self.make_tether(self.scene["rov_home"])
        self.run(SETTLE_TIME)
        return self.scene

    def rov_rotation(self, forward, yaw_deg, pitch_deg, roll_deg):
        f = geom.normalize(forward)
        left = np.cross(geom.Z_UP, f)
        R = np.column_stack([f, left, geom.Z_UP])
        R = geom.rotation(geom.Z_UP, np.radians(yaw_deg)) @ R
        R = geom.rotation(R[:, 1], np.radians(pitch_deg)) @ R
        R = geom.rotation(R[:, 0], np.radians(roll_deg)) @ R
        return R

    def make_tether(self, rov_pose, settle=1.0):
        """The ROV's tether: from the back of the ROV, a loop down and back, up to
        a float at the surface 2-6 m behind; a slightly heavy cable in the current."""
        start = geom.apply(rov_pose, self.tether_mount[None])[0]
        back = -rov_pose[:3, 0]
        back[2] = 0
        back = geom.normalize(back) if np.linalg.norm(back) > 1e-6 else np.array([1.0, 0, 0])
        lateral = np.cross(geom.Z_UP, back)
        end = start + back * self.u(2.0, 6.0) + lateral * self.u(-2.0, 2.0)
        end[2] = 0.0
        sag = start + back * self.u(0.3, 1.5) + lateral * self.u(-0.8, 0.8) - geom.Z_UP * self.u(0.0, 1.5)
        floor = self.scene["world"]["floor"] if self.scene else None
        if floor is not None:
            sag[2] = max(sag[2], float(floor(sag[0], sag[1])) + 0.2)
        pts = polyline([start, sag, (sag + end) / 2 + geom.Z_UP * 0.3, end], TETHER_PIECE)
        tether = Rope(pts, 0.0076, 1030.0, {0: pts[0], len(pts) - 1: pts[-1]},
                      floor=floor, current=self.scene["current"] if self.scene else (0, 0, 0),
                      obstacles=self.scene["world"]["obstacles"] if self.scene else (), iterations=20, bend=0.0)
        tether.run(settle)
        return tether

    def run(self, seconds):
        """Physics: the rope (and the ROV's tether) move for this long."""
        s = self.scene
        for _ in range(int(round(seconds / DT))):
            s["rope"].step(DT)
            s["tether"].step(DT)

    # ---------------------------------------------------------------- views
    def next(self):
        """Everything for the next picture (physics paused while it's taken)."""
        if self.scene is None or self.index % PICTURES_PER_SCENE == 0:
            self.new_scene()
        else:
            self.run(MOVE_TIME)
        view = self.new_view()
        view["index"] = self.index
        self.index += 1
        return view

    def new_view(self):
        s = self.scene
        rope_pts = s["rope"].x.copy()
        info = dict(s["info"])
        info["jaws_open"] = s["jaws_open"]
        tether = s["tether"]
        rov_pose = s["rov_home"]
        if self.rng.random() < ROV_VIEW_FRACTION:
            near, cam_R, cam_t, rov_pose, extra = self.rov_view(rope_pts)
            tether = self.make_tether(rov_pose)  # follows the ROV for this picture
        else:
            near, cam_R, cam_t, extra = self.orbit_view(rope_pts, tether.x)
        info.update(extra)
        cables, kinds = self.add_cables(near[0], near[1], cam_R, cam_t, info["view"])
        info["cables"] = kinds
        look = self.random_look()
        water = self.random_water()
        tether_colour = self.random_tether_colour()
        info.update({
            "rope_color": list(look["color"]), "construction": look["construction"], "tracer": look["tracer"] is not None,
            "fuzz": look["fuzz"], "tether_color": list(tether_colour), "visibility_m": water["fog_end"],
            "water_color": list(water["tint"]), "lamp": water["lamp"], "sun": water["sun"], "ambient": water["ambient"],
        })
        return {"camera_R": cam_R, "camera_t": cam_t, "rov_pose": rov_pose, "jaws_open": s["jaws_open"],
                "rope_points": rope_pts, "tether_points": tether.x.copy(), "cables": cables, "look": look,
                "water": water, "tether_color": tether_colour, "world": s["world"], "settings": info}

    def orbit_view(self, points, tether_pts):
        rng = self.rng
        world = self.scene["world"]
        target = None
        along = geom.Z_UP.copy()
        at_tether = len(tether_pts) > 0 and rng.random() < TETHER_VIEW_FRACTION
        if at_tether:
            target = tether_pts[int(rng.integers(len(tether_pts)))]
        for _ in range(0 if at_tether else 50):
            k = int(rng.integers(len(points) - 1))
            p = points[k] + (points[k + 1] - points[k]) * rng.random()
            if p[2] < -0.5:
                target = p
                along = geom.normalize(points[k + 1] - points[k])
                break
        if target is None:
            target = points[-1]
        position = target
        for _ in range(50):
            distance = float(np.exp(rng.uniform(np.log(MIN_DISTANCE), np.log(MAX_DISTANCE))))
            heading = rng.uniform(0, 2 * np.pi)
            elevation = np.radians(rng.uniform(-30, 30))
            position = target + distance * np.array([np.cos(elevation) * np.cos(heading),
                                                     np.cos(elevation) * np.sin(heading), np.sin(elevation)])
            position[2] = min(position[2], -0.3)
            position[2] = max(position[2], float(world["floor"](position[0], position[1])) + 0.15)
            if worlds.clear_of_obstacles(world, position[:2], 0.2):
                break
        R = geom.look_at(position, target)
        # turn so the rope isn't always in the middle, and sometimes out of view
        yaw = rng.uniform(-0.45, 0.45) * HFOV
        pitch = rng.uniform(-0.45, 0.45) * self.vfov
        looking_away = bool(rng.random() < NO_ROPE_FRACTION)
        if looking_away:
            yaw = rng.uniform(70, 180) * (1 if rng.random() < 0.5 else -1)
        R = geom.rotation(geom.Z_UP, np.radians(yaw)) @ R
        R = geom.rotation(R[:, 0], np.radians(pitch)) @ R
        R = geom.rotation(R[:, 2], np.radians(rng.uniform(-10, 10))) @ R
        extra = {"view": "orbit", "aimed_at_tether": bool(at_tether), "looking_away": looking_away}
        if at_tether:  # cables near the camera rather than the rope
            return (position - R[:, 2] * 1.0, along), R, position, extra  # 1 m ahead of the camera
        return (target, along), R, position, extra

    def rov_view(self, points):
        rng = self.rng
        bottom = points[:, 2].min()
        p, k = None, 0
        for _ in range(50):
            k = int(rng.integers(len(points) - 1))
            q = points[k] + (points[k + 1] - points[k]) * rng.random()
            if -0.7 > q[2] > bottom + 0.4:
                p = q
                break
        if p is None:
            k = len(points) // 2 - 1
            p = points[k]
        along = geom.normalize(points[k + 1] - points[k])
        heading = rng.uniform(0, 2 * np.pi)
        forward = np.array([np.cos(heading), np.sin(heading), 0.0])
        left = np.cross(geom.Z_UP, forward)
        r = rng.random()
        if r < 0.25:
            phase = "in_jaws"
            ahead, across, height = rng.uniform(-0.04, 0.04), rng.uniform(-0.03, 0.03), rng.uniform(-0.05, 0.05)
        elif r < 0.5:
            phase = "at_jaws"
            ahead, across, height = rng.uniform(0.06, 0.35), rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1)
        elif r < 0.92:
            phase = "approach"
            ahead = float(np.exp(rng.uniform(np.log(0.35), np.log(2.5))))
            across, height = rng.uniform(-0.5, 0.5) * ahead, rng.uniform(-0.3, 0.3)
        else:
            phase = "rope_beside"  # rope off to the side, maybe out of view
            ahead = rng.uniform(0.3, 2.0)
            across, height = rng.uniform(1.0, 2.0) * (1 if rng.random() < 0.5 else -1), rng.uniform(-0.3, 0.3)
        jaws_world = p - forward * ahead - left * across - geom.Z_UP * height
        R = self.rov_rotation(forward, rng.uniform(-10, 10), rng.uniform(-4, 4), rng.uniform(-4, 4))
        origin = jaws_world - R @ (self.camera_mount + JAWS_FROM_CAMERA)
        origin[2] = min(origin[2], -0.4)
        rov_pose = geom.pose(R, origin)
        # tilt the camera down (the real one tilts +/-90 deg; BlueSim's 45)
        tilt = rng.uniform(25, 45) if rng.random() < 0.85 else rng.uniform(0, 25)
        cam_R = R @ geom.camera_from_rov(tilt)
        cam_t = geom.apply(rov_pose, self.camera_mount[None])[0]
        extra = {"view": "rov", "phase": phase, "camera_tilt_deg": float(tilt), "aimed_at_tether": False, "looking_away": False}
        return (p, along), cam_R, cam_t, rov_pose, extra

    # ---------------------------------------------------------------- cables
    def random_unit(self):
        v = self.rng.normal(size=3)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else geom.Z_UP.copy()

    def add_cables(self, anchor, along, cam_R, cam_t, view):
        """Hard negatives: 0-2 tether-like cables (smooth, 6-16 mm, mostly
        yellow) crossing the rope, alongside it, or in front of the camera and
        gripper (some looped). Labelled tether."""
        rng = self.rng
        r = rng.random()
        n = 0 if r < 0.4 else (1 if r < 0.85 else 2)
        cables, kinds = [], []
        floor = self.scene["world"]["floor"]
        for _ in range(n):
            r = rng.random()
            if view == "rov":
                kind = "front" if r < 0.45 else ("cross" if r < 0.8 else "alongside")
            else:
                kind = "cross" if r < 0.45 else ("alongside" if r < 0.75 else "front")
            diameter = float(rng.uniform(0.006, 0.016))
            pts = self.cable_points(kind, anchor, along, cam_R, cam_t)
            pts[:, 2] = np.minimum(pts[:, 2], -diameter)
            pts[:, 2] = np.maximum(pts[:, 2], floor(pts[:, 0], pts[:, 1]) + diameter / 2)
            cables.append({"diameter": diameter, "points": pts, "color": self.random_tether_colour(),
                           "roughness": float(rng.uniform(0.3, 0.8)), "kind": kind})
            kinds.append(kind)
        return cables, kinds

    def cable_points(self, kind, anchor, along, cam_R, cam_t):
        rng = self.rng
        look = -cam_R[:, 2]
        loop = False
        if kind == "front":
            distance = rng.uniform(0.15, 0.8)
            centre = cam_t + distance * (look + cam_R[:, 0] * rng.uniform(-0.4, 0.4) + cam_R[:, 1] * rng.uniform(-0.3, 0.3))
            direction = geom.normalize(np.cross(look, self.random_unit()))
            length = rng.uniform(1.0, 3.0)
            bend = rng.uniform(0.1, 0.5)
            loop = bool(rng.random() < 0.5)
        else:
            away = geom.normalize(np.cross(along, self.random_unit()))
            clearance = ROPE_DIAMETER / 2.0 + 0.008 + 0.01
            centre = anchor + away * (clearance + rng.uniform(0.0, 0.35)) * (1 if rng.random() < 0.5 else -1)
            if kind == "cross":
                perp = geom.normalize(np.cross(along, away))
                direction = geom.normalize(perp + (along - perp) * rng.uniform(-0.5, 0.5))
                length = rng.uniform(2.0, 5.0)
                bend = rng.uniform(0.05, 0.3)
            else:
                direction = geom.normalize(along + self.random_unit() * 0.15)
                length = rng.uniform(1.5, 4.0)
                bend = rng.uniform(0.02, 0.15)
        p0 = centre - direction * length / 2.0
        p3 = centre + direction * length / 2.0
        if loop:  # control points crossed over: a loop near the middle
            p1 = p3 + self.random_unit() * bend * length * 0.5
            p2 = p0 + self.random_unit() * bend * length * 0.5
        else:
            p1 = p0 + (p3 - p0) / 3.0 + self.random_unit() * bend * length * 0.3
            p2 = p0 + (p3 - p0) * 2.0 / 3.0 + self.random_unit() * bend * length * 0.3
        return geom.bezier(p0, p1, p2, p3, 80)

    # ---------------------------------------------------------------- looks
    def random_look(self):
        """3-strand or braided (never smooth); 30% reds, 15% yellows, 35% marine
        rope colours, 20% anything; 30% with a tracer/fleck; some loose fibres."""
        rng = self.rng
        construction = "3-strand" if rng.random() < 0.5 else "braided"
        r = rng.random()
        if r < 0.3:
            color = hsv(rng.uniform(-0.05, 0.04), rng.uniform(0.6, 1.0), rng.uniform(0.35, 0.9))
        elif r < 0.45:
            color = hsv(rng.uniform(0.1, 0.18), rng.uniform(0.6, 1.0), rng.uniform(0.5, 1.0))
        elif r < 0.8:
            h, s, v = rgb_to_hsv(LOOK_COLOURS[int(rng.integers(len(LOOK_COLOURS)))])
            color = hsv(h + rng.uniform(-0.03, 0.03), s * rng.uniform(0.8, 1.2), max(0.03, v * rng.uniform(0.7, 1.2)))
        else:
            color = hsv(rng.random(), rng.uniform(0.0, 1.0), rng.uniform(0.1, 1.0))
        tracer = None
        if rng.random() < 0.3:
            tracer = TRACERS[int(rng.integers(len(TRACERS)))] if rng.random() < 0.8 else hsv(rng.random(), 0.8, 0.8)
        fuzz = float(rng.uniform(0.3, 1.0)) if construction == "3-strand" and rng.random() < 0.15 else 0.0
        return {"construction": construction, "color": tuple(float(c) for c in color), "tracer": tracer, "fuzz": fuzz,
                "lay_length": ROPE_DIAMETER * 3.3 * float(rng.uniform(0.9, 1.15)), "seed": int(rng.integers(1 << 30))}

    def random_tether_colour(self):
        rng = self.rng
        if rng.random() < 0.5:  # yellow, like a Fathom tether
            return hsv(rng.uniform(0.11, 0.15), rng.uniform(0.7, 1.0), rng.uniform(0.6, 1.0))
        h, s, v = rgb_to_hsv(TETHER_COLOURS[int(rng.integers(len(TETHER_COLOURS)))])
        return hsv(h, s, min(max(v * rng.uniform(0.7, 1.2), 0.02), 1.0))

    def random_water(self):
        """Water tint and visibility (2-30 m), ambient light, sun, the ROV's lamp
        (capture.gd set_looks), plus particles in the water."""
        rng = self.rng
        tint = hsv(rng.uniform(0.45, 0.62), rng.uniform(0.3, 0.9), rng.uniform(0.15, 0.7))
        begin = rng.uniform(0.0, 2.0)
        end = begin + float(np.exp(rng.uniform(np.log(2.0), np.log(30.0))))
        lighten = rng.uniform(0.0, 0.5)
        ambient_colour = tuple(c + (1 - c) * lighten for c in tint)
        return {"tint": tuple(float(c) for c in tint), "fog_begin": float(begin), "fog_end": float(end),
                "ambient_color": tuple(float(c) for c in ambient_colour), "ambient": float(rng.uniform(0.2, 1.2)),
                "lamp": float(rng.uniform(0.0, 4.0)) if rng.random() < 0.7 else 0.0,
                "sun": float(rng.uniform(0.0, 1.0)), "sun_tilt_deg": float(rng.uniform(0, 35)),
                "sun_heading_deg": float(rng.uniform(0, 360)),
                "particles": float(rng.uniform(0.0, 1.0)) if rng.random() < 0.6 else 0.0,
                "seed": int(rng.integers(1 << 30))}
