"""Checks that need no Isaac Sim: python -m pytest tests  (from rope_capture/)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ropecap import geom, labels, water  # noqa: E402
from ropecap.plan import HFOV, Planner  # noqa: E402
from ropecap.ropesim import Rope, polyline  # noqa: E402


def test_camera_model():
    k = geom.intrinsics(1920, 1080, HFOV)
    assert k["fx"] == pytest.approx(960 / np.tan(np.radians(40)))
    assert k["fy"] == k["fx"] and (k["cx"], k["cy"]) == (960, 540)
    for tilt in (0, 25, 45):
        R = geom.camera_from_rov(tilt)
        assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1)
    # a point straight ahead of an untilted camera, ROV facing +x
    R = geom.camera_from_rov(0)
    assert np.allclose(geom.to_camera(np.array([[2.0, 0, 0]]), R, np.zeros(3)), [[0, 0, 2]])
    # to the ROV's left is the camera's -right; up is up
    assert geom.to_camera(np.array([0.0, 1, 0]), R, np.zeros(3))[0] == pytest.approx(-1)
    assert geom.to_camera(np.array([0.0, 0, 1]), R, np.zeros(3))[1] == pytest.approx(1)
    # tilted down 30 deg: a point 30 deg below ahead is in the middle of the picture
    R = geom.camera_from_rov(30)
    p = np.array([np.cos(np.radians(30)), 0, -np.sin(np.radians(30))])
    assert np.allclose(geom.to_camera(p, R, np.zeros(3)), [0, 0, 1], atol=1e-9)


def test_look_at():
    R = geom.look_at(np.zeros(3), [1.0, 0, 0])
    assert np.allclose(geom.to_camera(np.array([3.0, 0, 0]), R, np.zeros(3)), [0, 0, 3])
    assert geom.to_camera(np.array([0, 0, 1.0]), R, np.zeros(3))[1] == pytest.approx(1)


def test_tube_is_closed_and_faces_out():
    trimesh = pytest.importorskip("trimesh")
    pts = np.column_stack([np.zeros(40), np.zeros(40), np.linspace(0, -2, 40)])
    for construction in ("3-strand", "braided"):
        v, n, f, uv = geom.tube(pts, 0.0254, 24, geom.strand_radius(0.0254, construction, 0.17))
        m = trimesh.Trimesh(v, f, process=True)
        assert m.is_watertight and m.volume > 0


def test_rope_physics():
    floor = lambda x, y: np.full(np.shape(x), -10.0)  # noqa: E731
    pts = polyline([[0, 0, 0], [0, 0, -10]], 0.1)
    hanging = Rope(pts, 0.0508, 1140, {0: pts[0]}, floor=floor, current=(0.5, 0, 0))
    hanging.run(3.0)
    assert hanging.stretch() < 1.02
    assert np.allclose(hanging.x[0], pts[0])  # still pinned
    assert hanging.x[-1, 0] > 0.3  # carried by the current
    leaning = polyline([[0.5, 0, -0.1], [0, 0, -10]], 0.1)  # the planner starts a standing rope leaning a little
    standing = Rope(leaning, 0.0508, 2000, {len(leaning) - 1: leaning[-1]}, floor=floor)
    standing.run(3.0)
    assert standing.x[:, 2].max() < -6  # lead-core falls over onto the floor
    assert standing.x[:-1, 2].min() >= -10 + 0.0254 - 1e-6  # (the last point is the anchor, on the floor)
    floating = Rope(leaning.copy(), 0.0508, 910, {len(leaning) - 1: leaning[-1]}, floor=floor)
    floating.run(3.0)
    assert floating.x[0, 2] > -1  # polypropylene floats up


def test_rope_stays_out_of_pilings():
    pts = polyline([[0, 0, 0], [0, 0, -8]], 0.1)
    rope = Rope(pts, 0.0508, 1140, {0: pts[0]}, current=(0.5, 0, 0), obstacles=[(0.6, 0.0, 0.3)])
    rope.run(3.0)
    assert np.hypot(rope.x[:, 0] - 0.6, rope.x[:, 1]).min() >= 0.3 + 0.0254 - 1e-3


def test_planner_is_repeatable_and_matches_bluesim_fields():
    a = [Planner(seed=7, size=(960, 540)).next()["settings"] for _ in range(1)]
    p1, p2 = Planner(seed=7, size=(960, 540)), Planner(seed=7, size=(960, 540))
    for _ in range(22):  # crosses a new rope (every 20)
        s1, s2 = p1.next()["settings"], p2.next()["settings"]
        assert s1 == s2
    assert a[0] == Planner(seed=7, size=(960, 540)).next()["settings"]
    s = p1.next()["settings"]
    for key in ("view", "rope", "setup", "length", "current", "construction", "jaws_open", "cables", "visibility_m"):
        assert key in s
    assert s["construction"] in ("3-strand", "braided")
    assert s["rope"] in ("polypropylene", "dyneema", "nylon", "polyester", "leadcore")


def test_look_odds():
    p = Planner(seed=3, size=(960, 540))
    looks = [p.random_look() for _ in range(4000)]
    assert 0.45 < np.mean([lk["construction"] == "3-strand" for lk in looks]) < 0.55
    assert 0.26 < np.mean([lk["tracer"] is not None for lk in looks]) < 0.34
    vis = [p.random_water()["fog_end"] for _ in range(2000)]
    assert min(vis) >= 2.0 and max(vis) <= 32.0


def test_rov_view_puts_rope_at_the_jaws():
    p = Planner(seed=5, size=(960, 540))
    p.new_scene()
    for _ in range(40):
        near, cam_R, cam_t, rov_pose, extra = p.rov_view(p.scene["rope"].x)
        if extra["phase"] != "in_jaws":
            continue
        jaws = geom.apply(rov_pose, (p.camera_mount + np.array([0.12, 0, -0.26]))[None])[0]
        if rov_pose[2, 3] < -0.41:  # not pushed down from the surface
            assert np.linalg.norm(jaws - near[0]) < 0.1


def test_labels_from_replicator():
    data = np.array([[0, 1, 2], [3, 4, 2]], np.uint32)
    id_to_labels = {"0": {"class": "BACKGROUND"}, "1": {"class": "UNLABELLED"}, "2": {"class": "rope"},
                    "3": {"class": "tether"}, "4": {"class": "rov,thing"}}
    assert labels.classes_from_semantic(data, id_to_labels).tolist() == [[0, 0, 1], [2, 3, 1]]


def test_water():
    w = {"tint": (0.1, 0.3, 0.4), "fog_begin": 0.0, "fog_end": 10.0, "lamp": 0.0, "particles": 0.0, "ambient": 1.0}
    k = geom.intrinsics(4, 2, HFOV)
    rgb = np.full((2, 4, 3), 200, np.uint8)
    dist = np.array([[0.001, 10.0, np.inf, 0.0], [5.0, 5.0, 5.0, 5.0]], np.float32)
    out = water.underwater(rgb, dist, w, k).astype(float) / 255
    assert np.allclose(out[0, 0], 200 / 255, atol=0.01)  # right at the camera: unchanged
    assert np.allclose(out[0, 2], w["tint"], atol=0.01)  # open water: the water's colour
    assert np.allclose(out[0, 3], w["tint"], atol=0.01)  # 0 = nothing hit, as Isaac Sim reports it
    assert water.transmission(np.array([10.0]), w)[0] == pytest.approx(0.02)


@pytest.mark.skipif(not os.path.isdir(os.path.expanduser(os.environ.get("BLUESIM", "~/bluesim"))), reason="needs BlueSim")
def test_build_rov(tmp_path):
    import subprocess

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, os.path.join(here, "build_rov.py"), "--bluesim",
                    os.environ.get("BLUESIM", "~/bluesim"), "--out", str(tmp_path)], check=True, cwd=here)
    import json

    meta = json.load(open(tmp_path / "rov.json"))
    assert np.allclose(meta["jaws_centre"], np.array(meta["camera_position"]) + [0.12, 0, -0.26])
    assert set(meta["jaws"]) == {"jaw_left", "jaw_right"}
