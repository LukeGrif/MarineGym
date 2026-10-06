"""Build the ROV (BlueROV2 Heavy + Newton gripper) for the Isaac Sim capture
from BlueSim's own models, so it looks the same in both simulators.

Reads from a BlueSim checkout (default ~/bluesim):
  vehicles/bluerovheavy/bluerov cleaned decimated.gltf   the frame and thrusters
  imports/newton/newton.dae                              the gripper body
  vehicles/bluerovheavy/BlueRovHeavy.tscn                where they sit, and the
                                                         two jaws (meshes in the scene)
and writes assets/rov.usd (+ assets/rov.json: camera mount, jaw hinges) in the
ROV frame used here: x forward, y left, z up, metres, origin at BlueSim's
BlueRov body origin. Nothing from BlueSim is changed or copied into this repo:
run this once on the machine that does the capture.

usage: python build_rov.py [--bluesim ~/bluesim] [--out assets] [--preview rov_preview.png]
Needs: numpy, trimesh, pycollada (pip install trimesh pycollada) and pxr (in
Isaac Sim's Python, or pip install usd-core).
"""
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from ropecap import geom

# newton.tscn: which of newton.dae's scene nodes (by order) are shown, and their material
NEWTON_SHOWN = {0: "blue", 1: "black", 2: "gray", 8: "gray", 9: "gray", 10: "black", 11: "blue", 16: "black"}
METALS = {"blue": (0.05, 0.22, 0.62), "black": (0.04, 0.04, 0.045), "gray": (0.45, 0.45, 0.46)}
JAW_TRAVEL_DEG = 35.0  # the hinge limits in BlueRovHeavy.tscn


def tscn_nodes(text):
    """{node name: {'parent':..., 'transform': 4x4 or None, 'mesh': id or None}}"""
    out = {}
    for chunk in text.split("\n[node ")[1:]:
        end = chunk.index("]\n")  # the header may span lines (groups=[ ... ]])
        header, body = chunk[:end], chunk[end + 2:].split("\n[", 1)[0]
        name = re.search(r'name="([^"]+)"', header).group(1)
        parent = re.search(r'parent="([^"]*)"', header)
        t = re.search(r"^transform = (Transform\([^)]*\))", body, re.M)
        mesh = re.search(r"^mesh = SubResource\( (\d+) \)", body, re.M)
        out[name] = {"parent": parent.group(1) if parent else None,
                     "transform": geom.godot_transform(t.group(1)) if t else np.eye(4),
                     "mesh": int(mesh.group(1)) if mesh else None}
    return out


def tscn_array_mesh(text, rid):
    """Decode a Godot 3 ArrayMesh sub-resource (float positions, compressed
    normals, 16-bit indices: the format the jaws are saved in)."""
    m = re.search(r'\[sub_resource type="ArrayMesh" id=%d\]\n(.*?)\n\n' % rid, text, re.S)
    body = m.group(1)
    fmt = int(re.search(r'"format": (\d+)', body).group(1))
    n = int(re.search(r'"vertex_count": (\d+)', body).group(1))
    data = np.array([int(x) for x in re.search(r'"array_data": PoolByteArray\( ([^)]*) \)', body).group(1).split(",")], np.uint8)
    idx = np.array([int(x) for x in re.search(r'"array_index_data": PoolByteArray\( ([^)]*) \)', body).group(1).split(",")], np.uint8)
    assert fmt & 1 and fmt & 2 and not fmt & 512, f"unexpected ArrayMesh format {fmt}"
    stride = len(data) // n
    assert stride == 16 and not fmt & (4 | 8 | 16 | 32 | 64 | 128), f"unexpected ArrayMesh layout {fmt}, stride {stride}"
    rec = data.reshape(n, stride)
    pos = rec[:, :12].copy().view("<f4").reshape(n, 3).astype(float)
    faces = idx.view("<u2").reshape(-1, 3).astype(int)[:, ::-1]  # Godot's front faces are clockwise
    return pos, faces


def gltf_meshes(path):
    """[(vertices, faces, rgb)] in the glTF's own frame (node transforms applied)."""
    scene = trimesh.load(path, force="scene")
    out = []
    for node in scene.graph.nodes_geometry:
        T, g = scene.graph[node]
        m = scene.geometry[g]
        rgb = (0.5, 0.5, 0.5)
        mat = getattr(m.visual, "material", None)
        if mat is not None and getattr(mat, "baseColorFactor", None) is not None:
            c = np.asarray(mat.baseColorFactor, float)
            rgb = tuple((c[:3] / 255.0 if c.max() > 1.0 else c[:3]).tolist())
        out.append((geom.apply(T, m.vertices), np.asarray(m.faces), rgb))
    return out


def newton_meshes(path):
    """The shown parts of newton.dae, in Godot's Y-up frame (Godot turns a Z-up
    Collada file round -90 deg about X when it imports it)."""
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    root = ET.parse(path).getroot()
    nodes = root.find("c:library_visual_scenes/c:visual_scene", ns).findall("c:node", ns)
    z_up = (root.find("c:asset/c:up_axis", ns).text or "Y_UP").strip() == "Z_UP"
    to_y_up = np.eye(4)
    if z_up:
        to_y_up[:3, :3] = geom.rotation([1, 0, 0], -np.pi / 2)
    geometries = {g.get("id"): g for g in root.find("c:library_geometries", ns)}
    out = []
    for i, colour in NEWTON_SHOWN.items():
        node = nodes[i]
        T = np.array([float(x) for x in node.find("c:matrix", ns).text.split()]).reshape(4, 4)
        g = geometries[node.find("c:instance_geometry", ns).get("url")[1:]]
        mesh = g.find("c:mesh", ns)
        sources = {s.get("id"): s for s in mesh.findall("c:source", ns)}
        vin = mesh.find("c:vertices/c:input[@semantic='POSITION']", ns).get("source")[1:]
        pos = np.array(sources[vin].find("c:float_array", ns).text.split(), float).reshape(-1, 3)
        faces = []
        for tri in mesh.findall("c:triangles", ns):
            stride = len(tri.findall("c:input", ns))
            off = int(tri.find("c:input[@semantic='VERTEX']", ns).get("offset"))
            p = np.array(tri.find("c:p", ns).text.split(), int).reshape(-1, stride)[:, off]
            faces.append(p.reshape(-1, 3))
        out.append((geom.apply(to_y_up @ T, pos), np.vstack(faces), METALS[colour]))
    return out


def merge(parts):
    vs, fs, cs, off = [], [], [], 0
    for v, f, c in parts:
        vs.append(v)
        fs.append(f + off)
        cs.append(np.tile(c, (len(f), 1)))
        off += len(v)
    return np.vstack(vs), np.vstack(fs), np.vstack(cs)


def write_usd(path, parts_by_name, meta):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/ROV")
    stage.SetDefaultPrim(root.GetPrim())
    looks = {}

    def material(rgb, metal):
        key = (tuple(np.round(rgb, 3)), metal)
        if key not in looks:
            p = f"/ROV/Looks/m{len(looks)}"
            mat = UsdShade.Material.Define(stage, p)
            sh = UsdShade.Shader.Define(stage, p + "/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6 if metal else 0.0)
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35 if metal else 0.5)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            looks[key] = mat
        return looks[key]

    for group, (parts, xform_meta) in parts_by_name.items():
        g = UsdGeom.Xform.Define(stage, f"/ROV/{group}")
        if xform_meta:
            g.GetPrim().SetCustomDataByKey("jaw", xform_meta)
        for k, (v, f, rgb) in enumerate(parts):
            mesh = UsdGeom.Mesh.Define(stage, f"/ROV/{group}/mesh_{k}")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1).astype(np.int32)))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, rgb))])
            metal = group != "body" or (max(rgb) - min(rgb) < 0.02 and max(rgb) > 0.05)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material(rgb, metal))
    root.GetPrim().SetCustomDataByKey("rov", json.loads(json.dumps(meta)))
    stage.GetRootLayer().Save()


def preview(path, parts, meta, size=(960, 540), tilt=35.0):
    """A quick look from the ROV's camera (points of the surfaces, nearest
    wins), to check the gripper and jaws come out where BlueSim has them."""
    import cv2

    R = geom.camera_from_rov(tilt)
    t = np.array(meta["camera_position"])
    k = geom.intrinsics(size[0], size[1], 80.0)
    img = np.full((size[1], size[0], 3), 40, np.uint8)
    depth = np.full((size[1], size[0]), np.inf)
    for v, f, rgb in parts:
        tm = trimesh.Trimesh(v, f, process=False)
        n = int(min(400000, max(2000, tm.area * 2e6)))
        pts, fi = trimesh.sample.sample_surface(tm, n, seed=1)
        cam = geom.to_camera(pts, R, t)
        uv, z = geom.project(cam, k)
        ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < size[0] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1] - 1)
        shade = 0.5 + 0.5 * np.abs(tm.face_normals[fi][:, 2])
        for (u, vv), zz, s in zip(uv[ok].astype(int), z[ok], shade[ok]):
            if zz < depth[vv, u]:
                depth[vv, u] = zz
                img[vv, u] = (np.array(rgb[::-1]) * 255 * s).clip(30, 255)
    jaws = geom.to_camera(np.array([meta["jaws_centre"]]), R, t)
    uv, _ = geom.project(jaws, k)
    cv2.drawMarker(img, (int(uv[0, 0]), int(uv[0, 1])), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bluesim", default="~/bluesim")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"))
    ap.add_argument("--preview", default=None, help="also write a picture from the ROV camera (tilted 35 deg)")
    args = ap.parse_args()
    bs = os.path.expanduser(args.bluesim)
    text = open(os.path.join(bs, "vehicles/bluerovheavy/BlueRovHeavy.tscn")).read()
    nodes = tscn_nodes(text)
    T_rov = nodes["BlueRov"]["transform"]  # BlueRov in the assembly
    inv_rov = np.linalg.inv(T_rov)
    C = np.eye(4)
    C[:3, :3] = geom.GODOT_TO_ROV

    def to_rov(T_in_rov_godot, v):  # godot BlueRov-local points -> ROV frame here
        return geom.apply(C @ T_in_rov_godot, v)

    # the frame: glTF, under BlueRov with its own transform
    T_model = nodes["bluerov cleaned decimated"]["transform"]
    body = [(to_rov(T_model, v), f, c) for v, f, c in gltf_meshes(os.path.join(bs, "vehicles/bluerovheavy/bluerov cleaned decimated.gltf"))]
    # the gripper body
    T_newton = nodes["newton"]["transform"]
    gripper = [(to_rov(T_newton, v), f, c) for v, f, c in newton_meshes(os.path.join(bs, "imports/newton/newton.dae"))]
    # the jaws: siblings of BlueRov in the assembly, mesh under RigidBody(2)/g2(g1)
    jaws = {}
    for side, body_name, mesh_name, joint in (("jaw_left", "RigidBody", "g2", "ljoint"), ("jaw_right", "RigidBody2", "g1", "rjoint")):
        v, f = tscn_array_mesh(text, nodes[mesh_name]["mesh"])
        T = inv_rov @ nodes[body_name]["transform"] @ nodes[mesh_name]["transform"]
        pts = to_rov(T, v)
        Tj = C @ inv_rov @ nodes[joint]["transform"]
        pivot = Tj[:3, 3]
        axis = geom.normalize(Tj[:3, 2])  # a Godot hinge turns about its Z
        jaws[side] = (pts, f, pivot, axis)
    # which way closes a jaw: the way that brings it towards the other one
    centre_y = np.mean([j[0][:, 1].mean() for j in jaws.values()])
    jaw_meta = {}
    for side, (pts, f, pivot, axis) in jaws.items():
        best = None
        for sign in (1.0, -1.0):
            R = geom.rotation(axis, sign * np.radians(JAW_TRAVEL_DEG))
            moved = (pts - pivot) @ R.T + pivot
            gap = np.abs(moved[:, 1].mean() - centre_y)
            if best is None or gap < best[0]:
                best = (gap, sign)
        jaw_meta[side] = {"pivot": pivot.tolist(), "axis": axis.tolist(), "closed_deg": best[1] * JAW_TRAVEL_DEG}
    cam_T = geom.godot_to_rov(nodes["Camera"]["transform"])
    cam_pos = cam_T[:3, 3]
    jaws_centre = cam_pos + np.array([0.12, 0.0, -0.26])  # capture.gd JAWS, from the camera
    meta = {"camera_position": cam_pos.tolist(), "jaws_centre": jaws_centre.tolist(), "jaws": jaw_meta,
            "lamp_position": cam_pos.tolist()}
    os.makedirs(args.out, exist_ok=True)
    body_m = merge(body)
    grip_m = merge(gripper)
    parts = {"body": ([p for p in body], None), "gripper": ([p for p in gripper], None)}
    for side, (pts, f, pivot, axis) in jaws.items():
        parts[side] = ([(pts, f, METALS["black"])], jaw_meta[side])
    write_usd(os.path.join(args.out, "rov.usd"), parts, meta)
    json.dump(meta, open(os.path.join(args.out, "rov.json"), "w"), indent=1)
    lo, hi = body_m[0].min(0), body_m[0].max(0)
    print(f"ROV frame {lo.round(3)} .. {hi.round(3)} m; gripper {grip_m[0].min(0).round(3)} .. {grip_m[0].max(0).round(3)}")
    for side, (pts, *_rest) in jaws.items():
        print(f"{side}: {pts.min(0).round(3)} .. {pts.max(0).round(3)}, closes {jaw_meta[side]['closed_deg']:+.0f} deg")
    print(f"camera at {cam_pos.round(3)}, jaws' middle at {jaws_centre.round(3)}; wrote {args.out}/rov.usd")
    if args.preview:
        allparts = body + gripper + [(j[0], j[1], (0.9, 0.9, 0.1)) for j in jaws.values()]
        preview(args.preview, allparts, meta)


if __name__ == "__main__":
    main()
