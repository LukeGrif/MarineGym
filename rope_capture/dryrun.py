"""A capture run without Isaac Sim, to check everything but the renderer:
the same plan, the same USD stage (isaac_scene.py, built with plain pxr), the
same label files, with a rough CPU picture (points of the surfaces, nearest
wins) instead of Isaac Sim's. For testing on a machine without an RTX GPU;
the pictures are NOT for training.

usage: python dryrun.py OUT [--count 20] [--size 480x270] [--seed 1] [--world seabed]
       [--rov assets/rov.usd] [--save-stage stage.usda]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def labels_of(prim):
    """The semantic class on this prim or the nearest ancestor that has one."""
    while prim and prim.IsValid():
        a = prim.GetAttribute("semantic:Semantics:params:semanticData")
        if a and a.Get():
            return a.Get()
        for attr in prim.GetAttributes():
            if attr.GetName().startswith("semantics:labels:") and attr.Get():
                return list(attr.Get())[0]
        prim = prim.GetParent()
    return None


def surface_colour(prim):
    from pxr import UsdShade

    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return np.array([0.5, 0.5, 0.5]), None
    surf = mat.GetPrim().GetPath().AppendChild("Surface")
    sh = UsdShade.Shader.Get(prim.GetStage(), surf)
    if not sh:  # build_rov.py's materials
        sh = UsdShade.Shader.Get(prim.GetStage(), mat.GetPrim().GetPath().AppendChild("Shader"))
    if not sh:
        return np.array([0.5, 0.5, 0.5]), None
    inp = sh.GetInput("diffuseColor")
    if inp.HasConnectedSource():
        tex = UsdShade.Shader.Get(prim.GetStage(), mat.GetPrim().GetPath().AppendChild("Texture"))
        path = tex.GetInput("file").Get().resolvedPath or tex.GetInput("file").Get().path
        return None, path
    return np.array(inp.Get()), None


def cpu_render(stage, cam_R, cam_t, k, size, lights, rng):
    """Rough picture, distance and classes from the stage's meshes."""
    import cv2
    from pxr import UsdGeom

    from ropecap import geom
    from ropecap.labels import CLASS_IDS

    w, h = size
    cache = UsdGeom.XformCache()
    pts_all, col_all, cls_all = [], [], []
    texcache = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh" or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        mesh = UsdGeom.Mesh(prim)
        v = np.array(mesh.GetPointsAttr().Get() or [], float)
        if len(v) == 0:
            continue
        f = np.array(mesh.GetFaceVertexIndicesAttr().Get(), int).reshape(-1, 3)
        M = np.array(cache.GetLocalToWorldTransform(prim)).T
        v = geom.apply(M, v)
        # only triangles in front of the camera and in the picture
        cv = geom.to_camera(v, cam_R, cam_t)
        puv, pz = geom.project(cv, k)
        tz = pz[f]
        tu, tv = puv[f, 0], puv[f, 1]
        keep = (tz.max(1) > 0.03) & ~((tz > 0.03).all(1) & ((tu.max(1) < 0) | (tu.min(1) >= w) | (tv.max(1) < 0) | (tv.min(1) >= h)))
        f = f[keep]
        if len(f) == 0:
            continue
        # points spread over the triangles, more where they're near the camera
        tri = v[f]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        dist = np.maximum(np.linalg.norm(tri.mean(1) - cam_t, axis=1), 0.05)
        want = area * (k["fx"] / dist) ** 2 * 0.8
        n = np.minimum(rng.poisson(np.minimum(want, 2000)), 2000)
        if n.sum() == 0:
            continue
        fi = np.repeat(np.arange(len(f)), n)
        if len(fi) > 3_000_000:
            fi = rng.choice(fi, 3_000_000, replace=False)
        a, b = rng.random(len(fi)), rng.random(len(fi))
        flip = a + b > 1
        a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
        p = tri[fi, 0] + a[:, None] * (tri[fi, 1] - tri[fi, 0]) + b[:, None] * (tri[fi, 2] - tri[fi, 0])
        normal = geom.normalize(np.cross(tri[fi, 1] - tri[fi, 0], tri[fi, 2] - tri[fi, 0]))
        rgb, tex = surface_colour(prim)
        if tex is not None:
            st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
            uv = np.array(st.Get(), float) if st and st.Get() is not None else np.zeros((len(v), 2))
            uvp = uv[f[fi, 0]] + a[:, None] * (uv[f[fi, 1]] - uv[f[fi, 0]]) + b[:, None] * (uv[f[fi, 2]] - uv[f[fi, 0]])
            if tex not in texcache:
                img = cv2.imread(tex)
                texcache[tex] = None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            t_img = texcache[tex]
            if t_img is None:
                col = np.full((len(p), 3), 0.5)
            else:
                th, tw = t_img.shape[:2]
                col = t_img[(np.mod(1 - uvp[:, 1], 1) * (th - 1)).astype(int), (np.mod(uvp[:, 0], 1) * (tw - 1)).astype(int)]
        else:
            col = np.tile(rgb, (len(p), 1))
        to_cam = geom.normalize(cam_t - p)
        lambert = np.abs(np.sum(normal * to_cam, 1))
        light = lights["ambient"] * 0.5 + lights["lamp"] / 4.0 * lambert / np.maximum(np.linalg.norm(p - cam_t, axis=1), 0.3) ** 2 \
            + lights["sun"] * 0.5 * np.abs(normal[:, 2])
        pts_all.append(p)
        col_all.append(col * np.clip(light, 0.05, 1.5)[:, None])
        label = labels_of(prim)
        cls_all.append(np.full(len(p), CLASS_IDS.get(label, 0), np.uint8))
    img = np.zeros((h, w, 3), np.float32)
    dist = np.full((h, w), np.inf, np.float32)
    classes = np.zeros((h, w), np.uint8)
    if pts_all:
        p = np.vstack(pts_all)
        col = np.vstack(col_all)
        cls = np.concatenate(cls_all)
        cam = geom.to_camera(p, cam_R, cam_t)
        uv, z = geom.project(cam, k)
        d = np.linalg.norm(p - cam_t, axis=1)
        ok = (z > 0.03) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        order = np.argsort(-d[ok])
        u, vv = uv[ok][order].astype(int).T
        img[vv, u] = col[ok][order]
        dist[vv, u] = d[ok][order]
        classes[vv, u] = cls[ok][order]
        # fill the gaps between the points (only inside surfaces, not the open water round them)
        valid = np.isfinite(dist).astype(np.uint8)
        inside = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
        for _ in range(3):
            empty = ~np.isfinite(dist)
            best = dist.copy()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
                nd = np.roll(dist, (dy, dx), (0, 1))
                take = empty & inside & (nd < best)
                best[take] = nd[take]
                img[take] = np.roll(img, (dy, dx), (0, 1))[take]
                classes[take] = np.roll(classes, (dy, dx), (0, 1))[take]
            dist = best
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), dist, classes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--size", default="480x270")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--world", default="mixed")
    ap.add_argument("--pictures-per-scene", type=int, default=None,
                    help="a new rope (and place) after this many pictures (default: as the capture, 20)")
    ap.add_argument("--rov", default=os.path.join(HERE, "assets", "rov.usd"))
    ap.add_argument("--save-stage", default=None, help="also save the last picture's stage (.usda)")
    args = ap.parse_args()
    args.size_wh = tuple(int(x) for x in args.size.lower().split("x"))
    args.ambient_scale, args.sun_scale, args.lamp_scale = 800.0, 2500.0, 60000.0
    from pxr import Usd

    from isaac_scene import IsaacScene
    from ropecap import labels, water
    from ropecap.plan import HFOV, Planner

    run = os.path.abspath(args.out)
    labels.folders(run)
    tex_dir = os.path.join(run, "_textures")
    os.makedirs(tex_dir, exist_ok=True)
    meta = json.load(open(os.path.splitext(args.rov)[0] + ".json"))
    stage = Usd.Stage.CreateInMemory()
    scene = IsaacScene(stage, args, meta, tex_dir)
    if args.pictures_per_scene:
        import ropecap.plan

        ropecap.plan.PICTURES_PER_SCENE = args.pictures_per_scene
    planner = Planner(seed=args.seed, size=args.size_wh, world_kind=args.world, rov_meta=meta)
    rng = np.random.default_rng(args.seed)
    current = None
    for i in range(args.count):
        view = planner.next()
        if current is not planner.scene:
            current = planner.scene
            scene.build_world(view["world"], np.random.default_rng(args.seed * 100003 + i))
        scene.show(view, i)
        rgb, dist, classes = cpu_render(stage, view["camera_R"], view["camera_t"], planner.k, args.size_wh,
                                        view["water"], rng)
        gone = water.transmission(dist, view["water"]) < 0.02
        classes[gone & (classes != 3)] = 0
        picture = water.underwater(rgb, dist, view["water"], planner.k)
        name = "%06d" % i
        lab = labels.label(view, name, args.size_wh, HFOV, {"simulator": "dryrun (not for training)"})
        labels.write(run, i, picture, classes, lab)
        scene.after_picture(i)
        s = view["settings"]
        print(f"{name} {s['view']:5s} {s.get('phase', ''):11s} rope px {int((classes == 1).sum()):6d} "
              f"tether px {int((classes == 2).sum()):5d} rov px {int((classes == 3).sum()):6d}")
    if args.save_stage:
        stage.GetRootLayer().Export(args.save_stage)
    shutil.rmtree(tex_dir, ignore_errors=True)
    open(os.path.join(run, "done"), "w").write("dryrun\n")


if __name__ == "__main__":
    main()
