"""The USD stage for the capture: the world (floor, pilings, walls, rocks, a
wreck or MarineGym's seabed), the ROV, the rope, its tether, the cables, the
camera and the lights, updated for every picture from a plan.Planner view.

Plain USD (pxr), so it can be built and checked without Isaac Sim (tests/).
Semantic labels: rope, tether (the ROV's tether and the cables), rov.
"""
import os

import cv2
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

from ropecap import geom, textures
from ropecap.plan import HFOV, ROPE_DIAMETER, TETHER_DIAMETER

ROPE_SIDES = 28
ROPE_RING_SPACING = 0.008  # m along the rope between rings of the mesh
FLOOR_SIZE = 80.0  # m square
FLOOR_CELLS = 120


def gf_matrix(T):
    """numpy 4x4 (column vectors) -> Gf.Matrix4d (row vectors)."""
    return Gf.Matrix4d(*[float(x) for x in np.asarray(T).T.reshape(-1)])


def set_label(prim, name):
    """Replicator's semantic class, with whichever API this Isaac Sim has."""
    try:
        from isaacsim.core.utils.semantics import add_labels

        add_labels(prim, labels=[name], instance_name="class")
        return
    except Exception:
        pass
    try:
        from pxr import UsdSemantics

        UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr().Set([name])
        return
    except Exception:
        pass
    prim.AddAppliedSchema("SemanticsAPI:Semantics")  # the older Semantics schema
    prim.CreateAttribute("semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String).Set("class")
    prim.CreateAttribute("semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String).Set(name)


def set_xform(prim, T):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(gf_matrix(T))


def set_mesh(mesh, verts, faces, normals=None, uv=None):
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(verts, np.float32)))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces, np.int32).reshape(-1)))
    if normals is not None:
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals, np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if uv is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uv, np.float32)))
    lo, hi = np.min(verts, 0), np.max(verts, 0)
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))


def new_mesh(stage, path):
    mesh = UsdGeom.Mesh.Define(stage, path)
    for make in (mesh.CreatePointsAttr, mesh.CreateFaceVertexCountsAttr, mesh.CreateFaceVertexIndicesAttr,
                 mesh.CreateNormalsAttr, mesh.CreateExtentAttr):
        make()
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    return mesh


def material(stage, path, colour=(0.5, 0.5, 0.5), texture=None, roughness=0.7, metallic=0.0):
    """A UsdPreviewSurface; with a texture, read through primvar st (repeating).
    Returns (material, texture shader or None, surface shader)."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    tex = None
    if texture is not None:
        reader = UsdShade.Shader.Define(stage, path + "/St")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, path + "/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, colour)))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat, tex, sh


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def write_png(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return os.path.abspath(path).replace("\\", "/")  # USD asset paths: forward slashes, also on Windows


def icosphere(subdiv=2):
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    v = [list(geom.normalize(x)) for x in v]
    for _ in range(subdiv):
        cache, nf = {}, []
        for a, b, c in f:
            m = []
            for x, y in ((a, b), (b, c), (c, a)):
                key = (min(x, y), max(x, y))
                if key not in cache:
                    cache[key] = len(v)
                    v.append(list(geom.normalize(np.add(v[x], v[y]) / 2)))
                m.append(cache[key])
            nf += [[a, m[0], m[2]], [b, m[1], m[0]], [c, m[2], m[1]], m]
        f = nf
    return np.array(v), np.array(f)


class IsaacScene:
    def __init__(self, stage, args, rov_meta, tex_dir, marinegym_world=None):
        self.stage = stage
        self.args = args
        self.meta = rov_meta
        self.tex_dir = tex_dir
        self.marinegym_world = marinegym_world
        self.size = args.size_wh
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        UsdGeom.Scope.Define(stage, "/World/Looks")
        # the camera: 80 deg across, square pixels (fx = fy = (W/2)/tan(40 deg))
        self.camera_path = "/World/Camera"
        cam = UsdGeom.Camera.Define(stage, self.camera_path)
        focal = 10.0
        aperture = 2 * focal * np.tan(np.radians(HFOV / 2))
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(float(aperture))
        cam.CreateVerticalApertureAttr(float(aperture * self.size[1] / self.size[0]))
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.03, 1000.0))
        cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        # the ROV's lamp, at the camera, a 45 deg cone
        lamp = UsdLux.SphereLight.Define(stage, self.camera_path + "/Lamp")
        lamp.CreateRadiusAttr(0.04)
        lamp.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        shaping = UsdLux.ShapingAPI.Apply(lamp.GetPrim())
        shaping.CreateShapingConeAngleAttr(45.0)
        shaping.CreateShapingConeSoftnessAttr(0.3)
        lamp.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.08, 0.06))  # below and just behind the lens: never in the picture
        lamp.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
        self.lamp = lamp
        self.ambient = UsdLux.DomeLight.Define(stage, "/World/Ambient")
        self.sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        self.sun.CreateAngleAttr(2.0)
        # the water surface, seen from below
        surf = new_mesh(stage, "/World/Surface")
        half = FLOOR_SIZE / 2
        set_mesh(surf, np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]]),
                 np.array([[0, 2, 1], [0, 3, 2]]), np.tile([0, 0, -1.0], (4, 1)))
        self.surface_mat, _, self.surface_shader = material(stage, "/World/Looks/Surface", (0.3, 0.4, 0.45), roughness=0.05)
        bind(surf.GetPrim(), self.surface_mat)
        self.surface = surf
        # the ROV (and its gripper), from build_rov.py
        rov = stage.DefinePrim("/World/ROV", "Xform")
        rov.GetReferences().AddReference(os.path.abspath(args.rov))
        set_label(rov, "rov")
        self.rov = rov
        # the rope, its texture changed for every picture
        self.rope = new_mesh(stage, "/World/Rope")
        set_label(self.rope.GetPrim(), "rope")
        blank = write_png(os.path.join(tex_dir, "blank.png"), np.full((4, 4, 3), 128, np.uint8))
        self.rope_mat, self.rope_tex, _ = material(stage, "/World/Looks/Rope", texture=blank, roughness=0.85)
        bind(self.rope.GetPrim(), self.rope_mat)
        # the ROV's tether and up to two cables like it (hard negatives)
        self.tether = new_mesh(stage, "/World/Tether")
        set_label(self.tether.GetPrim(), "tether")
        self.tether_mat, _, self.tether_shader = material(stage, "/World/Looks/Tether", roughness=0.45)
        bind(self.tether.GetPrim(), self.tether_mat)
        self.cables = []
        for c in range(2):
            m = new_mesh(stage, f"/World/Cable{c}")
            set_label(m.GetPrim(), "tether")
            mat, _, sh = material(stage, f"/World/Looks/Cable{c}", roughness=0.5)
            bind(m.GetPrim(), mat)
            self.cables.append((m, sh))
        self.rope_textures = []

    # ------------------------------------------------------------ world
    def build_world(self, world, rng):
        """A new layout: everything under /World/Props is rebuilt."""
        stage = self.stage
        if stage.GetPrimAtPath("/World/Props"):
            stage.RemovePrim("/World/Props")
        UsdGeom.Xform.Define(stage, "/World/Props")
        tag = "%06d" % int(rng.integers(1 << 30))
        UsdGeom.Imageable(self.surface).MakeVisible()
        if world.get("marinegym") and self.marinegym_world and os.path.exists(self.marinegym_world):
            # MarineGym's seabed: its water surface is 10 m up, its sand 5 m down: move it so its surface is ours
            ref = stage.DefinePrim("/World/Props/MarineGym", "Xform")
            ref.GetReferences().AddReference(os.path.abspath(self.marinegym_world))
            set_xform(ref, geom.pose(np.eye(3), [0.0, 0.0, -10.0]))
            UsdGeom.Imageable(self.surface).MakeInvisible()
        else:
            self.floor(world, rng, tag)
        for k, p in enumerate(world["pilings"]):
            self.piling(k, p, world, rng, tag)
        for k, wall in enumerate(world["walls"]):
            self.wall(k, wall, world, rng, tag)
        for k, rock in enumerate(world["rocks"]):
            self.rock(k, rock, world, rng, tag)
        for k, kelp in enumerate(world.get("kelp", [])):
            self.kelp(k, kelp, world)
        for k, d in enumerate(world.get("debris", [])):
            self.debris(k, d, world, rng, tag)
        if world.get("wreck"):
            wr = world["wreck"]
            ref = stage.DefinePrim("/World/Props/Wreck", "Xform")
            ref.GetReferences().AddReference(os.path.abspath(wr["path"]))
            z = float(world["floor"](wr["x"], wr["y"]))
            set_xform(ref, geom.pose(geom.rotation(geom.Z_UP, np.radians(wr["yaw_deg"])), [wr["x"], wr["y"], z]))

    def floor(self, world, rng, tag):
        n = FLOOR_CELLS
        xs = np.linspace(-FLOOR_SIZE / 2, FLOOR_SIZE / 2, n + 1)
        X, Y = np.meshgrid(xs, xs)
        Z = world["floor"](X, Y)
        verts = np.stack([X, Y, Z], -1).reshape(-1, 3)
        i, j = np.mgrid[0:n, 0:n]
        a = (i * (n + 1) + j).reshape(-1)
        faces = np.concatenate([np.stack([a, a + 1, a + n + 2], -1), np.stack([a, a + n + 2, a + n + 1], -1)])
        tile = rng.uniform(1.5, 4.0)
        mesh = new_mesh(self.stage, "/World/Props/Floor")
        normals = np.tile([0.0, 0.0, 1.0], (len(verts), 1))
        set_mesh(mesh, verts, faces, normals, verts[:, :2] / tile)
        tex = write_png(os.path.join(self.tex_dir, f"floor_{tag}.png"), textures.floor_texture(world["floor_style"], rng))
        mat, _, _ = material(self.stage, "/World/Props/FloorLook", texture=tex, roughness=0.9)
        bind(mesh.GetPrim(), mat)

    def piling(self, k, p, world, rng, tag):
        bottom = float(world["floor"](p["x"], p["y"])) - 0.5
        top = 1.0  # out of the water
        sides = 4 if p["square"] else 20
        line = np.column_stack([np.full(12, p["x"]), np.full(12, p["y"]), np.linspace(bottom, top, 12)])
        v, n, f, uv = geom.tube(line, p["radius"] * (1.3 if p["square"] else 1.0), sides)
        uv = uv * np.array([2 * np.pi * p["radius"], 1.0])  # metres round and along
        mesh = new_mesh(self.stage, f"/World/Props/Piling{k}")
        set_mesh(mesh, v, f, n, uv / 1.5)
        if k == 0:
            tex = write_png(os.path.join(self.tex_dir, f"piling_{tag}.png"), textures.surface_texture(p["material"], rng))
            self.piling_mat, _, _ = material(self.stage, "/World/Props/PilingLook", texture=tex, roughness=0.8)
        bind(mesh.GetPrim(), self.piling_mat)

    def wall(self, k, wall, world, rng, tag):
        c = np.array(wall["centre"], float)
        nrm = np.array(wall["normal"], float)
        along = np.array([-nrm[1], nrm[0]])
        half = wall["width"] / 2
        bottom = -world["depth"] - 2.0
        corners = []
        for s_ in (-half, half):
            for z in (bottom, 1.5):
                xy = c + along * s_
                corners.append([xy[0], xy[1], z])
        v = np.array(corners)
        f = np.array([[0, 1, 3], [0, 3, 2]])
        uv = np.array([[0, bottom], [0, 1.5], [wall["width"], bottom], [wall["width"], 1.5]]) / 3.0
        mesh = new_mesh(self.stage, f"/World/Props/Wall{k}")
        set_mesh(mesh, v, f, np.tile([nrm[0], nrm[1], 0.0], (4, 1)), uv)
        tex = write_png(os.path.join(self.tex_dir, f"wall{k}_{tag}.png"), textures.surface_texture(wall["material"], rng))
        mat, _, _ = material(self.stage, f"/World/Props/WallLook{k}", texture=tex, roughness=0.85)
        bind(mesh.GetPrim(), mat)

    def rock(self, k, rock, world, rng, tag):
        r2 = np.random.default_rng(rock["seed"])
        v, f = icosphere(2)
        bump = 1.0 + 0.25 * np.sin(v @ r2.normal(size=(3, 3)) * 3).sum(1) / 3
        scale = rock["size"] * np.array([r2.uniform(0.7, 1.3), r2.uniform(0.7, 1.3), r2.uniform(0.4, 0.8)])
        pts = v * bump[:, None] * scale
        z = float(world["floor"](rock["x"], rock["y"]))
        R = geom.rotation(geom.Z_UP, r2.uniform(0, 2 * np.pi))
        pts = pts @ R.T + [rock["x"], rock["y"], z - 0.2 * scale[2]]
        mesh = new_mesh(self.stage, f"/World/Props/Rock{k}")
        set_mesh(mesh, pts, f, geom.normalize(pts - pts.mean(0)), v[:, :2] * rock["size"])
        if k == 0:  # three rock looks per scene
            self.rock_mats = []
            for j in range(3):
                tex = write_png(os.path.join(self.tex_dir, f"rock{j}_{tag}.png"), textures.surface_texture("rock", rng))
                self.rock_mats.append(material(self.stage, f"/World/Props/RockLook{j}", texture=tex, roughness=0.9)[0])
        bind(mesh.GetPrim(), self.rock_mats[k % 3])

    def kelp(self, k, kelp, world):
        """A ribbon of weed or kelp standing up from the floor, swaying."""
        r2 = np.random.default_rng(kelp["seed"])
        n = 16
        t = np.linspace(0.0, 1.0, n)
        sway = r2.uniform(0.05, 0.4) * kelp["height"]
        a = r2.uniform(0, 2 * np.pi)
        lean = np.array([np.cos(a), np.sin(a)])
        z0 = float(world["floor"](kelp["x"], kelp["y"]))
        centre = np.column_stack([kelp["x"] + lean[0] * sway * t ** 1.5 + 0.05 * np.sin(6 * t + a),
                                  kelp["y"] + lean[1] * sway * t ** 1.5,
                                  z0 + kelp["height"] * t])
        side = np.array([-lean[1], lean[0], 0.0])
        width = kelp["width"] * (0.4 + 0.6 * np.sin(np.pi * np.clip(t * 1.1, 0, 1)))
        v = np.vstack([centre - side * width[:, None] / 2, centre + side * width[:, None] / 2])
        i = np.arange(n - 1)
        f = np.vstack([np.stack([i, i + n, i + 1], -1), np.stack([i + 1, i + n, i + n + 1], -1)])
        mesh = new_mesh(self.stage, f"/World/Props/Kelp{k}")
        normal = np.cross(side, [0.0, 0.0, 1.0])
        set_mesh(mesh, v, f, np.tile(normal, (len(v), 1)))
        mat, _, _ = material(self.stage, f"/World/Props/KelpLook{k}", tuple(np.clip(kelp["color"], 0.01, 1)), roughness=0.6)
        bind(mesh.GetPrim(), mat)

    def debris(self, k, d, world, rng, tag):
        """A mooring block, a pipe, a log or a tyre lying on the floor."""
        r2 = np.random.default_rng(d["seed"])
        z0 = float(world["floor"](d["x"], d["y"]))
        R = geom.rotation(geom.Z_UP, d["yaw"])
        size = d["size"]
        if d["kind"] == "block":
            c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (0, 1)], float)
            v = c * [size / 2, size / 2, size * 0.6]
            f = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                          [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
            n = geom.normalize(v - [0, 0, size * 0.3])
            uv = v[:, :2] + v[:, 2:3]
            look = "concrete"
        elif d["kind"] == "tyre":
            u, w_ = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, 2 * np.pi, 13))
            R0, r0 = size * 0.35, size * 0.12
            v = np.stack([(R0 + r0 * np.cos(w_)) * np.cos(u), (R0 + r0 * np.cos(w_)) * np.sin(u), r0 + r0 * np.sin(w_)], -1).reshape(-1, 3)
            n = geom.normalize(v - np.stack([R0 * np.cos(u), R0 * np.sin(u), np.full_like(u, r0)], -1).reshape(-1, 3))
            i, j = np.mgrid[0:12, 0:24]
            a = (i * 25 + j).reshape(-1)
            f = np.vstack([np.stack([a, a + 1, a + 26], -1), np.stack([a, a + 26, a + 25], -1)])
            uv = np.stack([u, w_], -1).reshape(-1, 2) / np.pi
            look = "rubber"
        else:  # pipe or log, lying down
            radius = size * (0.08 if d["kind"] == "pipe" else 0.12)
            length = size * r2.uniform(2.0, 5.0)
            line = np.column_stack([np.linspace(-length / 2, length / 2, 8), np.zeros(8), np.full(8, radius)])
            v, n, f, uv = geom.tube(line, radius, 16)
            uv = uv * [2 * np.pi * radius, 1.0]
            look = "steel" if d["kind"] == "pipe" else "timber"
        v = v @ R.T + [d["x"], d["y"], z0 - 0.03]
        n = n @ R.T
        mesh = new_mesh(self.stage, f"/World/Props/Debris{k}")
        set_mesh(mesh, v, f, n, uv)
        if look == "rubber":
            mat, _, _ = material(self.stage, f"/World/Props/DebrisLook{k}", (0.03, 0.03, 0.03), roughness=0.8)
        else:
            tex = write_png(os.path.join(self.tex_dir, f"debris{k}_{tag}.png"), textures.surface_texture(look, rng, 256))
            mat, _, _ = material(self.stage, f"/World/Props/DebrisLook{k}", texture=tex, roughness=0.8)
        bind(mesh.GetPrim(), mat)

    # ------------------------------------------------------------ per picture
    def show(self, view, index):
        """Put everything where this picture has it (physics is not running)."""
        args = self.args
        R, t = view["camera_R"], view["camera_t"]
        set_xform(self.stage.GetPrimAtPath(self.camera_path), geom.pose(R, t))
        set_xform(self.rov, view["rov_pose"])
        for side, jaw in self.meta["jaws"].items():
            angle = 0.0 if view["jaws_open"] else np.radians(jaw["closed_deg"])
            pivot = np.array(jaw["pivot"])
            T = geom.pose(geom.rotation(jaw["axis"], angle), pivot - geom.rotation(jaw["axis"], angle) @ pivot)
            prim = self.stage.GetPrimAtPath(f"/World/ROV/{side}")
            if prim:
                set_xform(prim, T)
        # rope: a strand-shaped tube along the physics points, a new look every picture
        look = view["look"]
        pts, s = geom.resample(view["rope_points"], ROPE_RING_SPACING)
        radius_fn = geom.strand_radius(ROPE_DIAMETER / 2, look["construction"], look["lay_length"])
        v, n, f, uv = geom.tube(pts, ROPE_DIAMETER / 2, ROPE_SIDES, radius_fn, along=s)
        from ropecap.textures import ROPE_TILE_LAYS

        uv[:, 1] /= look["lay_length"] * ROPE_TILE_LAYS
        set_mesh(self.rope, v, f, n, uv)
        tex = write_png(os.path.join(self.tex_dir, "rope_%06d.png" % index), textures.rope_texture(look))
        self.rope_tex.GetInput("file").Set(tex)
        self.rope_textures.append(tex)
        # tether and cables: smooth tubes
        tp, ts = geom.resample(view["tether_points"], 0.02)
        v, n, f, uv = geom.tube(tp, TETHER_DIAMETER / 2, 10, along=ts)
        set_mesh(self.tether, v, f, n)
        self.tether_shader.GetInput("diffuseColor").Set(Gf.Vec3f(*map(float, view["tether_color"])))
        for k, (mesh, sh) in enumerate(self.cables):
            if k < len(view["cables"]):
                c = view["cables"][k]
                v, n, f, uv = geom.tube(c["points"], c["diameter"] / 2, 10)
                set_mesh(mesh, v, f, n)
                sh.GetInput("diffuseColor").Set(Gf.Vec3f(*map(float, c["color"])))
                sh.GetInput("roughness").Set(float(c["roughness"]))
                UsdGeom.Imageable(mesh).MakeVisible()
            else:
                UsdGeom.Imageable(mesh).MakeInvisible()
        # light: ambient (tinted), sun from above, the ROV's lamp
        w = view["water"]
        self.ambient.CreateIntensityAttr().Set(float(w["ambient"] * args.ambient_scale))
        self.ambient.CreateColorAttr().Set(Gf.Vec3f(*map(float, w["ambient_color"])))
        self.sun.CreateIntensityAttr().Set(float(w["sun"] * args.sun_scale))
        self.sun.CreateColorAttr().Set(Gf.Vec3f(*map(float, 0.5 + 0.5 * np.array(w["tint"]))))
        tilt, heading = np.radians(w["sun_tilt_deg"]), np.radians(w["sun_heading_deg"])
        down = np.array([np.sin(tilt) * np.cos(heading), np.sin(tilt) * np.sin(heading), -np.cos(tilt)])
        set_xform(self.sun.GetPrim(), geom.pose(geom.look_at(np.zeros(3), down, up=[1.0, 0.0, 0.0] if abs(down[2]) > 0.99 else geom.Z_UP), [0, 0, 0]))
        self.lamp.CreateIntensityAttr().Set(float(w["lamp"] * args.lamp_scale))
        if w["lamp"] > 0:
            UsdGeom.Imageable(self.lamp).MakeVisible()
        else:
            UsdGeom.Imageable(self.lamp).MakeInvisible()
        self.surface_shader.GetInput("diffuseColor").Set(Gf.Vec3f(*map(float, w["tint"])))

    def after_picture(self, index):
        while len(self.rope_textures) > 3:  # keep the last few (the renderer may still hold them)
            old = self.rope_textures.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass
