"""Rope-detection training pictures from Isaac Sim, the same as BlueSim's
capture (bluesim/scripts/capture.gd) makes, into the same run folder layout:

  python capture.py OUT [--count 500] [--size 1920x1080] [--seed 1] [--world seabed]

OUT/images, classes (from Replicator's semantic segmentation: only what is
seen), masks, labels/*.json (BlueSim's fields), and OUT/done at the end.
Train with Rope_Detection's dataset/train_seg.py OUT (no make_masks.py needed).

Worlds: seabed (open floor, rocks; a new floor every rope), harbour (pilings,
quay walls), marinegym (MarineGym's EmptyMarine seabed), wreck (--wreck
model.usd placed near the rope). Run assets: build_rov.py first (the ROV and
gripper, from BlueSim's own models).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MARINEGYM_WORLD = os.path.join(HERE, "..", "marinegym", "robots", "assets", "usd", "worlds", "EmptyMarine.usd")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="the run's folder")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--size", default="1920x1080", help="picture size, e.g. 960x540")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--world", default="seabed", choices=["seabed", "harbour", "marinegym", "wreck"])
    ap.add_argument("--wreck", default=None, help="a wreck model (.usd/.usdz) for --world wreck")
    ap.add_argument("--rov", default=os.path.join(HERE, "assets", "rov.usd"), help="from build_rov.py")
    ap.add_argument("--subframes", type=int, default=8, help="render passes per picture (more = cleaner, slower)")
    ap.add_argument("--min-contrast", type=float, default=0.02,
                    help="rope/tether lost in the water (less contrast left than this) is not labelled; 0 labels all")
    ap.add_argument("--ambient-scale", type=float, default=800.0, help="dome light intensity for ambient 1.0")
    ap.add_argument("--sun-scale", type=float, default=2500.0, help="sun (distant light) intensity for sun 1.0")
    ap.add_argument("--lamp-scale", type=float, default=60000.0, help="ROV lamp intensity for lamp 1.0")
    ap.add_argument("--window", action="store_true", help="show Isaac Sim's window (default: headless)")
    ap.add_argument("--keep-raw", action="store_true", help="also save the picture before the water (raw/)")
    args = ap.parse_args()
    w, h = (int(x) for x in args.size.lower().split("x"))
    args.size_wh = (w, h)
    if args.world == "wreck" and not args.wreck:
        ap.error("--world wreck needs --wreck path/to/wreck.usd")
    return args


def main():
    args = parse_args()
    run = os.path.abspath(os.path.expanduser(args.out))
    if os.path.exists(os.path.join(run, "done")):
        print(f"{run} is already done")
        return
    if not os.path.exists(args.rov):
        sys.exit(f"no ROV model at {args.rov}: run  python build_rov.py --bluesim path/to/bluesim  first")
    from isaacsim import SimulationApp

    w, h = args.size_wh
    app = SimulationApp({"headless": not args.window, "width": w, "height": h, "renderer": "RaytracedLighting"})
    try:
        capture(app, args, run)
    finally:
        app.close()


def capture(app, args, run):
    import carb
    import omni.replicator.core as rep
    import omni.usd

    from ropecap import labels, water as water_mod
    from ropecap.plan import HFOV, Planner
    from isaac_scene import IsaacScene

    settings = carb.settings.get_settings()
    for key, value in [
        ("/rtx/post/histogram/enabled", False),  # no auto exposure: the lights decide
        ("/rtx/post/motionblur/enabled", False),
        ("/rtx/materialDb/syncLoads", True),  # textures loaded before the picture is taken
        ("/rtx/hydra/materialSyncLoads", True),
        ("/omni.kit.plugin/syncUsdLoads", True),
        ("/app/asyncRendering", False),
        ("/rtx/directLighting/sampledLighting/enabled", True),
        ("/rtx/post/dlss/execMode", 2),  # DLSS "quality", as Isaac Sim's own Replicator examples set it
        ("/omni/replicator/captureOnPlay", False),
    ]:
        settings.set(key, value)
    try:
        rep.orchestrator.set_capture_on_play(False)
    except Exception:
        pass

    w, h = args.size_wh
    meta = json.load(open(os.path.splitext(args.rov)[0] + ".json"))
    labels.folders(run)
    tex_dir = os.path.join(run, "_textures")
    os.makedirs(tex_dir, exist_ok=True)
    omni.usd.get_context().new_stage()
    for _ in range(3):
        app.update()
    stage = omni.usd.get_context().get_stage()
    scene = IsaacScene(stage, args, meta, tex_dir, MARINEGYM_WORLD)
    rp = rep.create.render_product(scene.camera_path, (w, h))
    rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
    dist_a = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    seg_a = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
    for a in (rgb_a, dist_a, seg_a):
        a.attach([rp])

    def render():
        try:
            rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0, pause_timeline=False)
        except TypeError:  # older Replicator
            rep.orchestrator.step(rt_subframes=args.subframes)

    from concurrent.futures import ThreadPoolExecutor

    saver = ThreadPoolExecutor(max_workers=1)
    pending = []
    planner = Planner(seed=args.seed, size=(w, h), world_kind=args.world, rov_meta=meta, wreck_path=args.wreck)
    k = planner.k
    start = time.time()
    render_time = 0.0
    scene_id = None
    for i in range(args.count):
        view = planner.next()  # physics runs here, then stops for the picture
        if scene_id is not planner.scene:
            scene_id = planner.scene
            scene.build_world(view["world"], np.random.default_rng(args.seed * 100003 + i))
        scene.show(view, i)
        t0 = time.time()
        if i == 0:
            render()  # the first pictures compile shaders and load textures
            render()
        render()
        rgb = np.asarray(rgb_a.get_data())[..., :3]
        dist = np.asarray(dist_a.get_data()).astype(np.float32)
        if dist.ndim == 3:
            dist = dist[..., 0]
        seg = seg_a.get_data()
        render_time += time.time() - t0
        classes = labels.classes_from_semantic(seg["data"], seg["info"].get("idToLabels", {}))
        if args.min_contrast > 0:  # lost in the water: not something the camera can see
            gone = water_mod.transmission(dist, view["water"]) < args.min_contrast
            classes[gone & (classes != 3)] = 0
        picture = water_mod.underwater(rgb, dist, view["water"], k, np.random.default_rng(view["water"]["seed"]))
        name = "%06d" % i
        if args.keep_raw:
            import cv2

            os.makedirs(os.path.join(run, "raw"), exist_ok=True)
            cv2.imwrite(os.path.join(run, "raw", name + ".png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        lab = labels.label(view, name, (w, h), HFOV)
        pending.append(saver.submit(labels.write, run, i, picture, classes, lab))  # saved while the next one renders
        while len(pending) > 2:
            pending.pop(0).result()
        scene.after_picture(i)
        if i == 0:
            steady_start = time.time()  # after the first picture (shader compiling, texture loading)
        if i % 20 == 0 or i == args.count - 1:
            done = i + 1
            rate = done / (time.time() - start) * 60
            print(f"Capture: {done} / {args.count}  ({rate:.1f} pictures/min, render {render_time / done:.2f} s each)", flush=True)
    for f in pending:
        f.result()
    saver.shutdown()
    try:
        rep.orchestrator.wait_until_complete()
    except Exception:
        pass
    elapsed = time.time() - start
    steady = (args.count - 1) / max(time.time() - steady_start, 1e-6) * 60 if args.count > 1 else None
    timing = {"pictures": args.count, "size": [w, h], "seconds": elapsed, "pictures_per_minute": args.count / elapsed * 60,
              "pictures_per_minute_after_first": steady,
              "render_seconds_per_picture": render_time / max(args.count, 1), "subframes": args.subframes}
    json.dump(timing, open(os.path.join(run, "timing.json"), "w"), indent=1)
    shutil.rmtree(tex_dir, ignore_errors=True)
    open(os.path.join(run, "done"), "w").write(json.dumps(timing) + "\n")
    print(f"Capture: done, {args.count} pictures in {run}, {timing['pictures_per_minute']:.1f} pictures/min"
          + (f" ({steady:.1f} after the first, which compiles the shaders)" if steady else ""))


if __name__ == "__main__":
    main()
