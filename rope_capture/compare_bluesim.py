"""Isaac Sim pictures next to BlueSim pictures of similar views (same view
type and phase, closest camera tilt), with their labels, and a few numbers
on how they differ (brightness, contrast, colour, how much rope is seen).

usage: python compare_bluesim.py ISAAC_RUN BLUESIM_RUN [--pairs 6] [--out compare.jpg]
(BlueSim's run needs classes/: run Rope_Detection's dataset/make_masks.py on it first.)
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def load(run):
    out = []
    for p in sorted(glob.glob(os.path.join(run, "labels", "*.json"))):
        lab = json.load(open(p))
        name = os.path.basename(p)[:-5]
        out.append((name, lab))
    return out


def overlay(run, name, width):
    image = cv2.imread(os.path.join(run, "images", name + ".png"))
    classes = cv2.imread(os.path.join(run, "classes", name + ".png"), cv2.IMREAD_GRAYSCALE)
    scale = width / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    over = image.copy()
    if classes is not None:
        classes = cv2.resize(classes, image.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        for k, col in ((1, (0, 255, 0)), (2, (0, 0, 255)), (3, (255, 0, 0))):
            over[classes == k] = (0.4 * over[classes == k] + 0.6 * np.array(col)).astype(np.uint8)
    return image, over, classes


def stats(run, items):
    rows = []
    for name, lab in items:
        img = cv2.imread(os.path.join(run, "images", name + ".png"))
        cls = cv2.imread(os.path.join(run, "classes", name + ".png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
        rope = 0.0 if cls is None else float((cls == 1).mean())
        rows.append([lab_img[..., 0].mean(), lab_img[..., 0].std(), lab_img[..., 1].mean() - 128,
                     lab_img[..., 2].mean() - 128, rope * 100])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("isaac")
    ap.add_argument("bluesim")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out", default="compare.jpg")
    args = ap.parse_args()
    isaac, blue = load(args.isaac), load(args.bluesim)
    rows, used = [], set()
    for name, lab in isaac:
        if len(rows) >= args.pairs:
            break
        s = lab["settings"]
        best = None
        for bname, blab in blue:
            b = blab["settings"]
            if bname in used or b.get("view") != s.get("view") or b.get("phase") != s.get("phase"):
                continue
            score = abs(b.get("camera_tilt_deg", 0) - s.get("camera_tilt_deg", 0)) + \
                5 * (b.get("aimed_at_tether") != s.get("aimed_at_tether")) + 5 * (b.get("looking_away") != s.get("looking_away"))
            if best is None or score < best[0]:
                best = (score, bname, blab)
        if best is None:
            continue
        used.add(best[1])
        a_img, a_over, _ = overlay(args.isaac, name, args.width)
        b_img, b_over, _ = overlay(args.bluesim, best[1], args.width)
        h = min(a_img.shape[0], b_img.shape[0])
        row = np.hstack([a_img[:h], a_over[:h], b_img[:h], b_over[:h]])
        text = f"Isaac {name} {s.get('view')} {s.get('phase', '')}  |  BlueSim {best[1]}"
        cv2.putText(row, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(row, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        rows.append(row)
    if rows:
        cv2.imwrite(args.out, np.vstack(rows))
        print(f"wrote {args.out}: {len(rows)} pairs (Isaac picture, labels | BlueSim picture, labels)")
    for title, run, items in (("Isaac Sim", args.isaac, isaac), ("BlueSim", args.bluesim, blue)):
        st = stats(run, items[:200])
        if len(st):
            m = st.mean(0)
            print(f"{title:9s} {len(st):4d} pictures: brightness {m[0]:5.1f}, contrast {m[1]:5.1f}, "
                  f"green-red {m[2]:+5.1f}, blue-yellow {m[3]:+5.1f} (Lab), rope {m[4]:4.1f}% of the picture")


if __name__ == "__main__":
    main()
