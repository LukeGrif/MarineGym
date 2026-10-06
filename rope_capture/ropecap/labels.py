"""The run folder, as BlueSim's capture writes it, so Rope_Detection's
scripts read it unchanged:
  images/NNNNNN.png    the picture
  classes/NNNNNN.png   0 background, 1 rope, 2 tether (the ROV's tether and the
                       tether-like cables), 3 ROV (gripper, jaws, frame): only
                       what is seen (from Replicator's semantic segmentation)
  masks/NNNNNN.png     rope seen = 255
  labels/NNNNNN.json   camera model, rope centre line, tether, cables, settings
  done                 when the run has finished
"""
import json
import os

import numpy as np

from . import geom

CLASS_IDS = {"rope": 1, "tether": 2, "rov": 3}


def label_class(entry):
    """Our class (1-3) for one of Replicator's idToLabels entries, else 0."""
    if isinstance(entry, dict):
        values = [str(v) for v in entry.values()]
    else:
        values = [str(entry)]
    for text in values:
        for part in text.replace(";", ",").split(","):
            part = part.strip().lower()
            if part in CLASS_IDS:
                return CLASS_IDS[part]
    return 0


def classes_from_semantic(data, id_to_labels):
    """Replicator's semantic_segmentation (ids per pixel, colorize off) -> classes."""
    data = np.asarray(data)
    if data.ndim == 3:  # some versions give (H, W, 1) or RGBA-packed ids
        data = data[..., 0]
    out = np.zeros(data.shape, np.uint8)
    for key, entry in (id_to_labels or {}).items():
        c = label_class(entry)
        if c:
            out[data == int(key)] = c
    return out


def folders(run):
    for sub in ("images", "classes", "masks", "labels"):
        os.makedirs(os.path.join(run, sub), exist_ok=True)


def label(view, name, size, hfov, extra=None):
    R, t = view["camera_R"], view["camera_t"]
    k = geom.intrinsics(size[0], size[1], hfov)
    tp = view["tether_points"]
    tether = [[geom.to_camera(a, R, t).tolist(), geom.to_camera(b, R, t).tolist()] for a, b in zip(tp[:-1], tp[1:])]
    out = {
        "image": f"images/{name}.png",
        "width": int(size[0]), "height": int(size[1]),
        "hfov_deg": hfov, "fx": k["fx"], "fy": k["fy"], "cx": k["cx"], "cy": k["cy"],
        "axes": "points are [right, up, ahead] in metres from the camera",
        "rope": {"diameter": 0.0508, "points": geom.to_camera(view["rope_points"], R, t).tolist()},
        "tether": {"diameter": 0.012, "segments": tether},
        "cables": [{"diameter": c["diameter"], "points": geom.to_camera(c["points"], R, t).tolist()} for c in view["cables"]],
        "classes": f"classes/{name}.png",
        "level": view["settings"].get("world", ""),
        "post": None,
        "simulator": "isaac_sim",
        "settings": view["settings"],
    }
    if extra:
        out.update(extra)
    return out


def write(run, index, rgb, classes, lab):
    import cv2

    name = "%06d" % index
    fast = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    cv2.imwrite(os.path.join(run, "images", name + ".png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), fast)
    cv2.imwrite(os.path.join(run, "classes", name + ".png"), classes)
    cv2.imwrite(os.path.join(run, "masks", name + ".png"), (classes == 1).astype(np.uint8) * 255)
    # the label last: watch_capture.py takes a picture once its label is there
    path = os.path.join(run, "labels", name + ".json")
    with open(path + ".tmp", "w") as f:
        json.dump(lab, f, default=_plain)
    os.replace(path + ".tmp", path)


def _plain(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))
