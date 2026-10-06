# Rope-detection training pictures from Isaac Sim

A second source of training pictures for the rope-detection network
([Rope_Detection](https://github.com/LukeGrif/Rope_Detection)), next to
BlueSim's capture (`bluesim/scripts/capture.gd`). It makes the same kinds of
picture with the same choices and odds, and writes the same run folders, so
the pictures drop straight into `dataset/train_seg.py`. BlueSim stays as it
is; this is an extra source.

```
OUT/images/NNNNNN.png    the picture (1920x1080 by default)
OUT/classes/NNNNNN.png   0 background, 1 rope, 2 tether (ROV tether + cables), 3 ROV
OUT/masks/NNNNNN.png     rope = 255
OUT/labels/NNNNNN.json   BlueSim's fields (camera, rope centre line, tether, cables, settings)
OUT/done                 written when the run finishes (with its timing)
```

`classes/` comes straight from Isaac Sim's Replicator semantic segmentation:
only what the camera actually sees is labelled. Rope behind a piling, the
gripper or a cable is not rope, and rope or tether lost in the water (less
than 2% contrast left) isn't labelled either (`--min-contrast 0` labels
it all, as BlueSim does). No `make_masks.py` step is needed.

## What's reproduced from BlueSim

| | |
|---|---|
| Camera | BlueROV2 Low-Light HD USB: 80° across, 1920x1080 (`--size 960x540`), pinhole, fx = fy = (W/2)/tan(40°), cx = W/2, cy = H/2 |
| ROV | BlueSim's own BlueROV2 Heavy, Newton gripper and jaws, converted by `build_rov.py` (same shapes and placement as BlueSim, which draws them at about twice real size) |
| Views | 60% from the ROV camera tilted down 25–45° (15% less), jaws 0.26 m below / 0.12 m ahead of the camera; phases approach (0.35–2.5 m), at the jaws, in the jaws, rope to one side; jaws open (60%) or closed per rope. 40% orbit views 0.25–4 m round the rope, a quarter aimed at the tether, 1 in 10 looking away |
| Rope | physics rope, 50.8 mm, never a rod: polypropylene 910, dyneema 975, nylon 1140, polyester 1380, lead-core 2000 kg/m³ in fresh water; surface-to-floor, hanging, standing, U (two surface points 3 m apart); 5/10/20 m; current 0/0.1/0.25/0.5 m/s from the left, right, ahead or behind; a new rope every 20 pictures, 3 s of physics to settle, 0.3 s between pictures |
| Rope look | new every picture: 3-strand or braided (shape and texture); 30% reds, 15% yellows, 35% marine rope colours, 20% anything; 30% tracer/fleck; some loose fibres |
| Tether | 7.6 mm Fathom-style, drawn 12 mm, from the back of the ROV to the surface, random colour (half yellow) |
| Hard negatives | 0–2 cables per picture, 6–16 mm, half yellow: crossing the rope, alongside it, or in front of the camera/gripper, half of those looped; labelled tether |
| Water and light | water tint, visibility 2–30 m, ambient light, sun, ROV lamp on (70%) or off |
| Repeatable | `--seed`: the same seed gives the same ropes, views and looks |
| Paused physics | the rope's physics only runs between pictures, never while one is taken, so the labels match exactly |

The rope physics (`ropecap/ropesim.py`) is BlueSim's rope-piece model
(weight minus buoyancy, Morison drag across and along, added mass, the
current) as a chain of points 10 cm apart. It lies on the floor, floats at
the surface and wraps round pilings. It runs in numpy, so it's the same on
every machine and needs no GPU physics.

**Underwater look.** Isaac Sim renders the scene, lit by its ambient,
sun and lamp lights, and the water is then added to every pixel from its
distance:
`I = J·exp(−β·d) + B·(1 − exp(−β·d))`. This is the image formation model
[OceanSim](https://github.com/umfieldrobotics/OceanSim) uses. Each colour
channel fades at its own rate into the water colour B, reaching 2% contrast
at the picture's visibility. On top of that come the lamp lighting up the
water in its beam and specks of suspended particles. MarineGym itself has
no underwater rendering.

**Worlds** (`--world`; a new layout, floor and textures every rope):
- `seabed`: sand, rippled sand, mud, gravel, rock, shell sand, silt and weed (now and then a pool floor), uneven, with rocks
- `harbour`: rows of round or square pilings (concrete, timber or steel, with marine growth), often a quay wall (concrete, stone or sheet piles); the rope hangs among them, so they hide parts of it
- `marinegym`: MarineGym's own `EmptyMarine.usd` seabed. Its materials download from NVIDIA's servers, so it needs internet access.
- `wreck`: a seabed plus a wreck model you give it: `--wreck path/to/wreck.usd` (MarineGym has none)

## Install (once)

Isaac Sim 5.x needs an RTX GPU with 16 GB or more. A GeForce RTX 5060 Ti
16 GB is fine; close other GPU-heavy programs while it runs.

**Windows** (PowerShell), in a folder of your choice:
```powershell
# Python 3.11 (e.g. from python.org), then a fresh environment for Isaac Sim:
py -3.11 -m venv isaac
.\isaac\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.*" --extra-index-url https://pypi.nvidia.com
git clone -b claude/beautiful-dijkstra-qs1jvt https://github.com/LukeGrif/MarineGym
git clone https://github.com/LukeGrif/bluesim
pip install -r MarineGym\rope_capture\requirements.txt
cd MarineGym\rope_capture
python build_rov.py --bluesim ..\..\bluesim      # writes assets\rov.usd + rov.json
python -m pytest tests                            # quick checks, no GPU needed
```
The first `import isaacsim` asks you to accept NVIDIA's licence (type `Yes`),
and the first run takes several minutes to compile shaders.

**Linux**: the same, with `python3.11 -m venv isaac; source isaac/bin/activate`.

## Run

**1. 20 test pictures** (at each size, for the speed):
```powershell
python capture.py C:\rope_isaac\test1080 --count 20 --seed 1
python capture.py C:\rope_isaac\test540 --count 20 --seed 1 --size 960x540
```
Each prints the pictures per minute as it goes and saves it in `timing.json`.
Add `--window` to watch Isaac Sim, and `--keep-raw` to also save the
pictures before the water is added (`raw/`).

**2. Check them** on the training machine, with Rope_Detection:
```bash
python3 dataset/watch_capture.py /path/to/test1080 --browse
```
Green must be on rope only; tethers and cables red; the gripper blue; no
green on rope hidden behind something.

**3. Next to BlueSim pictures:**
```bash
python3 compare_bluesim.py /path/to/test1080 ~/rope_sim/run1 --out compare.jpg
```

**4. A big set:** `.\collect_isaac.ps1 C:\rope_isaac` (Windows) or
`./collect_isaac.sh ~/rope_isaac` (Linux): 5 runs × 2000 pictures,
alternating seabed and harbour, seeds 1001 and up. Then train on both sources:
```bash
python3 dataset/train_seg.py ~/rope_sim/run* ~/rope_isaac/isaac_run*
```

## Tuning after the first run

The light levels are a first guess. If the pictures come out too dark or
too bright, change them:
`--ambient-scale` (default 800), `--sun-scale` (2500), `--lamp-scale`
(60000). Use `--keep-raw` to see the render before the water. `--subframes`
(default 8) trades speed for cleaner pictures.

## Without a GPU

`python dryrun.py OUT --count 20` goes through everything (the plan, the same
USD stage, the label files) with a rough CPU drawing in place of Isaac Sim's
render, to check changes on any machine (needs `pip install usd-core`). Its
pictures are only for checking, not for training.

## Files

- `capture.py`: the capture (Isaac Sim)
- `isaac_scene.py`: the USD stage: world, ROV, rope, tether, cables, camera, lights, labels
- `build_rov.py`: the ROV and gripper from BlueSim's models
- `ropecap/plan.py`: what's in each picture (capture.gd's choices); `ropesim.py`: rope physics; `worlds.py`: worlds; `textures.py`: rope, floor and piling textures; `water.py`: the water; `labels.py`: the run folder
- `dryrun.py`, `compare_bluesim.py`, `collect_isaac.sh` / `.ps1`, `tests/`
