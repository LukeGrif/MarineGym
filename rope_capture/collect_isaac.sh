#!/bin/bash
# Several Isaac Sim capture runs in one go (different seeds and worlds), like
# Rope_Detection's dataset/collect_sim.sh does for BlueSim. A run already done
# is skipped, so it can be stopped and started again.
#
# usage: ./collect_isaac.sh OUT [RUNS] [PICTURES_PER_RUN] [SIZE]
#   ./collect_isaac.sh ~/rope_isaac                 # 5 runs x 2000, 1920x1080
#   ./collect_isaac.sh ~/rope_isaac 2 500 960x540
# WORLDS="mixed" (default; e.g. "mixed reef"): taken in turn by the runs.
# PYTHON: the Python that has Isaac Sim (default: python).
# Seeds start at 1001, so they never repeat a BlueSim run's seed.
set -e
OUT=${1:?usage: collect_isaac.sh OUT [RUNS] [PICTURES_PER_RUN] [SIZE]}
RUNS=${2:-5}
COUNT=${3:-2000}
SIZE=${4:-1920x1080}
PYTHON=${PYTHON:-python}
read -r -a WORLDS <<< "${WORLDS:-mixed}"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
for i in $(seq 1 "$RUNS"); do
  RUN="$OUT/isaac_run$i"
  if [ -f "$RUN/done" ]; then echo "isaac_run$i already done"; continue; fi
  WORLD=${WORLDS[$(( (i - 1) % ${#WORLDS[@]} ))]}
  echo "isaac_run$i: $COUNT pictures ($SIZE), seed $((1000 + i)), world $WORLD"
  "$PYTHON" "$HERE/capture.py" "$RUN" --count "$COUNT" --size "$SIZE" --seed $((1000 + i)) --world "$WORLD"
done
echo "done: train with  python3 dataset/train_seg.py $OUT/isaac_run* (plus your BlueSim runs)"
