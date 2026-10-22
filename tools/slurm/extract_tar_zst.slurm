#!/bin/bash
# Unpack a .tar.zst after checking it fits the scratch quota (files + bytes).
#
#   sbatch --account=<acct> tools/slurm/extract_tar_zst.slurm <archive> <dest_dir> [splits.json]
#
# Pass 1 lists the archive (entry count, total size) without writing anything;
# if the result would not fit `showquota` (with a 5% / 5000-file margin) the
# job stops before extracting. Pass 2 extracts. With a splits.json (keys
# "episode_path" and "episodes"[].name), the extracted episode directories are
# checked against it.
#SBATCH --job-name=extract
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ARCHIVE=${1:?archive .tar.zst}
DEST=${2:?destination directory}
SPLITS=${3:-}
command -v zstd >/dev/null || module load zstd/1.5.5

# ---- pass 1: size the archive ----------------------------------------------
echo "[extract] listing $ARCHIVE"
read -r N_ENTRIES N_FILES N_BYTES < <(zstd -dc "$ARCHIVE" | tar -tvf - | awk '
  {n++; if (substr($1,1,1) == "-") {f++; b += $3}} END {printf "%d %d %.0f\n", n, f, b}')
echo "[extract] entries=$N_ENTRIES regular_files=$N_FILES unpacked_bytes=$N_BYTES ($(( N_BYTES / 1000000000 )) GB)"

# ---- quota check -------------------------------------------------------------
QLINE=$(showquota 2>/dev/null | grep -F "/scratch/user/$USER" || true)
if [ -n "$QLINE" ]; then
  # columns: path  used_space  space_limit  used_files  file_limit  (units like 268.4G / 1.0T)
  read -r _ USED_SP LIM_SP USED_F LIM_F <<<"$QLINE"
  to_bytes() { numfmt --from=iec "${1%B}"; }
  USED_B=$(to_bytes "$USED_SP"); LIM_B=$(to_bytes "$LIM_SP")
  NEED_F=$(( USED_F + N_ENTRIES + 5000 ))
  NEED_B=$(python3 -c "print(int($USED_B + $N_BYTES * 1.05))")
  echo "[extract] quota files: $USED_F used + $N_ENTRIES new (+5000 margin) = $NEED_F of $LIM_F"
  echo "[extract] quota bytes: $USED_SP used + $(( N_BYTES / 1000000000 )) GB new (+5%) of $LIM_SP"
  if [ "$NEED_F" -gt "$LIM_F" ] || [ "$NEED_B" -gt "$LIM_B" ]; then
    echo "[extract] ABORT: would exceed the scratch quota; nothing was extracted"
    exit 2
  fi
else
  echo "[extract] WARNING: showquota unavailable; skipping quota check"
fi

# ---- pass 2: extract ---------------------------------------------------------
mkdir -p "$DEST"
start=$(date +%s)
zstd -dc "$ARCHIVE" | tar -xf - -C "$DEST"
echo "[extract] extracted in $(( $(date +%s) - start ))s -> $DEST"

# ---- verify ------------------------------------------------------------------
if [ -n "$SPLITS" ]; then
  python3 - "$DEST" "$SPLITS" <<'EOF'
import json, os, sys
dest, splits = sys.argv[1], sys.argv[2]
d = json.load(open(splits))
root = os.path.join(dest, d["episode_path"].split("<name>")[0])
names = {e["name"]: e["split"] for e in d["episodes"]}
present = {n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n))}
missing = sorted(set(names) - present)
extra = sorted(present - set(names))
by_split = {}
for n in present & set(names):
    by_split[names[n]] = by_split.get(names[n], 0) + 1
print(f"[extract] episode dirs under {root}: {len(present)} (expected {len(names)})")
print(f"[extract] by split: {by_split}; missing={len(missing)} extra={len(extra)}")
if missing:
    print("[extract] first missing:", missing[:10])
EOF
fi
showquota | grep -F "/scratch/user/$USER" || true
