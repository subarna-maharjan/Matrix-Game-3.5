#!/bin/bash
# Re-unpack episodes deleted by tools/prune_episodes.py, optionally one split.
#
#   sbatch --account=<acct> tools/slurm/restore_episodes.slurm <dataset dir> [split]
#   e.g. data/hudnav-20k train
#
# Reads <dataset dir>/deleted_episodes.txt, extracts those episode folders from
# the archive, verifies them, and rewrites the tracking files to list only the
# episodes that are still deleted.
#SBATCH --job-name=mg35-restore
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
command -v zstd >/dev/null || module load zstd/1.5.5

DIR=${1:?dataset dir with deleted_episodes.txt}
SPLIT=${2:-}
ARCHIVE=$(ls "$DIR"/*.tar.zst | head -1)
TOP=$(head -1 "$DIR/restore_members.txt" | cut -d/ -f1)
LIST=$(mktemp)
awk -v s="$SPLIT" 's == "" || $2 == s {print $1}' "$DIR/deleted_episodes.txt" > "$LIST"
N=$(wc -l < "$LIST")
read -r _ _ _ USED_F LIM_F <<<"$(showquota | grep -F "/scratch/user/$USER")"
echo "[restore] $N episodes (${SPLIT:-all splits}) from $ARCHIVE; files used $USED_F of $LIM_F"
if [ $(( USED_F + N * 5 + 5000 )) -gt "$LIM_F" ]; then
  echo "[restore] ABORT: not enough file quota"; exit 2
fi

sed "s|^|$TOP/|; s|$|/|" "$LIST" > "$LIST.members"
start=$(date +%s)
zstd -dc "$ARCHIVE" | tar -xf - -C "$DIR" -T "$LIST.members"
echo "[restore] extracted in $(( $(date +%s) - start ))s"

missing=$(while read -r n; do [ -f "$DIR/$TOP/$n/rgb.mp4" ] && [ -f "$DIR/$TOP/$n/data.npz" ] || echo "$n"; done < "$LIST" | wc -l)
[ "$missing" -eq 0 ] || { echo "[restore] ERROR: $missing episodes incomplete"; exit 1; }

# keep tracking only what is still deleted
awk 'NR==FNR {done[$1]=1; next} !($1 in done)' "$LIST" "$DIR/deleted_episodes.txt" > "$DIR/deleted_episodes.txt.new"
mv "$DIR/deleted_episodes.txt.new" "$DIR/deleted_episodes.txt"
sed "s|^|$TOP/|; s|$|/|" <(cut -d' ' -f1 "$DIR/deleted_episodes.txt") > "$DIR/restore_members.txt"
REMAIN=$(wc -l < "$DIR/deleted_episodes.txt")
printf '\n## Restored %s (%s)\n\nRestored %s %s episodes; %s remain deleted (listed in deleted_episodes.txt / restore_members.txt).\n' \
    "$(date '+%Y-%m-%d %H:%M')" "$SPLIT" "$N" "${SPLIT:-all}" "$REMAIN" >> "$DIR/RESTORE.md"
rm -f "$LIST" "$LIST.members"
echo "[restore] done: $N restored, $REMAIN still deleted"
showquota | grep -F "/scratch/user/$USER" || true
