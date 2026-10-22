#!/bin/bash
# Download one file from a (gated) Hugging Face repo on a compute node, with
# resume, then verify its SHA-256.
#
#   sbatch --account=<acct> tools/slurm/download_hf_file.slurm \
#       <repo_type/owner/name> <file> <dest_dir> [sha256]
#   e.g. datasets/percool/luanti motworld-hudnav-20k.tar.zst data/hudnav-20k <sha>
#
# Reads the token from $HF_HOME/token (or ~/.cache/huggingface/token); it is
# never printed. Re-submitting resumes a partial download.
#SBATCH --job-name=hf-download
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

REPO=${1:?repo, e.g. datasets/percool/luanti}
FILE=${2:?file name in the repo}
DEST=${3:?destination directory}
SHA=${4:-}

module load WebProxy 2>/dev/null || true   # compute nodes have no direct internet
TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
[ -s "$TOKEN_FILE" ] || TOKEN_FILE="$HOME/.cache/huggingface/token"
[ -s "$TOKEN_FILE" ] || { echo "no Hugging Face token found"; exit 1; }

mkdir -p "$DEST"
OUT="$DEST/$FILE"
URL="https://huggingface.co/$REPO/resolve/main/$FILE"
echo "[download] $URL -> $OUT"
start=$(date +%s)
curl -fL --retry 20 --retry-delay 30 --retry-all-errors -C - \
    -H "Authorization: Bearer $(cat "$TOKEN_FILE")" \
    -o "$OUT.part" "$URL"
mv "$OUT.part" "$OUT"
secs=$(( $(date +%s) - start ))
bytes=$(stat -c %s "$OUT")
echo "[download] $bytes bytes in ${secs}s ($(( bytes / (secs > 0 ? secs : 1) / 1000000 )) MB/s)"

if [ -n "$SHA" ]; then
  got=$(sha256sum "$OUT" | cut -d' ' -f1)
  if [ "$got" = "$SHA" ]; then
    echo "[download] sha256 OK"
  else
    echo "[download] sha256 MISMATCH: expected $SHA got $got"; exit 1
  fi
fi
