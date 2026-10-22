#!/usr/bin/env python3
"""Delete unpacked episodes to free file-quota, recording how to restore them.

Deletes `unused` episodes first, then `train` episodes from the END of the
split (splits.json order), so the remaining train episodes stay a contiguous
prefix and every `eval` episode is kept. Before deleting it writes, next to
the splits file:

    deleted_episodes.txt   one "<name> <split>" line per deleted episode
    restore_members.txt    tar paths of those episode folders
    RESTORE.md             counts and the exact command to unpack them again

    python tools/prune_episodes.py --root data/hudnav-20k/OpenWorldCreativeHud-v0 \
        --splits data/hudnav-20k/motworld-hudnav-20k.splits.json --free-entries 60000
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

EPISODE_FILES = {"rgb.mp4", "data.npz", "level_metadata.json", "sha256.txt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--free-entries", type=int, required=True,
                        help="Files + directories to free (5 per episode).")
    parser.add_argument("--archive", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    splits_path = Path(args.splits).resolve()
    out_dir = splits_path.parent
    archive = Path(args.archive).resolve() if args.archive else next(out_dir.glob("*.tar.zst"), None)
    if not archive or not archive.is_file():
        sys.exit("ERROR: archive not found; refusing to delete without a way to restore")
    if (out_dir / "deleted_episodes.txt").exists():
        sys.exit(f"ERROR: {out_dir}/deleted_episodes.txt exists; restore or merge before pruning again")

    episodes = json.load(open(splits_path))["episodes"]
    present = [e for e in episodes if (root / e["name"]).is_dir()]
    unused = [e for e in present if e["split"] == "unused"]
    train = [e for e in present if e["split"] == "train"]
    n_episodes = -(-args.free_entries // (len(EPISODE_FILES) + 1))  # ceil
    victims = (unused + list(reversed(train)))[:n_episodes]
    if len(victims) < n_episodes:
        sys.exit(f"ERROR: only {len(victims)} unused+train episodes present, need {n_episodes}")
    kept_train = len(train) - sum(e["split"] == "train" for e in victims)

    # every target must be a plain episode folder under root
    for e in victims:
        d = root / e["name"]
        if d.parent != root or set(os.listdir(d)) - EPISODE_FILES:
            sys.exit(f"ERROR: unexpected contents in {d}; nothing deleted")

    top = root.name
    with open(out_dir / "deleted_episodes.txt", "w") as f:
        for e in victims:
            f.write(f"{e['name']} {e['split']}\n")
    # folder entries only: tar extracts a listed folder recursively, and also
    # listing its files makes tar report them as "not found" (exit status 2)
    with open(out_dir / "restore_members.txt", "w") as f:
        for e in victims:
            f.write(f"{top}/{e['name']}/\n")
    counts = {s: sum(e["split"] == s for e in victims) for s in ("unused", "train")}
    with open(out_dir / "RESTORE.md", "w") as f:
        f.write(f"""# Pruned episodes ({time.strftime('%Y-%m-%d %H:%M')})

Deleted {len(victims)} episodes ({len(victims) * (len(EPISODE_FILES) + 1)} files + folders)
from `{root}` to free file quota:

- unused: {counts['unused']} (all of them)
- train: {counts['train']} (the LAST ones in splits.json order)
- eval: 0 (all 200 kept)

Still unpacked: the first {kept_train} train episodes of splits.json and every eval episode.
Names of the deleted episodes: `deleted_episodes.txt`; tar paths: `restore_members.txt`.

## Restore (re-unpack exactly these episodes)

Reads the whole archive once and writes only the listed episodes (~{len(victims) * (len(EPISODE_FILES) + 1)} entries):

```bash
cd {out_dir}
zstd -dc {archive.name} | tar -xf - -T restore_members.txt
```

Afterwards delete deleted_episodes.txt, restore_members.txt and this file.
""")
    print(f"[prune] will delete {len(victims)} episodes: {counts}; train kept: {kept_train}")
    if args.dry_run:
        print("[prune] dry run: tracking files written, nothing deleted")
        return
    for i, e in enumerate(victims, 1):
        shutil.rmtree(root / e["name"])
        if i % 1000 == 0 or i == len(victims):
            print(f"[prune] deleted {i}/{len(victims)}", flush=True)


if __name__ == "__main__":
    main()
