# Sourced by the Slurm scripts: unpack the packed Python env to node-local
# /tmp (the env is ~42k files, too many for the scratch file quota) and set $PY.
#
#   source tools/slurm/lib_env.sh   # sets PY; the last job on the node removes the env
#
# Safe for several jobs on one node: the env is extracted to a temporary dir
# and renamed into place only when complete (marked by .complete), and every
# job registers itself under .users/; the last one to exit deletes the env.

MG35_ENV_TAR=${MG35_ENV_TAR:-.cache/env/mg35_env.tar}   # built by packing /tmp/mg35_env
MG35_ENV_DIR=${MG35_ENV_DIR:-/tmp/mg35_env}             # paths inside the env are absolute
MG35_ENV_LOCK=${MG35_ENV_DIR}.lock
_MG35_ME="${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_TASK_ID:-0}.$$"

(
  flock 9
  if [ ! -e "$MG35_ENV_DIR/.complete" ]; then
    rm -rf "$MG35_ENV_DIR" "$MG35_ENV_DIR.partial"
    echo "[env] unpacking $MG35_ENV_TAR -> $MG35_ENV_DIR"
    mkdir -p "$MG35_ENV_DIR.partial"
    tar -C "$MG35_ENV_DIR.partial" -xf "$MG35_ENV_TAR"
    # the tar holds mg35_env/...; absolute paths inside expect /tmp/mg35_env
    mv "$MG35_ENV_DIR.partial/mg35_env" "$MG35_ENV_DIR"
    rmdir "$MG35_ENV_DIR.partial"
    touch "$MG35_ENV_DIR/.complete"
  fi
  mkdir -p "$MG35_ENV_DIR/.users"
  touch "$MG35_ENV_DIR/.users/$_MG35_ME"
) 9>"$MG35_ENV_LOCK"

_mg35_release_env() {
  (
    flock 9
    rm -f "$MG35_ENV_DIR/.users/$_MG35_ME"
    if [ -z "$(ls -A "$MG35_ENV_DIR/.users" 2>/dev/null)" ]; then
      rm -rf "$MG35_ENV_DIR"
    fi
  ) 9>"$MG35_ENV_LOCK"
  return 0
}
trap _mg35_release_env EXIT

PY="$MG35_ENV_DIR/venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
