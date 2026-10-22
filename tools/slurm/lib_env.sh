# Sourced by the Slurm scripts: unpack the packed Python env to node-local
# /tmp (the env is ~42k files, too many for the scratch file quota) and set $PY.
#
#   source tools/slurm/lib_env.sh   # sets PY, removes the env on exit if we unpacked it

MG35_ENV_TAR=${MG35_ENV_TAR:-.cache/env/mg35_env.tar}   # built by packing /tmp/mg35_env
MG35_ENV_DIR=/tmp/mg35_env                               # paths inside the env are absolute

(
  flock 9
  if [ ! -x "$MG35_ENV_DIR/venv/bin/python" ]; then
    echo "[env] unpacking $MG35_ENV_TAR -> /tmp"
    tar -C /tmp -xf "$MG35_ENV_TAR"
    touch "$MG35_ENV_DIR/.created_by_${SLURM_JOB_ID:-$$}"
  fi
) 9>/tmp/mg35_env.lock

_MG35_CREATED_ENV=0
[ -e "$MG35_ENV_DIR/.created_by_${SLURM_JOB_ID:-$$}" ] && _MG35_CREATED_ENV=1
_mg35_cleanup_env() { [ "$_MG35_CREATED_ENV" = 1 ] && rm -rf "$MG35_ENV_DIR"; return 0; }
trap _mg35_cleanup_env EXIT

PY="$MG35_ENV_DIR/venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
