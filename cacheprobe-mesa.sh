#!/usr/bin/env bash
# cacheprobe-mesa.sh: runs ./build/cacheprobe on a Mesa driver (AMD RADV, Intel ANV) under run_cases.sh, which sets
# NVIDIA's shader cache variables for every step; this maps them to Mesa's:
#   OFF      (__GL_SHADER_DISK_CACHE=0)          -> MESA_SHADER_CACHE_DISABLE=true
#   ON-EMPTY (__GL_SHADER_DISK_CACHE_PATH=<dir>) -> MESA_SHADER_CACHE_DIR=<dir> (run_cases.sh empties it before each step)
# Device: CACHEPROBE_DEVICE=N picks physical device N; without it, the first device of any vendor, which may be a
# software device or an integrated GPU. Use: CACHEPROBE_DEVICE=0 CACHEPROBE=./cacheprobe-mesa.sh ./run_cases.sh
here=$(cd "$(dirname "$0")" && pwd)
if [ "${__GL_SHADER_DISK_CACHE:-}" = 0 ]; then
    export MESA_SHADER_CACHE_DISABLE=true
    unset MESA_SHADER_CACHE_DIR
elif [ -n "${__GL_SHADER_DISK_CACHE_PATH:-}" ]; then
    unset MESA_SHADER_CACHE_DISABLE
    export MESA_SHADER_CACHE_DIR="$__GL_SHADER_DISK_CACHE_PATH"
fi
if [ -n "${CACHEPROBE_DEVICE:-}" ]; then
    exec "$here/build/cacheprobe" --device "$CACHEPROBE_DEVICE" "$@"
fi
exec "$here/build/cacheprobe" --any-vendor "$@"
