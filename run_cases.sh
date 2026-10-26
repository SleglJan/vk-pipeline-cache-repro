#!/usr/bin/env bash
# run_cases.sh [statements] [fill-count] [huge-statements] [prefill]: the Linux driver for the cases in README.md.
# Needs ./build/cacheprobe (see README), python3 and glslc on PATH. Writes everything under ./out/<timestamp>/.
# Driver shader cache modes (NVIDIA, Linux): OFF = __GL_SHADER_DISK_CACHE=0; ON-EMPTY = the variable unset and
# __GL_SHADER_DISK_CACHE_PATH pointing at a fresh empty directory, so the user's own driver cache is never touched.
# Mesa drivers (AMD, Intel): CACHEPROBE=./cacheprobe-mesa.sh maps both modes to Mesa's cache variables (see README).
set -u
N=${1:-3000}
FILL=${2:-200}
HUGE=${3:-40000}
PREFILL=${4:-150}
HERE=$(cd "$(dirname "$0")" && pwd)
B=${CACHEPROBE:-$HERE/build/cacheprobe}
O=$HERE/out/$(date +%Y%m%d-%H%M%S)
mkdir -p "$O/nvcache" || exit 1
[ -x "$B" ] || { echo "missing $B (build first, or set CACHEPROBE)"; exit 1; }
OFF=(env __GL_SHADER_DISK_CACHE=0)
ON=(env -u __GL_SHADER_DISK_CACHE __GL_SHADER_DISK_CACHE_PATH="$O/nvcache")
fresh() { rm -rf "$O/nvcache"; mkdir -p "$O/nvcache"; }
S=$O/slow.spv; SM=$O/small.spv; HG=$O/huge.spv
L=$O/cases.log
(
    echo "run_cases $(date '+%F %T %Z') statements=$N fill=$FILL huge=$HUGE prefill=$PREFILL"
    python3 "$HERE/gen_slow.py" "$N" "$O/slow.comp" || exit 1
    python3 "$HERE/gen_slow.py" "$HUGE" "$O/huge.comp" || exit 1
    printf '#version 450\nlayout(local_size_x = 64) in;\nlayout(constant_id = 0) const uint K = 1u;\nlayout(set = 0, binding = 0) buffer Out { uint data[]; } o;\nvoid main() { o.data[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x * K; }\n' > "$O/small.comp"
    for f in slow small huge; do glslc -fshader-stage=compute -O "$O/$f.comp" -o "$O/$f.spv" || exit 1; done
    echo "slow.spv $(stat -c %s "$S") bytes, small.spv $(stat -c %s "$SM") bytes, huge.spv $(stat -c %s "$HG") bytes"

    echo "#### A: one pipeline, cache OFF: export in process 1, load in fresh processes"
    echo "== A0 control: no cache";                        "${OFF[@]}" "$B" load "$S" none
    for opts in "" "--subgroup32" "--full" "--uab --push --sets2" "--spec 8" "--full --uab --push --sets2 --spec 8"; do
        # shellcheck disable=SC2086
        echo "== A1 export ($opts)";                      "${OFF[@]}" "$B" $opts export "$S" "$O/i.bin" "$O/ii.bin" || { echo "export failed"; continue; }
        echo "== A2 load (ii) ($opts)";                   "${OFF[@]}" "$B" $opts load "$S" "$O/ii.bin"
        echo "== A3 load (i) ($opts)";                    "${OFF[@]}" "$B" $opts load "$S" "$O/i.bin"
        echo "== A4 load (i) after an unrelated pipeline ($opts)"; "${OFF[@]}" "$B" $opts load "$S" "$O/i.bin" "$SM"
    done

    echo "#### B: a $FILL-variant cache (OFF writer) and a single-pipeline file next to it"
    echo "== B1 fill $FILL variants";                     "${OFF[@]}" "$B" --spec 8 fill "$S" "$FILL" "$O/big.bin" || exit 1
    echo "== B2 export variant 999";                      "${OFF[@]}" "$B" --spec 8 --seed 999 export "$S" "$O/i999.bin" "$O/ii999.bin" || exit 1
    echo "== B3 control: variant 999, no cache";          "${OFF[@]}" "$B" --spec 8 --seed 999 load "$S" none
    echo "== B4 variant 999 from (ii) alone";             "${OFF[@]}" "$B" --spec 8 --seed 999 load "$S" "$O/ii999.bin"
    echo "== B5 big cache imported first, then (ii)";     "${OFF[@]}" "$B" --spec 8 --seed 999 --pre "$O/big.bin" load "$S" "$O/ii999.bin"
    echo "== B6 (ii) first, then the big cache";          "${OFF[@]}" "$B" --spec 8 --seed 999 --post "$O/big.bin" load "$S" "$O/ii999.bin"
    echo "== B7 OFF reader: probes of the big cache";     for seed in 0 1 50 100 150 $((FILL - 1)); do "${OFF[@]}" "$B" --spec 8 probe "$S" "$O/big.bin" "$seed"; done
    echo "== B8 ON-EMPTY reader: probes of the big cache"; for seed in 0 20 50 80 100 130 150 180 $((FILL - 1)); do fresh; "${ON[@]}" "$B" --spec 8 probe "$S" "$O/big.bin" "$seed"; done

    echo "#### C: cache ON-EMPTY on the reading side"
    fresh; echo "== C1 control: no cache";                "${ON[@]}" "$B" --spec 8 --seed 999 load "$S" none
    fresh; echo "== C2 (ii) alone";                       "${ON[@]}" "$B" --spec 8 --seed 999 load "$S" "$O/ii999.bin"
    fresh; echo "== C3 (i) alone";                        "${ON[@]}" "$B" --spec 8 --seed 999 load "$S" "$O/i999.bin"
    fresh; echo "== C4 big cache first, then (ii)";       "${ON[@]}" "$B" --spec 8 --seed 999 --pre "$O/big.bin" load "$S" "$O/ii999.bin"

    echo "#### D: eight threads compiling cold variants during the creation"
    echo "== D1 OFF (ii), busy";                          "${OFF[@]}" "$B" --spec 8 --seed 999 --busy 8 load "$S" "$O/ii999.bin"
    echo "== D2 OFF big first, (ii), busy";               "${OFF[@]}" "$B" --spec 8 --seed 999 --busy 8 --pre "$O/big.bin" load "$S" "$O/ii999.bin"
    fresh; echo "== D3 ON-EMPTY big first, (ii), busy";   "${ON[@]}" "$B" --spec 8 --seed 999 --busy 8 --pre "$O/big.bin" load "$S" "$O/ii999.bin"

    echo "#### E: cache ON-EMPTY on the writing side too"
    fresh; echo "== E1 export variant 777";               "${ON[@]}" "$B" --spec 8 --seed 777 export "$S" "$O/i777.bin" "$O/ii777.bin"
    fresh; echo "== E2 load (ii)";                        "${ON[@]}" "$B" --spec 8 --seed 777 load "$S" "$O/ii777.bin"
    fresh; echo "== E3 load (i)";                         "${ON[@]}" "$B" --spec 8 --seed 777 load "$S" "$O/i777.bin"
    fresh; echo "== E4 load (ii), big first, busy";       "${ON[@]}" "$B" --spec 8 --seed 777 --busy 8 --pre "$O/big.bin" load "$S" "$O/ii777.bin"

    echo "#### F: a large pipeline ($HUGE statements)"
    echo "== F0 control: no cache";                       "${OFF[@]}" "$B" --spec 8 --seed 555 load "$HG" none
    echo "== F1 export OFF, variant 555";                 "${OFF[@]}" "$B" --spec 8 --seed 555 export "$HG" "$O/hoff_i.bin" "$O/hoff_ii.bin" || exit 1
    fresh; echo "== F2 export ON-EMPTY, variant 556";     "${ON[@]}" "$B" --spec 8 --seed 556 export "$HG" "$O/hon_i.bin" "$O/hon_ii.bin" || exit 1
    echo "== F3 OFF (ii) alone";                          "${OFF[@]}" "$B" --spec 8 --seed 555 load "$HG" "$O/hoff_ii.bin"
    fresh; echo "== F4 ON-EMPTY (ii) alone";              "${ON[@]}" "$B" --spec 8 --seed 556 load "$HG" "$O/hon_ii.bin"
    echo "== F5 OFF big first, then (ii)";                "${OFF[@]}" "$B" --spec 8 --seed 555 --pre "$O/big.bin" load "$HG" "$O/hoff_ii.bin"
    fresh; echo "== F6 ON-EMPTY big first, then (ii)";    "${ON[@]}" "$B" --spec 8 --seed 556 --pre "$O/big.bin" load "$HG" "$O/hon_ii.bin"
    echo "== F7 OFF (ii) first, then big";                "${OFF[@]}" "$B" --spec 8 --seed 555 --post "$O/big.bin" load "$HG" "$O/hoff_ii.bin"
    echo "== F8 OFF big first, then (i)";                 "${OFF[@]}" "$B" --spec 8 --seed 555 --pre "$O/big.bin" load "$HG" "$O/hoff_i.bin"

    echo "#### G: a process that compiled $PREFILL variants first"
    echo "== G1 OFF: prefill, then import the large (ii) and create"; "${OFF[@]}" "$B" --spec 8 --seed 555 --prefill "$PREFILL" load "$HG" "$O/hoff_ii.bin" "$S"
    echo "== G2 OFF: prefill, then import big and create its variant 0"; "${OFF[@]}" "$B" --spec 8 --prefill "$PREFILL" probe "$S" "$O/big.bin" 0
    fresh; echo "== G3 ON-EMPTY: prefill, then import the large (ii) and create"; "${ON[@]}" "$B" --spec 8 --seed 556 --prefill "$PREFILL" load "$HG" "$O/hon_ii.bin" "$S"
    fresh; echo "== G4 ON-EMPTY: prefill, then import big and create its variant 0"; "${ON[@]}" "$B" --spec 8 --prefill "$PREFILL" probe "$S" "$O/big.bin" 0
    echo "done $(date '+%T %Z')"
) > "$L" 2>&1
rc=$?
grep -E '^(run_cases|slow\.spv|#### |== |device |.*creation|fill variant|fill:|prefill:|busy:|\(i|cache .*initial [1-9]|done|export failed|cacheprobe:|missing)' "$L" | sed -E "s#$O/##g"
echo "full log: ${L#"$HERE"/}"
exit $rc
