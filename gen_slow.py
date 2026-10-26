"""gen_slow.py <statements> <out.comp>: a synthetic compute shader whose straight-line body is long enough to take the
NVIDIA compiler a while. Eight specialization constants (ids 0-7) feed the loop bound and the arithmetic, so a creation
may supply one or all of them. No title data: every constant below is generated here."""
import sys

n = int(sys.argv[1])
lines = [
    "#version 450",
    "layout(local_size_x = 64) in;",
]
lines += [f"layout(constant_id = {i}) const uint K{i} = {i + 1}u;" for i in range(8)]
lines += [
    "layout(set = 0, binding = 0) buffer Out { uint data[]; } o;",
    "void main() {",
    "    uint id = gl_GlobalInvocationID.x;",
    "    float v = float(id) * 0.001;",
    "    float w = float(id ^ 0x5bd1e995u) * 0.002;",
    "    uint h = id * 2654435761u + K1 * K2 + K3 * K4 + K5 * K6 + K7;",
    "    for (uint k = 0u; k < K0; ++k) {",
]
for i in range(n):
    a = 1.0 + (i % 97) * 0.0001
    b = (i * 7919) % 1009 * 0.001
    c = ((i * 104729) % 4294967291) | 1
    lines.append(f"        v = fma(v, {a:.6f}, w + {b:.6f} * float(k + K{i % 8}));")
    lines.append(f"        w = fract(w * 1.37 + v * {1.0 + (i % 13) * 0.01:.4f});")
    lines.append(f"        h = (h ^ floatBitsToUint(v)) * {c}u + uint(w * 255.0);")
    if i % 50 == 49:
        lines.append(f"        if ((h & {1 << (i % 31)}u) != 0u) v = -v;")
lines += [
    "    }",
    "    o.data[id] = h ^ floatBitsToUint(v + w);",
    "}",
]
with open(sys.argv[2], "w") as out:
    out.write("\n".join(lines) + "\n")
