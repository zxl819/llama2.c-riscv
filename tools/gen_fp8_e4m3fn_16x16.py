#!/usr/bin/env python3
import math
from typing import List, Tuple

# FP8 E4M3FN decode (bias=7). exp=0 => subnormals/zero, exp=15 => NaN (no Inf).
# This matches the C helper in fp8_e4m3fn.h.

def fp8e4m3fn_to_f32(x: int) -> float:
    x &= 0xFF
    sign = -1.0 if (x & 0x80) else 1.0
    exp = (x >> 3) & 0x0F
    mant = x & 0x07
    if exp == 0x0F:
        return float('nan')
    if exp == 0:
        if mant == 0:
            return sign * 0.0
        # subnormal: value = sign * 2^(1-bias) * (mant/2^3)
        return sign * (2.0 ** (1 - 7)) * (mant / 8.0)
    # normal: value = sign * 2^(exp-bias) * (1 + mant/2^3)
    return sign * (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)


def allowed_codes(exclude_subnormals_and_zero: bool = True) -> List[int]:
    codes: List[int] = []
    for b in range(256):
        exp = (b >> 3) & 0x0F
        mant = b & 0x07
        if exp == 0x0F:
            continue  # NaN
        if exclude_subnormals_and_zero and exp == 0:
            continue
        # If not excluding exp==0, still exclude +/-0 as "special"? Up to caller.
        # We keep the strict mode by default.
        codes.append(b)
    return codes


def fill_16x16(codes: List[int], offset: int = 0) -> List[int]:
    out: List[int] = []
    n = len(codes)
    for i in range(256):
        out.append(codes[(i + offset) % n])
    return out


def gemm_ref_A_Bt(A: List[int], B: List[int], M: int, K: int, N: int) -> List[float]:
    # Hardware behavior: C = A * B^T
    # A is [M x K] row-major
    # B is [N x K] row-major
    Af = [fp8e4m3fn_to_f32(v) for v in A]
    Bf = [fp8e4m3fn_to_f32(v) for v in B]
    C = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for kk in range(K):
                acc += Af[i * K + kk] * Bf[j * K + kk]
            C[i * N + j] = acc
    return C


def bit_coverage(vals: List[int]) -> Tuple[int, List[int]]:
    # Returns (mask, per-bit counts of ones)
    ones = [0] * 8
    for v in vals:
        for b in range(8):
            ones[b] += (v >> b) & 1
    mask = 0
    for b in range(8):
        if ones[b] != 0 and ones[b] != len(vals):
            mask |= (1 << b)
    return mask, ones


def format_u8_array(name: str, vals: List[int], cols: int = 16) -> str:
    lines = []
    lines.append(f"__attribute__((section(\".{name}\"), aligned(64))) static const uint8_t {name}_fp8[M*K] = {{")
    for r in range(0, len(vals), cols):
        chunk = vals[r:r+cols]
        lines.append("  " + ", ".join(f"0x{v:02X}" for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def format_u8_array_B(name: str, vals: List[int], cols: int = 16) -> str:
    # B is NxK (row-major) for A*B^T behavior.
    lines = []
    lines.append(f"__attribute__((section(\".{name}\"), aligned(64))) static const uint8_t {name}_fp8[N*K] = {{")
    for r in range(0, len(vals), cols):
        chunk = vals[r:r+cols]
        lines.append("  " + ", ".join(f"0x{v:02X}" for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def format_f32_array(name: str, vals: List[float], cols: int = 8) -> str:
    lines = []
    lines.append(f"__attribute__((aligned(64))) static const float {name}[M*N] = {{")
    for r in range(0, len(vals), cols):
        chunk = vals[r:r+cols]
        lines.append("  " + ", ".join(f"{v:.8g}f" for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def main():
    M = K = N = 16
    codes = allowed_codes(exclude_subnormals_and_zero=True)
    if len(codes) != 224:
        raise SystemExit(f"Expected 224 normal finite codes, got {len(codes)}")

    A = fill_16x16(codes, offset=0)   # [M x K]
    B = fill_16x16(codes, offset=37)  # [N x K]

    # Sanity: ensure we didn't include NaNs and exp!=0 in strict mode
    for v in A + B:
        exp = (v >> 3) & 0x0F
        if exp == 0 or exp == 0x0F:
            raise SystemExit("Found excluded exp in generated codes")

    maskA, onesA = bit_coverage(A)
    maskB, onesB = bit_coverage(B)
    # Each bit should have both 0 and 1 across the 256 elements.
    if maskA != 0xFF or maskB != 0xFF:
        raise SystemExit(f"Bit coverage failed: maskA=0x{maskA:02X} maskB=0x{maskB:02X}")

    D = gemm_ref_A_Bt(A, B, M, K, N)

    print("#pragma once")
    print("#include <stdint.h>")
    print("")
    print("#define M 16")
    print("#define K 16")
    print("#define N 16")
    print("")
    print("// FP8(E4M3FN) 16x16 test data (bytes).")
    print("// Generation policy: enumerate ALL normal finite E4M3FN encodings (exp=1..14),")
    print("// excluding subnormals/zero (exp=0) and excluding NaNs (exp=15).")
    print("// There are 224 such encodings; we fill 16x16=256 entries by wrapping (repeat first 32).")
    print("// This ensures full bit-position coverage (each of 8 bits appears as 0 and 1).")
    print("")
    print("// A: [M x K] row-major")
    print(format_u8_array("matA", A, cols=16).replace("matA_fp8[M*K]", "A_fp8[M*K]"))
    print("")
    print("// B: [N x K] row-major (kernel computes A * B^T)")
    print(format_u8_array_B("matB", B, cols=16).replace("matB_fp8[N*K]", "B_fp8[N*K]"))
    print("")
    print("// C: accumulator initial value (fp32)")
    print("__attribute__((section(\".matC\"), aligned(64))) static float C[M*N] = {0};")
    print("")
    print("// D_ref: golden fp32 result for C = A * B^T (row-major)")
    print(format_f32_array("D_ref", D, cols=8))


if __name__ == "__main__":
    main()
