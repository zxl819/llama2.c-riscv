/* riscv_xxiangshaname.h - RISC-V Xiangshan AME Intrinsics */

#ifndef __RISCV_XXIANGSHANAME_H
#define __RISCV_XXIANGSHANAME_H

#include <stdint.h>
#include <stddef.h>

#ifndef __riscv_xxiangshaname
#error "Xiangshan AME intrinsics require the xxiangshaname extension."
#endif

#define __rvm_overloaded static inline __attribute__((__always_inline__, __nodebug__, __overloadable__))

#ifndef __RISCV_VECTOR_H
typedef __rvv_int8m1_t    vint8m1_t;
typedef __rvv_uint8m1_t   vuint8m1_t;
typedef __rvv_int16m1_t   vint16m1_t;
typedef __rvv_uint16m1_t  vuint16m1_t;
typedef __rvv_int32m1_t   vint32m1_t;
typedef __rvv_uint32m1_t  vuint32m1_t;
typedef __rvv_int64m1_t   vint64m1_t;
typedef __rvv_uint64m1_t  vuint64m1_t;
typedef __rvv_float16m1_t vfloat16m1_t;
typedef __rvv_float32m1_t vfloat32m1_t;
typedef __rvv_float64m1_t vfloat64m1_t;
typedef __rvv_int8m2_t    vint8m2_t;
typedef __rvv_uint8m2_t   vuint8m2_t;
typedef __rvv_int16m2_t   vint16m2_t;
typedef __rvv_uint16m2_t  vuint16m2_t;
typedef __rvv_int32m2_t   vint32m2_t;
typedef __rvv_uint32m2_t  vuint32m2_t;
typedef __rvv_int64m2_t   vint64m2_t;
typedef __rvv_uint64m2_t  vuint64m2_t;
typedef __rvv_float16m2_t vfloat16m2_t;
typedef __rvv_float32m2_t vfloat32m2_t;
typedef __rvv_float64m2_t vfloat64m2_t;
typedef __rvv_int8m4_t    vint8m4_t;
typedef __rvv_uint8m4_t   vuint8m4_t;
typedef __rvv_int16m4_t   vint16m4_t;
typedef __rvv_uint16m4_t  vuint16m4_t;
typedef __rvv_int32m4_t   vint32m4_t;
typedef __rvv_uint32m4_t  vuint32m4_t;
typedef __rvv_int64m4_t   vint64m4_t;
typedef __rvv_uint64m4_t  vuint64m4_t;
typedef __rvv_float16m4_t vfloat16m4_t;
typedef __rvv_float32m4_t vfloat32m4_t;
typedef __rvv_float64m4_t vfloat64m4_t;
#endif

typedef __rvv_int8m16_t    mint8_t;
typedef __rvv_uint8m16_t   muint8_t;
typedef __rvv_int16m16_t   mint16_t;
typedef __rvv_uint16m16_t  muint16_t;
typedef __rvv_int32m16_t   mint32_t;
typedef __rvv_uint32m16_t  muint32_t;
typedef __rvv_int64m16_t   mint64_t;
typedef __rvv_uint64m16_t  muint64_t;
typedef __rvv_float16m16_t mfloat16_t;
typedef __rvv_float32m16_t mfloat32_t;
typedef __rvv_float64m16_t mfloat64_t;

__rvm_overloaded mint8_t mla_m(const int8_t * op0, size_t op1) {
  return __builtin_riscv_mla_m_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mla_m(const uint8_t * op0, size_t op1) {
  return __builtin_riscv_mla_m_u8m1(op0, op1);
}
__rvm_overloaded void msa_m(mint8_t op0, const int8_t * op1, size_t op2) {
  return __builtin_riscv_msa_m_i8m1(op0, op1, op2);
}
__rvm_overloaded void msa_m(muint8_t op0, const uint8_t * op1, size_t op2) {
  return __builtin_riscv_msa_m_u8m1(op0, op1, op2);
}
__rvm_overloaded mint8_t mbcar_m(mint8_t op0) {
  return __builtin_riscv_mbcar_m_i8m1(op0);
}
__rvm_overloaded muint8_t mbcar_m(muint8_t op0) {
  return __builtin_riscv_mbcar_m_u8m1(op0);
}
__rvm_overloaded mint8_t mand_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mand_mm_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mand_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mand_mm_u8m1(op0, op1);
}
__rvm_overloaded int8_t mmv_x_a(mint8_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_a_i8m1(op0, op1);
}
__rvm_overloaded int16_t mmv_x_a(mint16_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_a_i16m1(op0, op1);
}
__rvm_overloaded int32_t mmv_x_a(mint32_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_a_i32m1(op0, op1);
}
__rvm_overloaded int64_t mmv_x_a(mint64_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_a_i64m1(op0, op1);
}
__rvm_overloaded mint16_t mla_m(const int16_t * op0, size_t op1) {
  return __builtin_riscv_mla_m_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mla_m(const uint16_t * op0, size_t op1) {
  return __builtin_riscv_mla_m_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mla_m(const int32_t * op0, size_t op1) {
  return __builtin_riscv_mla_m_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mla_m(const uint32_t * op0, size_t op1) {
  return __builtin_riscv_mla_m_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mla_m(const int64_t * op0, size_t op1) {
  return __builtin_riscv_mla_m_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mla_m(const uint64_t * op0, size_t op1) {
  return __builtin_riscv_mla_m_u64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mla_m(const _Float16 * op0, size_t op1) {
  return __builtin_riscv_mla_m_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mla_m(const float * op0, size_t op1) {
  return __builtin_riscv_mla_m_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mla_m(const double * op0, size_t op1) {
  return __builtin_riscv_mla_m_f64m1(op0, op1);
}
__rvm_overloaded mint8_t mlb_m(const int8_t * op0, size_t op1) {
  return __builtin_riscv_mlb_m_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mlb_m(const uint8_t * op0, size_t op1) {
  return __builtin_riscv_mlb_m_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mlb_m(const int16_t * op0, size_t op1) {
  return __builtin_riscv_mlb_m_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mlb_m(const uint16_t * op0, size_t op1) {
  return __builtin_riscv_mlb_m_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mlb_m(const int32_t * op0, size_t op1) {
  return __builtin_riscv_mlb_m_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mlb_m(const uint32_t * op0, size_t op1) {
  return __builtin_riscv_mlb_m_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mlb_m(const int64_t * op0, size_t op1) {
  return __builtin_riscv_mlb_m_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mlb_m(const uint64_t * op0, size_t op1) {
  return __builtin_riscv_mlb_m_u64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mlb_m(const _Float16 * op0, size_t op1) {
  return __builtin_riscv_mlb_m_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mlb_m(const float * op0, size_t op1) {
  return __builtin_riscv_mlb_m_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mlb_m(const double * op0, size_t op1) {
  return __builtin_riscv_mlb_m_f64m1(op0, op1);
}
__rvm_overloaded mint8_t mlc_m(const int8_t * op0, size_t op1) {
  return __builtin_riscv_mlc_m_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mlc_m(const uint8_t * op0, size_t op1) {
  return __builtin_riscv_mlc_m_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mlc_m(const int16_t * op0, size_t op1) {
  return __builtin_riscv_mlc_m_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mlc_m(const uint16_t * op0, size_t op1) {
  return __builtin_riscv_mlc_m_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mlc_m(const int32_t * op0, size_t op1) {
  return __builtin_riscv_mlc_m_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mlc_m(const uint32_t * op0, size_t op1) {
  return __builtin_riscv_mlc_m_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mlc_m(const int64_t * op0, size_t op1) {
  return __builtin_riscv_mlc_m_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mlc_m(const uint64_t * op0, size_t op1) {
  return __builtin_riscv_mlc_m_u64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mlc_m(const _Float16 * op0, size_t op1) {
  return __builtin_riscv_mlc_m_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mlc_m(const float * op0, size_t op1) {
  return __builtin_riscv_mlc_m_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mlc_m(const double * op0, size_t op1) {
  return __builtin_riscv_mlc_m_f64m1(op0, op1);
}
__rvm_overloaded mint8_t mlat_m(const int8_t * op0, size_t op1) {
  return __builtin_riscv_mlat_m_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mlat_m(const uint8_t * op0, size_t op1) {
  return __builtin_riscv_mlat_m_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mlat_m(const int16_t * op0, size_t op1) {
  return __builtin_riscv_mlat_m_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mlat_m(const uint16_t * op0, size_t op1) {
  return __builtin_riscv_mlat_m_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mlat_m(const int32_t * op0, size_t op1) {
  return __builtin_riscv_mlat_m_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mlat_m(const uint32_t * op0, size_t op1) {
  return __builtin_riscv_mlat_m_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mlat_m(const int64_t * op0, size_t op1) {
  return __builtin_riscv_mlat_m_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mlat_m(const uint64_t * op0, size_t op1) {
  return __builtin_riscv_mlat_m_u64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mlat_m(const _Float16 * op0, size_t op1) {
  return __builtin_riscv_mlat_m_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mlat_m(const float * op0, size_t op1) {
  return __builtin_riscv_mlat_m_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mlat_m(const double * op0, size_t op1) {
  return __builtin_riscv_mlat_m_f64m1(op0, op1);
}
__rvm_overloaded mint8_t mlbt_m(const int8_t * op0, size_t op1) {
  return __builtin_riscv_mlbt_m_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mlbt_m(const uint8_t * op0, size_t op1) {
  return __builtin_riscv_mlbt_m_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mlbt_m(const int16_t * op0, size_t op1) {
  return __builtin_riscv_mlbt_m_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mlbt_m(const uint16_t * op0, size_t op1) {
  return __builtin_riscv_mlbt_m_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mlbt_m(const int32_t * op0, size_t op1) {
  return __builtin_riscv_mlbt_m_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mlbt_m(const uint32_t * op0, size_t op1) {
  return __builtin_riscv_mlbt_m_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mlbt_m(const int64_t * op0, size_t op1) {
  return __builtin_riscv_mlbt_m_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mlbt_m(const uint64_t * op0, size_t op1) {
  return __builtin_riscv_mlbt_m_u64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mlbt_m(const _Float16 * op0, size_t op1) {
  return __builtin_riscv_mlbt_m_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mlbt_m(const float * op0, size_t op1) {
  return __builtin_riscv_mlbt_m_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mlbt_m(const double * op0, size_t op1) {
  return __builtin_riscv_mlbt_m_f64m1(op0, op1);
}
__rvm_overloaded mint8_t mlct_m(const int8_t * op0, size_t op1) {
  return __builtin_riscv_mlct_m_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mlct_m(const uint8_t * op0, size_t op1) {
  return __builtin_riscv_mlct_m_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mlct_m(const int16_t * op0, size_t op1) {
  return __builtin_riscv_mlct_m_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mlct_m(const uint16_t * op0, size_t op1) {
  return __builtin_riscv_mlct_m_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mlct_m(const int32_t * op0, size_t op1) {
  return __builtin_riscv_mlct_m_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mlct_m(const uint32_t * op0, size_t op1) {
  return __builtin_riscv_mlct_m_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mlct_m(const int64_t * op0, size_t op1) {
  return __builtin_riscv_mlct_m_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mlct_m(const uint64_t * op0, size_t op1) {
  return __builtin_riscv_mlct_m_u64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mlct_m(const _Float16 * op0, size_t op1) {
  return __builtin_riscv_mlct_m_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mlct_m(const float * op0, size_t op1) {
  return __builtin_riscv_mlct_m_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mlct_m(const double * op0, size_t op1) {
  return __builtin_riscv_mlct_m_f64m1(op0, op1);
}
__rvm_overloaded mint8_t mltr_m(const int8_t * op0, size_t op1) {
  return __builtin_riscv_mltr_m_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mltr_m(const uint8_t * op0, size_t op1) {
  return __builtin_riscv_mltr_m_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mltr_m(const int16_t * op0, size_t op1) {
  return __builtin_riscv_mltr_m_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mltr_m(const uint16_t * op0, size_t op1) {
  return __builtin_riscv_mltr_m_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mltr_m(const int32_t * op0, size_t op1) {
  return __builtin_riscv_mltr_m_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mltr_m(const uint32_t * op0, size_t op1) {
  return __builtin_riscv_mltr_m_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mltr_m(const int64_t * op0, size_t op1) {
  return __builtin_riscv_mltr_m_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mltr_m(const uint64_t * op0, size_t op1) {
  return __builtin_riscv_mltr_m_u64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mltr_m(const _Float16 * op0, size_t op1) {
  return __builtin_riscv_mltr_m_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mltr_m(const float * op0, size_t op1) {
  return __builtin_riscv_mltr_m_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mltr_m(const double * op0, size_t op1) {
  return __builtin_riscv_mltr_m_f64m1(op0, op1);
}
__rvm_overloaded mint8_t mlacc_m(const int8_t * op0, size_t op1) {
  return __builtin_riscv_mlacc_m_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mlacc_m(const uint8_t * op0, size_t op1) {
  return __builtin_riscv_mlacc_m_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mlacc_m(const int16_t * op0, size_t op1) {
  return __builtin_riscv_mlacc_m_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mlacc_m(const uint16_t * op0, size_t op1) {
  return __builtin_riscv_mlacc_m_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mlacc_m(const int32_t * op0, size_t op1) {
  return __builtin_riscv_mlacc_m_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mlacc_m(const uint32_t * op0, size_t op1) {
  return __builtin_riscv_mlacc_m_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mlacc_m(const int64_t * op0, size_t op1) {
  return __builtin_riscv_mlacc_m_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mlacc_m(const uint64_t * op0, size_t op1) {
  return __builtin_riscv_mlacc_m_u64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mlacc_m(const _Float16 * op0, size_t op1) {
  return __builtin_riscv_mlacc_m_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mlacc_m(const float * op0, size_t op1) {
  return __builtin_riscv_mlacc_m_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mlacc_m(const double * op0, size_t op1) {
  return __builtin_riscv_mlacc_m_f64m1(op0, op1);
}
__rvm_overloaded void msa_m(mint16_t op0, const int16_t * op1, size_t op2) {
  return __builtin_riscv_msa_m_i16m1(op0, op1, op2);
}
__rvm_overloaded void msa_m(muint16_t op0, const uint16_t * op1, size_t op2) {
  return __builtin_riscv_msa_m_u16m1(op0, op1, op2);
}
__rvm_overloaded void msa_m(mint32_t op0, const int32_t * op1, size_t op2) {
  return __builtin_riscv_msa_m_i32m1(op0, op1, op2);
}
__rvm_overloaded void msa_m(muint32_t op0, const uint32_t * op1, size_t op2) {
  return __builtin_riscv_msa_m_u32m1(op0, op1, op2);
}
__rvm_overloaded void msa_m(mint64_t op0, const int64_t * op1, size_t op2) {
  return __builtin_riscv_msa_m_i64m1(op0, op1, op2);
}
__rvm_overloaded void msa_m(muint64_t op0, const uint64_t * op1, size_t op2) {
  return __builtin_riscv_msa_m_u64m1(op0, op1, op2);
}
__rvm_overloaded void msa_m(mfloat16_t op0, const _Float16 * op1, size_t op2) {
  return __builtin_riscv_msa_m_f16m1(op0, op1, op2);
}
__rvm_overloaded void msa_m(mfloat32_t op0, const float * op1, size_t op2) {
  return __builtin_riscv_msa_m_f32m1(op0, op1, op2);
}
__rvm_overloaded void msa_m(mfloat64_t op0, const double * op1, size_t op2) {
  return __builtin_riscv_msa_m_f64m1(op0, op1, op2);
}
__rvm_overloaded void msb_m(mint8_t op0, const int8_t * op1, size_t op2) {
  return __builtin_riscv_msb_m_i8m1(op0, op1, op2);
}
__rvm_overloaded void msb_m(muint8_t op0, const uint8_t * op1, size_t op2) {
  return __builtin_riscv_msb_m_u8m1(op0, op1, op2);
}
__rvm_overloaded void msb_m(mint16_t op0, const int16_t * op1, size_t op2) {
  return __builtin_riscv_msb_m_i16m1(op0, op1, op2);
}
__rvm_overloaded void msb_m(muint16_t op0, const uint16_t * op1, size_t op2) {
  return __builtin_riscv_msb_m_u16m1(op0, op1, op2);
}
__rvm_overloaded void msb_m(mint32_t op0, const int32_t * op1, size_t op2) {
  return __builtin_riscv_msb_m_i32m1(op0, op1, op2);
}
__rvm_overloaded void msb_m(muint32_t op0, const uint32_t * op1, size_t op2) {
  return __builtin_riscv_msb_m_u32m1(op0, op1, op2);
}
__rvm_overloaded void msb_m(mint64_t op0, const int64_t * op1, size_t op2) {
  return __builtin_riscv_msb_m_i64m1(op0, op1, op2);
}
__rvm_overloaded void msb_m(muint64_t op0, const uint64_t * op1, size_t op2) {
  return __builtin_riscv_msb_m_u64m1(op0, op1, op2);
}
__rvm_overloaded void msb_m(mfloat16_t op0, const _Float16 * op1, size_t op2) {
  return __builtin_riscv_msb_m_f16m1(op0, op1, op2);
}
__rvm_overloaded void msb_m(mfloat32_t op0, const float * op1, size_t op2) {
  return __builtin_riscv_msb_m_f32m1(op0, op1, op2);
}
__rvm_overloaded void msb_m(mfloat64_t op0, const double * op1, size_t op2) {
  return __builtin_riscv_msb_m_f64m1(op0, op1, op2);
}
__rvm_overloaded void msc_m(mint8_t op0, const int8_t * op1, size_t op2) {
  return __builtin_riscv_msc_m_i8m1(op0, op1, op2);
}
__rvm_overloaded void msc_m(muint8_t op0, const uint8_t * op1, size_t op2) {
  return __builtin_riscv_msc_m_u8m1(op0, op1, op2);
}
__rvm_overloaded void msc_m(mint16_t op0, const int16_t * op1, size_t op2) {
  return __builtin_riscv_msc_m_i16m1(op0, op1, op2);
}
__rvm_overloaded void msc_m(muint16_t op0, const uint16_t * op1, size_t op2) {
  return __builtin_riscv_msc_m_u16m1(op0, op1, op2);
}
__rvm_overloaded void msc_m(mint32_t op0, const int32_t * op1, size_t op2) {
  return __builtin_riscv_msc_m_i32m1(op0, op1, op2);
}
__rvm_overloaded void msc_m(muint32_t op0, const uint32_t * op1, size_t op2) {
  return __builtin_riscv_msc_m_u32m1(op0, op1, op2);
}
__rvm_overloaded void msc_m(mint64_t op0, const int64_t * op1, size_t op2) {
  return __builtin_riscv_msc_m_i64m1(op0, op1, op2);
}
__rvm_overloaded void msc_m(muint64_t op0, const uint64_t * op1, size_t op2) {
  return __builtin_riscv_msc_m_u64m1(op0, op1, op2);
}
__rvm_overloaded void msc_m(mfloat16_t op0, const _Float16 * op1, size_t op2) {
  return __builtin_riscv_msc_m_f16m1(op0, op1, op2);
}
__rvm_overloaded void msc_m(mfloat32_t op0, const float * op1, size_t op2) {
  return __builtin_riscv_msc_m_f32m1(op0, op1, op2);
}
__rvm_overloaded void msc_m(mfloat64_t op0, const double * op1, size_t op2) {
  return __builtin_riscv_msc_m_f64m1(op0, op1, op2);
}
__rvm_overloaded void msat_m(mint8_t op0, const int8_t * op1, size_t op2) {
  return __builtin_riscv_msat_m_i8m1(op0, op1, op2);
}
__rvm_overloaded void msat_m(muint8_t op0, const uint8_t * op1, size_t op2) {
  return __builtin_riscv_msat_m_u8m1(op0, op1, op2);
}
__rvm_overloaded void msat_m(mint16_t op0, const int16_t * op1, size_t op2) {
  return __builtin_riscv_msat_m_i16m1(op0, op1, op2);
}
__rvm_overloaded void msat_m(muint16_t op0, const uint16_t * op1, size_t op2) {
  return __builtin_riscv_msat_m_u16m1(op0, op1, op2);
}
__rvm_overloaded void msat_m(mint32_t op0, const int32_t * op1, size_t op2) {
  return __builtin_riscv_msat_m_i32m1(op0, op1, op2);
}
__rvm_overloaded void msat_m(muint32_t op0, const uint32_t * op1, size_t op2) {
  return __builtin_riscv_msat_m_u32m1(op0, op1, op2);
}
__rvm_overloaded void msat_m(mint64_t op0, const int64_t * op1, size_t op2) {
  return __builtin_riscv_msat_m_i64m1(op0, op1, op2);
}
__rvm_overloaded void msat_m(muint64_t op0, const uint64_t * op1, size_t op2) {
  return __builtin_riscv_msat_m_u64m1(op0, op1, op2);
}
__rvm_overloaded void msat_m(mfloat16_t op0, const _Float16 * op1, size_t op2) {
  return __builtin_riscv_msat_m_f16m1(op0, op1, op2);
}
__rvm_overloaded void msat_m(mfloat32_t op0, const float * op1, size_t op2) {
  return __builtin_riscv_msat_m_f32m1(op0, op1, op2);
}
__rvm_overloaded void msat_m(mfloat64_t op0, const double * op1, size_t op2) {
  return __builtin_riscv_msat_m_f64m1(op0, op1, op2);
}
__rvm_overloaded void msbt_m(mint8_t op0, const int8_t * op1, size_t op2) {
  return __builtin_riscv_msbt_m_i8m1(op0, op1, op2);
}
__rvm_overloaded void msbt_m(muint8_t op0, const uint8_t * op1, size_t op2) {
  return __builtin_riscv_msbt_m_u8m1(op0, op1, op2);
}
__rvm_overloaded void msbt_m(mint16_t op0, const int16_t * op1, size_t op2) {
  return __builtin_riscv_msbt_m_i16m1(op0, op1, op2);
}
__rvm_overloaded void msbt_m(muint16_t op0, const uint16_t * op1, size_t op2) {
  return __builtin_riscv_msbt_m_u16m1(op0, op1, op2);
}
__rvm_overloaded void msbt_m(mint32_t op0, const int32_t * op1, size_t op2) {
  return __builtin_riscv_msbt_m_i32m1(op0, op1, op2);
}
__rvm_overloaded void msbt_m(muint32_t op0, const uint32_t * op1, size_t op2) {
  return __builtin_riscv_msbt_m_u32m1(op0, op1, op2);
}
__rvm_overloaded void msbt_m(mint64_t op0, const int64_t * op1, size_t op2) {
  return __builtin_riscv_msbt_m_i64m1(op0, op1, op2);
}
__rvm_overloaded void msbt_m(muint64_t op0, const uint64_t * op1, size_t op2) {
  return __builtin_riscv_msbt_m_u64m1(op0, op1, op2);
}
__rvm_overloaded void msbt_m(mfloat16_t op0, const _Float16 * op1, size_t op2) {
  return __builtin_riscv_msbt_m_f16m1(op0, op1, op2);
}
__rvm_overloaded void msbt_m(mfloat32_t op0, const float * op1, size_t op2) {
  return __builtin_riscv_msbt_m_f32m1(op0, op1, op2);
}
__rvm_overloaded void msbt_m(mfloat64_t op0, const double * op1, size_t op2) {
  return __builtin_riscv_msbt_m_f64m1(op0, op1, op2);
}
__rvm_overloaded void msct_m(mint8_t op0, const int8_t * op1, size_t op2) {
  return __builtin_riscv_msct_m_i8m1(op0, op1, op2);
}
__rvm_overloaded void msct_m(muint8_t op0, const uint8_t * op1, size_t op2) {
  return __builtin_riscv_msct_m_u8m1(op0, op1, op2);
}
__rvm_overloaded void msct_m(mint16_t op0, const int16_t * op1, size_t op2) {
  return __builtin_riscv_msct_m_i16m1(op0, op1, op2);
}
__rvm_overloaded void msct_m(muint16_t op0, const uint16_t * op1, size_t op2) {
  return __builtin_riscv_msct_m_u16m1(op0, op1, op2);
}
__rvm_overloaded void msct_m(mint32_t op0, const int32_t * op1, size_t op2) {
  return __builtin_riscv_msct_m_i32m1(op0, op1, op2);
}
__rvm_overloaded void msct_m(muint32_t op0, const uint32_t * op1, size_t op2) {
  return __builtin_riscv_msct_m_u32m1(op0, op1, op2);
}
__rvm_overloaded void msct_m(mint64_t op0, const int64_t * op1, size_t op2) {
  return __builtin_riscv_msct_m_i64m1(op0, op1, op2);
}
__rvm_overloaded void msct_m(muint64_t op0, const uint64_t * op1, size_t op2) {
  return __builtin_riscv_msct_m_u64m1(op0, op1, op2);
}
__rvm_overloaded void msct_m(mfloat16_t op0, const _Float16 * op1, size_t op2) {
  return __builtin_riscv_msct_m_f16m1(op0, op1, op2);
}
__rvm_overloaded void msct_m(mfloat32_t op0, const float * op1, size_t op2) {
  return __builtin_riscv_msct_m_f32m1(op0, op1, op2);
}
__rvm_overloaded void msct_m(mfloat64_t op0, const double * op1, size_t op2) {
  return __builtin_riscv_msct_m_f64m1(op0, op1, op2);
}
__rvm_overloaded void mstr_m(mint8_t op0, const int8_t * op1, size_t op2) {
  return __builtin_riscv_mstr_m_i8m1(op0, op1, op2);
}
__rvm_overloaded void mstr_m(muint8_t op0, const uint8_t * op1, size_t op2) {
  return __builtin_riscv_mstr_m_u8m1(op0, op1, op2);
}
__rvm_overloaded void mstr_m(mint16_t op0, const int16_t * op1, size_t op2) {
  return __builtin_riscv_mstr_m_i16m1(op0, op1, op2);
}
__rvm_overloaded void mstr_m(muint16_t op0, const uint16_t * op1, size_t op2) {
  return __builtin_riscv_mstr_m_u16m1(op0, op1, op2);
}
__rvm_overloaded void mstr_m(mint32_t op0, const int32_t * op1, size_t op2) {
  return __builtin_riscv_mstr_m_i32m1(op0, op1, op2);
}
__rvm_overloaded void mstr_m(muint32_t op0, const uint32_t * op1, size_t op2) {
  return __builtin_riscv_mstr_m_u32m1(op0, op1, op2);
}
__rvm_overloaded void mstr_m(mint64_t op0, const int64_t * op1, size_t op2) {
  return __builtin_riscv_mstr_m_i64m1(op0, op1, op2);
}
__rvm_overloaded void mstr_m(muint64_t op0, const uint64_t * op1, size_t op2) {
  return __builtin_riscv_mstr_m_u64m1(op0, op1, op2);
}
__rvm_overloaded void mstr_m(mfloat16_t op0, const _Float16 * op1, size_t op2) {
  return __builtin_riscv_mstr_m_f16m1(op0, op1, op2);
}
__rvm_overloaded void mstr_m(mfloat32_t op0, const float * op1, size_t op2) {
  return __builtin_riscv_mstr_m_f32m1(op0, op1, op2);
}
__rvm_overloaded void mstr_m(mfloat64_t op0, const double * op1, size_t op2) {
  return __builtin_riscv_mstr_m_f64m1(op0, op1, op2);
}
__rvm_overloaded void msacc_m(mint8_t op0, const int8_t * op1, size_t op2) {
  return __builtin_riscv_msacc_m_i8m1(op0, op1, op2);
}
__rvm_overloaded void msacc_m(muint8_t op0, const uint8_t * op1, size_t op2) {
  return __builtin_riscv_msacc_m_u8m1(op0, op1, op2);
}
__rvm_overloaded void msacc_m(mint16_t op0, const int16_t * op1, size_t op2) {
  return __builtin_riscv_msacc_m_i16m1(op0, op1, op2);
}
__rvm_overloaded void msacc_m(muint16_t op0, const uint16_t * op1, size_t op2) {
  return __builtin_riscv_msacc_m_u16m1(op0, op1, op2);
}
__rvm_overloaded void msacc_m(mint32_t op0, const int32_t * op1, size_t op2) {
  return __builtin_riscv_msacc_m_i32m1(op0, op1, op2);
}
__rvm_overloaded void msacc_m(muint32_t op0, const uint32_t * op1, size_t op2) {
  return __builtin_riscv_msacc_m_u32m1(op0, op1, op2);
}
__rvm_overloaded void msacc_m(mint64_t op0, const int64_t * op1, size_t op2) {
  return __builtin_riscv_msacc_m_i64m1(op0, op1, op2);
}
__rvm_overloaded void msacc_m(muint64_t op0, const uint64_t * op1, size_t op2) {
  return __builtin_riscv_msacc_m_u64m1(op0, op1, op2);
}
__rvm_overloaded void msacc_m(mfloat16_t op0, const _Float16 * op1, size_t op2) {
  return __builtin_riscv_msacc_m_f16m1(op0, op1, op2);
}
__rvm_overloaded void msacc_m(mfloat32_t op0, const float * op1, size_t op2) {
  return __builtin_riscv_msacc_m_f32m1(op0, op1, op2);
}
__rvm_overloaded void msacc_m(mfloat64_t op0, const double * op1, size_t op2) {
  return __builtin_riscv_msacc_m_f64m1(op0, op1, op2);
}
__rvm_overloaded mint8_t mmv_a_t(mint8_t op0, size_t op1) {
  return __builtin_riscv_mmv_a_t_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mmv_a_t(muint8_t op0, size_t op1) {
  return __builtin_riscv_mmv_a_t_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mmv_a_t(mint16_t op0, size_t op1) {
  return __builtin_riscv_mmv_a_t_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mmv_a_t(muint16_t op0, size_t op1) {
  return __builtin_riscv_mmv_a_t_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mmv_a_t(mint32_t op0, size_t op1) {
  return __builtin_riscv_mmv_a_t_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mmv_a_t(muint32_t op0, size_t op1) {
  return __builtin_riscv_mmv_a_t_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mmv_a_t(mint64_t op0, size_t op1) {
  return __builtin_riscv_mmv_a_t_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mmv_a_t(muint64_t op0, size_t op1) {
  return __builtin_riscv_mmv_a_t_u64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mmv_a_t(mfloat16_t op0, size_t op1) {
  return __builtin_riscv_mmv_a_t_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mmv_a_t(mfloat32_t op0, size_t op1) {
  return __builtin_riscv_mmv_a_t_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mmv_a_t(mfloat64_t op0, size_t op1) {
  return __builtin_riscv_mmv_a_t_f64m1(op0, op1);
}
__rvm_overloaded mint8_t mmv_t_a(mint8_t op0, size_t op1) {
  return __builtin_riscv_mmv_t_a_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mmv_t_a(muint8_t op0, size_t op1) {
  return __builtin_riscv_mmv_t_a_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mmv_t_a(mint16_t op0, size_t op1) {
  return __builtin_riscv_mmv_t_a_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mmv_t_a(muint16_t op0, size_t op1) {
  return __builtin_riscv_mmv_t_a_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mmv_t_a(mint32_t op0, size_t op1) {
  return __builtin_riscv_mmv_t_a_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mmv_t_a(muint32_t op0, size_t op1) {
  return __builtin_riscv_mmv_t_a_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mmv_t_a(mint64_t op0, size_t op1) {
  return __builtin_riscv_mmv_t_a_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mmv_t_a(muint64_t op0, size_t op1) {
  return __builtin_riscv_mmv_t_a_u64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mmv_t_a(mfloat16_t op0, size_t op1) {
  return __builtin_riscv_mmv_t_a_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mmv_t_a(mfloat32_t op0, size_t op1) {
  return __builtin_riscv_mmv_t_a_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mmv_t_a(mfloat64_t op0, size_t op1) {
  return __builtin_riscv_mmv_t_a_f64m1(op0, op1);
}
__rvm_overloaded uint8_t mmv_x_a(muint8_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_a_u8m1(op0, op1);
}
__rvm_overloaded uint16_t mmv_x_a(muint16_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_a_u16m1(op0, op1);
}
__rvm_overloaded uint32_t mmv_x_a(muint32_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_a_u32m1(op0, op1);
}
__rvm_overloaded uint64_t mmv_x_a(muint64_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_a_u64m1(op0, op1);
}
__rvm_overloaded mint8_t mmv_a_x(mint8_t op0, int8_t op1, size_t op2) {
  return __builtin_riscv_mmv_a_x_i8m1(op0, op1, op2);
}
__rvm_overloaded mint16_t mmv_a_x(mint16_t op0, int16_t op1, size_t op2) {
  return __builtin_riscv_mmv_a_x_i16m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mmv_a_x(mint32_t op0, int32_t op1, size_t op2) {
  return __builtin_riscv_mmv_a_x_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mmv_a_x(mint64_t op0, int64_t op1, size_t op2) {
  return __builtin_riscv_mmv_a_x_i64m1(op0, op1, op2);
}
__rvm_overloaded muint8_t mmv_a_x(muint8_t op0, uint8_t op1, size_t op2) {
  return __builtin_riscv_mmv_a_x_u8m1(op0, op1, op2);
}
__rvm_overloaded muint16_t mmv_a_x(muint16_t op0, uint16_t op1, size_t op2) {
  return __builtin_riscv_mmv_a_x_u16m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mmv_a_x(muint32_t op0, uint32_t op1, size_t op2) {
  return __builtin_riscv_mmv_a_x_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mmv_a_x(muint64_t op0, uint64_t op1, size_t op2) {
  return __builtin_riscv_mmv_a_x_u64m1(op0, op1, op2);
}
__rvm_overloaded _Float16 mfmv_f_a(mfloat16_t op0, size_t op1) {
  return __builtin_riscv_mfmv_f_a_f16m1(op0, op1);
}
__rvm_overloaded float mfmv_f_a(mfloat32_t op0, size_t op1) {
  return __builtin_riscv_mfmv_f_a_f32m1(op0, op1);
}
__rvm_overloaded double mfmv_f_a(mfloat64_t op0, size_t op1) {
  return __builtin_riscv_mfmv_f_a_f64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfmv_a_f(mfloat16_t op0, _Float16 op1, size_t op2) {
  return __builtin_riscv_mfmv_a_f_f16m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfmv_a_f(mfloat32_t op0, float op1, size_t op2) {
  return __builtin_riscv_mfmv_a_f_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfmv_a_f(mfloat64_t op0, double op1, size_t op2) {
  return __builtin_riscv_mfmv_a_f_f64m1(op0, op1, op2);
}
__rvm_overloaded int8_t mmv_x_t(mint8_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_t_i8m1(op0, op1);
}
__rvm_overloaded int16_t mmv_x_t(mint16_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_t_i16m1(op0, op1);
}
__rvm_overloaded int32_t mmv_x_t(mint32_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_t_i32m1(op0, op1);
}
__rvm_overloaded int64_t mmv_x_t(mint64_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_t_i64m1(op0, op1);
}
__rvm_overloaded uint8_t mmv_x_t(muint8_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_t_u8m1(op0, op1);
}
__rvm_overloaded uint16_t mmv_x_t(muint16_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_t_u16m1(op0, op1);
}
__rvm_overloaded uint32_t mmv_x_t(muint32_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_t_u32m1(op0, op1);
}
__rvm_overloaded uint64_t mmv_x_t(muint64_t op0, size_t op1) {
  return __builtin_riscv_mmv_x_t_u64m1(op0, op1);
}
__rvm_overloaded mint8_t mmv_t_x(mint8_t op0, int8_t op1, size_t op2) {
  return __builtin_riscv_mmv_t_x_i8m1(op0, op1, op2);
}
__rvm_overloaded mint16_t mmv_t_x(mint16_t op0, int16_t op1, size_t op2) {
  return __builtin_riscv_mmv_t_x_i16m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mmv_t_x(mint32_t op0, int32_t op1, size_t op2) {
  return __builtin_riscv_mmv_t_x_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mmv_t_x(mint64_t op0, int64_t op1, size_t op2) {
  return __builtin_riscv_mmv_t_x_i64m1(op0, op1, op2);
}
__rvm_overloaded muint8_t mmv_t_x(muint8_t op0, uint8_t op1, size_t op2) {
  return __builtin_riscv_mmv_t_x_u8m1(op0, op1, op2);
}
__rvm_overloaded muint16_t mmv_t_x(muint16_t op0, uint16_t op1, size_t op2) {
  return __builtin_riscv_mmv_t_x_u16m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mmv_t_x(muint32_t op0, uint32_t op1, size_t op2) {
  return __builtin_riscv_mmv_t_x_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mmv_t_x(muint64_t op0, uint64_t op1, size_t op2) {
  return __builtin_riscv_mmv_t_x_u64m1(op0, op1, op2);
}
__rvm_overloaded _Float16 mfmv_f_t(mfloat16_t op0, size_t op1) {
  return __builtin_riscv_mfmv_f_t_f16m1(op0, op1);
}
__rvm_overloaded float mfmv_f_t(mfloat32_t op0, size_t op1) {
  return __builtin_riscv_mfmv_f_t_f32m1(op0, op1);
}
__rvm_overloaded double mfmv_f_t(mfloat64_t op0, size_t op1) {
  return __builtin_riscv_mfmv_f_t_f64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfmv_t_f(mfloat16_t op0, _Float16 op1, size_t op2) {
  return __builtin_riscv_mfmv_t_f_f16m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfmv_t_f(mfloat32_t op0, float op1, size_t op2) {
  return __builtin_riscv_mfmv_t_f_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfmv_t_f(mfloat64_t op0, double op1, size_t op2) {
  return __builtin_riscv_mfmv_t_f_f64m1(op0, op1, op2);
}
__rvm_overloaded mint16_t mbcar_m(mint16_t op0) {
  return __builtin_riscv_mbcar_m_i16m1(op0);
}
__rvm_overloaded muint16_t mbcar_m(muint16_t op0) {
  return __builtin_riscv_mbcar_m_u16m1(op0);
}
__rvm_overloaded mint32_t mbcar_m(mint32_t op0) {
  return __builtin_riscv_mbcar_m_i32m1(op0);
}
__rvm_overloaded muint32_t mbcar_m(muint32_t op0) {
  return __builtin_riscv_mbcar_m_u32m1(op0);
}
__rvm_overloaded mint64_t mbcar_m(mint64_t op0) {
  return __builtin_riscv_mbcar_m_i64m1(op0);
}
__rvm_overloaded muint64_t mbcar_m(muint64_t op0) {
  return __builtin_riscv_mbcar_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mbcar_m(mfloat16_t op0) {
  return __builtin_riscv_mbcar_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mbcar_m(mfloat32_t op0) {
  return __builtin_riscv_mbcar_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mbcar_m(mfloat64_t op0) {
  return __builtin_riscv_mbcar_m_f64m1(op0);
}
__rvm_overloaded mint8_t mbcbr_m(mint8_t op0) {
  return __builtin_riscv_mbcbr_m_i8m1(op0);
}
__rvm_overloaded muint8_t mbcbr_m(muint8_t op0) {
  return __builtin_riscv_mbcbr_m_u8m1(op0);
}
__rvm_overloaded mint16_t mbcbr_m(mint16_t op0) {
  return __builtin_riscv_mbcbr_m_i16m1(op0);
}
__rvm_overloaded muint16_t mbcbr_m(muint16_t op0) {
  return __builtin_riscv_mbcbr_m_u16m1(op0);
}
__rvm_overloaded mint32_t mbcbr_m(mint32_t op0) {
  return __builtin_riscv_mbcbr_m_i32m1(op0);
}
__rvm_overloaded muint32_t mbcbr_m(muint32_t op0) {
  return __builtin_riscv_mbcbr_m_u32m1(op0);
}
__rvm_overloaded mint64_t mbcbr_m(mint64_t op0) {
  return __builtin_riscv_mbcbr_m_i64m1(op0);
}
__rvm_overloaded muint64_t mbcbr_m(muint64_t op0) {
  return __builtin_riscv_mbcbr_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mbcbr_m(mfloat16_t op0) {
  return __builtin_riscv_mbcbr_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mbcbr_m(mfloat32_t op0) {
  return __builtin_riscv_mbcbr_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mbcbr_m(mfloat64_t op0) {
  return __builtin_riscv_mbcbr_m_f64m1(op0);
}
__rvm_overloaded mint8_t mbccr_m(mint8_t op0) {
  return __builtin_riscv_mbccr_m_i8m1(op0);
}
__rvm_overloaded muint8_t mbccr_m(muint8_t op0) {
  return __builtin_riscv_mbccr_m_u8m1(op0);
}
__rvm_overloaded mint16_t mbccr_m(mint16_t op0) {
  return __builtin_riscv_mbccr_m_i16m1(op0);
}
__rvm_overloaded muint16_t mbccr_m(muint16_t op0) {
  return __builtin_riscv_mbccr_m_u16m1(op0);
}
__rvm_overloaded mint32_t mbccr_m(mint32_t op0) {
  return __builtin_riscv_mbccr_m_i32m1(op0);
}
__rvm_overloaded muint32_t mbccr_m(muint32_t op0) {
  return __builtin_riscv_mbccr_m_u32m1(op0);
}
__rvm_overloaded mint64_t mbccr_m(mint64_t op0) {
  return __builtin_riscv_mbccr_m_i64m1(op0);
}
__rvm_overloaded muint64_t mbccr_m(muint64_t op0) {
  return __builtin_riscv_mbccr_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mbccr_m(mfloat16_t op0) {
  return __builtin_riscv_mbccr_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mbccr_m(mfloat32_t op0) {
  return __builtin_riscv_mbccr_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mbccr_m(mfloat64_t op0) {
  return __builtin_riscv_mbccr_m_f64m1(op0);
}
__rvm_overloaded mint8_t mbcac_m(mint8_t op0) {
  return __builtin_riscv_mbcac_m_i8m1(op0);
}
__rvm_overloaded muint8_t mbcac_m(muint8_t op0) {
  return __builtin_riscv_mbcac_m_u8m1(op0);
}
__rvm_overloaded mint16_t mbcac_m(mint16_t op0) {
  return __builtin_riscv_mbcac_m_i16m1(op0);
}
__rvm_overloaded muint16_t mbcac_m(muint16_t op0) {
  return __builtin_riscv_mbcac_m_u16m1(op0);
}
__rvm_overloaded mint32_t mbcac_m(mint32_t op0) {
  return __builtin_riscv_mbcac_m_i32m1(op0);
}
__rvm_overloaded muint32_t mbcac_m(muint32_t op0) {
  return __builtin_riscv_mbcac_m_u32m1(op0);
}
__rvm_overloaded mint64_t mbcac_m(mint64_t op0) {
  return __builtin_riscv_mbcac_m_i64m1(op0);
}
__rvm_overloaded muint64_t mbcac_m(muint64_t op0) {
  return __builtin_riscv_mbcac_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mbcac_m(mfloat16_t op0) {
  return __builtin_riscv_mbcac_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mbcac_m(mfloat32_t op0) {
  return __builtin_riscv_mbcac_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mbcac_m(mfloat64_t op0) {
  return __builtin_riscv_mbcac_m_f64m1(op0);
}
__rvm_overloaded mint8_t mbcbc_m(mint8_t op0) {
  return __builtin_riscv_mbcbc_m_i8m1(op0);
}
__rvm_overloaded muint8_t mbcbc_m(muint8_t op0) {
  return __builtin_riscv_mbcbc_m_u8m1(op0);
}
__rvm_overloaded mint16_t mbcbc_m(mint16_t op0) {
  return __builtin_riscv_mbcbc_m_i16m1(op0);
}
__rvm_overloaded muint16_t mbcbc_m(muint16_t op0) {
  return __builtin_riscv_mbcbc_m_u16m1(op0);
}
__rvm_overloaded mint32_t mbcbc_m(mint32_t op0) {
  return __builtin_riscv_mbcbc_m_i32m1(op0);
}
__rvm_overloaded muint32_t mbcbc_m(muint32_t op0) {
  return __builtin_riscv_mbcbc_m_u32m1(op0);
}
__rvm_overloaded mint64_t mbcbc_m(mint64_t op0) {
  return __builtin_riscv_mbcbc_m_i64m1(op0);
}
__rvm_overloaded muint64_t mbcbc_m(muint64_t op0) {
  return __builtin_riscv_mbcbc_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mbcbc_m(mfloat16_t op0) {
  return __builtin_riscv_mbcbc_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mbcbc_m(mfloat32_t op0) {
  return __builtin_riscv_mbcbc_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mbcbc_m(mfloat64_t op0) {
  return __builtin_riscv_mbcbc_m_f64m1(op0);
}
__rvm_overloaded mint8_t mbccc_m(mint8_t op0) {
  return __builtin_riscv_mbccc_m_i8m1(op0);
}
__rvm_overloaded muint8_t mbccc_m(muint8_t op0) {
  return __builtin_riscv_mbccc_m_u8m1(op0);
}
__rvm_overloaded mint16_t mbccc_m(mint16_t op0) {
  return __builtin_riscv_mbccc_m_i16m1(op0);
}
__rvm_overloaded muint16_t mbccc_m(muint16_t op0) {
  return __builtin_riscv_mbccc_m_u16m1(op0);
}
__rvm_overloaded mint32_t mbccc_m(mint32_t op0) {
  return __builtin_riscv_mbccc_m_i32m1(op0);
}
__rvm_overloaded muint32_t mbccc_m(muint32_t op0) {
  return __builtin_riscv_mbccc_m_u32m1(op0);
}
__rvm_overloaded mint64_t mbccc_m(mint64_t op0) {
  return __builtin_riscv_mbccc_m_i64m1(op0);
}
__rvm_overloaded muint64_t mbccc_m(muint64_t op0) {
  return __builtin_riscv_mbccc_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mbccc_m(mfloat16_t op0) {
  return __builtin_riscv_mbccc_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mbccc_m(mfloat32_t op0) {
  return __builtin_riscv_mbccc_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mbccc_m(mfloat64_t op0) {
  return __builtin_riscv_mbccc_m_f64m1(op0);
}
__rvm_overloaded mint8_t mbcae_m(mint8_t op0) {
  return __builtin_riscv_mbcae_m_i8m1(op0);
}
__rvm_overloaded muint8_t mbcae_m(muint8_t op0) {
  return __builtin_riscv_mbcae_m_u8m1(op0);
}
__rvm_overloaded mint16_t mbcae_m(mint16_t op0) {
  return __builtin_riscv_mbcae_m_i16m1(op0);
}
__rvm_overloaded muint16_t mbcae_m(muint16_t op0) {
  return __builtin_riscv_mbcae_m_u16m1(op0);
}
__rvm_overloaded mint32_t mbcae_m(mint32_t op0) {
  return __builtin_riscv_mbcae_m_i32m1(op0);
}
__rvm_overloaded muint32_t mbcae_m(muint32_t op0) {
  return __builtin_riscv_mbcae_m_u32m1(op0);
}
__rvm_overloaded mint64_t mbcae_m(mint64_t op0) {
  return __builtin_riscv_mbcae_m_i64m1(op0);
}
__rvm_overloaded muint64_t mbcae_m(muint64_t op0) {
  return __builtin_riscv_mbcae_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mbcae_m(mfloat16_t op0) {
  return __builtin_riscv_mbcae_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mbcae_m(mfloat32_t op0) {
  return __builtin_riscv_mbcae_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mbcae_m(mfloat64_t op0) {
  return __builtin_riscv_mbcae_m_f64m1(op0);
}
__rvm_overloaded mint8_t mbcbe_m(mint8_t op0) {
  return __builtin_riscv_mbcbe_m_i8m1(op0);
}
__rvm_overloaded muint8_t mbcbe_m(muint8_t op0) {
  return __builtin_riscv_mbcbe_m_u8m1(op0);
}
__rvm_overloaded mint16_t mbcbe_m(mint16_t op0) {
  return __builtin_riscv_mbcbe_m_i16m1(op0);
}
__rvm_overloaded muint16_t mbcbe_m(muint16_t op0) {
  return __builtin_riscv_mbcbe_m_u16m1(op0);
}
__rvm_overloaded mint32_t mbcbe_m(mint32_t op0) {
  return __builtin_riscv_mbcbe_m_i32m1(op0);
}
__rvm_overloaded muint32_t mbcbe_m(muint32_t op0) {
  return __builtin_riscv_mbcbe_m_u32m1(op0);
}
__rvm_overloaded mint64_t mbcbe_m(mint64_t op0) {
  return __builtin_riscv_mbcbe_m_i64m1(op0);
}
__rvm_overloaded muint64_t mbcbe_m(muint64_t op0) {
  return __builtin_riscv_mbcbe_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mbcbe_m(mfloat16_t op0) {
  return __builtin_riscv_mbcbe_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mbcbe_m(mfloat32_t op0) {
  return __builtin_riscv_mbcbe_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mbcbe_m(mfloat64_t op0) {
  return __builtin_riscv_mbcbe_m_f64m1(op0);
}
__rvm_overloaded mint8_t mbcce_m(mint8_t op0) {
  return __builtin_riscv_mbcce_m_i8m1(op0);
}
__rvm_overloaded muint8_t mbcce_m(muint8_t op0) {
  return __builtin_riscv_mbcce_m_u8m1(op0);
}
__rvm_overloaded mint16_t mbcce_m(mint16_t op0) {
  return __builtin_riscv_mbcce_m_i16m1(op0);
}
__rvm_overloaded muint16_t mbcce_m(muint16_t op0) {
  return __builtin_riscv_mbcce_m_u16m1(op0);
}
__rvm_overloaded mint32_t mbcce_m(mint32_t op0) {
  return __builtin_riscv_mbcce_m_i32m1(op0);
}
__rvm_overloaded muint32_t mbcce_m(muint32_t op0) {
  return __builtin_riscv_mbcce_m_u32m1(op0);
}
__rvm_overloaded mint64_t mbcce_m(mint64_t op0) {
  return __builtin_riscv_mbcce_m_i64m1(op0);
}
__rvm_overloaded muint64_t mbcce_m(muint64_t op0) {
  return __builtin_riscv_mbcce_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mbcce_m(mfloat16_t op0) {
  return __builtin_riscv_mbcce_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mbcce_m(mfloat32_t op0) {
  return __builtin_riscv_mbcce_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mbcce_m(mfloat64_t op0) {
  return __builtin_riscv_mbcce_m_f64m1(op0);
}
__rvm_overloaded mint8_t mta_m(mint8_t op0) {
  return __builtin_riscv_mta_m_i8m1(op0);
}
__rvm_overloaded muint8_t mta_m(muint8_t op0) {
  return __builtin_riscv_mta_m_u8m1(op0);
}
__rvm_overloaded mint16_t mta_m(mint16_t op0) {
  return __builtin_riscv_mta_m_i16m1(op0);
}
__rvm_overloaded muint16_t mta_m(muint16_t op0) {
  return __builtin_riscv_mta_m_u16m1(op0);
}
__rvm_overloaded mint32_t mta_m(mint32_t op0) {
  return __builtin_riscv_mta_m_i32m1(op0);
}
__rvm_overloaded muint32_t mta_m(muint32_t op0) {
  return __builtin_riscv_mta_m_u32m1(op0);
}
__rvm_overloaded mint64_t mta_m(mint64_t op0) {
  return __builtin_riscv_mta_m_i64m1(op0);
}
__rvm_overloaded muint64_t mta_m(muint64_t op0) {
  return __builtin_riscv_mta_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mta_m(mfloat16_t op0) {
  return __builtin_riscv_mta_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mta_m(mfloat32_t op0) {
  return __builtin_riscv_mta_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mta_m(mfloat64_t op0) {
  return __builtin_riscv_mta_m_f64m1(op0);
}
__rvm_overloaded mint8_t mtb_m(mint8_t op0) {
  return __builtin_riscv_mtb_m_i8m1(op0);
}
__rvm_overloaded muint8_t mtb_m(muint8_t op0) {
  return __builtin_riscv_mtb_m_u8m1(op0);
}
__rvm_overloaded mint16_t mtb_m(mint16_t op0) {
  return __builtin_riscv_mtb_m_i16m1(op0);
}
__rvm_overloaded muint16_t mtb_m(muint16_t op0) {
  return __builtin_riscv_mtb_m_u16m1(op0);
}
__rvm_overloaded mint32_t mtb_m(mint32_t op0) {
  return __builtin_riscv_mtb_m_i32m1(op0);
}
__rvm_overloaded muint32_t mtb_m(muint32_t op0) {
  return __builtin_riscv_mtb_m_u32m1(op0);
}
__rvm_overloaded mint64_t mtb_m(mint64_t op0) {
  return __builtin_riscv_mtb_m_i64m1(op0);
}
__rvm_overloaded muint64_t mtb_m(muint64_t op0) {
  return __builtin_riscv_mtb_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mtb_m(mfloat16_t op0) {
  return __builtin_riscv_mtb_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mtb_m(mfloat32_t op0) {
  return __builtin_riscv_mtb_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mtb_m(mfloat64_t op0) {
  return __builtin_riscv_mtb_m_f64m1(op0);
}
__rvm_overloaded mint8_t mtc_m(mint8_t op0) {
  return __builtin_riscv_mtc_m_i8m1(op0);
}
__rvm_overloaded muint8_t mtc_m(muint8_t op0) {
  return __builtin_riscv_mtc_m_u8m1(op0);
}
__rvm_overloaded mint16_t mtc_m(mint16_t op0) {
  return __builtin_riscv_mtc_m_i16m1(op0);
}
__rvm_overloaded muint16_t mtc_m(muint16_t op0) {
  return __builtin_riscv_mtc_m_u16m1(op0);
}
__rvm_overloaded mint32_t mtc_m(mint32_t op0) {
  return __builtin_riscv_mtc_m_i32m1(op0);
}
__rvm_overloaded muint32_t mtc_m(muint32_t op0) {
  return __builtin_riscv_mtc_m_u32m1(op0);
}
__rvm_overloaded mint64_t mtc_m(mint64_t op0) {
  return __builtin_riscv_mtc_m_i64m1(op0);
}
__rvm_overloaded muint64_t mtc_m(muint64_t op0) {
  return __builtin_riscv_mtc_m_u64m1(op0);
}
__rvm_overloaded mfloat16_t mtc_m(mfloat16_t op0) {
  return __builtin_riscv_mtc_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mtc_m(mfloat32_t op0) {
  return __builtin_riscv_mtc_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mtc_m(mfloat64_t op0) {
  return __builtin_riscv_mtc_m_f64m1(op0);
}
__rvm_overloaded mint16_t mand_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mand_mm_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mand_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mand_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mand_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mand_mm_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mand_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mand_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mand_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mand_mm_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mand_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_mand_mm_u64m1(op0, op1);
}
__rvm_overloaded mint8_t mor_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mor_mm_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mor_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mor_mm_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mor_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mor_mm_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mor_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mor_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mor_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mor_mm_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mor_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mor_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mor_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mor_mm_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mor_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_mor_mm_u64m1(op0, op1);
}
__rvm_overloaded mint8_t mxor_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mxor_mm_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mxor_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mxor_mm_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mxor_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mxor_mm_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mxor_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mxor_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mxor_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mxor_mm_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mxor_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mxor_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mxor_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mxor_mm_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mxor_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_mxor_mm_u64m1(op0, op1);
}
__rvm_overloaded mint8_t msll_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msll_mm_i8m1(op0, op1);
}
__rvm_overloaded muint8_t msll_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msll_mm_u8m1(op0, op1);
}
__rvm_overloaded mint16_t msll_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msll_mm_i16m1(op0, op1);
}
__rvm_overloaded muint16_t msll_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msll_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t msll_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msll_mm_i32m1(op0, op1);
}
__rvm_overloaded muint32_t msll_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msll_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t msll_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msll_mm_i64m1(op0, op1);
}
__rvm_overloaded muint64_t msll_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msll_mm_u64m1(op0, op1);
}
__rvm_overloaded mint8_t msll_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msll_b_mm_i8m1(op0, op1);
}
__rvm_overloaded muint8_t msll_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msll_b_mm_u8m1(op0, op1);
}
__rvm_overloaded mint16_t msll_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msll_h_mm_i16m1(op0, op1);
}
__rvm_overloaded muint16_t msll_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msll_h_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t msll_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msll_w_mm_i32m1(op0, op1);
}
__rvm_overloaded muint32_t msll_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msll_w_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t msll_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msll_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded muint64_t msll_dw_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msll_dw_mm_u64m1(op0, op1);
}
__rvm_overloaded mint8_t msrl_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msrl_mm_i8m1(op0, op1);
}
__rvm_overloaded muint8_t msrl_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msrl_mm_u8m1(op0, op1);
}
__rvm_overloaded mint16_t msrl_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msrl_mm_i16m1(op0, op1);
}
__rvm_overloaded muint16_t msrl_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msrl_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t msrl_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msrl_mm_i32m1(op0, op1);
}
__rvm_overloaded muint32_t msrl_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msrl_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t msrl_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msrl_mm_i64m1(op0, op1);
}
__rvm_overloaded muint64_t msrl_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msrl_mm_u64m1(op0, op1);
}
__rvm_overloaded mint8_t msrl_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msrl_b_mm_i8m1(op0, op1);
}
__rvm_overloaded muint8_t msrl_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msrl_b_mm_u8m1(op0, op1);
}
__rvm_overloaded mint16_t msrl_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msrl_h_mm_i16m1(op0, op1);
}
__rvm_overloaded muint16_t msrl_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msrl_h_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t msrl_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msrl_w_mm_i32m1(op0, op1);
}
__rvm_overloaded muint32_t msrl_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msrl_w_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t msrl_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msrl_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded muint64_t msrl_dw_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msrl_dw_mm_u64m1(op0, op1);
}
__rvm_overloaded mint8_t msra_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msra_mm_i8m1(op0, op1);
}
__rvm_overloaded muint8_t msra_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msra_mm_u8m1(op0, op1);
}
__rvm_overloaded mint16_t msra_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msra_mm_i16m1(op0, op1);
}
__rvm_overloaded muint16_t msra_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msra_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t msra_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msra_mm_i32m1(op0, op1);
}
__rvm_overloaded muint32_t msra_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msra_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t msra_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msra_mm_i64m1(op0, op1);
}
__rvm_overloaded muint64_t msra_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msra_mm_u64m1(op0, op1);
}
__rvm_overloaded mint8_t msra_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msra_b_mm_i8m1(op0, op1);
}
__rvm_overloaded muint8_t msra_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msra_b_mm_u8m1(op0, op1);
}
__rvm_overloaded mint16_t msra_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msra_h_mm_i16m1(op0, op1);
}
__rvm_overloaded muint16_t msra_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msra_h_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t msra_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msra_w_mm_i32m1(op0, op1);
}
__rvm_overloaded muint32_t msra_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msra_w_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t msra_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msra_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded muint64_t msra_dw_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msra_dw_mm_u64m1(op0, op1);
}
__rvm_overloaded vint8m1_t mla_v(const int8_t * op0, size_t op1) {
  return __builtin_riscv_mla_v_i8m1(op0, op1);
}
__rvm_overloaded vuint8m1_t mla_v(const uint8_t * op0, size_t op1) {
  return __builtin_riscv_mla_v_u8m1(op0, op1);
}
__rvm_overloaded vint16m1_t mla_v(const int16_t * op0, size_t op1) {
  return __builtin_riscv_mla_v_i16m1(op0, op1);
}
__rvm_overloaded vuint16m1_t mla_v(const uint16_t * op0, size_t op1) {
  return __builtin_riscv_mla_v_u16m1(op0, op1);
}
__rvm_overloaded vint32m1_t mla_v(const int32_t * op0, size_t op1) {
  return __builtin_riscv_mla_v_i32m1(op0, op1);
}
__rvm_overloaded vuint32m1_t mla_v(const uint32_t * op0, size_t op1) {
  return __builtin_riscv_mla_v_u32m1(op0, op1);
}
__rvm_overloaded vint64m1_t mla_v(const int64_t * op0, size_t op1) {
  return __builtin_riscv_mla_v_i64m1(op0, op1);
}
__rvm_overloaded vuint64m1_t mla_v(const uint64_t * op0, size_t op1) {
  return __builtin_riscv_mla_v_u64m1(op0, op1);
}
__rvm_overloaded vfloat16m1_t mla_v(const _Float16 * op0, size_t op1) {
  return __builtin_riscv_mla_v_f16m1(op0, op1);
}
__rvm_overloaded vfloat32m1_t mla_v(const float * op0, size_t op1) {
  return __builtin_riscv_mla_v_f32m1(op0, op1);
}
__rvm_overloaded vfloat64m1_t mla_v(const double * op0, size_t op1) {
  return __builtin_riscv_mla_v_f64m1(op0, op1);
}
__rvm_overloaded vint8m1_t mlb_v(const int8_t * op0, size_t op1) {
  return __builtin_riscv_mlb_v_i8m1(op0, op1);
}
__rvm_overloaded vuint8m1_t mlb_v(const uint8_t * op0, size_t op1) {
  return __builtin_riscv_mlb_v_u8m1(op0, op1);
}
__rvm_overloaded vint16m1_t mlb_v(const int16_t * op0, size_t op1) {
  return __builtin_riscv_mlb_v_i16m1(op0, op1);
}
__rvm_overloaded vuint16m1_t mlb_v(const uint16_t * op0, size_t op1) {
  return __builtin_riscv_mlb_v_u16m1(op0, op1);
}
__rvm_overloaded vint32m1_t mlb_v(const int32_t * op0, size_t op1) {
  return __builtin_riscv_mlb_v_i32m1(op0, op1);
}
__rvm_overloaded vuint32m1_t mlb_v(const uint32_t * op0, size_t op1) {
  return __builtin_riscv_mlb_v_u32m1(op0, op1);
}
__rvm_overloaded vint64m1_t mlb_v(const int64_t * op0, size_t op1) {
  return __builtin_riscv_mlb_v_i64m1(op0, op1);
}
__rvm_overloaded vuint64m1_t mlb_v(const uint64_t * op0, size_t op1) {
  return __builtin_riscv_mlb_v_u64m1(op0, op1);
}
__rvm_overloaded vfloat16m1_t mlb_v(const _Float16 * op0, size_t op1) {
  return __builtin_riscv_mlb_v_f16m1(op0, op1);
}
__rvm_overloaded vfloat32m1_t mlb_v(const float * op0, size_t op1) {
  return __builtin_riscv_mlb_v_f32m1(op0, op1);
}
__rvm_overloaded vfloat64m1_t mlb_v(const double * op0, size_t op1) {
  return __builtin_riscv_mlb_v_f64m1(op0, op1);
}
__rvm_overloaded vint8m1_t mlc_v(const int8_t * op0, size_t op1) {
  return __builtin_riscv_mlc_v_i8m1(op0, op1);
}
__rvm_overloaded vuint8m1_t mlc_v(const uint8_t * op0, size_t op1) {
  return __builtin_riscv_mlc_v_u8m1(op0, op1);
}
__rvm_overloaded vint16m1_t mlc_v(const int16_t * op0, size_t op1) {
  return __builtin_riscv_mlc_v_i16m1(op0, op1);
}
__rvm_overloaded vuint16m1_t mlc_v(const uint16_t * op0, size_t op1) {
  return __builtin_riscv_mlc_v_u16m1(op0, op1);
}
__rvm_overloaded vint32m1_t mlc_v(const int32_t * op0, size_t op1) {
  return __builtin_riscv_mlc_v_i32m1(op0, op1);
}
__rvm_overloaded vuint32m1_t mlc_v(const uint32_t * op0, size_t op1) {
  return __builtin_riscv_mlc_v_u32m1(op0, op1);
}
__rvm_overloaded vint64m1_t mlc_v(const int64_t * op0, size_t op1) {
  return __builtin_riscv_mlc_v_i64m1(op0, op1);
}
__rvm_overloaded vuint64m1_t mlc_v(const uint64_t * op0, size_t op1) {
  return __builtin_riscv_mlc_v_u64m1(op0, op1);
}
__rvm_overloaded vfloat16m1_t mlc_v(const _Float16 * op0, size_t op1) {
  return __builtin_riscv_mlc_v_f16m1(op0, op1);
}
__rvm_overloaded vfloat32m1_t mlc_v(const float * op0, size_t op1) {
  return __builtin_riscv_mlc_v_f32m1(op0, op1);
}
__rvm_overloaded vfloat64m1_t mlc_v(const double * op0, size_t op1) {
  return __builtin_riscv_mlc_v_f64m1(op0, op1);
}
__rvm_overloaded void msa_v(vint8m1_t op0, const int8_t * op1, size_t op2) {
  return __builtin_riscv_msa_v_i8m1(op0, op1, op2);
}
__rvm_overloaded void msa_v(vuint8m1_t op0, const uint8_t * op1, size_t op2) {
  return __builtin_riscv_msa_v_u8m1(op0, op1, op2);
}
__rvm_overloaded void msa_v(vint16m1_t op0, const int16_t * op1, size_t op2) {
  return __builtin_riscv_msa_v_i16m1(op0, op1, op2);
}
__rvm_overloaded void msa_v(vuint16m1_t op0, const uint16_t * op1, size_t op2) {
  return __builtin_riscv_msa_v_u16m1(op0, op1, op2);
}
__rvm_overloaded void msa_v(vint32m1_t op0, const int32_t * op1, size_t op2) {
  return __builtin_riscv_msa_v_i32m1(op0, op1, op2);
}
__rvm_overloaded void msa_v(vuint32m1_t op0, const uint32_t * op1, size_t op2) {
  return __builtin_riscv_msa_v_u32m1(op0, op1, op2);
}
__rvm_overloaded void msa_v(vint64m1_t op0, const int64_t * op1, size_t op2) {
  return __builtin_riscv_msa_v_i64m1(op0, op1, op2);
}
__rvm_overloaded void msa_v(vuint64m1_t op0, const uint64_t * op1, size_t op2) {
  return __builtin_riscv_msa_v_u64m1(op0, op1, op2);
}
__rvm_overloaded void msa_v(vfloat16m1_t op0, const _Float16 * op1, size_t op2) {
  return __builtin_riscv_msa_v_f16m1(op0, op1, op2);
}
__rvm_overloaded void msa_v(vfloat32m1_t op0, const float * op1, size_t op2) {
  return __builtin_riscv_msa_v_f32m1(op0, op1, op2);
}
__rvm_overloaded void msa_v(vfloat64m1_t op0, const double * op1, size_t op2) {
  return __builtin_riscv_msa_v_f64m1(op0, op1, op2);
}
__rvm_overloaded void msb_v(vint8m1_t op0, const int8_t * op1, size_t op2) {
  return __builtin_riscv_msb_v_i8m1(op0, op1, op2);
}
__rvm_overloaded void msb_v(vuint8m1_t op0, const uint8_t * op1, size_t op2) {
  return __builtin_riscv_msb_v_u8m1(op0, op1, op2);
}
__rvm_overloaded void msb_v(vint16m1_t op0, const int16_t * op1, size_t op2) {
  return __builtin_riscv_msb_v_i16m1(op0, op1, op2);
}
__rvm_overloaded void msb_v(vuint16m1_t op0, const uint16_t * op1, size_t op2) {
  return __builtin_riscv_msb_v_u16m1(op0, op1, op2);
}
__rvm_overloaded void msb_v(vint32m1_t op0, const int32_t * op1, size_t op2) {
  return __builtin_riscv_msb_v_i32m1(op0, op1, op2);
}
__rvm_overloaded void msb_v(vuint32m1_t op0, const uint32_t * op1, size_t op2) {
  return __builtin_riscv_msb_v_u32m1(op0, op1, op2);
}
__rvm_overloaded void msb_v(vint64m1_t op0, const int64_t * op1, size_t op2) {
  return __builtin_riscv_msb_v_i64m1(op0, op1, op2);
}
__rvm_overloaded void msb_v(vuint64m1_t op0, const uint64_t * op1, size_t op2) {
  return __builtin_riscv_msb_v_u64m1(op0, op1, op2);
}
__rvm_overloaded void msb_v(vfloat16m1_t op0, const _Float16 * op1, size_t op2) {
  return __builtin_riscv_msb_v_f16m1(op0, op1, op2);
}
__rvm_overloaded void msb_v(vfloat32m1_t op0, const float * op1, size_t op2) {
  return __builtin_riscv_msb_v_f32m1(op0, op1, op2);
}
__rvm_overloaded void msb_v(vfloat64m1_t op0, const double * op1, size_t op2) {
  return __builtin_riscv_msb_v_f64m1(op0, op1, op2);
}
__rvm_overloaded void msc_v(vint8m1_t op0, const int8_t * op1, size_t op2) {
  return __builtin_riscv_msc_v_i8m1(op0, op1, op2);
}
__rvm_overloaded void msc_v(vuint8m1_t op0, const uint8_t * op1, size_t op2) {
  return __builtin_riscv_msc_v_u8m1(op0, op1, op2);
}
__rvm_overloaded void msc_v(vint16m1_t op0, const int16_t * op1, size_t op2) {
  return __builtin_riscv_msc_v_i16m1(op0, op1, op2);
}
__rvm_overloaded void msc_v(vuint16m1_t op0, const uint16_t * op1, size_t op2) {
  return __builtin_riscv_msc_v_u16m1(op0, op1, op2);
}
__rvm_overloaded void msc_v(vint32m1_t op0, const int32_t * op1, size_t op2) {
  return __builtin_riscv_msc_v_i32m1(op0, op1, op2);
}
__rvm_overloaded void msc_v(vuint32m1_t op0, const uint32_t * op1, size_t op2) {
  return __builtin_riscv_msc_v_u32m1(op0, op1, op2);
}
__rvm_overloaded void msc_v(vint64m1_t op0, const int64_t * op1, size_t op2) {
  return __builtin_riscv_msc_v_i64m1(op0, op1, op2);
}
__rvm_overloaded void msc_v(vuint64m1_t op0, const uint64_t * op1, size_t op2) {
  return __builtin_riscv_msc_v_u64m1(op0, op1, op2);
}
__rvm_overloaded void msc_v(vfloat16m1_t op0, const _Float16 * op1, size_t op2) {
  return __builtin_riscv_msc_v_f16m1(op0, op1, op2);
}
__rvm_overloaded void msc_v(vfloat32m1_t op0, const float * op1, size_t op2) {
  return __builtin_riscv_msc_v_f32m1(op0, op1, op2);
}
__rvm_overloaded void msc_v(vfloat64m1_t op0, const double * op1, size_t op2) {
  return __builtin_riscv_msc_v_f64m1(op0, op1, op2);
}
__rvm_overloaded vint8m1_t mmvar_v_m(mint8_t op0, size_t op1) {
  return __builtin_riscv_mmvar_v_m_i8m1(op0, op1);
}
__rvm_overloaded vuint8m1_t mmvar_v_m(muint8_t op0, size_t op1) {
  return __builtin_riscv_mmvar_v_m_u8m1(op0, op1);
}
__rvm_overloaded mint8_t mmvar_m_v(vint8m1_t op0, size_t op1) {
  return __builtin_riscv_mmvar_m_v_i8m1(op0, op1);
}
__rvm_overloaded muint8_t mmvar_m_v(vuint8m1_t op0, size_t op1) {
  return __builtin_riscv_mmvar_m_v_u8m1(op0, op1);
}
__rvm_overloaded vint16m1_t mmvar_v_m(mint16_t op0, size_t op1) {
  return __builtin_riscv_mmvar_v_m_i16m1(op0, op1);
}
__rvm_overloaded vuint16m1_t mmvar_v_m(muint16_t op0, size_t op1) {
  return __builtin_riscv_mmvar_v_m_u16m1(op0, op1);
}
__rvm_overloaded vint32m1_t mmvar_v_m(mint32_t op0, size_t op1) {
  return __builtin_riscv_mmvar_v_m_i32m1(op0, op1);
}
__rvm_overloaded vuint32m1_t mmvar_v_m(muint32_t op0, size_t op1) {
  return __builtin_riscv_mmvar_v_m_u32m1(op0, op1);
}
__rvm_overloaded vint64m1_t mmvar_v_m(mint64_t op0, size_t op1) {
  return __builtin_riscv_mmvar_v_m_i64m1(op0, op1);
}
__rvm_overloaded vuint64m1_t mmvar_v_m(muint64_t op0, size_t op1) {
  return __builtin_riscv_mmvar_v_m_u64m1(op0, op1);
}
__rvm_overloaded vfloat16m1_t mmvar_v_m(mfloat16_t op0, size_t op1) {
  return __builtin_riscv_mmvar_v_m_f16m1(op0, op1);
}
__rvm_overloaded vfloat32m1_t mmvar_v_m(mfloat32_t op0, size_t op1) {
  return __builtin_riscv_mmvar_v_m_f32m1(op0, op1);
}
__rvm_overloaded vfloat64m1_t mmvar_v_m(mfloat64_t op0, size_t op1) {
  return __builtin_riscv_mmvar_v_m_f64m1(op0, op1);
}
__rvm_overloaded mint16_t mmvar_m_v(vint16m1_t op0, size_t op1) {
  return __builtin_riscv_mmvar_m_v_i16m1(op0, op1);
}
__rvm_overloaded muint16_t mmvar_m_v(vuint16m1_t op0, size_t op1) {
  return __builtin_riscv_mmvar_m_v_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mmvar_m_v(vint32m1_t op0, size_t op1) {
  return __builtin_riscv_mmvar_m_v_i32m1(op0, op1);
}
__rvm_overloaded muint32_t mmvar_m_v(vuint32m1_t op0, size_t op1) {
  return __builtin_riscv_mmvar_m_v_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mmvar_m_v(vint64m1_t op0, size_t op1) {
  return __builtin_riscv_mmvar_m_v_i64m1(op0, op1);
}
__rvm_overloaded muint64_t mmvar_m_v(vuint64m1_t op0, size_t op1) {
  return __builtin_riscv_mmvar_m_v_u64m1(op0, op1);
}
__rvm_overloaded mfloat16_t mmvar_m_v(vfloat16m1_t op0, size_t op1) {
  return __builtin_riscv_mmvar_m_v_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mmvar_m_v(vfloat32m1_t op0, size_t op1) {
  return __builtin_riscv_mmvar_m_v_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mmvar_m_v(vfloat64m1_t op0, size_t op1) {
  return __builtin_riscv_mmvar_m_v_f64m1(op0, op1);
}
__rvm_overloaded mint8_t madd_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_madd_b_mm_i8m1(op0, op1);
}
__rvm_overloaded mint64_t madd_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_madd_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t madd_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_madd_h_mm_i16m1(op0, op1);
}
__rvm_overloaded mint8_t madd_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_madd_mm_i8m1(op0, op1);
}
__rvm_overloaded mint16_t madd_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_madd_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t madd_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_madd_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t madd_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_madd_mm_i64m1(op0, op1);
}
__rvm_overloaded mint32_t madd_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_madd_w_mm_i32m1(op0, op1);
}
__rvm_overloaded muint8_t maddu_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_maddu_b_mm_u8m1(op0, op1);
}
__rvm_overloaded muint64_t maddu_dw_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_maddu_dw_mm_u64m1(op0, op1);
}
__rvm_overloaded muint16_t maddu_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_maddu_h_mm_u16m1(op0, op1);
}
__rvm_overloaded muint8_t maddu_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_maddu_mm_u8m1(op0, op1);
}
__rvm_overloaded muint16_t maddu_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_maddu_mm_u16m1(op0, op1);
}
__rvm_overloaded muint32_t maddu_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_maddu_mm_u32m1(op0, op1);
}
__rvm_overloaded muint64_t maddu_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_maddu_mm_u64m1(op0, op1);
}
__rvm_overloaded muint32_t maddu_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_maddu_w_mm_u32m1(op0, op1);
}
__rvm_overloaded mint8_t mcvt_b_ub_m(muint8_t op0) {
  return __builtin_riscv_mcvt_b_ub_m_i8m1(op0);
}
__rvm_overloaded mint64_t mcvt_dw_udw_m(muint64_t op0) {
  return __builtin_riscv_mcvt_dw_udw_m_i64m1(op0);
}
__rvm_overloaded mint16_t mcvt_h_uh_m(muint16_t op0) {
  return __builtin_riscv_mcvt_h_uh_m_i16m1(op0);
}
__rvm_overloaded muint8_t mcvt_ub_b_m(mint8_t op0) {
  return __builtin_riscv_mcvt_ub_b_m_u8m1(op0);
}
__rvm_overloaded muint64_t mcvt_udw_dw_m(mint64_t op0) {
  return __builtin_riscv_mcvt_udw_dw_m_u64m1(op0);
}
__rvm_overloaded muint16_t mcvt_uh_h_m(mint16_t op0) {
  return __builtin_riscv_mcvt_uh_h_m_u16m1(op0);
}
__rvm_overloaded muint32_t mcvt_uw_w_m(mint32_t op0) {
  return __builtin_riscv_mcvt_uw_w_m_u32m1(op0);
}
__rvm_overloaded mint32_t mcvt_w_uw_m(muint32_t op0) {
  return __builtin_riscv_mcvt_w_uw_m_i32m1(op0);
}
__rvm_overloaded mint8_t mcvt_x_xu_m(muint8_t op0) {
  return __builtin_riscv_mcvt_x_xu_m_i8m1(op0);
}
__rvm_overloaded mint16_t mcvt_x_xu_m(muint16_t op0) {
  return __builtin_riscv_mcvt_x_xu_m_i16m1(op0);
}
__rvm_overloaded mint32_t mcvt_x_xu_m(muint32_t op0) {
  return __builtin_riscv_mcvt_x_xu_m_i32m1(op0);
}
__rvm_overloaded mint64_t mcvt_x_xu_m(muint64_t op0) {
  return __builtin_riscv_mcvt_x_xu_m_i64m1(op0);
}
__rvm_overloaded muint8_t mcvt_xu_x_m(mint8_t op0) {
  return __builtin_riscv_mcvt_xu_x_m_u8m1(op0);
}
__rvm_overloaded muint16_t mcvt_xu_x_m(mint16_t op0) {
  return __builtin_riscv_mcvt_xu_x_m_u16m1(op0);
}
__rvm_overloaded muint32_t mcvt_xu_x_m(mint32_t op0) {
  return __builtin_riscv_mcvt_xu_x_m_u32m1(op0);
}
__rvm_overloaded muint64_t mcvt_xu_x_m(mint64_t op0) {
  return __builtin_riscv_mcvt_xu_x_m_u64m1(op0);
}
__rvm_overloaded mfloat64_t mfadd_d_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfadd_d_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfadd_f_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfadd_f_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfadd_hf_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfadd_hf_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfadd_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfadd_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfadd_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfadd_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfadd_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfadd_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfcvt_d_dw_m(mint64_t op0) {
  return __builtin_riscv_mfcvt_d_dw_m_f64m1(op0);
}
__rvm_overloaded mint64_t mfcvt_dw_d_m(mfloat64_t op0) {
  return __builtin_riscv_mfcvt_dw_d_m_i64m1(op0);
}
__rvm_overloaded mfloat32_t mfcvt_f_w_m(mint32_t op0) {
  return __builtin_riscv_mfcvt_f_w_m_f32m1(op0);
}
__rvm_overloaded mfloat16_t mfcvt_f_x_m(mint16_t op0) {
  return __builtin_riscv_mfcvt_f_x_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mfcvt_f_x_m(mint32_t op0) {
  return __builtin_riscv_mfcvt_f_x_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mfcvt_f_x_m(mint64_t op0) {
  return __builtin_riscv_mfcvt_f_x_m_f64m1(op0);
}
__rvm_overloaded mint16_t mfcvt_h_hf_m(mfloat16_t op0) {
  return __builtin_riscv_mfcvt_h_hf_m_i16m1(op0);
}
__rvm_overloaded mfloat16_t mfcvt_hf_h_m(mint16_t op0) {
  return __builtin_riscv_mfcvt_hf_h_m_f16m1(op0);
}
__rvm_overloaded mint32_t mfcvt_w_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfcvt_w_f_m_i32m1(op0);
}
__rvm_overloaded mint16_t mfcvt_x_f_m(mfloat16_t op0) {
  return __builtin_riscv_mfcvt_x_f_m_i16m1(op0);
}
__rvm_overloaded mint32_t mfcvt_x_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfcvt_x_f_m_i32m1(op0);
}
__rvm_overloaded mint64_t mfcvt_x_f_m(mfloat64_t op0) {
  return __builtin_riscv_mfcvt_x_f_m_i64m1(op0);
}
__rvm_overloaded mfloat64_t mfcvtu_d_dw_m(muint64_t op0) {
  return __builtin_riscv_mfcvtu_d_dw_m_f64m1(op0);
}
__rvm_overloaded muint64_t mfcvtu_dw_d_m(mfloat64_t op0) {
  return __builtin_riscv_mfcvtu_dw_d_m_u64m1(op0);
}
__rvm_overloaded mfloat32_t mfcvtu_f_w_m(muint32_t op0) {
  return __builtin_riscv_mfcvtu_f_w_m_f32m1(op0);
}
__rvm_overloaded mfloat16_t mfcvtu_f_x_m(muint16_t op0) {
  return __builtin_riscv_mfcvtu_f_x_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mfcvtu_f_x_m(muint32_t op0) {
  return __builtin_riscv_mfcvtu_f_x_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mfcvtu_f_x_m(muint64_t op0) {
  return __builtin_riscv_mfcvtu_f_x_m_f64m1(op0);
}
__rvm_overloaded muint16_t mfcvtu_h_hf_m(mfloat16_t op0) {
  return __builtin_riscv_mfcvtu_h_hf_m_u16m1(op0);
}
__rvm_overloaded mfloat16_t mfcvtu_hf_h_m(muint16_t op0) {
  return __builtin_riscv_mfcvtu_hf_h_m_f16m1(op0);
}
__rvm_overloaded muint32_t mfcvtu_w_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfcvtu_w_f_m_u32m1(op0);
}
__rvm_overloaded muint16_t mfcvtu_x_f_m(mfloat16_t op0) {
  return __builtin_riscv_mfcvtu_x_f_m_u16m1(op0);
}
__rvm_overloaded muint32_t mfcvtu_x_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfcvtu_x_f_m_u32m1(op0);
}
__rvm_overloaded muint64_t mfcvtu_x_f_m(mfloat64_t op0) {
  return __builtin_riscv_mfcvtu_x_f_m_u64m1(op0);
}
__rvm_overloaded mfloat64_t mfdiv_d_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfdiv_d_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfdiv_f_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfdiv_f_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfdiv_hf_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfdiv_hf_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfdiv_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfdiv_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfdiv_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfdiv_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfdiv_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfdiv_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfma_d_mm(mfloat64_t op0, mfloat64_t op1, mfloat64_t op2) {
  return __builtin_riscv_mfma_d_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfma_f_mm(mfloat32_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfma_f_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat16_t mfma_hf_mm(mfloat16_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfma_hf_mm_f16m1(op0, op1, op2);
}
__rvm_overloaded mfloat16_t mfma_mm(mfloat16_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfma_mm_f16m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfma_mm(mfloat32_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfma_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfma_mm(mfloat64_t op0, mfloat64_t op1, mfloat64_t op2) {
  return __builtin_riscv_mfma_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfma_spa_d_mm(mfloat64_t op0, mfloat64_t op1, mfloat64_t op2) {
  return __builtin_riscv_mfma_spa_d_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfma_spa_f_mm(mfloat32_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfma_spa_f_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat16_t mfma_spa_hf_mm(mfloat16_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfma_spa_hf_mm_f16m1(op0, op1, op2);
}
__rvm_overloaded mfloat16_t mfma_spa_mm(mfloat16_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfma_spa_mm_f16m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfma_spa_mm(mfloat32_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfma_spa_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfma_spa_mm(mfloat64_t op0, mfloat64_t op1, mfloat64_t op2) {
  return __builtin_riscv_mfma_spa_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfma_spb_d_mm(mfloat64_t op0, mfloat64_t op1, mfloat64_t op2) {
  return __builtin_riscv_mfma_spb_d_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfma_spb_f_mm(mfloat32_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfma_spb_f_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat16_t mfma_spb_hf_mm(mfloat16_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfma_spb_hf_mm_f16m1(op0, op1, op2);
}
__rvm_overloaded mfloat16_t mfma_spb_mm(mfloat16_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfma_spb_mm_f16m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfma_spb_mm(mfloat32_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfma_spb_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfma_spb_mm(mfloat64_t op0, mfloat64_t op1, mfloat64_t op2) {
  return __builtin_riscv_mfma_spb_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfmax_d_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfmax_d_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfmax_f_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfmax_f_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfmax_hf_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfmax_hf_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfmax_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfmax_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfmax_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfmax_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfmax_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfmax_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfmin_d_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfmin_d_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfmin_f_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfmin_f_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfmin_hf_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfmin_hf_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfmin_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfmin_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfmin_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfmin_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfmin_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfmin_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfmul_d_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfmul_d_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfmul_f_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfmul_f_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfmul_hf_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfmul_hf_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfmul_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfmul_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfmul_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfmul_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfmul_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfmul_mm_f64m1(op0, op1);
}
__rvm_overloaded mint8_t mfncvt_b_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfncvt_b_f_m_i8m1(op0);
}
__rvm_overloaded mint8_t mfncvt_b_hf_m(mfloat16_t op0) {
  return __builtin_riscv_mfncvt_b_hf_m_i8m1(op0);
}
__rvm_overloaded mfloat32_t mfncvt_f_d_m(mfloat64_t op0) {
  return __builtin_riscv_mfncvt_f_d_m_f32m1(op0);
}
__rvm_overloaded mfloat32_t mfncvt_f_dw_m(mint64_t op0) {
  return __builtin_riscv_mfncvt_f_dw_m_f32m1(op0);
}
__rvm_overloaded mfloat16_t mfncvt_f_fw_m(mfloat32_t op0) {
  return __builtin_riscv_mfncvt_f_fw_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mfncvt_f_fw_m(mfloat64_t op0) {
  return __builtin_riscv_mfncvt_f_fw_m_f32m1(op0);
}
__rvm_overloaded mfloat16_t mfncvt_f_xw_m(mint32_t op0) {
  return __builtin_riscv_mfncvt_f_xw_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mfncvt_f_xw_m(mint64_t op0) {
  return __builtin_riscv_mfncvt_f_xw_m_f32m1(op0);
}
__rvm_overloaded mint16_t mfncvt_h_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfncvt_h_f_m_i16m1(op0);
}
__rvm_overloaded mfloat16_t mfncvt_hf_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfncvt_hf_f_m_f16m1(op0);
}
__rvm_overloaded mfloat16_t mfncvt_hf_w_m(mint32_t op0) {
  return __builtin_riscv_mfncvt_hf_w_m_f16m1(op0);
}
__rvm_overloaded mint32_t mfncvt_w_d_m(mfloat64_t op0) {
  return __builtin_riscv_mfncvt_w_d_m_i32m1(op0);
}
__rvm_overloaded mint8_t mfncvt_x_fq_m(mfloat32_t op0) {
  return __builtin_riscv_mfncvt_x_fq_m_i8m1(op0);
}
__rvm_overloaded mint8_t mfncvt_x_fw_m(mfloat16_t op0) {
  return __builtin_riscv_mfncvt_x_fw_m_i8m1(op0);
}
__rvm_overloaded mint16_t mfncvt_x_fw_m(mfloat32_t op0) {
  return __builtin_riscv_mfncvt_x_fw_m_i16m1(op0);
}
__rvm_overloaded mint32_t mfncvt_x_fw_m(mfloat64_t op0) {
  return __builtin_riscv_mfncvt_x_fw_m_i32m1(op0);
}
__rvm_overloaded muint8_t mfncvtu_b_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfncvtu_b_f_m_u8m1(op0);
}
__rvm_overloaded muint8_t mfncvtu_b_hf_m(mfloat16_t op0) {
  return __builtin_riscv_mfncvtu_b_hf_m_u8m1(op0);
}
__rvm_overloaded mfloat32_t mfncvtu_f_dw_m(muint64_t op0) {
  return __builtin_riscv_mfncvtu_f_dw_m_f32m1(op0);
}
__rvm_overloaded mfloat16_t mfncvtu_f_xw_m(muint32_t op0) {
  return __builtin_riscv_mfncvtu_f_xw_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mfncvtu_f_xw_m(muint64_t op0) {
  return __builtin_riscv_mfncvtu_f_xw_m_f32m1(op0);
}
__rvm_overloaded muint16_t mfncvtu_h_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfncvtu_h_f_m_u16m1(op0);
}
__rvm_overloaded mfloat16_t mfncvtu_hf_w_m(muint32_t op0) {
  return __builtin_riscv_mfncvtu_hf_w_m_f16m1(op0);
}
__rvm_overloaded muint32_t mfncvtu_w_d_m(mfloat64_t op0) {
  return __builtin_riscv_mfncvtu_w_d_m_u32m1(op0);
}
__rvm_overloaded muint8_t mfncvtu_x_fq_m(mfloat32_t op0) {
  return __builtin_riscv_mfncvtu_x_fq_m_u8m1(op0);
}
__rvm_overloaded muint8_t mfncvtu_x_fw_m(mfloat16_t op0) {
  return __builtin_riscv_mfncvtu_x_fw_m_u8m1(op0);
}
__rvm_overloaded muint16_t mfncvtu_x_fw_m(mfloat32_t op0) {
  return __builtin_riscv_mfncvtu_x_fw_m_u16m1(op0);
}
__rvm_overloaded muint32_t mfncvtu_x_fw_m(mfloat64_t op0) {
  return __builtin_riscv_mfncvtu_x_fw_m_u32m1(op0);
}
__rvm_overloaded mfloat64_t mfsqrt_d_m(mfloat64_t op0) {
  return __builtin_riscv_mfsqrt_d_m_f64m1(op0);
}
__rvm_overloaded mfloat32_t mfsqrt_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfsqrt_f_m_f32m1(op0);
}
__rvm_overloaded mfloat16_t mfsqrt_hf_m(mfloat16_t op0) {
  return __builtin_riscv_mfsqrt_hf_m_f16m1(op0);
}
__rvm_overloaded mfloat16_t mfsqrt_m(mfloat16_t op0) {
  return __builtin_riscv_mfsqrt_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mfsqrt_m(mfloat32_t op0) {
  return __builtin_riscv_mfsqrt_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mfsqrt_m(mfloat64_t op0) {
  return __builtin_riscv_mfsqrt_m_f64m1(op0);
}
__rvm_overloaded mfloat64_t mfsub_d_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfsub_d_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfsub_f_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfsub_f_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfsub_hf_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfsub_hf_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat16_t mfsub_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfsub_mm_f16m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfsub_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfsub_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfsub_mm(mfloat64_t op0, mfloat64_t op1) {
  return __builtin_riscv_mfsub_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfwadd_f_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfwadd_f_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfwadd_hf_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfwadd_hf_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfwadd_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfwadd_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfwadd_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfwadd_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfwcvt_d_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfwcvt_d_f_m_f64m1(op0);
}
__rvm_overloaded mfloat64_t mfwcvt_d_w_m(mint32_t op0) {
  return __builtin_riscv_mfwcvt_d_w_m_f64m1(op0);
}
__rvm_overloaded mint64_t mfwcvt_dw_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfwcvt_dw_f_m_i64m1(op0);
}
__rvm_overloaded mfloat32_t mfwcvt_f_b_m(mint8_t op0) {
  return __builtin_riscv_mfwcvt_f_b_m_f32m1(op0);
}
__rvm_overloaded mfloat32_t mfwcvt_f_h_m(mint16_t op0) {
  return __builtin_riscv_mfwcvt_f_h_m_f32m1(op0);
}
__rvm_overloaded mfloat32_t mfwcvt_f_hf_m(mfloat16_t op0) {
  return __builtin_riscv_mfwcvt_f_hf_m_f32m1(op0);
}
__rvm_overloaded mfloat32_t mfwcvt_fq_x_m(mint8_t op0) {
  return __builtin_riscv_mfwcvt_fq_x_m_f32m1(op0);
}
__rvm_overloaded mfloat32_t mfwcvt_fw_f_m(mfloat16_t op0) {
  return __builtin_riscv_mfwcvt_fw_f_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mfwcvt_fw_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfwcvt_fw_f_m_f64m1(op0);
}
__rvm_overloaded mfloat16_t mfwcvt_fw_x_m(mint8_t op0) {
  return __builtin_riscv_mfwcvt_fw_x_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mfwcvt_fw_x_m(mint16_t op0) {
  return __builtin_riscv_mfwcvt_fw_x_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mfwcvt_fw_x_m(mint32_t op0) {
  return __builtin_riscv_mfwcvt_fw_x_m_f64m1(op0);
}
__rvm_overloaded mfloat16_t mfwcvt_hf_b_m(mint8_t op0) {
  return __builtin_riscv_mfwcvt_hf_b_m_f16m1(op0);
}
__rvm_overloaded mint32_t mfwcvt_w_hf_m(mfloat16_t op0) {
  return __builtin_riscv_mfwcvt_w_hf_m_i32m1(op0);
}
__rvm_overloaded mint32_t mfwcvt_xw_f_m(mfloat16_t op0) {
  return __builtin_riscv_mfwcvt_xw_f_m_i32m1(op0);
}
__rvm_overloaded mint64_t mfwcvt_xw_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfwcvt_xw_f_m_i64m1(op0);
}
__rvm_overloaded mfloat64_t mfwcvtu_d_w_m(muint32_t op0) {
  return __builtin_riscv_mfwcvtu_d_w_m_f64m1(op0);
}
__rvm_overloaded muint64_t mfwcvtu_dw_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfwcvtu_dw_f_m_u64m1(op0);
}
__rvm_overloaded mfloat32_t mfwcvtu_f_b_m(muint8_t op0) {
  return __builtin_riscv_mfwcvtu_f_b_m_f32m1(op0);
}
__rvm_overloaded mfloat32_t mfwcvtu_f_h_m(muint16_t op0) {
  return __builtin_riscv_mfwcvtu_f_h_m_f32m1(op0);
}
__rvm_overloaded mfloat32_t mfwcvtu_fq_x_m(muint8_t op0) {
  return __builtin_riscv_mfwcvtu_fq_x_m_f32m1(op0);
}
__rvm_overloaded mfloat16_t mfwcvtu_fw_x_m(muint8_t op0) {
  return __builtin_riscv_mfwcvtu_fw_x_m_f16m1(op0);
}
__rvm_overloaded mfloat32_t mfwcvtu_fw_x_m(muint16_t op0) {
  return __builtin_riscv_mfwcvtu_fw_x_m_f32m1(op0);
}
__rvm_overloaded mfloat64_t mfwcvtu_fw_x_m(muint32_t op0) {
  return __builtin_riscv_mfwcvtu_fw_x_m_f64m1(op0);
}
__rvm_overloaded mfloat16_t mfwcvtu_hf_b_m(muint8_t op0) {
  return __builtin_riscv_mfwcvtu_hf_b_m_f16m1(op0);
}
__rvm_overloaded muint32_t mfwcvtu_w_hf_m(mfloat16_t op0) {
  return __builtin_riscv_mfwcvtu_w_hf_m_u32m1(op0);
}
__rvm_overloaded muint32_t mfwcvtu_xw_f_m(mfloat16_t op0) {
  return __builtin_riscv_mfwcvtu_xw_f_m_u32m1(op0);
}
__rvm_overloaded muint64_t mfwcvtu_xw_f_m(mfloat32_t op0) {
  return __builtin_riscv_mfwcvtu_xw_f_m_u64m1(op0);
}
__rvm_overloaded mfloat64_t mfwma_f_mm(mfloat64_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfwma_f_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfwma_hf_mm(mfloat32_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfwma_hf_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfwma_mm(mfloat32_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfwma_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfwma_mm(mfloat64_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfwma_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfwma_spa_f_mm(mfloat64_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfwma_spa_f_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfwma_spa_hf_mm(mfloat32_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfwma_spa_hf_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfwma_spa_mm(mfloat32_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfwma_spa_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfwma_spa_mm(mfloat64_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfwma_spa_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfwma_spb_f_mm(mfloat64_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfwma_spb_f_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfwma_spb_hf_mm(mfloat32_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfwma_spb_hf_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat32_t mfwma_spb_mm(mfloat32_t op0, mfloat16_t op1, mfloat16_t op2) {
  return __builtin_riscv_mfwma_spb_mm_f32m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfwma_spb_mm(mfloat64_t op0, mfloat32_t op1, mfloat32_t op2) {
  return __builtin_riscv_mfwma_spb_mm_f64m1(op0, op1, op2);
}
__rvm_overloaded mfloat64_t mfwmul_f_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfwmul_f_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfwmul_hf_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfwmul_hf_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfwmul_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfwmul_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfwmul_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfwmul_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfwsub_f_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfwsub_f_mm_f64m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfwsub_hf_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfwsub_hf_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat32_t mfwsub_mm(mfloat16_t op0, mfloat16_t op1) {
  return __builtin_riscv_mfwsub_mm_f32m1(op0, op1);
}
__rvm_overloaded mfloat64_t mfwsub_mm(mfloat32_t op0, mfloat32_t op1) {
  return __builtin_riscv_mfwsub_mm_f64m1(op0, op1);
}
__rvm_overloaded mint64_t mma_dw_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_mma_dw_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint16_t mma_h_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mma_h_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint16_t mma_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mma_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mma_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mma_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mma_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_mma_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mma_spa_dw_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_mma_spa_dw_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint16_t mma_spa_h_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mma_spa_h_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint16_t mma_spa_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mma_spa_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mma_spa_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mma_spa_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mma_spa_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_mma_spa_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mma_spa_w_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mma_spa_w_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mma_spb_dw_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_mma_spb_dw_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint16_t mma_spb_h_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mma_spb_h_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint16_t mma_spb_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mma_spb_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mma_spb_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mma_spb_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mma_spb_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_mma_spb_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mma_spb_w_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mma_spb_w_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mma_w_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mma_w_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mmau_dw_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_mmau_dw_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint16_t mmau_h_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mmau_h_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint16_t mmau_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mmau_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mmau_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mmau_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mmau_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_mmau_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mmau_spa_dw_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_mmau_spa_dw_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint16_t mmau_spa_h_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mmau_spa_h_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint16_t mmau_spa_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mmau_spa_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mmau_spa_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mmau_spa_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mmau_spa_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_mmau_spa_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mmau_spa_w_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mmau_spa_w_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mmau_spb_dw_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_mmau_spb_dw_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint16_t mmau_spb_h_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mmau_spb_h_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint16_t mmau_spb_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mmau_spb_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mmau_spb_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mmau_spb_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mmau_spb_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_mmau_spb_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mmau_spb_w_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mmau_spb_w_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mmau_w_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mmau_w_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded mint8_t mmax_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mmax_b_mm_i8m1(op0, op1);
}
__rvm_overloaded mint64_t mmax_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mmax_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t mmax_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mmax_h_mm_i16m1(op0, op1);
}
__rvm_overloaded mint8_t mmax_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mmax_mm_i8m1(op0, op1);
}
__rvm_overloaded mint16_t mmax_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mmax_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mmax_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mmax_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t mmax_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mmax_mm_i64m1(op0, op1);
}
__rvm_overloaded mint32_t mmax_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mmax_w_mm_i32m1(op0, op1);
}
__rvm_overloaded muint8_t mmaxu_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mmaxu_b_mm_u8m1(op0, op1);
}
__rvm_overloaded muint64_t mmaxu_dw_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_mmaxu_dw_mm_u64m1(op0, op1);
}
__rvm_overloaded muint16_t mmaxu_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mmaxu_h_mm_u16m1(op0, op1);
}
__rvm_overloaded muint8_t mmaxu_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mmaxu_mm_u8m1(op0, op1);
}
__rvm_overloaded muint16_t mmaxu_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mmaxu_mm_u16m1(op0, op1);
}
__rvm_overloaded muint32_t mmaxu_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mmaxu_mm_u32m1(op0, op1);
}
__rvm_overloaded muint64_t mmaxu_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_mmaxu_mm_u64m1(op0, op1);
}
__rvm_overloaded muint32_t mmaxu_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mmaxu_w_mm_u32m1(op0, op1);
}
__rvm_overloaded mint8_t mmin_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mmin_b_mm_i8m1(op0, op1);
}
__rvm_overloaded mint64_t mmin_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mmin_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t mmin_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mmin_h_mm_i16m1(op0, op1);
}
__rvm_overloaded mint8_t mmin_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mmin_mm_i8m1(op0, op1);
}
__rvm_overloaded mint16_t mmin_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mmin_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mmin_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mmin_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t mmin_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mmin_mm_i64m1(op0, op1);
}
__rvm_overloaded mint32_t mmin_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mmin_w_mm_i32m1(op0, op1);
}
__rvm_overloaded muint8_t mminu_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mminu_b_mm_u8m1(op0, op1);
}
__rvm_overloaded muint64_t mminu_dw_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_mminu_dw_mm_u64m1(op0, op1);
}
__rvm_overloaded muint16_t mminu_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mminu_h_mm_u16m1(op0, op1);
}
__rvm_overloaded muint8_t mminu_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mminu_mm_u8m1(op0, op1);
}
__rvm_overloaded muint16_t mminu_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mminu_mm_u16m1(op0, op1);
}
__rvm_overloaded muint32_t mminu_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mminu_mm_u32m1(op0, op1);
}
__rvm_overloaded muint64_t mminu_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_mminu_mm_u64m1(op0, op1);
}
__rvm_overloaded muint32_t mminu_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mminu_w_mm_u32m1(op0, op1);
}
__rvm_overloaded mint8_t mmul_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mmul_b_mm_i8m1(op0, op1);
}
__rvm_overloaded mint64_t mmul_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mmul_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t mmul_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mmul_h_mm_i16m1(op0, op1);
}
__rvm_overloaded mint8_t mmul_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mmul_mm_i8m1(op0, op1);
}
__rvm_overloaded mint16_t mmul_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mmul_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mmul_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mmul_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t mmul_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mmul_mm_i64m1(op0, op1);
}
__rvm_overloaded mint32_t mmul_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mmul_w_mm_i32m1(op0, op1);
}
__rvm_overloaded mint8_t mmulh_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mmulh_b_mm_i8m1(op0, op1);
}
__rvm_overloaded mint64_t mmulh_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mmulh_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t mmulh_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mmulh_h_mm_i16m1(op0, op1);
}
__rvm_overloaded mint8_t mmulh_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mmulh_mm_i8m1(op0, op1);
}
__rvm_overloaded mint16_t mmulh_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mmulh_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mmulh_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mmulh_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t mmulh_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mmulh_mm_i64m1(op0, op1);
}
__rvm_overloaded mint32_t mmulh_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mmulh_w_mm_i32m1(op0, op1);
}
__rvm_overloaded mint8_t mmulhsu_b_mm(mint8_t op0, muint8_t op1) {
  return __builtin_riscv_mmulhsu_b_mm_i8m1(op0, op1);
}
__rvm_overloaded mint64_t mmulhsu_dw_mm(mint64_t op0, muint64_t op1) {
  return __builtin_riscv_mmulhsu_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t mmulhsu_h_mm(mint16_t op0, muint16_t op1) {
  return __builtin_riscv_mmulhsu_h_mm_i16m1(op0, op1);
}
__rvm_overloaded mint8_t mmulhsu_mm(mint8_t op0, muint8_t op1) {
  return __builtin_riscv_mmulhsu_mm_i8m1(op0, op1);
}
__rvm_overloaded mint16_t mmulhsu_mm(mint16_t op0, muint16_t op1) {
  return __builtin_riscv_mmulhsu_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mmulhsu_mm(mint32_t op0, muint32_t op1) {
  return __builtin_riscv_mmulhsu_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t mmulhsu_mm(mint64_t op0, muint64_t op1) {
  return __builtin_riscv_mmulhsu_mm_i64m1(op0, op1);
}
__rvm_overloaded mint32_t mmulhsu_w_mm(mint32_t op0, muint32_t op1) {
  return __builtin_riscv_mmulhsu_w_mm_i32m1(op0, op1);
}
__rvm_overloaded muint8_t mmulhu_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mmulhu_b_mm_u8m1(op0, op1);
}
__rvm_overloaded muint64_t mmulhu_dw_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_mmulhu_dw_mm_u64m1(op0, op1);
}
__rvm_overloaded muint16_t mmulhu_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mmulhu_h_mm_u16m1(op0, op1);
}
__rvm_overloaded muint8_t mmulhu_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mmulhu_mm_u8m1(op0, op1);
}
__rvm_overloaded muint16_t mmulhu_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mmulhu_mm_u16m1(op0, op1);
}
__rvm_overloaded muint32_t mmulhu_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mmulhu_mm_u32m1(op0, op1);
}
__rvm_overloaded muint64_t mmulhu_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_mmulhu_mm_u64m1(op0, op1);
}
__rvm_overloaded muint32_t mmulhu_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mmulhu_w_mm_u32m1(op0, op1);
}
__rvm_overloaded mint8_t mncvt_b_h_m(mint16_t op0) {
  return __builtin_riscv_mncvt_b_h_m_i8m1(op0);
}
__rvm_overloaded mint8_t mncvt_b_w_m(mint32_t op0) {
  return __builtin_riscv_mncvt_b_w_m_i8m1(op0);
}
__rvm_overloaded mint16_t mncvt_h_w_m(mint32_t op0) {
  return __builtin_riscv_mncvt_h_w_m_i16m1(op0);
}
__rvm_overloaded mint32_t mncvt_w_dw_m(mint64_t op0) {
  return __builtin_riscv_mncvt_w_dw_m_i32m1(op0);
}
__rvm_overloaded mint8_t mncvt_x_xq_m(mint32_t op0) {
  return __builtin_riscv_mncvt_x_xq_m_i8m1(op0);
}
__rvm_overloaded mint8_t mncvt_x_xw_m(mint16_t op0) {
  return __builtin_riscv_mncvt_x_xw_m_i8m1(op0);
}
__rvm_overloaded mint16_t mncvt_x_xw_m(mint32_t op0) {
  return __builtin_riscv_mncvt_x_xw_m_i16m1(op0);
}
__rvm_overloaded mint32_t mncvt_x_xw_m(mint64_t op0) {
  return __builtin_riscv_mncvt_x_xw_m_i32m1(op0);
}
__rvm_overloaded muint8_t mncvtu_b_h_m(muint16_t op0) {
  return __builtin_riscv_mncvtu_b_h_m_u8m1(op0);
}
__rvm_overloaded muint8_t mncvtu_b_w_m(muint32_t op0) {
  return __builtin_riscv_mncvtu_b_w_m_u8m1(op0);
}
__rvm_overloaded muint16_t mncvtu_h_w_m(muint32_t op0) {
  return __builtin_riscv_mncvtu_h_w_m_u16m1(op0);
}
__rvm_overloaded muint32_t mncvtu_w_dw_m(muint64_t op0) {
  return __builtin_riscv_mncvtu_w_dw_m_u32m1(op0);
}
__rvm_overloaded muint8_t mncvtu_x_xq_m(muint32_t op0) {
  return __builtin_riscv_mncvtu_x_xq_m_u8m1(op0);
}
__rvm_overloaded muint8_t mncvtu_x_xw_m(muint16_t op0) {
  return __builtin_riscv_mncvtu_x_xw_m_u8m1(op0);
}
__rvm_overloaded muint16_t mncvtu_x_xw_m(muint32_t op0) {
  return __builtin_riscv_mncvtu_x_xw_m_u16m1(op0);
}
__rvm_overloaded muint32_t mncvtu_x_xw_m(muint64_t op0) {
  return __builtin_riscv_mncvtu_x_xw_m_u32m1(op0);
}
__rvm_overloaded mint32_t mqma_b_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_mqma_b_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mqma_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_mqma_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mqma_spa_b_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_mqma_spa_b_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mqma_spa_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_mqma_spa_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mqma_spb_b_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_mqma_spb_b_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mqma_spb_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_mqma_spb_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mqmau_b_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_mqmau_b_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mqmau_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_mqmau_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mqmau_spa_b_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_mqmau_spa_b_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mqmau_spa_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_mqmau_spa_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mqmau_spb_b_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_mqmau_spb_b_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mqmau_spb_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_mqmau_spb_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded mint8_t msadd_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msadd_b_mm_i8m1(op0, op1);
}
__rvm_overloaded mint64_t msadd_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msadd_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t msadd_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msadd_h_mm_i16m1(op0, op1);
}
__rvm_overloaded mint8_t msadd_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msadd_mm_i8m1(op0, op1);
}
__rvm_overloaded mint16_t msadd_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msadd_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t msadd_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msadd_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t msadd_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msadd_mm_i64m1(op0, op1);
}
__rvm_overloaded mint32_t msadd_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msadd_w_mm_i32m1(op0, op1);
}
__rvm_overloaded muint8_t msaddu_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msaddu_b_mm_u8m1(op0, op1);
}
__rvm_overloaded muint64_t msaddu_dw_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msaddu_dw_mm_u64m1(op0, op1);
}
__rvm_overloaded muint16_t msaddu_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msaddu_h_mm_u16m1(op0, op1);
}
__rvm_overloaded muint8_t msaddu_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msaddu_mm_u8m1(op0, op1);
}
__rvm_overloaded muint16_t msaddu_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msaddu_mm_u16m1(op0, op1);
}
__rvm_overloaded muint32_t msaddu_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msaddu_mm_u32m1(op0, op1);
}
__rvm_overloaded muint64_t msaddu_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msaddu_mm_u64m1(op0, op1);
}
__rvm_overloaded muint32_t msaddu_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msaddu_w_mm_u32m1(op0, op1);
}
__rvm_overloaded size_t msetba(size_t op0) {
  return __builtin_riscv_msetba(op0);
}
__rvm_overloaded size_t msetdsp(size_t op0) {
  return __builtin_riscv_msetdsp(op0);
}
__rvm_overloaded size_t msetdspi(size_t op0) {
  return __builtin_riscv_msetdspi(op0);
}
__rvm_overloaded size_t msetfp16(size_t op0) {
  return __builtin_riscv_msetfp16(op0);
}
__rvm_overloaded size_t msetfp32(size_t op0) {
  return __builtin_riscv_msetfp32(op0);
}
__rvm_overloaded size_t msetfp64(size_t op0) {
  return __builtin_riscv_msetfp64(op0);
}
__rvm_overloaded size_t msetint8(size_t op0) {
  return __builtin_riscv_msetint8(op0);
}
__rvm_overloaded size_t msetint16(size_t op0) {
  return __builtin_riscv_msetint16(op0);
}
__rvm_overloaded size_t msetint32(size_t op0) {
  return __builtin_riscv_msetint32(op0);
}
__rvm_overloaded size_t msetint64(size_t op0) {
  return __builtin_riscv_msetint64(op0);
}
__rvm_overloaded size_t msetpadval(size_t op0) {
  return __builtin_riscv_msetpadval(op0);
}
__rvm_overloaded size_t msetsew(size_t op0) {
  return __builtin_riscv_msetsew(op0);
}
__rvm_overloaded size_t msettilek(size_t op0) {
  return __builtin_riscv_msettilek(op0);
}
__rvm_overloaded size_t msettileki(size_t op0) {
  return __builtin_riscv_msettileki(op0);
}
__rvm_overloaded size_t msettilem(size_t op0) {
  return __builtin_riscv_msettilem(op0);
}
__rvm_overloaded size_t msettilemi(size_t op0) {
  return __builtin_riscv_msettilemi(op0);
}
__rvm_overloaded size_t msettilen(size_t op0) {
  return __builtin_riscv_msettilen(op0);
}
__rvm_overloaded size_t msettileni(size_t op0) {
  return __builtin_riscv_msettileni(op0);
}
__rvm_overloaded size_t msettsp(size_t op0) {
  return __builtin_riscv_msettsp(op0);
}
__rvm_overloaded size_t msettspi(size_t op0) {
  return __builtin_riscv_msettspi(op0);
}
__rvm_overloaded size_t msettype(size_t op0) {
  return __builtin_riscv_msettype(op0);
}
__rvm_overloaded size_t msettypehi(size_t op0) {
  return __builtin_riscv_msettypehi(op0);
}
__rvm_overloaded size_t msettypei(size_t op0) {
  return __builtin_riscv_msettypei(op0);
}
__rvm_overloaded mint64_t msma_dw_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_msma_dw_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint16_t msma_h_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_msma_h_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint16_t msma_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_msma_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint32_t msma_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_msma_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t msma_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_msma_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint64_t msma_spa_dw_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_msma_spa_dw_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint16_t msma_spa_h_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_msma_spa_h_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint16_t msma_spa_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_msma_spa_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint32_t msma_spa_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_msma_spa_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t msma_spa_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_msma_spa_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint32_t msma_spa_w_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_msma_spa_w_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t msma_spb_dw_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_msma_spb_dw_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint16_t msma_spb_h_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_msma_spb_h_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint16_t msma_spb_mm(mint16_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_msma_spb_mm_i16m1(op0, op1, op2);
}
__rvm_overloaded mint32_t msma_spb_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_msma_spb_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t msma_spb_mm(mint64_t op0, mint64_t op1, mint64_t op2) {
  return __builtin_riscv_msma_spb_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint32_t msma_spb_w_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_msma_spb_w_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t msma_w_mm(mint32_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_msma_w_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t msmau_dw_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_msmau_dw_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint16_t msmau_h_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_msmau_h_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint16_t msmau_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_msmau_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msmau_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_msmau_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t msmau_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_msmau_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint64_t msmau_spa_dw_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_msmau_spa_dw_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint16_t msmau_spa_h_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_msmau_spa_h_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint16_t msmau_spa_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_msmau_spa_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msmau_spa_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_msmau_spa_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t msmau_spa_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_msmau_spa_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msmau_spa_w_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_msmau_spa_w_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t msmau_spb_dw_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_msmau_spb_dw_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint16_t msmau_spb_h_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_msmau_spb_h_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint16_t msmau_spb_mm(muint16_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_msmau_spb_mm_u16m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msmau_spb_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_msmau_spb_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t msmau_spb_mm(muint64_t op0, muint64_t op1, muint64_t op2) {
  return __builtin_riscv_msmau_spb_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msmau_spb_w_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_msmau_spb_w_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msmau_w_mm(muint32_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_msmau_w_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded mint8_t msmul_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msmul_b_mm_i8m1(op0, op1);
}
__rvm_overloaded mint64_t msmul_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msmul_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t msmul_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msmul_h_mm_i16m1(op0, op1);
}
__rvm_overloaded mint8_t msmul_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msmul_mm_i8m1(op0, op1);
}
__rvm_overloaded mint16_t msmul_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msmul_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t msmul_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msmul_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t msmul_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msmul_mm_i64m1(op0, op1);
}
__rvm_overloaded mint32_t msmul_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msmul_w_mm_i32m1(op0, op1);
}
__rvm_overloaded mint8_t msmulsu_b_mm(mint8_t op0, muint8_t op1) {
  return __builtin_riscv_msmulsu_b_mm_i8m1(op0, op1);
}
__rvm_overloaded mint64_t msmulsu_dw_mm(mint64_t op0, muint64_t op1) {
  return __builtin_riscv_msmulsu_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t msmulsu_h_mm(mint16_t op0, muint16_t op1) {
  return __builtin_riscv_msmulsu_h_mm_i16m1(op0, op1);
}
__rvm_overloaded mint8_t msmulsu_mm(mint8_t op0, muint8_t op1) {
  return __builtin_riscv_msmulsu_mm_i8m1(op0, op1);
}
__rvm_overloaded mint16_t msmulsu_mm(mint16_t op0, muint16_t op1) {
  return __builtin_riscv_msmulsu_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t msmulsu_mm(mint32_t op0, muint32_t op1) {
  return __builtin_riscv_msmulsu_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t msmulsu_mm(mint64_t op0, muint64_t op1) {
  return __builtin_riscv_msmulsu_mm_i64m1(op0, op1);
}
__rvm_overloaded mint32_t msmulsu_w_mm(mint32_t op0, muint32_t op1) {
  return __builtin_riscv_msmulsu_w_mm_i32m1(op0, op1);
}
__rvm_overloaded muint8_t msmulu_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msmulu_b_mm_u8m1(op0, op1);
}
__rvm_overloaded muint64_t msmulu_dw_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msmulu_dw_mm_u64m1(op0, op1);
}
__rvm_overloaded muint16_t msmulu_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msmulu_h_mm_u16m1(op0, op1);
}
__rvm_overloaded muint8_t msmulu_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msmulu_mm_u8m1(op0, op1);
}
__rvm_overloaded muint16_t msmulu_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msmulu_mm_u16m1(op0, op1);
}
__rvm_overloaded muint32_t msmulu_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msmulu_mm_u32m1(op0, op1);
}
__rvm_overloaded muint64_t msmulu_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msmulu_mm_u64m1(op0, op1);
}
__rvm_overloaded muint32_t msmulu_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msmulu_w_mm_u32m1(op0, op1);
}
__rvm_overloaded mint32_t msqma_b_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_msqma_b_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t msqma_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_msqma_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t msqma_spa_b_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_msqma_spa_b_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t msqma_spa_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_msqma_spa_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t msqma_spb_b_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_msqma_spb_b_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t msqma_spb_mm(mint32_t op0, mint8_t op1, mint8_t op2) {
  return __builtin_riscv_msqma_spb_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msqmau_b_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_msqmau_b_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msqmau_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_msqmau_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msqmau_spa_b_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_msqmau_spa_b_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msqmau_spa_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_msqmau_spa_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msqmau_spb_b_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_msqmau_spb_b_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t msqmau_spb_mm(muint32_t op0, muint8_t op1, muint8_t op2) {
  return __builtin_riscv_msqmau_spb_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded mint8_t mssub_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mssub_b_mm_i8m1(op0, op1);
}
__rvm_overloaded mint64_t mssub_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mssub_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t mssub_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mssub_h_mm_i16m1(op0, op1);
}
__rvm_overloaded mint8_t mssub_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mssub_mm_i8m1(op0, op1);
}
__rvm_overloaded mint16_t mssub_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mssub_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mssub_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mssub_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t mssub_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_mssub_mm_i64m1(op0, op1);
}
__rvm_overloaded mint32_t mssub_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mssub_w_mm_i32m1(op0, op1);
}
__rvm_overloaded mint8_t mssubu_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mssubu_b_mm_u8m1(op0, op1);
}
__rvm_overloaded mint64_t mssubu_dw_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_mssubu_dw_mm_u64m1(op0, op1);
}
__rvm_overloaded mint16_t mssubu_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mssubu_h_mm_u16m1(op0, op1);
}
__rvm_overloaded mint8_t mssubu_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mssubu_mm_u8m1(op0, op1);
}
__rvm_overloaded mint16_t mssubu_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mssubu_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mssubu_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mssubu_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mssubu_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_mssubu_mm_u64m1(op0, op1);
}
__rvm_overloaded mint32_t mssubu_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mssubu_w_mm_u32m1(op0, op1);
}
__rvm_overloaded mint8_t msub_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msub_b_mm_i8m1(op0, op1);
}
__rvm_overloaded mint64_t msub_dw_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msub_dw_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t msub_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msub_h_mm_i16m1(op0, op1);
}
__rvm_overloaded mint8_t msub_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_msub_mm_i8m1(op0, op1);
}
__rvm_overloaded mint16_t msub_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_msub_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t msub_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msub_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t msub_mm(mint64_t op0, mint64_t op1) {
  return __builtin_riscv_msub_mm_i64m1(op0, op1);
}
__rvm_overloaded mint32_t msub_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_msub_w_mm_i32m1(op0, op1);
}
__rvm_overloaded mint8_t msubu_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msubu_b_mm_u8m1(op0, op1);
}
__rvm_overloaded mint64_t msubu_dw_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msubu_dw_mm_u64m1(op0, op1);
}
__rvm_overloaded mint16_t msubu_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msubu_h_mm_u16m1(op0, op1);
}
__rvm_overloaded mint8_t msubu_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_msubu_mm_u8m1(op0, op1);
}
__rvm_overloaded mint16_t msubu_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_msubu_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t msubu_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msubu_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t msubu_mm(muint64_t op0, muint64_t op1) {
  return __builtin_riscv_msubu_mm_u64m1(op0, op1);
}
__rvm_overloaded mint32_t msubu_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_msubu_w_mm_u32m1(op0, op1);
}
__rvm_overloaded mint32_t mswma_h_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mswma_h_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mswma_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mswma_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mswma_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mswma_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mswma_spa_h_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mswma_spa_h_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mswma_spa_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mswma_spa_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mswma_spa_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mswma_spa_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mswma_spa_w_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mswma_spa_w_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mswma_spb_h_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mswma_spb_h_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mswma_spb_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mswma_spb_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mswma_spb_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mswma_spb_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mswma_spb_w_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mswma_spb_w_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mswma_w_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mswma_w_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mswmau_h_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mswmau_h_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mswmau_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mswmau_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mswmau_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mswmau_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mswmau_spa_h_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mswmau_spa_h_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mswmau_spa_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mswmau_spa_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mswmau_spa_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mswmau_spa_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mswmau_spa_w_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mswmau_spa_w_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mswmau_spb_h_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mswmau_spb_h_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mswmau_spb_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mswmau_spb_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mswmau_spb_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mswmau_spb_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mswmau_spb_w_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mswmau_spb_w_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mswmau_w_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mswmau_w_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded size_t munsetfp16(size_t op0) {
  return __builtin_riscv_munsetfp16(op0);
}
__rvm_overloaded size_t munsetfp32(size_t op0) {
  return __builtin_riscv_munsetfp32(op0);
}
__rvm_overloaded size_t munsetfp64(size_t op0) {
  return __builtin_riscv_munsetfp64(op0);
}
__rvm_overloaded size_t munsetint8(size_t op0) {
  return __builtin_riscv_munsetint8(op0);
}
__rvm_overloaded size_t munsetint16(size_t op0) {
  return __builtin_riscv_munsetint16(op0);
}
__rvm_overloaded size_t munsetint32(size_t op0) {
  return __builtin_riscv_munsetint32(op0);
}
__rvm_overloaded size_t munsetint64(size_t op0) {
  return __builtin_riscv_munsetint64(op0);
}
__rvm_overloaded mint16_t mwadd_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mwadd_b_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mwadd_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mwadd_h_mm_i32m1(op0, op1);
}
__rvm_overloaded mint16_t mwadd_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mwadd_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mwadd_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mwadd_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t mwadd_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mwadd_mm_i64m1(op0, op1);
}
__rvm_overloaded mint64_t mwadd_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mwadd_w_mm_i64m1(op0, op1);
}
__rvm_overloaded muint16_t mwaddu_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mwaddu_b_mm_u16m1(op0, op1);
}
__rvm_overloaded muint32_t mwaddu_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mwaddu_h_mm_u32m1(op0, op1);
}
__rvm_overloaded muint16_t mwaddu_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mwaddu_mm_u16m1(op0, op1);
}
__rvm_overloaded muint32_t mwaddu_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mwaddu_mm_u32m1(op0, op1);
}
__rvm_overloaded muint64_t mwaddu_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mwaddu_mm_u64m1(op0, op1);
}
__rvm_overloaded muint64_t mwaddu_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mwaddu_w_mm_u64m1(op0, op1);
}
__rvm_overloaded mint64_t mwcvt_dw_w_m(mint32_t op0) {
  return __builtin_riscv_mwcvt_dw_w_m_i64m1(op0);
}
__rvm_overloaded mint16_t mwcvt_h_b_m(mint8_t op0) {
  return __builtin_riscv_mwcvt_h_b_m_i16m1(op0);
}
__rvm_overloaded mint32_t mwcvt_w_b_m(mint8_t op0) {
  return __builtin_riscv_mwcvt_w_b_m_i32m1(op0);
}
__rvm_overloaded mint32_t mwcvt_w_h_m(mint16_t op0) {
  return __builtin_riscv_mwcvt_w_h_m_i32m1(op0);
}
__rvm_overloaded mint32_t mwcvt_xq_x_m(mint8_t op0) {
  return __builtin_riscv_mwcvt_xq_x_m_i32m1(op0);
}
__rvm_overloaded mint16_t mwcvt_xw_x_m(mint8_t op0) {
  return __builtin_riscv_mwcvt_xw_x_m_i16m1(op0);
}
__rvm_overloaded mint32_t mwcvt_xw_x_m(mint16_t op0) {
  return __builtin_riscv_mwcvt_xw_x_m_i32m1(op0);
}
__rvm_overloaded mint64_t mwcvt_xw_x_m(mint32_t op0) {
  return __builtin_riscv_mwcvt_xw_x_m_i64m1(op0);
}
__rvm_overloaded muint64_t mwcvtu_dw_w_m(muint32_t op0) {
  return __builtin_riscv_mwcvtu_dw_w_m_u64m1(op0);
}
__rvm_overloaded muint16_t mwcvtu_h_b_m(muint8_t op0) {
  return __builtin_riscv_mwcvtu_h_b_m_u16m1(op0);
}
__rvm_overloaded muint32_t mwcvtu_w_b_m(muint8_t op0) {
  return __builtin_riscv_mwcvtu_w_b_m_u32m1(op0);
}
__rvm_overloaded muint32_t mwcvtu_w_h_m(muint16_t op0) {
  return __builtin_riscv_mwcvtu_w_h_m_u32m1(op0);
}
__rvm_overloaded muint32_t mwcvtu_xq_x_m(muint8_t op0) {
  return __builtin_riscv_mwcvtu_xq_x_m_u32m1(op0);
}
__rvm_overloaded muint16_t mwcvtu_xw_x_m(muint8_t op0) {
  return __builtin_riscv_mwcvtu_xw_x_m_u16m1(op0);
}
__rvm_overloaded muint32_t mwcvtu_xw_x_m(muint16_t op0) {
  return __builtin_riscv_mwcvtu_xw_x_m_u32m1(op0);
}
__rvm_overloaded muint64_t mwcvtu_xw_x_m(muint32_t op0) {
  return __builtin_riscv_mwcvtu_xw_x_m_u64m1(op0);
}
__rvm_overloaded mint32_t mwma_h_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mwma_h_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mwma_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mwma_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mwma_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mwma_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mwma_spa_h_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mwma_spa_h_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mwma_spa_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mwma_spa_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mwma_spa_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mwma_spa_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mwma_spa_w_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mwma_spa_w_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mwma_spb_h_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mwma_spb_h_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint32_t mwma_spb_mm(mint32_t op0, mint16_t op1, mint16_t op2) {
  return __builtin_riscv_mwma_spb_mm_i32m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mwma_spb_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mwma_spb_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mwma_spb_w_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mwma_spb_w_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded mint64_t mwma_w_mm(mint64_t op0, mint32_t op1, mint32_t op2) {
  return __builtin_riscv_mwma_w_mm_i64m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mwmau_h_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mwmau_h_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mwmau_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mwmau_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mwmau_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mwmau_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mwmau_spa_h_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mwmau_spa_h_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mwmau_spa_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mwmau_spa_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mwmau_spa_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mwmau_spa_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mwmau_spa_w_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mwmau_spa_w_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mwmau_spb_h_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mwmau_spb_h_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint32_t mwmau_spb_mm(muint32_t op0, muint16_t op1, muint16_t op2) {
  return __builtin_riscv_mwmau_spb_mm_u32m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mwmau_spb_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mwmau_spb_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mwmau_spb_w_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mwmau_spb_w_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded muint64_t mwmau_w_mm(muint64_t op0, muint32_t op1, muint32_t op2) {
  return __builtin_riscv_mwmau_w_mm_u64m1(op0, op1, op2);
}
__rvm_overloaded mint16_t mwmul_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mwmul_b_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mwmul_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mwmul_h_mm_i32m1(op0, op1);
}
__rvm_overloaded mint16_t mwmul_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mwmul_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mwmul_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mwmul_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t mwmul_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mwmul_mm_i64m1(op0, op1);
}
__rvm_overloaded mint64_t mwmul_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mwmul_w_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t mwmulsu_b_mm(mint8_t op0, muint8_t op1) {
  return __builtin_riscv_mwmulsu_b_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mwmulsu_h_mm(mint16_t op0, muint16_t op1) {
  return __builtin_riscv_mwmulsu_h_mm_i32m1(op0, op1);
}
__rvm_overloaded mint16_t mwmulsu_mm(mint8_t op0, muint8_t op1) {
  return __builtin_riscv_mwmulsu_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mwmulsu_mm(mint16_t op0, muint16_t op1) {
  return __builtin_riscv_mwmulsu_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t mwmulsu_mm(mint32_t op0, muint32_t op1) {
  return __builtin_riscv_mwmulsu_mm_i64m1(op0, op1);
}
__rvm_overloaded mint64_t mwmulsu_w_mm(mint32_t op0, muint32_t op1) {
  return __builtin_riscv_mwmulsu_w_mm_i64m1(op0, op1);
}
__rvm_overloaded muint16_t mwmulu_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mwmulu_b_mm_u16m1(op0, op1);
}
__rvm_overloaded muint32_t mwmulu_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mwmulu_h_mm_u32m1(op0, op1);
}
__rvm_overloaded muint16_t mwmulu_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mwmulu_mm_u16m1(op0, op1);
}
__rvm_overloaded muint32_t mwmulu_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mwmulu_mm_u32m1(op0, op1);
}
__rvm_overloaded muint64_t mwmulu_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mwmulu_mm_u64m1(op0, op1);
}
__rvm_overloaded muint64_t mwmulu_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mwmulu_w_mm_u64m1(op0, op1);
}
__rvm_overloaded mint16_t mwsub_b_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mwsub_b_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mwsub_h_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mwsub_h_mm_i32m1(op0, op1);
}
__rvm_overloaded mint16_t mwsub_mm(mint8_t op0, mint8_t op1) {
  return __builtin_riscv_mwsub_mm_i16m1(op0, op1);
}
__rvm_overloaded mint32_t mwsub_mm(mint16_t op0, mint16_t op1) {
  return __builtin_riscv_mwsub_mm_i32m1(op0, op1);
}
__rvm_overloaded mint64_t mwsub_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mwsub_mm_i64m1(op0, op1);
}
__rvm_overloaded mint64_t mwsub_w_mm(mint32_t op0, mint32_t op1) {
  return __builtin_riscv_mwsub_w_mm_i64m1(op0, op1);
}
__rvm_overloaded mint16_t mwsubu_b_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mwsubu_b_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mwsubu_h_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mwsubu_h_mm_u32m1(op0, op1);
}
__rvm_overloaded mint16_t mwsubu_mm(muint8_t op0, muint8_t op1) {
  return __builtin_riscv_mwsubu_mm_u16m1(op0, op1);
}
__rvm_overloaded mint32_t mwsubu_mm(muint16_t op0, muint16_t op1) {
  return __builtin_riscv_mwsubu_mm_u32m1(op0, op1);
}
__rvm_overloaded mint64_t mwsubu_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mwsubu_mm_u64m1(op0, op1);
}
__rvm_overloaded mint64_t mwsubu_w_mm(muint32_t op0, muint32_t op1) {
  return __builtin_riscv_mwsubu_w_mm_u64m1(op0, op1);
}
#endif // __RISCV_XXIANGSHANAME_H
