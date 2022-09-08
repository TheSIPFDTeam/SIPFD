/* Autogenerated: word_by_word_montgomery fp100 64 0x98256f1482164ffffffffffff */
/* curve description: fp100 */
/* machine_wordsize = 64 (from "64") */
/* requested operations: (all) */
/* m = 0x98256f1482164ffffffffffff (from "0x98256f1482164ffffffffffff") */
/*                                                                    */
/* NOTE: In addition to the bounds specified above each function, all */
/*   functions synthesized for this Montgomery arithmetic require the */
/*   input to be strictly less than the prime modulus (m), and also   */
/*   require the input to be in the unique saturated representation.  */
/*   All functions also ensure that these two properties are true of  */
/*   return values.                                                   */
/*  */
/* Computed values: */
/*   eval z = z[0] + (z[1] << 64) */
/*   bytes_eval z = z[0] + (z[1] << 8) + (z[2] << 16) + (z[3] << 24) + (z[4] << 32) + (z[5] << 40) + (z[6] << 48) + (z[7] << 56) + (z[8] << 64) + (z[9] << 72) + (z[10] << 80) + (z[11] << 88) + (z[12] << 96) */
/*   twos_complement_eval z = let x1 := z[0] + (z[1] << 64) in */
/*                            if x1 & (2^128-1) < 2^127 then x1 & (2^128-1) else (x1 & (2^128-1)) - 2^128 */

#include <stdint.h>
typedef unsigned char fiat_fp100_uint1;
typedef signed char fiat_fp100_int1;
#if defined(__GNUC__) || defined(__clang__)
#  define FIAT_FP100_FIAT_EXTENSION __extension__
#  define FIAT_FP100_FIAT_INLINE __inline__
#else
#  define FIAT_FP100_FIAT_EXTENSION
#  define FIAT_FP100_FIAT_INLINE
#endif

FIAT_FP100_FIAT_EXTENSION typedef signed __int128 fiat_fp100_int128;
FIAT_FP100_FIAT_EXTENSION typedef unsigned __int128 fiat_fp100_uint128;

/* The type fiat_fp100_montgomery_domain_field_element is a field element in the Montgomery domain. */
/* Bounds: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]] */
typedef uint64_t fiat_fp100_montgomery_domain_field_element[2];

/* The type fiat_fp100_non_montgomery_domain_field_element is a field element NOT in the Montgomery domain. */
/* Bounds: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]] */
typedef uint64_t fiat_fp100_non_montgomery_domain_field_element[2];

#if (-1 & 3) != 3
#error "This code only works on a two's complement system"
#endif


/*
 * The function fiat_fp100_addcarryx_u64 is an addition with carry.
 *
 * Postconditions:
 *   out1 = (arg1 + arg2 + arg3) mod 2^64
 *   out2 = ⌊(arg1 + arg2 + arg3) / 2^64⌋
 *
 * Input Bounds:
 *   arg1: [0x0 ~> 0x1]
 *   arg2: [0x0 ~> 0xffffffffffffffff]
 *   arg3: [0x0 ~> 0xffffffffffffffff]
 * Output Bounds:
 *   out1: [0x0 ~> 0xffffffffffffffff]
 *   out2: [0x0 ~> 0x1]
 */
void fiat_fp100_addcarryx_u64(uint64_t* out1, fiat_fp100_uint1* out2, fiat_fp100_uint1 arg1, uint64_t arg2, uint64_t arg3) {
  fiat_fp100_uint128 x1;
  uint64_t x2;
  fiat_fp100_uint1 x3;
  x1 = ((arg1 + (fiat_fp100_uint128)arg2) + arg3);
  x2 = (uint64_t)(x1 & UINT64_C(0xffffffffffffffff));
  x3 = (fiat_fp100_uint1)(x1 >> 64);
  *out1 = x2;
  *out2 = x3;
}

/*
 * The function fiat_fp100_subborrowx_u64 is a subtraction with borrow.
 *
 * Postconditions:
 *   out1 = (-arg1 + arg2 + -arg3) mod 2^64
 *   out2 = -⌊(-arg1 + arg2 + -arg3) / 2^64⌋
 *
 * Input Bounds:
 *   arg1: [0x0 ~> 0x1]
 *   arg2: [0x0 ~> 0xffffffffffffffff]
 *   arg3: [0x0 ~> 0xffffffffffffffff]
 * Output Bounds:
 *   out1: [0x0 ~> 0xffffffffffffffff]
 *   out2: [0x0 ~> 0x1]
 */
void fiat_fp100_subborrowx_u64(uint64_t* out1, fiat_fp100_uint1* out2, fiat_fp100_uint1 arg1, uint64_t arg2, uint64_t arg3) {
  fiat_fp100_int128 x1;
  fiat_fp100_int1 x2;
  uint64_t x3;
  x1 = ((arg2 - (fiat_fp100_int128)arg1) - arg3);
  x2 = (fiat_fp100_int1)(x1 >> 64);
  x3 = (uint64_t)(x1 & UINT64_C(0xffffffffffffffff));
  *out1 = x3;
  *out2 = (fiat_fp100_uint1)(0x0 - x2);
}

/*
 * The function fiat_fp100_mulx_u64 is a multiplication, returning the full double-width result.
 *
 * Postconditions:
 *   out1 = (arg1 * arg2) mod 2^64
 *   out2 = ⌊arg1 * arg2 / 2^64⌋
 *
 * Input Bounds:
 *   arg1: [0x0 ~> 0xffffffffffffffff]
 *   arg2: [0x0 ~> 0xffffffffffffffff]
 * Output Bounds:
 *   out1: [0x0 ~> 0xffffffffffffffff]
 *   out2: [0x0 ~> 0xffffffffffffffff]
 */
void fiat_fp100_mulx_u64(uint64_t* out1, uint64_t* out2, uint64_t arg1, uint64_t arg2) {
  fiat_fp100_uint128 x1;
  uint64_t x2;
  uint64_t x3;
  x1 = ((fiat_fp100_uint128)arg1 * arg2);
  x2 = (uint64_t)(x1 & UINT64_C(0xffffffffffffffff));
  x3 = (uint64_t)(x1 >> 64);
  *out1 = x2;
  *out2 = x3;
}

/*
 * The function fiat_fp100_cmovznz_u64 is a single-word conditional move.
 *
 * Postconditions:
 *   out1 = (if arg1 = 0 then arg2 else arg3)
 *
 * Input Bounds:
 *   arg1: [0x0 ~> 0x1]
 *   arg2: [0x0 ~> 0xffffffffffffffff]
 *   arg3: [0x0 ~> 0xffffffffffffffff]
 * Output Bounds:
 *   out1: [0x0 ~> 0xffffffffffffffff]
 */
void fiat_fp100_cmovznz_u64(uint64_t* out1, fiat_fp100_uint1 arg1, uint64_t arg2, uint64_t arg3) {
  fiat_fp100_uint1 x1;
  uint64_t x2;
  uint64_t x3;
  x1 = (!(!arg1));
  x2 = ((fiat_fp100_int1)(0x0 - x1) & UINT64_C(0xffffffffffffffff));
  x3 = ((x2 & arg3) | ((~x2) & arg2));
  *out1 = x3;
}

/*
 * The function fiat_fp100_mul multiplies two field elements in the Montgomery domain.
 *
 * Preconditions:
 *   0 ≤ eval arg1 < m
 *   0 ≤ eval arg2 < m
 * Postconditions:
 *   eval (from_montgomery out1) mod m = (eval (from_montgomery arg1) * eval (from_montgomery arg2)) mod m
 *   0 ≤ eval out1 < m
 *
 */
void fiat_fp100_mul(fiat_fp100_montgomery_domain_field_element out1, const fiat_fp100_montgomery_domain_field_element arg1, const fiat_fp100_montgomery_domain_field_element arg2) {
  uint64_t x1;
  uint64_t x2;
  uint64_t x3;
  uint64_t x4;
  uint64_t x5;
  uint64_t x6;
  uint64_t x7;
  fiat_fp100_uint1 x8;
  uint64_t x9;
  uint64_t x10;
  uint64_t x11;
  uint64_t x12;
  uint64_t x13;
  uint64_t x14;
  uint64_t x15;
  uint64_t x16;
  fiat_fp100_uint1 x17;
  uint64_t x18;
  uint64_t x19;
  fiat_fp100_uint1 x20;
  uint64_t x21;
  fiat_fp100_uint1 x22;
  uint64_t x23;
  fiat_fp100_uint1 x24;
  uint64_t x25;
  uint64_t x26;
  uint64_t x27;
  uint64_t x28;
  uint64_t x29;
  fiat_fp100_uint1 x30;
  uint64_t x31;
  uint64_t x32;
  fiat_fp100_uint1 x33;
  uint64_t x34;
  fiat_fp100_uint1 x35;
  uint64_t x36;
  fiat_fp100_uint1 x37;
  uint64_t x38;
  uint64_t x39;
  uint64_t x40;
  uint64_t x41;
  uint64_t x42;
  uint64_t x43;
  uint64_t x44;
  fiat_fp100_uint1 x45;
  uint64_t x46;
  uint64_t x47;
  fiat_fp100_uint1 x48;
  uint64_t x49;
  fiat_fp100_uint1 x50;
  uint64_t x51;
  fiat_fp100_uint1 x52;
  uint64_t x53;
  uint64_t x54;
  fiat_fp100_uint1 x55;
  uint64_t x56;
  fiat_fp100_uint1 x57;
  uint64_t x58;
  fiat_fp100_uint1 x59;
  uint64_t x60;
  uint64_t x61;
  x1 = (arg1[1]);
  x2 = (arg1[0]);
  fiat_fp100_mulx_u64(&x3, &x4, x2, (arg2[1]));
  fiat_fp100_mulx_u64(&x5, &x6, x2, (arg2[0]));
  fiat_fp100_addcarryx_u64(&x7, &x8, 0x0, x6, x3);
  x9 = (x8 + x4);
  fiat_fp100_mulx_u64(&x10, &x11, x5, UINT64_C(0x2165000000000001));
  fiat_fp100_mulx_u64(&x12, &x13, x10, UINT64_C(0x98256f148));
  fiat_fp100_mulx_u64(&x14, &x15, x10, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_addcarryx_u64(&x16, &x17, 0x0, x15, x12);
  x18 = (x17 + x13);
  fiat_fp100_addcarryx_u64(&x19, &x20, 0x0, x5, x14);
  fiat_fp100_addcarryx_u64(&x21, &x22, x20, x7, x16);
  fiat_fp100_addcarryx_u64(&x23, &x24, x22, x9, x18);
  fiat_fp100_mulx_u64(&x25, &x26, x1, (arg2[1]));
  fiat_fp100_mulx_u64(&x27, &x28, x1, (arg2[0]));
  fiat_fp100_addcarryx_u64(&x29, &x30, 0x0, x28, x25);
  x31 = (x30 + x26);
  fiat_fp100_addcarryx_u64(&x32, &x33, 0x0, x21, x27);
  fiat_fp100_addcarryx_u64(&x34, &x35, x33, x23, x29);
  fiat_fp100_addcarryx_u64(&x36, &x37, x35, x24, x31);
  fiat_fp100_mulx_u64(&x38, &x39, x32, UINT64_C(0x2165000000000001));
  fiat_fp100_mulx_u64(&x40, &x41, x38, UINT64_C(0x98256f148));
  fiat_fp100_mulx_u64(&x42, &x43, x38, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_addcarryx_u64(&x44, &x45, 0x0, x43, x40);
  x46 = (x45 + x41);
  fiat_fp100_addcarryx_u64(&x47, &x48, 0x0, x32, x42);
  fiat_fp100_addcarryx_u64(&x49, &x50, x48, x34, x44);
  fiat_fp100_addcarryx_u64(&x51, &x52, x50, x36, x46);
  x53 = ((uint64_t)x52 + x37);
  fiat_fp100_subborrowx_u64(&x54, &x55, 0x0, x49, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_subborrowx_u64(&x56, &x57, x55, x51, UINT64_C(0x98256f148));
  fiat_fp100_subborrowx_u64(&x58, &x59, x57, x53, 0x0);
  fiat_fp100_cmovznz_u64(&x60, x59, x54, x49);
  fiat_fp100_cmovznz_u64(&x61, x59, x56, x51);
  out1[0] = x60;
  out1[1] = x61;
}

/*
 * The function fiat_fp100_square squares a field element in the Montgomery domain.
 *
 * Preconditions:
 *   0 ≤ eval arg1 < m
 * Postconditions:
 *   eval (from_montgomery out1) mod m = (eval (from_montgomery arg1) * eval (from_montgomery arg1)) mod m
 *   0 ≤ eval out1 < m
 *
 */
void fiat_fp100_square(fiat_fp100_montgomery_domain_field_element out1, const fiat_fp100_montgomery_domain_field_element arg1) {
  uint64_t x1;
  uint64_t x2;
  uint64_t x3;
  uint64_t x4;
  uint64_t x5;
  uint64_t x6;
  uint64_t x7;
  fiat_fp100_uint1 x8;
  uint64_t x9;
  uint64_t x10;
  uint64_t x11;
  uint64_t x12;
  uint64_t x13;
  uint64_t x14;
  uint64_t x15;
  uint64_t x16;
  fiat_fp100_uint1 x17;
  uint64_t x18;
  uint64_t x19;
  fiat_fp100_uint1 x20;
  uint64_t x21;
  fiat_fp100_uint1 x22;
  uint64_t x23;
  fiat_fp100_uint1 x24;
  uint64_t x25;
  uint64_t x26;
  uint64_t x27;
  uint64_t x28;
  uint64_t x29;
  fiat_fp100_uint1 x30;
  uint64_t x31;
  uint64_t x32;
  fiat_fp100_uint1 x33;
  uint64_t x34;
  fiat_fp100_uint1 x35;
  uint64_t x36;
  fiat_fp100_uint1 x37;
  uint64_t x38;
  uint64_t x39;
  uint64_t x40;
  uint64_t x41;
  uint64_t x42;
  uint64_t x43;
  uint64_t x44;
  fiat_fp100_uint1 x45;
  uint64_t x46;
  uint64_t x47;
  fiat_fp100_uint1 x48;
  uint64_t x49;
  fiat_fp100_uint1 x50;
  uint64_t x51;
  fiat_fp100_uint1 x52;
  uint64_t x53;
  uint64_t x54;
  fiat_fp100_uint1 x55;
  uint64_t x56;
  fiat_fp100_uint1 x57;
  uint64_t x58;
  fiat_fp100_uint1 x59;
  uint64_t x60;
  uint64_t x61;
  x1 = (arg1[1]);
  x2 = (arg1[0]);
  fiat_fp100_mulx_u64(&x3, &x4, x2, (arg1[1]));
  fiat_fp100_mulx_u64(&x5, &x6, x2, (arg1[0]));
  fiat_fp100_addcarryx_u64(&x7, &x8, 0x0, x6, x3);
  x9 = (x8 + x4);
  fiat_fp100_mulx_u64(&x10, &x11, x5, UINT64_C(0x2165000000000001));
  fiat_fp100_mulx_u64(&x12, &x13, x10, UINT64_C(0x98256f148));
  fiat_fp100_mulx_u64(&x14, &x15, x10, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_addcarryx_u64(&x16, &x17, 0x0, x15, x12);
  x18 = (x17 + x13);
  fiat_fp100_addcarryx_u64(&x19, &x20, 0x0, x5, x14);
  fiat_fp100_addcarryx_u64(&x21, &x22, x20, x7, x16);
  fiat_fp100_addcarryx_u64(&x23, &x24, x22, x9, x18);
  fiat_fp100_mulx_u64(&x25, &x26, x1, (arg1[1]));
  fiat_fp100_mulx_u64(&x27, &x28, x1, (arg1[0]));
  fiat_fp100_addcarryx_u64(&x29, &x30, 0x0, x28, x25);
  x31 = (x30 + x26);
  fiat_fp100_addcarryx_u64(&x32, &x33, 0x0, x21, x27);
  fiat_fp100_addcarryx_u64(&x34, &x35, x33, x23, x29);
  fiat_fp100_addcarryx_u64(&x36, &x37, x35, x24, x31);
  fiat_fp100_mulx_u64(&x38, &x39, x32, UINT64_C(0x2165000000000001));
  fiat_fp100_mulx_u64(&x40, &x41, x38, UINT64_C(0x98256f148));
  fiat_fp100_mulx_u64(&x42, &x43, x38, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_addcarryx_u64(&x44, &x45, 0x0, x43, x40);
  x46 = (x45 + x41);
  fiat_fp100_addcarryx_u64(&x47, &x48, 0x0, x32, x42);
  fiat_fp100_addcarryx_u64(&x49, &x50, x48, x34, x44);
  fiat_fp100_addcarryx_u64(&x51, &x52, x50, x36, x46);
  x53 = ((uint64_t)x52 + x37);
  fiat_fp100_subborrowx_u64(&x54, &x55, 0x0, x49, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_subborrowx_u64(&x56, &x57, x55, x51, UINT64_C(0x98256f148));
  fiat_fp100_subborrowx_u64(&x58, &x59, x57, x53, 0x0);
  fiat_fp100_cmovznz_u64(&x60, x59, x54, x49);
  fiat_fp100_cmovznz_u64(&x61, x59, x56, x51);
  out1[0] = x60;
  out1[1] = x61;
}

/*
 * The function fiat_fp100_add adds two field elements in the Montgomery domain.
 *
 * Preconditions:
 *   0 ≤ eval arg1 < m
 *   0 ≤ eval arg2 < m
 * Postconditions:
 *   eval (from_montgomery out1) mod m = (eval (from_montgomery arg1) + eval (from_montgomery arg2)) mod m
 *   0 ≤ eval out1 < m
 *
 */
void fiat_fp100_add(fiat_fp100_montgomery_domain_field_element out1, const fiat_fp100_montgomery_domain_field_element arg1, const fiat_fp100_montgomery_domain_field_element arg2) {
  uint64_t x1;
  fiat_fp100_uint1 x2;
  uint64_t x3;
  fiat_fp100_uint1 x4;
  uint64_t x5;
  fiat_fp100_uint1 x6;
  uint64_t x7;
  fiat_fp100_uint1 x8;
  uint64_t x9;
  fiat_fp100_uint1 x10;
  uint64_t x11;
  uint64_t x12;
  fiat_fp100_addcarryx_u64(&x1, &x2, 0x0, (arg1[0]), (arg2[0]));
  fiat_fp100_addcarryx_u64(&x3, &x4, x2, (arg1[1]), (arg2[1]));
  fiat_fp100_subborrowx_u64(&x5, &x6, 0x0, x1, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_subborrowx_u64(&x7, &x8, x6, x3, UINT64_C(0x98256f148));
  fiat_fp100_subborrowx_u64(&x9, &x10, x8, x4, 0x0);
  fiat_fp100_cmovznz_u64(&x11, x10, x5, x1);
  fiat_fp100_cmovznz_u64(&x12, x10, x7, x3);
  out1[0] = x11;
  out1[1] = x12;
}

/*
 * The function fiat_fp100_sub subtracts two field elements in the Montgomery domain.
 *
 * Preconditions:
 *   0 ≤ eval arg1 < m
 *   0 ≤ eval arg2 < m
 * Postconditions:
 *   eval (from_montgomery out1) mod m = (eval (from_montgomery arg1) - eval (from_montgomery arg2)) mod m
 *   0 ≤ eval out1 < m
 *
 */
void fiat_fp100_sub(fiat_fp100_montgomery_domain_field_element out1, const fiat_fp100_montgomery_domain_field_element arg1, const fiat_fp100_montgomery_domain_field_element arg2) {
  uint64_t x1;
  fiat_fp100_uint1 x2;
  uint64_t x3;
  fiat_fp100_uint1 x4;
  uint64_t x5;
  uint64_t x6;
  fiat_fp100_uint1 x7;
  uint64_t x8;
  fiat_fp100_uint1 x9;
  fiat_fp100_subborrowx_u64(&x1, &x2, 0x0, (arg1[0]), (arg2[0]));
  fiat_fp100_subborrowx_u64(&x3, &x4, x2, (arg1[1]), (arg2[1]));
  fiat_fp100_cmovznz_u64(&x5, x4, 0x0, UINT64_C(0xffffffffffffffff));
  fiat_fp100_addcarryx_u64(&x6, &x7, 0x0, x1, (x5 & UINT64_C(0x2164ffffffffffff)));
  fiat_fp100_addcarryx_u64(&x8, &x9, x7, x3, (x5 & UINT64_C(0x98256f148)));
  out1[0] = x6;
  out1[1] = x8;
}

/*
 * The function fiat_fp100_opp negates a field element in the Montgomery domain.
 *
 * Preconditions:
 *   0 ≤ eval arg1 < m
 * Postconditions:
 *   eval (from_montgomery out1) mod m = -eval (from_montgomery arg1) mod m
 *   0 ≤ eval out1 < m
 *
 */
void fiat_fp100_opp(fiat_fp100_montgomery_domain_field_element out1, const fiat_fp100_montgomery_domain_field_element arg1) {
  uint64_t x1;
  fiat_fp100_uint1 x2;
  uint64_t x3;
  fiat_fp100_uint1 x4;
  uint64_t x5;
  uint64_t x6;
  fiat_fp100_uint1 x7;
  uint64_t x8;
  fiat_fp100_uint1 x9;
  fiat_fp100_subborrowx_u64(&x1, &x2, 0x0, 0x0, (arg1[0]));
  fiat_fp100_subborrowx_u64(&x3, &x4, x2, 0x0, (arg1[1]));
  fiat_fp100_cmovznz_u64(&x5, x4, 0x0, UINT64_C(0xffffffffffffffff));
  fiat_fp100_addcarryx_u64(&x6, &x7, 0x0, x1, (x5 & UINT64_C(0x2164ffffffffffff)));
  fiat_fp100_addcarryx_u64(&x8, &x9, x7, x3, (x5 & UINT64_C(0x98256f148)));
  out1[0] = x6;
  out1[1] = x8;
}

/*
 * The function fiat_fp100_from_montgomery translates a field element out of the Montgomery domain.
 *
 * Preconditions:
 *   0 ≤ eval arg1 < m
 * Postconditions:
 *   eval out1 mod m = (eval arg1 * ((2^64)⁻¹ mod m)^2) mod m
 *   0 ≤ eval out1 < m
 *
 */
void fiat_fp100_from_montgomery(fiat_fp100_non_montgomery_domain_field_element out1, const fiat_fp100_montgomery_domain_field_element arg1) {
  uint64_t x1;
  uint64_t x2;
  uint64_t x3;
  uint64_t x4;
  uint64_t x5;
  uint64_t x6;
  uint64_t x7;
  uint64_t x8;
  fiat_fp100_uint1 x9;
  uint64_t x10;
  fiat_fp100_uint1 x11;
  uint64_t x12;
  fiat_fp100_uint1 x13;
  uint64_t x14;
  fiat_fp100_uint1 x15;
  uint64_t x16;
  uint64_t x17;
  uint64_t x18;
  uint64_t x19;
  uint64_t x20;
  uint64_t x21;
  uint64_t x22;
  fiat_fp100_uint1 x23;
  uint64_t x24;
  fiat_fp100_uint1 x25;
  uint64_t x26;
  fiat_fp100_uint1 x27;
  uint64_t x28;
  uint64_t x29;
  fiat_fp100_uint1 x30;
  uint64_t x31;
  fiat_fp100_uint1 x32;
  uint64_t x33;
  fiat_fp100_uint1 x34;
  uint64_t x35;
  uint64_t x36;
  x1 = (arg1[0]);
  fiat_fp100_mulx_u64(&x2, &x3, x1, UINT64_C(0x2165000000000001));
  fiat_fp100_mulx_u64(&x4, &x5, x2, UINT64_C(0x98256f148));
  fiat_fp100_mulx_u64(&x6, &x7, x2, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_addcarryx_u64(&x8, &x9, 0x0, x7, x4);
  fiat_fp100_addcarryx_u64(&x10, &x11, 0x0, x1, x6);
  fiat_fp100_addcarryx_u64(&x12, &x13, x11, 0x0, x8);
  fiat_fp100_addcarryx_u64(&x14, &x15, 0x0, x12, (arg1[1]));
  fiat_fp100_mulx_u64(&x16, &x17, x14, UINT64_C(0x2165000000000001));
  fiat_fp100_mulx_u64(&x18, &x19, x16, UINT64_C(0x98256f148));
  fiat_fp100_mulx_u64(&x20, &x21, x16, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_addcarryx_u64(&x22, &x23, 0x0, x21, x18);
  fiat_fp100_addcarryx_u64(&x24, &x25, 0x0, x14, x20);
  fiat_fp100_addcarryx_u64(&x26, &x27, x25, (x15 + (x13 + (x9 + x5))), x22);
  x28 = (x27 + (x23 + x19));
  fiat_fp100_subborrowx_u64(&x29, &x30, 0x0, x26, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_subborrowx_u64(&x31, &x32, x30, x28, UINT64_C(0x98256f148));
  fiat_fp100_subborrowx_u64(&x33, &x34, x32, 0x0, 0x0);
  fiat_fp100_cmovznz_u64(&x35, x34, x29, x26);
  fiat_fp100_cmovznz_u64(&x36, x34, x31, x28);
  out1[0] = x35;
  out1[1] = x36;
}

/*
 * The function fiat_fp100_to_montgomery translates a field element into the Montgomery domain.
 *
 * Preconditions:
 *   0 ≤ eval arg1 < m
 * Postconditions:
 *   eval (from_montgomery out1) mod m = eval arg1 mod m
 *   0 ≤ eval out1 < m
 *
 */
void fiat_fp100_to_montgomery(fiat_fp100_montgomery_domain_field_element out1, const fiat_fp100_non_montgomery_domain_field_element arg1) {
  uint64_t x1;
  uint64_t x2;
  uint64_t x3;
  uint64_t x4;
  uint64_t x5;
  uint64_t x6;
  uint64_t x7;
  fiat_fp100_uint1 x8;
  uint64_t x9;
  uint64_t x10;
  uint64_t x11;
  uint64_t x12;
  uint64_t x13;
  uint64_t x14;
  uint64_t x15;
  fiat_fp100_uint1 x16;
  uint64_t x17;
  fiat_fp100_uint1 x18;
  uint64_t x19;
  fiat_fp100_uint1 x20;
  uint64_t x21;
  uint64_t x22;
  uint64_t x23;
  uint64_t x24;
  uint64_t x25;
  fiat_fp100_uint1 x26;
  uint64_t x27;
  fiat_fp100_uint1 x28;
  uint64_t x29;
  fiat_fp100_uint1 x30;
  uint64_t x31;
  uint64_t x32;
  uint64_t x33;
  uint64_t x34;
  uint64_t x35;
  uint64_t x36;
  uint64_t x37;
  fiat_fp100_uint1 x38;
  uint64_t x39;
  fiat_fp100_uint1 x40;
  uint64_t x41;
  fiat_fp100_uint1 x42;
  uint64_t x43;
  uint64_t x44;
  fiat_fp100_uint1 x45;
  uint64_t x46;
  fiat_fp100_uint1 x47;
  uint64_t x48;
  fiat_fp100_uint1 x49;
  uint64_t x50;
  uint64_t x51;
  x1 = (arg1[1]);
  x2 = (arg1[0]);
  fiat_fp100_mulx_u64(&x3, &x4, x2, UINT32_C(0xe4ed25ef));
  fiat_fp100_mulx_u64(&x5, &x6, x2, UINT64_C(0xdca5f9809fe37f5d));
  fiat_fp100_addcarryx_u64(&x7, &x8, 0x0, x6, x3);
  fiat_fp100_mulx_u64(&x9, &x10, x5, UINT64_C(0x2165000000000001));
  fiat_fp100_mulx_u64(&x11, &x12, x9, UINT64_C(0x98256f148));
  fiat_fp100_mulx_u64(&x13, &x14, x9, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_addcarryx_u64(&x15, &x16, 0x0, x14, x11);
  fiat_fp100_addcarryx_u64(&x17, &x18, 0x0, x5, x13);
  fiat_fp100_addcarryx_u64(&x19, &x20, x18, x7, x15);
  fiat_fp100_mulx_u64(&x21, &x22, x1, UINT32_C(0xe4ed25ef));
  fiat_fp100_mulx_u64(&x23, &x24, x1, UINT64_C(0xdca5f9809fe37f5d));
  fiat_fp100_addcarryx_u64(&x25, &x26, 0x0, x24, x21);
  fiat_fp100_addcarryx_u64(&x27, &x28, 0x0, x19, x23);
  fiat_fp100_addcarryx_u64(&x29, &x30, x28, ((x20 + (x8 + x4)) + (x16 + x12)), x25);
  fiat_fp100_mulx_u64(&x31, &x32, x27, UINT64_C(0x2165000000000001));
  fiat_fp100_mulx_u64(&x33, &x34, x31, UINT64_C(0x98256f148));
  fiat_fp100_mulx_u64(&x35, &x36, x31, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_addcarryx_u64(&x37, &x38, 0x0, x36, x33);
  fiat_fp100_addcarryx_u64(&x39, &x40, 0x0, x27, x35);
  fiat_fp100_addcarryx_u64(&x41, &x42, x40, x29, x37);
  x43 = ((x42 + (x30 + (x26 + x22))) + (x38 + x34));
  fiat_fp100_subborrowx_u64(&x44, &x45, 0x0, x41, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_subborrowx_u64(&x46, &x47, x45, x43, UINT64_C(0x98256f148));
  fiat_fp100_subborrowx_u64(&x48, &x49, x47, 0x0, 0x0);
  fiat_fp100_cmovznz_u64(&x50, x49, x44, x41);
  fiat_fp100_cmovznz_u64(&x51, x49, x46, x43);
  out1[0] = x50;
  out1[1] = x51;
}

/*
 * The function fiat_fp100_nonzero outputs a single non-zero word if the input is non-zero and zero otherwise.
 *
 * Preconditions:
 *   0 ≤ eval arg1 < m
 * Postconditions:
 *   out1 = 0 ↔ eval (from_montgomery arg1) mod m = 0
 *
 * Input Bounds:
 *   arg1: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 * Output Bounds:
 *   out1: [0x0 ~> 0xffffffffffffffff]
 */
void fiat_fp100_nonzero(uint64_t* out1, const uint64_t arg1[2]) {
  uint64_t x1;
  x1 = ((arg1[0]) | (arg1[1]));
  *out1 = x1;
}

/*
 * The function fiat_fp100_selectznz is a multi-limb conditional select.
 *
 * Postconditions:
 *   out1 = (if arg1 = 0 then arg2 else arg3)
 *
 * Input Bounds:
 *   arg1: [0x0 ~> 0x1]
 *   arg2: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 *   arg3: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 * Output Bounds:
 *   out1: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 */
void fiat_fp100_selectznz(uint64_t out1[2], fiat_fp100_uint1 arg1, const uint64_t arg2[2], const uint64_t arg3[2]) {
  uint64_t x1;
  uint64_t x2;
  fiat_fp100_cmovznz_u64(&x1, arg1, (arg2[0]), (arg3[0]));
  fiat_fp100_cmovznz_u64(&x2, arg1, (arg2[1]), (arg3[1]));
  out1[0] = x1;
  out1[1] = x2;
}

/*
 * The function fiat_fp100_to_bytes serializes a field element NOT in the Montgomery domain to bytes in little-endian order.
 *
 * Preconditions:
 *   0 ≤ eval arg1 < m
 * Postconditions:
 *   out1 = map (λ x, ⌊((eval arg1 mod m) mod 2^(8 * (x + 1))) / 2^(8 * x)⌋) [0..12]
 *
 * Input Bounds:
 *   arg1: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xfffffffff]]
 * Output Bounds:
 *   out1: [[0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xf]]
 */
void fiat_fp100_to_bytes(uint8_t out1[13], const uint64_t arg1[2]) {
  uint64_t x1;
  uint64_t x2;
  uint8_t x3;
  uint64_t x4;
  uint8_t x5;
  uint64_t x6;
  uint8_t x7;
  uint64_t x8;
  uint8_t x9;
  uint64_t x10;
  uint8_t x11;
  uint64_t x12;
  uint8_t x13;
  uint64_t x14;
  uint8_t x15;
  uint8_t x16;
  uint8_t x17;
  uint64_t x18;
  uint8_t x19;
  uint64_t x20;
  uint8_t x21;
  uint64_t x22;
  uint8_t x23;
  uint8_t x24;
  x1 = (arg1[1]);
  x2 = (arg1[0]);
  x3 = (uint8_t)(x2 & UINT8_C(0xff));
  x4 = (x2 >> 8);
  x5 = (uint8_t)(x4 & UINT8_C(0xff));
  x6 = (x4 >> 8);
  x7 = (uint8_t)(x6 & UINT8_C(0xff));
  x8 = (x6 >> 8);
  x9 = (uint8_t)(x8 & UINT8_C(0xff));
  x10 = (x8 >> 8);
  x11 = (uint8_t)(x10 & UINT8_C(0xff));
  x12 = (x10 >> 8);
  x13 = (uint8_t)(x12 & UINT8_C(0xff));
  x14 = (x12 >> 8);
  x15 = (uint8_t)(x14 & UINT8_C(0xff));
  x16 = (uint8_t)(x14 >> 8);
  x17 = (uint8_t)(x1 & UINT8_C(0xff));
  x18 = (x1 >> 8);
  x19 = (uint8_t)(x18 & UINT8_C(0xff));
  x20 = (x18 >> 8);
  x21 = (uint8_t)(x20 & UINT8_C(0xff));
  x22 = (x20 >> 8);
  x23 = (uint8_t)(x22 & UINT8_C(0xff));
  x24 = (uint8_t)(x22 >> 8);
  out1[0] = x3;
  out1[1] = x5;
  out1[2] = x7;
  out1[3] = x9;
  out1[4] = x11;
  out1[5] = x13;
  out1[6] = x15;
  out1[7] = x16;
  out1[8] = x17;
  out1[9] = x19;
  out1[10] = x21;
  out1[11] = x23;
  out1[12] = x24;
}

/*
 * The function fiat_fp100_from_bytes deserializes a field element NOT in the Montgomery domain from bytes in little-endian order.
 *
 * Preconditions:
 *   0 ≤ bytes_eval arg1 < m
 * Postconditions:
 *   eval out1 mod m = bytes_eval arg1 mod m
 *   0 ≤ eval out1 < m
 *
 * Input Bounds:
 *   arg1: [[0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xff], [0x0 ~> 0xf]]
 * Output Bounds:
 *   out1: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xfffffffff]]
 */
void fiat_fp100_from_bytes(uint64_t out1[2], const uint8_t arg1[13]) {
  uint64_t x1;
  uint64_t x2;
  uint64_t x3;
  uint64_t x4;
  uint8_t x5;
  uint64_t x6;
  uint64_t x7;
  uint64_t x8;
  uint64_t x9;
  uint64_t x10;
  uint64_t x11;
  uint64_t x12;
  uint8_t x13;
  uint64_t x14;
  uint64_t x15;
  uint64_t x16;
  uint64_t x17;
  uint64_t x18;
  uint64_t x19;
  uint64_t x20;
  uint64_t x21;
  uint64_t x22;
  uint64_t x23;
  uint64_t x24;
  x1 = ((uint64_t)(arg1[12]) << 32);
  x2 = ((uint64_t)(arg1[11]) << 24);
  x3 = ((uint64_t)(arg1[10]) << 16);
  x4 = ((uint64_t)(arg1[9]) << 8);
  x5 = (arg1[8]);
  x6 = ((uint64_t)(arg1[7]) << 56);
  x7 = ((uint64_t)(arg1[6]) << 48);
  x8 = ((uint64_t)(arg1[5]) << 40);
  x9 = ((uint64_t)(arg1[4]) << 32);
  x10 = ((uint64_t)(arg1[3]) << 24);
  x11 = ((uint64_t)(arg1[2]) << 16);
  x12 = ((uint64_t)(arg1[1]) << 8);
  x13 = (arg1[0]);
  x14 = (x12 + (uint64_t)x13);
  x15 = (x11 + x14);
  x16 = (x10 + x15);
  x17 = (x9 + x16);
  x18 = (x8 + x17);
  x19 = (x7 + x18);
  x20 = (x6 + x19);
  x21 = (x4 + (uint64_t)x5);
  x22 = (x3 + x21);
  x23 = (x2 + x22);
  x24 = (x1 + x23);
  out1[0] = x20;
  out1[1] = x24;
}

/*
 * The function fiat_fp100_set_one returns the field element one in the Montgomery domain.
 *
 * Postconditions:
 *   eval (from_montgomery out1) mod m = 1 mod m
 *   0 ≤ eval out1 < m
 *
 */
void fiat_fp100_set_one(fiat_fp100_montgomery_domain_field_element out1) {
  out1[0] = UINT64_C(0x4d9a00001aebe56e);
  out1[1] = UINT64_C(0x16833e36b);
}

/*
 * The function fiat_fp100_msat returns the saturated representation of the prime modulus.
 *
 * Postconditions:
 *   twos_complement_eval out1 = m
 *   0 ≤ eval out1 < m
 *
 * Output Bounds:
 *   out1: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 */
void fiat_fp100_msat(uint64_t out1[3]) {
  out1[0] = UINT64_C(0x2164ffffffffffff);
  out1[1] = UINT64_C(0x98256f148);
  out1[2] = 0x0;
}

/*
 * The function fiat_fp100_divstep_precomp returns the precomputed value for Bernstein-Yang-inversion (in montgomery form).
 *
 * Postconditions:
 *   eval (from_montgomery out1) = ⌊(m - 1) / 2⌋^(if ⌊log2 m⌋ + 1 < 46 then ⌊(49 * (⌊log2 m⌋ + 1) + 80) / 17⌋ else ⌊(49 * (⌊log2 m⌋ + 1) + 57) / 17⌋)
 *   0 ≤ eval out1 < m
 *
 * Output Bounds:
 *   out1: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 */
void fiat_fp100_divstep_precomp(uint64_t out1[2]) {
  out1[0] = UINT64_C(0x74eb624af446d8a4);
  out1[1] = UINT64_C(0x76c0c8ed8);
}

/*
 * The function fiat_fp100_divstep computes a divstep.
 *
 * Preconditions:
 *   0 ≤ eval arg4 < m
 *   0 ≤ eval arg5 < m
 * Postconditions:
 *   out1 = (if 0 < arg1 ∧ (twos_complement_eval arg3) is odd then 1 - arg1 else 1 + arg1)
 *   twos_complement_eval out2 = (if 0 < arg1 ∧ (twos_complement_eval arg3) is odd then twos_complement_eval arg3 else twos_complement_eval arg2)
 *   twos_complement_eval out3 = (if 0 < arg1 ∧ (twos_complement_eval arg3) is odd then ⌊(twos_complement_eval arg3 - twos_complement_eval arg2) / 2⌋ else ⌊(twos_complement_eval arg3 + (twos_complement_eval arg3 mod 2) * twos_complement_eval arg2) / 2⌋)
 *   eval (from_montgomery out4) mod m = (if 0 < arg1 ∧ (twos_complement_eval arg3) is odd then (2 * eval (from_montgomery arg5)) mod m else (2 * eval (from_montgomery arg4)) mod m)
 *   eval (from_montgomery out5) mod m = (if 0 < arg1 ∧ (twos_complement_eval arg3) is odd then (eval (from_montgomery arg4) - eval (from_montgomery arg4)) mod m else (eval (from_montgomery arg5) + (twos_complement_eval arg3 mod 2) * eval (from_montgomery arg4)) mod m)
 *   0 ≤ eval out5 < m
 *   0 ≤ eval out5 < m
 *   0 ≤ eval out2 < m
 *   0 ≤ eval out3 < m
 *
 * Input Bounds:
 *   arg1: [0x0 ~> 0xffffffffffffffff]
 *   arg2: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 *   arg3: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 *   arg4: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 *   arg5: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 * Output Bounds:
 *   out1: [0x0 ~> 0xffffffffffffffff]
 *   out2: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 *   out3: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 *   out4: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 *   out5: [[0x0 ~> 0xffffffffffffffff], [0x0 ~> 0xffffffffffffffff]]
 */
void fiat_fp100_divstep(uint64_t* out1, uint64_t out2[3], uint64_t out3[3], uint64_t out4[2], uint64_t out5[2], uint64_t arg1, const uint64_t arg2[3], const uint64_t arg3[3], const uint64_t arg4[2], const uint64_t arg5[2]) {
  uint64_t x1;
  fiat_fp100_uint1 x2;
  fiat_fp100_uint1 x3;
  uint64_t x4;
  fiat_fp100_uint1 x5;
  uint64_t x6;
  uint64_t x7;
  uint64_t x8;
  uint64_t x9;
  uint64_t x10;
  fiat_fp100_uint1 x11;
  uint64_t x12;
  fiat_fp100_uint1 x13;
  uint64_t x14;
  fiat_fp100_uint1 x15;
  uint64_t x16;
  uint64_t x17;
  uint64_t x18;
  uint64_t x19;
  uint64_t x20;
  uint64_t x21;
  fiat_fp100_uint1 x22;
  uint64_t x23;
  fiat_fp100_uint1 x24;
  uint64_t x25;
  fiat_fp100_uint1 x26;
  uint64_t x27;
  fiat_fp100_uint1 x28;
  uint64_t x29;
  fiat_fp100_uint1 x30;
  uint64_t x31;
  uint64_t x32;
  uint64_t x33;
  fiat_fp100_uint1 x34;
  uint64_t x35;
  fiat_fp100_uint1 x36;
  uint64_t x37;
  uint64_t x38;
  fiat_fp100_uint1 x39;
  uint64_t x40;
  fiat_fp100_uint1 x41;
  uint64_t x42;
  uint64_t x43;
  fiat_fp100_uint1 x44;
  uint64_t x45;
  uint64_t x46;
  uint64_t x47;
  uint64_t x48;
  fiat_fp100_uint1 x49;
  uint64_t x50;
  fiat_fp100_uint1 x51;
  uint64_t x52;
  fiat_fp100_uint1 x53;
  uint64_t x54;
  uint64_t x55;
  uint64_t x56;
  fiat_fp100_uint1 x57;
  uint64_t x58;
  fiat_fp100_uint1 x59;
  uint64_t x60;
  fiat_fp100_uint1 x61;
  uint64_t x62;
  fiat_fp100_uint1 x63;
  uint64_t x64;
  fiat_fp100_uint1 x65;
  uint64_t x66;
  fiat_fp100_uint1 x67;
  uint64_t x68;
  uint64_t x69;
  uint64_t x70;
  uint64_t x71;
  uint64_t x72;
  uint64_t x73;
  uint64_t x74;
  fiat_fp100_addcarryx_u64(&x1, &x2, 0x0, (~arg1), 0x1);
  x3 = (fiat_fp100_uint1)((fiat_fp100_uint1)(x1 >> 63) & (fiat_fp100_uint1)((arg3[0]) & 0x1));
  fiat_fp100_addcarryx_u64(&x4, &x5, 0x0, (~arg1), 0x1);
  fiat_fp100_cmovznz_u64(&x6, x3, arg1, x4);
  fiat_fp100_cmovznz_u64(&x7, x3, (arg2[0]), (arg3[0]));
  fiat_fp100_cmovznz_u64(&x8, x3, (arg2[1]), (arg3[1]));
  fiat_fp100_cmovznz_u64(&x9, x3, (arg2[2]), (arg3[2]));
  fiat_fp100_addcarryx_u64(&x10, &x11, 0x0, 0x1, (~(arg2[0])));
  fiat_fp100_addcarryx_u64(&x12, &x13, x11, 0x0, (~(arg2[1])));
  fiat_fp100_addcarryx_u64(&x14, &x15, x13, 0x0, (~(arg2[2])));
  fiat_fp100_cmovznz_u64(&x16, x3, (arg3[0]), x10);
  fiat_fp100_cmovznz_u64(&x17, x3, (arg3[1]), x12);
  fiat_fp100_cmovznz_u64(&x18, x3, (arg3[2]), x14);
  fiat_fp100_cmovznz_u64(&x19, x3, (arg4[0]), (arg5[0]));
  fiat_fp100_cmovznz_u64(&x20, x3, (arg4[1]), (arg5[1]));
  fiat_fp100_addcarryx_u64(&x21, &x22, 0x0, x19, x19);
  fiat_fp100_addcarryx_u64(&x23, &x24, x22, x20, x20);
  fiat_fp100_subborrowx_u64(&x25, &x26, 0x0, x21, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_subborrowx_u64(&x27, &x28, x26, x23, UINT64_C(0x98256f148));
  fiat_fp100_subborrowx_u64(&x29, &x30, x28, x24, 0x0);
  x31 = (arg4[1]);
  x32 = (arg4[0]);
  fiat_fp100_subborrowx_u64(&x33, &x34, 0x0, 0x0, x32);
  fiat_fp100_subborrowx_u64(&x35, &x36, x34, 0x0, x31);
  fiat_fp100_cmovznz_u64(&x37, x36, 0x0, UINT64_C(0xffffffffffffffff));
  fiat_fp100_addcarryx_u64(&x38, &x39, 0x0, x33, (x37 & UINT64_C(0x2164ffffffffffff)));
  fiat_fp100_addcarryx_u64(&x40, &x41, x39, x35, (x37 & UINT64_C(0x98256f148)));
  fiat_fp100_cmovznz_u64(&x42, x3, (arg5[0]), x38);
  fiat_fp100_cmovznz_u64(&x43, x3, (arg5[1]), x40);
  x44 = (fiat_fp100_uint1)(x16 & 0x1);
  fiat_fp100_cmovznz_u64(&x45, x44, 0x0, x7);
  fiat_fp100_cmovznz_u64(&x46, x44, 0x0, x8);
  fiat_fp100_cmovznz_u64(&x47, x44, 0x0, x9);
  fiat_fp100_addcarryx_u64(&x48, &x49, 0x0, x16, x45);
  fiat_fp100_addcarryx_u64(&x50, &x51, x49, x17, x46);
  fiat_fp100_addcarryx_u64(&x52, &x53, x51, x18, x47);
  fiat_fp100_cmovznz_u64(&x54, x44, 0x0, x19);
  fiat_fp100_cmovznz_u64(&x55, x44, 0x0, x20);
  fiat_fp100_addcarryx_u64(&x56, &x57, 0x0, x42, x54);
  fiat_fp100_addcarryx_u64(&x58, &x59, x57, x43, x55);
  fiat_fp100_subborrowx_u64(&x60, &x61, 0x0, x56, UINT64_C(0x2164ffffffffffff));
  fiat_fp100_subborrowx_u64(&x62, &x63, x61, x58, UINT64_C(0x98256f148));
  fiat_fp100_subborrowx_u64(&x64, &x65, x63, x59, 0x0);
  fiat_fp100_addcarryx_u64(&x66, &x67, 0x0, x6, 0x1);
  x68 = ((x48 >> 1) | ((x50 << 63) & UINT64_C(0xffffffffffffffff)));
  x69 = ((x50 >> 1) | ((x52 << 63) & UINT64_C(0xffffffffffffffff)));
  x70 = ((x52 & UINT64_C(0x8000000000000000)) | (x52 >> 1));
  fiat_fp100_cmovznz_u64(&x71, x30, x25, x21);
  fiat_fp100_cmovznz_u64(&x72, x30, x27, x23);
  fiat_fp100_cmovznz_u64(&x73, x65, x60, x56);
  fiat_fp100_cmovznz_u64(&x74, x65, x62, x58);
  *out1 = x66;
  out2[0] = x7;
  out2[1] = x8;
  out2[2] = x9;
  out3[0] = x68;
  out3[1] = x69;
  out3[2] = x70;
  out4[0] = x71;
  out4[1] = x72;
  out5[0] = x73;
  out5[1] = x74;
}
