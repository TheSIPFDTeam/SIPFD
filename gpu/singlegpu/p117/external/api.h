#ifndef _API_H_
#define _API_H_

#include <stdio.h>
#include "config.h"

// GF(p)
typedef digit_t fp_t[NWORDS_FIELD];
// The prime field api is in pX0X/pX0X_api.h where X0X denotes the bitlength of p

// GF(p²)
typedef fp_t fp2_t[2];

//void fp2_random(fp2_t x);
void fp2_set_one(fp2_t x);
void fp2_copy(fp2_t b, const fp2_t a);
//int fp2_compare(const fp2_t b, const fp2_t a);
//int fp2_iszero(const fp2_t x);

void fp2_mul(fp2_t out1, const fp2_t arg1, const fp2_t arg2);
void fp2_sqr(fp2_t out1, const fp2_t arg1);
void fp2_add(fp2_t out1, const fp2_t arg1, const fp2_t arg2);
void fp2_sub(fp2_t out1, const fp2_t arg1, const fp2_t arg2);
//void fp2_neg(fp2_t out1, const fp2_t arg1);
//void fp2_conj(fp2_t out1, const fp2_t arg1);
//void fp2_pow(fp2_t c, const fp2_t a, const fp_t e);
//void fp2_inv(fp2_t x);

// Differential arithmetic (x-only projective point)
typedef fp2_t proj_t[2];

#include "mont.h"

typedef struct {
    uint64_t k;         // k : integer scalar for computing the kernel R = P + [k]Q
    uint8_t c;      // c : single bit that determines the side (c = 0 is initial curve, c = 1 is public key curve)
} point_t;

#endif
