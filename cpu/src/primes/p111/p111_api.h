#ifndef _P111_API_H_
#define _P111_API_H_

#include <stdint.h>
#include "config.h"
#if defined(_shortw_)
#include "../../shortw/api.h"
#elif defined(_mont_)
#include "../../mont/api.h"
#endif

// GF(p)
void fiat_fp111_mul(fp_t out1, const fp_t arg1, const fp_t arg2);
void fiat_fp111_square(fp_t out1, const fp_t arg1);
void fiat_fp111_add(fp_t out1, const fp_t arg1, const fp_t arg2);
void fiat_fp111_sub(fp_t out1, const fp_t arg1, const fp_t arg2);
void fiat_fp111_opp(fp_t out1, const fp_t arg1);
void fiat_fp111_from_montgomery(fp_t out1, const fp_t arg1);
void fiat_fp111_to_montgomery(fp_t out1, const fp_t arg1);
void fiat_fp111_nonzero(digit_t* out1, const digit_t arg1[NWORDS_FIELD]);
void fiat_fp111_to_bytes(uint8_t out1[NBYTES_FIELD], const digit_t arg1[NWORDS_FIELD]);
void fiat_fp111_from_bytes(digit_t out1[NWORDS_FIELD], const uint8_t arg1[NBYTES_FIELD]);
void fiat_fp111_set_one(fp_t out1);

void fiat_fp111_random(fp_t x);
void fiat_fp111_copy(fp_t b, const fp_t a);
int fiat_fp111_compare(const fp_t b, const fp_t a);
int fiat_fp111_iszero(const fp_t x);
void fiat_fp111_string(char x_string[2*NBYTES_FIELD + 1], const fp_t x);
void fiat_fp111_printf(const fp_t x);
void fiat_fp111_pow(fp_t c, const fp_t a, const fp_t e);
void fiat_fp111_inv(fp_t x);

#endif
