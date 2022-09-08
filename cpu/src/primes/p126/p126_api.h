#ifndef _P126_API_H_
#define _P126_API_H_

#include <stdint.h>
#include "config.h"
#if defined(_shortw_)
#include "../../shortw/api.h"
#elif defined(_mont_)
#include "../../mont/api.h"
#endif

// GF(p)
void fiat_fp126_mul(fp_t out1, const fp_t arg1, const fp_t arg2);
void fiat_fp126_square(fp_t out1, const fp_t arg1);
void fiat_fp126_add(fp_t out1, const fp_t arg1, const fp_t arg2);
void fiat_fp126_sub(fp_t out1, const fp_t arg1, const fp_t arg2);
void fiat_fp126_opp(fp_t out1, const fp_t arg1);
void fiat_fp126_from_montgomery(fp_t out1, const fp_t arg1);
void fiat_fp126_to_montgomery(fp_t out1, const fp_t arg1);
void fiat_fp126_nonzero(digit_t* out1, const digit_t arg1[NWORDS_FIELD]);
void fiat_fp126_to_bytes(uint8_t out1[NBYTES_FIELD], const digit_t arg1[NWORDS_FIELD]);
void fiat_fp126_from_bytes(digit_t out1[NWORDS_FIELD], const uint8_t arg1[NBYTES_FIELD]);
void fiat_fp126_set_one(fp_t out1);

void fiat_fp126_random(fp_t x);
void fiat_fp126_copy(fp_t b, const fp_t a);
int fiat_fp126_compare(const fp_t b, const fp_t a);
int fiat_fp126_iszero(const fp_t x);
void fiat_fp126_string(char x_string[2*NBYTES_FIELD + 1], const fp_t x);
void fiat_fp126_printf(const fp_t x);
void fiat_fp126_pow(fp_t c, const fp_t a, const fp_t e);
void fiat_fp126_inv(fp_t x);

#endif
