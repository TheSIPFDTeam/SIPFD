#ifndef _P236_ASM_H_
#define _P236_ASM_H_


#include <stdbool.h>
#include <string.h>
#include "config.h"
#if defined(_shortw_)
#include "../../shortw/api.h"
#elif defined(_mont_)
#include "../../mont/api.h"
#endif


extern const fp_t uintbig_1;
extern const fp_t fp_0, fp_1, r_squared_mod_p;
bool uintbig_add(fp_t x, fp_t const y, fp_t const z); /* returns carry */
bool uintbig_sub(fp_t x, fp_t const y, fp_t const z); /* returns borrow */


void fp_mul(fp_t out1, const fp_t arg1, const fp_t arg2);
void fp_sqr(fp_t out1, const fp_t arg1);
void fp_add(fp_t out1, const fp_t arg1, const fp_t arg2);
void fp_sub(fp_t out1, const fp_t arg1, const fp_t arg2);
void fp_random(fp_t x);
void fp_copy(fp_t b, const fp_t a);
void fp_pow(fp_t c, const fp_t a, const fp_t e);
void fp_inv(fp_t x);


static inline void from_montgomery(fp_t c, const fp_t a)
{fp_mul(c, a, uintbig_1);}


static inline void to_montgomery(fp_t c, const fp_t a)
{fp_mul(c, a, r_squared_mod_p);}


static inline void from_bytes(digit_t out1[NWORDS_FIELD], const uint8_t arg1[NBYTES_FIELD])
{
	for (size_t i = 0; i < NBYTES_FIELD; i++)
		out1[i>>3] += arg1[i] << (i & 0x7);
}


static inline void to_bytes(uint8_t out1[NBYTES_FIELD], const digit_t arg1[NWORDS_FIELD])
{
	for (size_t i = 0; i < NBYTES_FIELD; i++)
		out1[i] = arg1[i>>3] & (0xFF << ( i & 0x7));
}


static inline void fp_set_one(fp_t c)
{
    for (size_t i = 0; i < NWORDS_FIELD; i++)
        c[i] = fp_1[i];
}


static inline void fp_set_zero(fp_t c)
{
    for (size_t i = 0; i < NWORDS_FIELD; i++)
        c[i] = 0;
}


static inline void fp_neg(fp_t c, const fp_t a)
{fp_sub(c, fp_0, a);}


static inline void fp_nonzero(digit_t* c, const digit_t a[NWORDS_FIELD])
{
    int i;
    digit_t out = 0;
    for (i = 0; i < NWORDS_FIELD; i++)
        out = out | a[i];
    *c = out;
}


#endif
