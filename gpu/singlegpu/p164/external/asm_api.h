#ifndef _ASM_H_
#define _ASM_H_

#include <stdbool.h>
#include <string.h>
#include "api.h"

extern const fp_t uintbig_1;
extern const fp_t fp_0, fp_1, r_squared_mod_p;
bool uintbig_add(fp_t x, fp_t const y, fp_t const z); /* returns carry */
bool uintbig_sub(fp_t x, fp_t const y, fp_t const z); /* returns borrow */

void fp_mul(fp_t out1, const fp_t arg1, const fp_t arg2);
void fp_sqr(fp_t out1, const fp_t arg1);
void fp_add(fp_t out1, const fp_t arg1, const fp_t arg2);
void fp_sub(fp_t out1, const fp_t arg1, const fp_t arg2);
void fp_copy(fp_t b, const fp_t a);

#endif
