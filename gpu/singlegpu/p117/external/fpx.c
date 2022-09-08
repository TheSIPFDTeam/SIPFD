#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "api.h"
#include "asm_api.h"

/* ------------------------------------------------------------- *
 *  fp2_copy()
 *  inputs: a projective Edwards y-coordinates of y(P)=YP/ZP;
 *  output: a copy of the projective Edwards's y-coordinate of y(P)
 * ------------------------------------------------------------- */
void fp2_copy(fp2_t x, const fp2_t y)
{
    fp_copy(x[0], y[0]);
    fp_copy(x[1], y[1]);
}

/* ------------------------------------------------------------- *
 *  fp2_add()
 *  inputs: two elements a and b of GF(p^2);
 *  output: a + b
 * ------------------------------------------------------------- */
void fp2_add(fp2_t c, const fp2_t a, const fp2_t b)
{
    fp_add(c[0], a[0], b[0]);
    fp_add(c[1], a[1], b[1]);
}   // 2 ADDS in Fp

/* ------------------------------------------------------------- *
 *  fp2_sub()
 *  inputs: two elements a and b of GF(p^2);
 *  output: a - b
 * ------------------------------------------------------------- */
void fp2_sub(fp2_t c, const fp2_t a, const fp2_t b)
{
    fp_sub(c[0], a[0], b[0]);
    fp_sub(c[1], a[1], b[1]);
}   // 2 ADDS (SUBS) in Fp

/* ------------------------------------------------------------- *
 *  fp2_mul()
 *  inputs: two elements a and b of GF(p^2);
 *  output: a * b
 * ------------------------------------------------------------- */
void fp2_mul(fp2_t c, const fp2_t a, const fp2_t b)
{
    fp_t z0, z1, z2, z3, tmp;
    fp_add(z0, a[0], a[1]);	// a[0] + a[1]
    fp_add(z1, b[0], b[1]);	// b[0] + b[1]
    fp_mul(tmp, z0, z1);		// (a[0] + a[1]) * (b[0] + b[1])
    fp_mul(z2, a[0], b[0]);	// a[0] * b[0]
    fp_mul(z3, a[1], b[1]);	// a[1] * b[1]
    fp_sub(c[0], z2, z3);	//  a[0] * b[0] -  a[1] * b[1]
    fp_sub(c[1], tmp, z2);	//  (a[0] + a[1]) * (b[0] + b[1]) - a[0] * b[0]
    fp_sub(c[1], c[1], z3); //  (a[0] + a[1]) * (b[0] + b[1]) - a[0] * b[0] - a[1] * b[1] = a[1] * b[0] + a[0] * b[1]
}   // 3 MULS + 5 ADDS in Fp

/* ------------------------------------------------------------- *
 *  fp2_sqr()
 *  inputs: an elements a of GF(p^2);
 *  output: a ^ 2
 * ------------------------------------------------------------- */
void fp2_sqr(fp2_t b, const fp2_t a)
{
    fp_t z0, z1, z2;
    fp_add(z0, a[0], a[0]);	// 2 * a[0]
    fp_add(z1, a[0], a[1]);	// a[0] + a[1]
    fp_sub(z2, a[0], a[1]);	// a[0] - a[1]
    fp_mul(b[0], z1, z2);	// (a[0] + a[1]) * (a[0] - a[1]) = a[0]^2 - a[1]^2
    fp_mul(b[1], z0, a[1]);	// 2 * a[0] * a[1]
}   // 2 MULS + 3 ADDS in Fp

