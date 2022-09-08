#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include "api.h"

// GF(p)
#include "fp.h"

__host__ __device__ limb_t fp_nonzero(fp_t a) {
    int i;
    limb_t acc = 0;

    for (i = 0; i < NWORDS_FIELD; i++)
        acc |= a[i];

    return acc;
}

/* ------------------------------------------------------------- *
 * random_mod_A()
 * output: a random element 0 <= a < 2^e2
 * ------------------------------------------------------------- */
__device__ void random_mod_A(fp_t x, curandStatePhilox4_32_10_t *state) {
    randombytes(x, EXP2_MASK_GPU, EXPONENT2_BITS, state);
}

/* ------------------------------------------------------------- *
 * random_mod_B()
 * output: a random element 0 <= a < 3^e3
 * ------------------------------------------------------------- */
__device__ void random_mod_B(fp_t x, curandStatePhilox4_32_10_t *state) {
    randombytes(x, EXP3_MASK_GPU, EXPONENT3_BITS, state);
}

/* ------------------------------------------------------------- *
 * fp_random()
 * output: a random element 0 <= a < p of GF(p)
 * ------------------------------------------------------------- */
__device__ void fp_random(fp_t x, curandStatePhilox4_32_10_t *state) {
    randombytes(x, MASK, NBITS_FIELD, state);
    fp_mul(x, x, mont_one);
}

__device__ void from_montgomery(fp_t c, fp_t a) {
    fp_mul(c, a, bigone);
}

__device__ void fp_copy(fp_t b, fp_t a)
{
    /*
     * Copying field element
     */
    int i;
    for (i = 0; i < NWORDS_FIELD; i++)
        b[i] = a[i];
}

__host__ __device__ int fp_compare(fp_t a, fp_t b)
{
    /* ------------------------------------------------------------- *
     * fp_compare()
     *  inputs: two integer numbers x and y, and the number of 64-bits
     *          words of x and y.
     *  outputs:
     *           +1 if x > y,
     *           -1 if x < y, or
     *            0 if x = y.
     * ------------------------------------------------------------- */
    //return memcmp(a, b, NBITS_TO_NBYTES(NBITS_FIELD));
    int i;
    for (i=NWORDS_FIELD-1; i >= 0; i--)
    {
        if (a[i] != b[i])
            return a[i] > b[i] ? 1 : -1;
    }
    return 0;
}

__host__ __device__ int fp_iszero(fp_t a)
{
    return (int)(fp_nonzero(a) == 0);
}

// I think I don't need this fnc so far
int8_t ct_compare(const uint8_t *a, const uint8_t *b, unsigned int len)
{
    // Compare two byte arrays in constant time.
    // Returns 0 if the byte arrays are equal, -1 otherwise.
    uint8_t r = 0;

    for (unsigned int i = 0; i < len; i++)
        r |= a[i] ^ b[i];

    return (int8_t)((-(int32_t)r) >> (8*sizeof(uint32_t)-1));
}

// NO
//void fp_string(char x_string[2*NBYTES_FIELD + 1], const fp_t x)
//{
//    /*
//     * From field element into string
//     */
//    int i, j = 0;
//    uint8_t x_bytes[NBYTES_FIELD] = {0};
//    char tmp[3];
//    to_bytes(x_bytes, x);
//
//    x_string[0] = '\0';
//    for(i = (NBYTES_FIELD - 1); i > -1; i--)
//    {
//        //printf("%02X", x_bytes[i]);
//        tmp[0] = '\0';
//        sprintf(tmp, "%02X", x_bytes[i]);
//        tmp[2] = '\0';
//        strcat(x_string, tmp);
//        j += 2;
//        x_string[j] = '\0';
//    }
//    //printf("\n");
//}

// NO
//void fp_printf(const fp_t x)
//{
//    /*
//     * Printing field element
//     */
//    int i;
//    fp_t tmp;
//    from_montgomery(tmp, x);
//    printf("0x");
//    for(i = (NWORDS_FIELD-1); i > -1; i--)
//        TOHEX(tmp[i]);
//    printf(";\n");
//}

__device__ void fp_pow(fp_t c, fp_t a, fp_t exp)
{
    /* ------------------------------------------------------------- *
     * fp_pow()
     * inputs: an elements a of F_p, and an element b < p of F_p
     * output: a ^ b
     * ------------------------------------------------------------- */
    int i, j;
    limb_t flag;

    fp_t tmp;
    fp_copy(c, mont_one);
    fp_copy(tmp, a);

    for(i = 0; i < NWORDS_FIELD; i++)
    {
        flag = 1;
        for(j = 0; j < RADIX; j++)
        {
            if( (flag & exp[i]) != 0 )
            {
                fp_mul(c, tmp, c);
//                fp2mul_counter += 1;
            }

            fp_sqr(tmp, tmp);
//            fp2sqr_counter += 1;
            flag <<= 1;
        }
    }
}

__device__ void fp_inv(fp_t a)
{
    /* ------------------------------------------------------------- *
     * fp_inv()
     * inputs: an elements a of F_p;
     * output: a^-1
     * ------------------------------------------------------------- */
    fp_t tmp;
    fp_pow(tmp, a, P_MINUS_TWO);
    fp_copy(a, tmp);
}

__device__ void fp_neg(fp_t a, fp_t b) {
    //fp_mul(a, b, mont_minus_one);
    fp_t fp_0 = {0};
    fp_sub(a, fp_0, b);
}

//void fp2_printf(const fp2_t x)
//{
//    fp2_t tmp = {0};
//    from_montgomery(tmp[0], x[0]);
//    from_montgomery(tmp[1], x[1]);
//    int i;
//    printf("0x");
//    for(i = (NWORDS_FIELD-1); i > -1; i--)
//        TOHEX(tmp[0][i]);
//    printf(" + i * 0x");
//    for(i = (NWORDS_FIELD-1); i > -1; i--)
//        TOHEX(tmp[1][i]);
//    printf(";\n");
//}

__device__ void fp2_random(fp2_t x, curandStatePhilox4_32_10_t *state)
{
    fp_random(x[0], state);
    fp_random(x[1], state);
}

__host__ __device__ void fp2_set_one(fp2_t x)
{
    fp_t tmp = {0};
    fp_copy(x[0], mont_one);
    fp_copy(x[1], tmp);
}

__device__ int fp2_iszero(fp2_t x)
{
    return fp_iszero(x[0]) & fp_iszero(x[1]);
}

__device__ void fp2_copy(fp2_t x, fp2_t y)
{
    /* ------------------------------------------------------------- *
     *  fp2_copy()
     *  inputs: a projective Edwards y-coordinates of y(P)=YP/ZP;
     *  output: a copy of the projective Edwards's y-coordinate of y(P)
     * ------------------------------------------------------------- */
    fp_copy(x[0], y[0]);
    fp_copy(x[1], y[1]);
}

__host__ __device__ int fp2_compare(fp2_t a, fp2_t b)
{
    /* ----------------------------------------------------------------------------- *
     *  fp2_compare()
     *  Inputs: two elements of fp2_t a and b
     *  Output:
     *           0  if a = b,
     *          -1  if a < b, or
     *           1  if a > b.
     * ----------------------------------------------------------------------------- */
    int local_1st = fp_compare(a[0], b[0]);

    if(local_1st != 0)
        return local_1st;
    else
        return fp_compare(a[1], b[1]); // local_2nd
}

__device__ int fp2_compare_conj(fp2_t a, fp2_t b)
{
    /* ----------------------------------------------------------------------------- *
     *  fp2_compare()
     *  Inputs: two elements of fp2_t a and b
     *  Output:
     *           0  if a = b or b conj,
     *          -1  if a < b, or
     *           1  if a > b.
     * ----------------------------------------------------------------------------- */
    int local_1st = fp_compare(a[0],b[0]);

    if(local_1st == 0) {
        fp_t neg_b, neg_a;
        fp_copy(neg_b, b[1]);
        fp_copy(neg_a, a[1]);
        if (neg_b[0] & 1)
            fp_neg(neg_b, neg_b);
        if (neg_a[0] & 1)
            fp_neg(neg_a, neg_a);
        
        int local_2nd = fp_compare(neg_a, neg_b);
        return local_2nd;
    }
    return local_1st;
}

__device__ void fp2_add(fp2_t c, fp2_t a, fp2_t b)
{
    /* ------------------------------------------------------------- *
     *  fp2_add()
     *  inputs: two elements a and b of GF(p^2);
     *  output: a + b
     * ------------------------------------------------------------- */
    fp_add(c[0], a[0], b[0]);
    fp_add(c[1], a[1], b[1]);
//    fp2add_counter += 2;
}   // 2 ADDS in Fp

__device__ void fp2_sub(fp2_t c, fp2_t a, fp2_t b)
{
    /* ------------------------------------------------------------- *
     *  fp2_sub()
     *  inputs: two elements a and b of GF(p^2);
     *  output: a - b
     * ------------------------------------------------------------- */
    fp_sub(c[0], a[0], b[0]);
    fp_sub(c[1], a[1], b[1]);
//    fp2add_counter += 2;
}   // 2 ADDS (SUBS) in Fp

__device__ void fp2_mul(fp2_t c, fp2_t a, fp2_t b)
{
    /* ------------------------------------------------------------- *
     *  fp2_mul()
     *  inputs: two elements a and b of GF(p^2);
     *  output: a * b
     * ------------------------------------------------------------- */
    fp_t z0, z1, z2, z3, tmp;
    fp_add(z0, a[0], a[1]);	// a[0] + a[1]
    fp_add(z1, b[0], b[1]);	// b[0] + b[1]
    fp_mul(tmp, z0, z1);		// (a[0] + a[1]) * (b[0] + b[1])
    fp_mul(z2, a[0], b[0]);	// a[0] * b[0]
    fp_mul(z3, a[1], b[1]);	// a[1] * b[1]
    fp_sub(c[0], z2, z3);	//  a[0] * b[0] -  a[1] * b[1]
    fp_sub(c[1], tmp, z2);	//  (a[0] + a[1]) * (b[0] + b[1]) - a[0] * b[0]
    fp_sub(c[1], c[1], z3); //  (a[0] + a[1]) * (b[0] + b[1]) - a[0] * b[0] - a[1] * b[1] = a[1] * b[0] + a[0] * b[1]

//    fp2add_counter += 5;
//    fp2mul_counter += 3;
}   // 3 MULS + 5 ADDS in Fp

__device__ void fp2_sqr(fp2_t b, fp2_t a)
{
    /* ------------------------------------------------------------- *
     *  fp2_sqr()
     *  inputs: an elements a of GF(p^2);
     *  output: a ^ 2
     * ------------------------------------------------------------- */
    fp_t z0, z1, z2;
    fp_add(z0, a[0], a[0]);	// 2 * a[0]
    fp_add(z1, a[0], a[1]);	// a[0] + a[1]
    fp_sub(z2, a[0], a[1]);	// a[0] - a[1]
    fp_mul(b[0], z1, z2);	// (a[0] + a[1]) * (a[0] - a[1]) = a[0]^2 - a[1]^2
    fp_mul(b[1], z0, a[1]);	// 2 * a[0] * a[1]

 //   fp2add_counter += 3;
 //   fp2mul_counter += 2;
}   // 2 MULS + 3 ADDS in Fp

__device__ void fp2_inv(fp2_t b)
{
    /* ------------------------------------------------------------- *
     * fp2_inv()
     * inputs: an elements a of GF(p^2);
     * output: a^-1
     * ------------------------------------------------------------- */
    fp2_t a;
    fp2_copy(a, b);
    fp_t N0, N1, S1, S2, zero = {0};
    fp_sqr(N0, a[0]);	// a[0] ^ 2
    fp_sqr(N1, a[1]);	// a[1] ^ 2
    fp_add(S1, N0, N1);// a[0] ^ 2 + a[1] ^ 2 = Norm(a[0] + a[1] * i)
    fp_inv(S1);		// 1 / (a[0] ^ 2 + a[1] ^ 2)
    fp_sub(S2, zero, a[1]);	// -a[1]
    fp_mul(b[0], S1, a[0]);	//  a[0] / (a[0] ^ 2 + a[1] ^ 2)
    fp_mul(b[1], S1, S2);   // -a[1] / (a[0] ^ 2 + a[1] ^ 2)

 //   fp2add_counter += 2;
 //   fp2sqr_counter += 2;
 //   fp2mul_counter += 2;
}   // 1 INV + 2 MULS + 2 SQR + 2 ADDS in Fp

__device__ void fp2_pow(fp2_t c, fp2_t a, fp_t exp)
{
    /* ------------------------------------------------------------- *
     * fp2_pow()
     * inputs: an elements a of F_{p^2}, and an element b < p of F_p
     * output: a ^ b
     * ------------------------------------------------------------- */
    int i, j;
    limb_t flag;

    fp2_t tmp;
    fp2_set_one(c);
    fp2_copy(tmp, a);

    for(i = 0; i < NWORDS_FIELD; i++)
    {
        flag = 1;
        for(j = 0; j < RADIX; j++)
        {
            if( (flag & exp[i]) != 0 )
                fp2_mul(c, tmp, c);

            fp2_sqr(tmp, tmp);
            flag <<= 1;
        }
    }
}

__device__ void fp2_neg(fp2_t y, fp2_t x)
{
    fp_neg(y[0], x[0]);
    fp_neg(y[1], x[1]);
}

__device__ void fp2_conj(fp2_t y, fp2_t x)
{
    fp_copy(y[0], x[0]);
    fp_neg(y[1], x[1]);
}

__device__ int fp2_issquare(fp2_t b, fp2_t a)
{
    /* ------------------------------------------------------------- *
     * fp2_issquare()
     * inputs: an elements a of F_{p^2};
     * output: 1 if a is a quadratic residue, or 0 otherwise.
     * ------------------------------------------------------------- */
    fp2_t a1, alpha, alpha_conjugated, a0, minus_one, zero = {0}, x0;
    fp2_set_one(minus_one);			//  1
    fp2_neg(minus_one, minus_one);	// -1

    fp2_pow(a1, a, P_MINUS_THREE_QUARTERS);		// a1 <- a^( [p-3]/4)
    fp2_sqr(alpha, a1);				            // a1^2
    fp2_mul(alpha, alpha, a);			        // alpha <- a1^2 * a

    fp2_conj(alpha_conjugated, alpha);
    fp2_mul(a0, alpha_conjugated, alpha);	// a0 <- alpha^p * alpha

    if (fp2_compare(minus_one, a0) == 0)
        return 0;

    fp2_mul(x0, a1, a);

    if (fp2_compare(minus_one, alpha) == 0)
    {
        // b <- (i * x0) = (-x[1]) + (x[0] * i)
        fp_sub(b[0], zero[0], x0[1]);
        fp_add(b[1], zero[0], x0[0]);
    }
    else
    {
        fp2_sub(alpha, alpha, minus_one);       // (alpha + 1)
        fp2_pow(b, alpha, P_MINUS_ONE_HALVES);  // (alpha + 1)^([p-1]/2)
        fp2_mul(b, b, x0);                      // (alpha + 1)^([p-1]/2) * x0
    }
    return 1;
}
