#ifndef _API_H_
#define _API_H_

//#define FROBENIUS

// GF(p)
typedef digit_t fp_t[NWORDS_FIELD];
// The prime field api is in pX0X/pX0X_api.h where X0X denotes the bitlength of p

// GF(p²)
typedef fp_t fp2_t[2];

void fp2_random(fp2_t x);
void fp2_set_one(fp2_t x);
void fp2_copy(fp2_t b, const fp2_t a);
int fp2_compare(const fp2_t b, const fp2_t a);
int fp2_iszero(const fp2_t x);

void fp2_mul(fp2_t out1, const fp2_t arg1, const fp2_t arg2);
void fp2_square(fp2_t out1, const fp2_t arg1);
void fp2_add(fp2_t out1, const fp2_t arg1, const fp2_t arg2);
void fp2_sub(fp2_t out1, const fp2_t arg1, const fp2_t arg2);
void fp2_neg(fp2_t out1, const fp2_t arg1);
void fp2_conj(fp2_t out1, const fp2_t arg1);
void fp2_pow(fp2_t c, const fp2_t a, const fp_t e);
void fp2_inv(fp2_t x);
int fp2_issquare(fp2_t b, const fp2_t a);

void fp2_printf(const fp2_t x);

// Differential arithmetic (x-only projective point)
typedef fp2_t proj_t[2];
//digit_t fp2mul_counter = 0, fp2sqr_counter = 0, fp2add_counter = 0;

#include "mont.h"

// utils
void rsh(fp_t x);   // Right Shift
void lsh(fp_t x);   // Left Shift

typedef struct {
    uint64_t k;         // k : integer scalar for computing the kernel R = P + [k]Q
    uint8_t c;      // c : single bit that determines the side (c = 0 is initial curve, c = 1 is public key curve)
} point_t;

int point_compare(const point_t *a, const point_t *c);

typedef struct {
    int pc_depth;
    fp2_t ***pc_table;
    uint8_t c;
    digit_t NONCE;          // Pseudo Random Number (nonce)
    // <P[c],Q[c]> = torsion-(deg^e) subgroup of E[c], PQ[c] = P[c] - Q[c]
    proj_t BASIS[2][3];
    proj_t E[2];               // Either the domain or codomain curve in deg representation
    proj_t notE[2];            // Curves in not-deg representation
    proj_t A2[2];              // Curves in deg=2 representation
    // torsion-2^e2 point basis <PA, QA> on E[c], PQA = PA - QA
    proj_t PA;
    proj_t QA;
    proj_t PQA;
    // torsion-3^e3 point basis <P3, Q3> on E[c], PQB = PB - QB
    proj_t PB;
    proj_t QB;
    proj_t PQB;
    // Fixed values determined by deg
    void (*xmul_deg)();	    // either xdbl or xtpl
    void (*xmule_deg)();    // either xdble or xtple
    void (*xmul_notdeg)();  // either xtpl or xdbl
    void (*xmule_notdeg)(); // either xtple or xdble
    void (*xeval)();        // either xeval_2 or xeval_3
    void (*xisog)();        // either xisog_2 or xisog_3
    void (*xisoge_1st)();   // either xisog_2e_1st or xisog_3e_1st
    void (*xisoge_2nd)();   // either xisog_2e_2nd or xisog_3e_2nd
    digit_t deg;            // either 2 or 3
    fp_t size;              // either BOUND2 or BOUND3
    digit_t *strategy;      // strategy evaluation: either 2^EXPONENT2 or 3^EXPONENT3
    digit_t e[2];           // exponent halves: e[0] + e[1] is either EXPONENT2 or EXPONENT3
    digit_t not_e;          // either EXPONENT3 or EXPONENT2
    digit_t ebits[2];       // log2(e)
    digit_t ebits_max;
    fp_t bound[2];          // deg ^ (e - 1)
    digit_t *S[2];          // strategy evaluation: deg^e
    digit_t *S_PC[2];          // strategy evaluation: deg^e
    // Values concerning the parallel computations
    uint64_t runtime[2];        // deg ^ (e - 1) / (2^t): 2^t determines the #threads
    int cores;
} ctx_mitm_t;

void random_mod_A(fp_t x);
void random_mod_B(fp_t x);
void _h_(proj_t G, const proj_t P[3], const proj_t A2, const point_t g, const digit_t deg, const digit_t e);
void _gn_(point_t *g, const fp2_t jinv, const digit_t NONCE, const digit_t deg, const fp_t bound, const digit_t ebits);
void _fn_(point_t *y, fp2_t j, const point_t x, const ctx_mitm_t conf);

void init_context_mitm(ctx_mitm_t *context, digit_t deg);
void random_instance(ctx_mitm_t *context, digit_t deg);
void undo_2isog(ctx_mitm_t *context);
void collision_printf(point_t collision[2], const ctx_mitm_t context);

#if defined(_mitm_)
#include "mitm.h"
#endif
#if defined(_vowgcs_)
#include "vowgcs.h"
#endif

#endif
