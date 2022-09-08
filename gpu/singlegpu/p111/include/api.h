#ifndef _API_H_
#define _API_H_ 1

#include <cuda.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>
#include "config.h"

#define GPUID 0
#define RUNS 100
#define RUNS_BENCH (1 << 15)
#define RUNS_BENCH_S (1 << 12)

#ifdef NFNC
#define NFUNCTIONS 30
#endif

typedef limb_t fp_t[NWORDS_FIELD];
// GF(p²)
typedef fp_t fp2_t[2];
// Differential arithmetic (x-only projective point)
typedef fp2_t proj_t[2];

typedef struct {
    // k : integer scalar for computing the kernel R = P + [k]Q
    uint64_t k; 
    // c : single bit that determines the side (c = 0 is initial curve, c = 1 is public key curve)
    uint8_t c; 
} point_t;

#include "rng.h"
#include "mont.h"

#include "mitm.h"
#include "vowgcs.h"

#include "utils.h"

extern __device__ void fp_neg(fp_t a, fp_t b);
extern __host__ __device__ limb_t fp_nonzero(fp_t a);
extern __device__ void fp_random(fp_t x, curandStatePhilox4_32_10_t *state);
extern __host__ __device__ void fp_copy(fp_t b, fp_t a);
extern __host__ __device__ int fp_compare(fp_t b, fp_t a);
extern __host__ __device__ int fp_iszero(fp_t x);
extern __device__ void fp_pow(fp_t c, fp_t a, fp_t e);
extern __device__ void fp_inv(fp_t x);
extern __device__ void from_montgomery(fp_t c, fp_t a);

extern __host__ __device__ void fp2_set_one(fp2_t x);
extern __host__ __device__ void fp2_copy(fp2_t b, fp2_t a);
extern __host__ __device__ int fp2_compare(fp2_t b, fp2_t a);

extern __device__ void fp2_random(fp2_t x, curandStatePhilox4_32_10_t *state);
extern __device__ int fp2_iszero(fp2_t x);

extern __device__ void fp2_mul(fp2_t out1, fp2_t arg1, fp2_t arg2);
extern __device__ void fp2_sqr(fp2_t out1, fp2_t arg1);
extern __device__ void fp2_add(fp2_t out1, fp2_t arg1, fp2_t arg2);
extern __device__ void fp2_sub(fp2_t out1, fp2_t arg1, fp2_t arg2);
extern __device__ void fp2_neg(fp2_t out1, fp2_t arg1);
extern __device__ void fp2_pow(fp2_t c, fp2_t a, fp_t e);
extern __device__ void fp2_inv(fp2_t x);

extern __device__ void fp2_conj(fp2_t out1, fp2_t arg1);
extern __device__ int fp2_issquare(fp2_t b, fp2_t a);

extern __device__ void random_mod_A(fp_t x, curandStatePhilox4_32_10_t *state);
extern __device__ void random_mod_B(fp_t x, curandStatePhilox4_32_10_t *state);

void to_montgomery(fp_t out1, const fp_t arg1);
//void fp_nonzero(limb_t* out1, const limb_t arg1);
void to_bytes(uint8_t out1[NBYTES_FIELD], const limb_t arg1);
void from_bytes(limb_t out1, const uint8_t arg1[NBYTES_FIELD]);
void fp_set_one(fp_t out1);

void fp_string(char x_string[2*NBYTES_FIELD + 1], const fp_t x);
void fp_printf(const fp_t x);
void randombytes(void *x, size_t l);

void fp2_printf(const fp2_t x);

//uint32_t fp2mul_counter = 0, fp2sqr_counter = 0, fp2add_counter = 0;

int point_compare(const point_t *a, const point_t *c);

// Instance from instance.cu
extern __device__ proj_t const_E[2];
extern __device__ proj_t const_A2[2];
extern __device__ proj_t const_BASIS[2][3];

extern const proj_t h_E[2];
extern const proj_t h_A2[2];
extern const proj_t h_BASIS[2][3];
#endif
