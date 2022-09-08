#ifndef _UTILS_H_
#define _UTILS_H_

#include "api.h"

void randombytes(void *x, size_t l);

extern "C" __global__ void collision_printf(point_t collision[2], ctx_t *context);
extern __device__ void random_instance(ctx_t *context, limb_t deg, curandStatePhilox4_32_10_t *state);
extern __device__ void _fn_(point_t *y, fp2_t j, point_t *x, ctx_t *context,
        limb_t S[2][EXP0], fp2_t *P0, fp2_t *Q0, fp2_t *PQ0, fp2_t *E0, fp2_t *Z0,
        fp2_t *P1, fp2_t *Q1, fp2_t *PQ1, fp2_t *E1, fp2_t *Z1, limb_t *expo, limb_t *ebits);
extern __device__ void _gn_(point_t *g, fp2_t jinv, uint64_t NONCE);
extern __device__ void _h_(proj_t G, proj_t P[3], proj_t A2, point_t *g, limb_t e);

extern __device__ void rsh(fp_t x); // Right Shift
extern __device__ void lsh(fp_t x); // Left Shift

#endif
