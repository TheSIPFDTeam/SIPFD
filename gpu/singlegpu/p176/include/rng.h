#ifndef RNG_H
#define RNG_H

#include <cuda.h>
#include <curand_kernel.h>

extern __global__ void setup_rand(curandStatePhilox4_32_10_t *state, uint64_t seed);
extern __device__ void randombytes(fp_t a, uint32_t mask, uint32_t nbits, curandStatePhilox4_32_10_t *state);
extern __device__ void init_rand(curandStatePhilox4_32_10_t *state, uint64_t seed, int seq);

#endif
