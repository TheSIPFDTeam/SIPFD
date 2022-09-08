#include "api.h"

__global__ void setup_rand(curandStatePhilox4_32_10_t *state, uint64_t seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(seed, i, 0, &state[i]);
}

__device__ void init_rand(curandStatePhilox4_32_10_t *state, uint64_t seed, int seq) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    if (seq)
        curand_init(seed, seq, 0, &state[t]);
    else
        curand_init(seed, t, 0, &state[t]);
}

/* IMPORTANT: this function just generates random numbers for 2, 3, and 4 words */
__device__ void randombytes(fp_t a, uint32_t mask, uint32_t nbits, curandStatePhilox4_32_10_t *state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    uint4 r[2];

    /* Copy state to local memory for efficiency */
    curandStatePhilox4_32_10_t localState = state[i];

    /* Generate pseudo-random unsigned ints */
    r[0] = curand4(&localState);
    r[1] = curand4(&localState);
    
    /* Store results */
<rng>

    /* Copy state back to global memory */
    state[i] = localState;
}

