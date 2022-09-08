#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0x02A0B06FFFFFFFFF, 0x0000000000000E28 };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x02A0B07000000001, 0x7AAFBA9EC59A3F28 };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x68B83F2624F57D33, 0x0000000000000486 };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0x99E87149DB0A82CC, 0x00000000000009A1 };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0x02A0B06FFFFFFFFD, 0x0000000000000E28 };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x00A82C1BFFFFFFFF, 0x000000000000038A };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x01505837FFFFFFFF, 0x0000000000000714 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0000000800000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000010000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000010000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x000000026F7C52B3, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x000000000000E6A9, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x000000000000E6A9, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x68B83F2624F57D33, 0x0000000000000486 };

__device__ limb_t STRATEGY2[35] = { 16, 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_0[17] = { 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_1[17] = { 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[16] = { 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[16] = { 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_PC_0[6] = { 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_PC_1[6] = { 3, 2, 1, 1, 1, 1};

__device__ uint32_t STRATEGY3[21] = { 8, 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[10] = { 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_1[10] = { 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};

