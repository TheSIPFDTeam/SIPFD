#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0xA667023FFFFFFFFF, 0x000000000002A205 };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0xA667024000000001, 0xAF6CD6DDCF87B205 };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x099D787B32616D4B, 0x0000000000023046 };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0x9CC989C4CD9E92B4, 0x00000000000071BF };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0xA667023FFFFFFFFD, 0x000000000002A205 };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x6999C08FFFFFFFFF, 0x000000000000A881 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xD333811FFFFFFFFF, 0x0000000000015102 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0000002000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000020000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000020000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x00000015EB5EE84B, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x000000000002B3FB, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x000000000002B3FB, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x099D787B32616D4B, 0x0000000000023046 };

__device__ limb_t STRATEGY2[37] = { 17, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_0[18] = { 8, 4, 3, 2, 1, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_1[18] = { 8, 4, 3, 2, 1, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[17] = { 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[17] = { 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_PC_0[6] = { 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_PC_1[6] = { 3, 2, 1, 1, 1, 1};

__device__ uint32_t STRATEGY3[23] = { 9, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[11] = { 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_1[11] = { 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1};

