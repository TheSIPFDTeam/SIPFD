#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0x212361F3FFFFFFFF, 0x0000000000000092 };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x212361F400000001, 0x4DCB37B4D632D122 };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x9B1F75192E2E875F, 0x0000000000000012 };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0x8603ECDAD1D178A0, 0x000000000000007F };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0x212361F3FFFFFFFD, 0x0000000000000092 };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x8848D87CFFFFFFFF, 0x0000000000000024 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x1091B0F9FFFFFFFF, 0x0000000000000049 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0000000200000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000008000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000008000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x000000026F7C52B3, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x000000000000E6A9, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x000000000000E6A9, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x9B1F75192E2E875F, 0x0000000000000012 };

__device__ limb_t STRATEGY2[33] = { 16, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_0[16] = { 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_1[16] = { 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[15] = { 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[15] = { 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};

__device__ limb_t STRATEGY2_PC_0[6] = { 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_PC_1[6] = { 3, 2, 1, 1, 1, 1};

__device__ uint32_t STRATEGY3[21] = { 8, 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[10] = { 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_1[10] = { 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};

