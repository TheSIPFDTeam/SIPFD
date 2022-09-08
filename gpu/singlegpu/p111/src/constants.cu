#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0x944FFFFFFFFFFFFF, 0x0000510E67901461 };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x9450000000000001, 0x3A8CEA0E67901461 };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0xDE20000000032886, 0x00001B970A4158CF };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0xB62FFFFFFFFCD779, 0x000035775D4EBB91 };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0x944FFFFFFFFFFFFD, 0x0000510E67901461 };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x6513FFFFFFFFFFFF, 0x0000144399E40518 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xCA27FFFFFFFFFFFF, 0x0000288733C80A30 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0008000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000001000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000001000000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x000231C54B5F6A2B, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x0000000000DAF26B, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x0000000000DAF26B, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0xDE20000000032886, 0x00001B970A4158CF };

__device__ limb_t STRATEGY2[51] = { 21, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_0[25] = { 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_1[25] = { 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[24] = { 10, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[24] = { 10, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_PC_0[10] = { 4, 3, 2, 1, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_PC_1[10] = { 4, 3, 2, 1, 1, 1, 1, 2, 1, 1};

__device__ uint32_t STRATEGY3[31] = { 11, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[15] = { 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
//const uint32_t STRATEGY3_1[15] = { 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};

