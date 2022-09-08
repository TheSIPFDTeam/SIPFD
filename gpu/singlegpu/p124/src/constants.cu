#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0x83FFFFFFFFFFFFFF, 0x0F2258788E0F2886 };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x8400000000000001, 0x833258788E0F2886 };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0xC000000000000010, 0x0DDA78771F0D7797 };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0xC3FFFFFFFFFFFFEF, 0x0147E0016F01B0EE };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0x83FFFFFFFFFFFFFD, 0x0F2258788E0F2886 };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xA0FFFFFFFFFFFFFF, 0x03C8961E2383CA21 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x41FFFFFFFFFFFFFF, 0x07912C3C47079443 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0200000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000008000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000008000000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x02153E468B91C6D1, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x0000000017179149, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x0000000007B285C3, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0xC000000000000010, 0x0DDA78771F0D7797 };

__device__ limb_t STRATEGY2[57] = { 27, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_0[28] = { 13, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_1[28] = { 13, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[27] = { 12, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[27] = { 12, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};

__device__ limb_t STRATEGY2_PC_0[11] = { 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_PC_1[11] = { 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};

__device__ uint32_t STRATEGY3[36] = { 13, 9, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
//const uint32_t STRATEGY3_0[17] = { 6, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
//const uint32_t STRATEGY3_1[18] = { 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1};

