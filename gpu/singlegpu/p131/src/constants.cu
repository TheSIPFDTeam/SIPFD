#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x397B18CF129F1E8C, 0x0000000000000006 };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x0000000000000001, 0x397B18CF129F1E8D, 0x19E69B38CACB59AF };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x2920A8A1210118C1, 0xCCEA8DE7BF9FBFB3, 0x0000000000000001 };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0xD6DF575EDEFEE73E, 0x6C908AE752FF5ED9, 0x0000000000000004 };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFD, 0x397B18CF129F1E8C, 0x0000000000000006 };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x3FFFFFFFFFFFFFFF, 0x8E5EC633C4A7C7A3, 0x0000000000000001 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x7FFFFFFFFFFFFFFF, 0x1CBD8C67894F8F46, 0x0000000000000003 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x8000000000000000, 0x0000000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000040000000, 0x0000000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000040000000, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x063FBAD3A2B55473, 0x0000000000000000, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x0000000017179149, 0x0000000000000000, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x0000000017179149, 0x0000000000000000, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x2920A8A1210118C1, 0xCCEA8DE7BF9FBFB3, 0x0000000000000001 };

__device__ limb_t STRATEGY2[63] = { 27, 16, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_0[31] = { 15, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_1[31] = { 15, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[30] = { 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[30] = { 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1};

__device__ limb_t STRATEGY2_PC_0[13] = { 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_PC_1[13] = { 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1};

__device__ uint32_t STRATEGY3[37] = { 13, 8, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
//const uint32_t STRATEGY3_0[18] = { 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1};
//const uint32_t STRATEGY3_1[18] = { 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1};

