#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0x71F2B3FFFFFFFFFF, 0x00000000015625FD };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x71F2B40000000001, 0xC477F100CFE625FD };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x7393F0BF8AE90394, 0x0000000000923A36 };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0xFE5EC3407516FC6B, 0x0000000000C3EBC6 };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0x71F2B3FFFFFFFFFD, 0x00000000015625FD };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x5C7CACFFFFFFFFFF, 0x000000000055897F };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xB8F959FFFFFFFFFF, 0x0000000000AB12FE };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0000020000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000080000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000080000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x000000C546562AA3, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x0000000000081BF1, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x0000000000081BF1, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x7393F0BF8AE90394, 0x0000000000923A36 };

__device__ limb_t STRATEGY2[41] = { 18, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_0[20] = { 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_1[20] = { 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[19] = { 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[19] = { 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_PC_0[7] = { 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_PC_1[7] = { 4, 2, 1, 1, 2, 1, 1};

__device__ uint32_t STRATEGY3[25] = { 9, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[12] = { 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
//const uint32_t STRATEGY3_1[12] = { 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};

