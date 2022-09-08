#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x0257E5F3D63080BF, 0x0000000000000CCC };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x0000000000000001, 0x0257E5F3D63080C0, 0x9C210CF188C09CCC };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x0014013C6A3568E9, 0x238A4BB431CCD140, 0x000000000000032E };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0xFFEBFEC395CA9716, 0xDECD9A3FA463AF7F, 0x000000000000099D };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFD, 0x0257E5F3D63080BF, 0x0000000000000CCC };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x0095F97CF58C202F, 0x0000000000000333 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x012BF2F9EB18405F, 0x0000000000000666 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0000000000000000, 0x0000000000000020, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000200000000, 0x0000000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000200000000, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x12BF307AE81FFD59, 0x0000000000000000, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x000000004546B3DB, 0x0000000000000000, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x0000000017179149, 0x0000000000000000, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x0014013C6A3568E9, 0x238A4BB431CCD140, 0x000000000000032E };

__device__ limb_t STRATEGY2[69] = { 32, 17, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 15, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_0[34] = { 15, 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_1[34] = { 15, 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[33] = { 16, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[33] = { 16, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};

__device__ limb_t STRATEGY2_PC_0[14] = { 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_PC_1[14] = { 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};

__device__ uint32_t STRATEGY3[38] = { 13, 9, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
//const uint32_t STRATEGY3_0[18] = { 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1};
//const uint32_t STRATEGY3_1[19] = { 7, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1};

