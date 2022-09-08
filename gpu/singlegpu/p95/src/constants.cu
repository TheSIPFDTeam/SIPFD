#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0xA44F8FFFFFFFFFFF, 0x000000005EEDC896 };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0xA44F900000000001, 0xE7B2C93A8FEDC896 };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x28D22002B25EA74E, 0x000000004783D593 };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0x7B7D6FFD4DA158B1, 0x000000001769F303 };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0xA44F8FFFFFFFFFFD, 0x000000005EEDC896 };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xA913E3FFFFFFFFFF, 0x0000000017BB7225 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x5227C7FFFFFFFFFF, 0x000000002F76E44B };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0000080000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000100000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000100000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x000006EF79077FBB, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x00000000001853D3, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x00000000001853D3, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x28D22002B25EA74E, 0x000000004783D593 };

__device__ limb_t STRATEGY2[43] = { 18, 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_0[21] = { 9, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_1[21] = { 9, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[20] = { 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[20] = { 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_PC_0[8] = { 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_PC_1[8] = { 4, 2, 1, 1, 1, 2, 1, 1};

__device__ uint32_t STRATEGY3[27] = { 9, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[13] = { 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1};
//const uint32_t STRATEGY3_1[13] = { 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1};

