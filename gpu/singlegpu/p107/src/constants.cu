#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0x0AA3FFFFFFFFFFFF, 0x00000437A72CDB60 };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x0AA4000000000001, 0x93713D47A72CDB60 };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x8C540000003CB373, 0x000002981EF2CDFA };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0x7E4FFFFFFFC34C8C, 0x0000019F883A0D65 };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0x0AA3FFFFFFFFFFFD, 0x00000437A72CDB60 };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x02A8FFFFFFFFFFFF, 0x0000010DE9CB36D8 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x0551FFFFFFFFFFFF, 0x0000021BD3966DB0 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0002000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000800000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000800000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x000231C54B5F6A2B, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x0000000000DAF26B, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x0000000000DAF26B, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x8C540000003CB373, 0x000002981EF2CDFA };

__device__ limb_t STRATEGY2[49] = { 22, 12, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 9, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_0[24] = { 10, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_1[24] = { 10, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[23] = { 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[23] = { 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_PC_0[9] = { 4, 2, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_PC_1[9] = { 4, 2, 2, 1, 1, 1, 2, 1, 1};

__device__ uint32_t STRATEGY3[31] = { 11, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[15] = { 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
//const uint32_t STRATEGY3_1[15] = { 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};

