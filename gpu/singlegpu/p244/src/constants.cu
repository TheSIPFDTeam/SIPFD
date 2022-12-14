#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x4BFFFFFFFFFFFFFF, 0x8191DDD880E8B63F, 0x000AB058140F738F };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x0000000000000001, 0x4C00000000000000, 0x8191DDD880E8B63F, 0x7E9AB058140F738F };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x00000000000017F3, 0xDC00000000000000, 0xE79BF7E8BABB5216, 0x0002B29791F36A23 };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0xFFFFFFFFFFFFE80C, 0x6FFFFFFFFFFFFFFF, 0x99F5E5EFC62D6428, 0x0007FDC0821C096B };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFD, 0x4BFFFFFFFFFFFFFF, 0x8191DDD880E8B63F, 0x000AB058140F738F };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0xD2FFFFFFFFFFFFFF, 0xE0647776203A2D8F, 0x0002AC160503DCE3 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0xA5FFFFFFFFFFFFFF, 0xC0C8EEEC40745B1F, 0x0005582C0A07B9C7 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0000000000000000, 0x0200000000000000, 0x0000000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x1FD29F05F9E837D9, 0x00007B6A43A7EF90, 0x0000000000000000, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x00B1BF6CD930979B, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x003B3FCEF3103289, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x00000000000017F3, 0xDC00000000000000, 0xE79BF7E8BABB5216, 0x0002B29791F36A23 };

__device__ limb_t STRATEGY2[121] = { 55, 29, 17, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 13, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 24, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_0[60] = { 27, 16, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_1[60] = { 27, 16, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[59] = { 27, 15, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[59] = { 27, 15, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};

__device__ limb_t STRATEGY2_PC_0[39] = { 17, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_PC_1[39] = { 17, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};

__device__ uint32_t STRATEGY3[70] = { 23, 15, 12, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 8, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[34] = { 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
//const uint32_t STRATEGY3_1[35] = { 13, 8, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};

