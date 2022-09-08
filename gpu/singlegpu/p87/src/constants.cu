#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0x667C50FFFFFFFFFF, 0x00000000005EC8CB };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x667C510000000001, 0x5DBFC2EA91FFC8CB };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x30165FB36C10EDB3, 0x000000000053CEF5 };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0x3665F14C93EF124C, 0x00000000000AF9D6 };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0x667C50FFFFFFFFFD, 0x00000000005EC8CB };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xD99F143FFFFFFFFF, 0x000000000017B232 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xB33E287FFFFFFFFF, 0x00000000002F6465 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0000008000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000040000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000040000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x000000C546562AA3, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x0000000000081BF1, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x0000000000081BF1, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x30165FB36C10EDB3, 0x000000000053CEF5 };

__device__ limb_t STRATEGY2[39] = { 17, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_0[19] = { 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_1[19] = { 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[18] = { 8, 4, 3, 2, 1, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[18] = { 8, 4, 3, 2, 1, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_PC_0[7] = { 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_PC_1[7] = { 4, 2, 1, 1, 2, 1, 1};

__device__ uint32_t STRATEGY3[25] = { 9, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[12] = { 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
//const uint32_t STRATEGY3_1[12] = { 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};

