#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0x383FFFFFFFFFFFFF, 0x0017AA3C6895382F };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x3840000000000001, 0xB3F3BA3C6895382F };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x93C0000000000AD1, 0x0007A898C9FB4940 };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0xA47FFFFFFFFFF52E, 0x001001A39E99EEEE };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0x383FFFFFFFFFFFFD, 0x0017AA3C6895382F };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xCE0FFFFFFFFFFFFF, 0x0005EA8F1A254E0B };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x9C1FFFFFFFFFFFFF, 0x000BD51E344A9C17 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0020000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000000002000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000000002000000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x0013BFEFA65ABB83, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x000000000290D741, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x000000000290D741, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x93C0000000000AD1, 0x0007A898C9FB4940 };

__device__ limb_t STRATEGY2[53] = { 23, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_0[26] = { 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_1[26] = { 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[25] = { 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[25] = { 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};

__device__ limb_t STRATEGY2_PC_0[10] = { 4, 3, 2, 1, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_PC_1[10] = { 4, 3, 2, 1, 1, 1, 1, 2, 1, 1};

__device__ uint32_t STRATEGY3[33] = { 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
//const uint32_t STRATEGY3_0[16] = { 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
//const uint32_t STRATEGY3_1[16] = { 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};

