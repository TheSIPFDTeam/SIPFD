#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x61BFFFFFFFFFFFFF, 0x9EF3CA40CDEF4B2E, 0x00000B731BC6D479 };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x0000000000000001, 0x61C0000000000000, 0x9EF3CA40CDEF4B2E, 0xC6531B731BC6D479 };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x0000000000165BD8, 0x4600000000000000, 0xD7D69D8DB862AD9E, 0x000008EAA041D4ED };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0xFFFFFFFFFFE9A427, 0x1BBFFFFFFFFFFFFF, 0xC71D2CB3158C9D90, 0x000002887B84FF8B };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFD, 0x61BFFFFFFFFFFFFF, 0x9EF3CA40CDEF4B2E, 0x00000B731BC6D479 };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x986FFFFFFFFFFFFF, 0x67BCF290337BD2CB, 0x000002DCC6F1B51E };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x30DFFFFFFFFFFFFF, 0xCF79E52066F7A597, 0x000005B98DE36A3C };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0000000000000000, 0x0020000000000000, 0x0000000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0200000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0200000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x0A9B8A57534D67F3, 0x000029236BE2A530, 0x0000000000000000, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x003B3FCEF3103289, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x003B3FCEF3103289, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x0000000000165BD8, 0x4600000000000000, 0xD7D69D8DB862AD9E, 0x000008EAA041D4ED };

__device__ limb_t STRATEGY2[117] = { 51, 29, 17, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 13, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 22, 13, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 9, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_0[58] = { 27, 15, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_1[58] = { 27, 15, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[57] = { 27, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[57] = { 27, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};

__device__ limb_t STRATEGY2_PC_0[35] = { 16, 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_PC_1[35] = { 16, 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};

__device__ uint32_t STRATEGY3[69] = { 23, 17, 11, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 8, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[34] = { 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
//const uint32_t STRATEGY3_1[34] = { 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};

