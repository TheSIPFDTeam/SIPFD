#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0xF0680FCA98D3FFFF, 0x000000085B8D5B04 };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x0000000000000001, 0xF0680FCA98D40000, 0x61CC6F985B8D5B04 };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x000000001EA17625, 0x9D5B5FF2315C0000, 0x00000006CA562E2D };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0xFFFFFFFFE15E89DA, 0x530CAFD86777FFFF, 0x0000000191372CD7 };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFD, 0xF0680FCA98D3FFFF, 0x000000085B8D5B04 };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x3C1A03F2A634FFFF, 0x0000000216E356C1 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x783407E54C69FFFF, 0x000000042DC6AD82 };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0000000000000000, 0x0000000000020000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0000008000000000, 0x0000000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0000008000000000, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x75F97DF71BDC05F9, 0x00000000000001E0, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x00000015EB5EE84B, 0x0000000000000000, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x000000074E74F819, 0x0000000000000000, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x000000001EA17625, 0x9D5B5FF2315C0000, 0x00000006CA562E2D };

__device__ limb_t STRATEGY2[81] = { 34, 21, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 16, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
__device__ limb_t STRATEGY2_0[40] = { 17, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_1[40] = { 17, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[39] = { 17, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[39] = { 17, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_PC_0[17] = { 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_PC_1[17] = { 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};

__device__ uint32_t STRATEGY3[46] = { 17, 11, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[22] = { 8, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_1[23] = { 9, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1};

