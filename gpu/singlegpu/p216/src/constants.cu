#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x81538FFFFFFFFFFF, 0xB16F1FC2ACE470D3, 0x00000000009A6555 };
__constant__ limb_t __mu[NWORDS_FIELD] = { 0x0000000000000001, 0x8153900000000000, 0xB16F1FC2ACE470D3, 0x0F14B266B19A6555 };
__constant__ limb_t mont_one[NWORDS_FIELD] = { 0x000001A877A768C4, 0x1985C00000000000, 0x714CC0F0B5C5FC14, 0x000000000031989F };
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = { 0xFFFFFE578858973B, 0x67CDCFFFFFFFFFFF, 0x40225ED1F71E74BF, 0x000000000068CCB6 };
__constant__ limb_t bigone[NWORDS_FIELD] = { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFD, 0x81538FFFFFFFFFFF, 0xB16F1FC2ACE470D3, 0x00000000009A6555 };
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0xE054E3FFFFFFFFFF, 0x6C5BC7F0AB391C34, 0x0000000000269955 };
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0xC0A9C7FFFFFFFFFF, 0xD8B78FE156723869, 0x00000000004D32AA };

__device__ limb_t BOUND2[NWORDS_FIELD] = { 0x0000000000000000, 0x0000080000000000, 0x0000000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = { 0x0010000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = { 0x0010000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

__device__ limb_t BOUND3[NWORDS_FIELD] = { 0x7932278C797EBD01, 0x0000002B56D4AF8F, 0x0000000000000000, 0x0000000000000000 };
__device__ limb_t BOUND3_0[NWORDS_FIELD] = { 0x0006954FE21E3E81, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
__device__ limb_t BOUND3_1[NWORDS_FIELD] = { 0x000231C54B5F6A2B, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

const limb_t h_mont_one[NWORDS_FIELD] = { 0x000001A877A768C4, 0x1985C00000000000, 0x714CC0F0B5C5FC14, 0x000000000031989F };

__device__ limb_t STRATEGY2[107] = { 48, 27, 15, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 21, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_0[53] = { 23, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_1[53] = { 23, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_REDUCED_0[52] = { 22, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 9, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
__device__ limb_t STRATEGY2_REDUCED_1[52] = { 22, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 9, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};

__device__ limb_t STRATEGY2_PC_0[27] = { 12, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
__device__ limb_t STRATEGY2_PC_1[27] = { 12, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};

__device__ uint32_t STRATEGY3[64] = { 22, 15, 9, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 8, 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_0[31] = { 11, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};
//const uint32_t STRATEGY3_1[32] = { 12, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1};

