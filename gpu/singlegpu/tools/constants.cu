#include "api.h"

__constant__ limb_t __p[NWORDS_FIELD] = <prime>;
__constant__ limb_t __mu[NWORDS_FIELD] = <mu>;
__constant__ limb_t mont_one[NWORDS_FIELD] = <montone>;
__constant__ limb_t mont_minus_one[NWORDS_FIELD] = <mminusone>;
__constant__ limb_t bigone[NWORDS_FIELD] = <bigone>;

__device__ limb_t P_MINUS_TWO[NWORDS_FIELD] = <p_minus_two>;
__device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = <p_minus_three_q>;
__device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = <p_min_one_h>;

__device__ limb_t BOUND2[NWORDS_FIELD] = <bound2>;
__constant__ limb_t BOUND2_0[NWORDS_FIELD] = <bound21>;
__constant__ limb_t BOUND2_1[NWORDS_FIELD] = <bound20>;

__device__ limb_t BOUND3[NWORDS_FIELD] = <bound3>;
__device__ limb_t BOUND3_0[NWORDS_FIELD] = <bound31>;
__device__ limb_t BOUND3_1[NWORDS_FIELD] = <bound30>;

const limb_t h_mont_one[NWORDS_FIELD] = <montone>;

__device__ limb_t STRATEGY2[<e2min1>] = <strategy2>
__device__ limb_t STRATEGY2_0[<ceildive2>] = <strategy21>
__device__ limb_t STRATEGY2_1[<e2div21>] = <strategy20>

__device__ limb_t STRATEGY2_REDUCED_0[<ceildive22>] = <strategy2_red_1>
__device__ limb_t STRATEGY2_REDUCED_1[<e2div22>] = <strategy2_red_0>

__device__ limb_t STRATEGY2_PC_0[<pc_strategy_size_0>] = <strategy2_pc_0>
__device__ limb_t STRATEGY2_PC_1[<pc_strategy_size_1>] = <strategy2_pc_1>

__device__ uint32_t STRATEGY3[<e3min1>] = <strategy3>
