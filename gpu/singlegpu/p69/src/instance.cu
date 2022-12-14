#include "api.h"

/* +++ Fixed instance for p69 +++ */

#if RADIX == 64

__device__ proj_t const_E[2] = {{{{0xb12d912bd3bbfc,0xa},{0x3c187ce432397e77,0x2}},{{0x232f65e70d602393,0x12},{0xed2380492aa1bfb4,0xc}}},{{{0x40f195866f9fb594,0x10},{0x27458d10e38503bc,0x10}},{{0xf89a220e122938ea,0x3},{0x794816ef4e983f77,0xd}}}};
__device__ proj_t const_A2[2] = {{{{0xb12d912bd3bbfc,0xa},{0x3c187ce432397e77,0x2}},{{0x232f65e70d602393,0x12},{0xed2380492aa1bfb4,0xc}}},{{{0x40f195866f9fb594,0x10},{0x27458d10e38503bc,0x10}},{{0xf89a220e122938ea,0x3},{0x794816ef4e983f77,0xd}}}};
__device__ proj_t const_BASIS[2][3] = {{{{{0x30e6c9c00c3f642,0x10},{0x356e02f6014c1679,0x12}},{{0xe0670d20ad7f4ae3,0x2},{0xbc80674e54c7eb51,0x3}}},{{{0xea45741583986df5,0x0},{0x3bf72684894ae7c4,0x6}},{{0x10cba751913aebc6,0xa},{0x7802e9eda33de5e6,0xb}}},{{{0x49e18c4672df122f,0x6},{0xf162c90642accd8c,0xd}},{{0x5be8d143fa2749f1,0x9},{0x3401e6d09ebd608f,0x3}}}},{{{{0x80fbc99cee947bf8,0xf},{0x7594c274d54ac957,0x8}},{{0xe130032664feeb38,0xf},{0x10b36dbf247fbbc4,0x0}}},{{{0x2b25c17854c1c390,0x10},{0x244ab2aae85fe507,0x5}},{{0x82df5622386a0f58,0x4},{0xc2004194e01e932,0x8}}},{{{0x314c9339e82cf4a2,0x10},{0xe20cb2911d6a55e0,0x8}},{{0xb26c2e7c4a242a69,0xb},{0x9f82169d741c4402,0x2}}}}};

#elif RADIX == 32

#else
#error "Not implemented"
#endif

const proj_t h_E[2] = {{{{0xb12d912bd3bbfc,0xa},{0x3c187ce432397e77,0x2}},{{0x232f65e70d602393,0x12},{0xed2380492aa1bfb4,0xc}}},{{{0x40f195866f9fb594,0x10},{0x27458d10e38503bc,0x10}},{{0xf89a220e122938ea,0x3},{0x794816ef4e983f77,0xd}}}};
const proj_t h_A2[2] = {{{{0xb12d912bd3bbfc,0xa},{0x3c187ce432397e77,0x2}},{{0x232f65e70d602393,0x12},{0xed2380492aa1bfb4,0xc}}},{{{0x40f195866f9fb594,0x10},{0x27458d10e38503bc,0x10}},{{0xf89a220e122938ea,0x3},{0x794816ef4e983f77,0xd}}}};
const proj_t h_BASIS[2][3] = {{{{{0x30e6c9c00c3f642,0x10},{0x356e02f6014c1679,0x12}},{{0xe0670d20ad7f4ae3,0x2},{0xbc80674e54c7eb51,0x3}}},{{{0xea45741583986df5,0x0},{0x3bf72684894ae7c4,0x6}},{{0x10cba751913aebc6,0xa},{0x7802e9eda33de5e6,0xb}}},{{{0x49e18c4672df122f,0x6},{0xf162c90642accd8c,0xd}},{{0x5be8d143fa2749f1,0x9},{0x3401e6d09ebd608f,0x3}}}},{{{{0x80fbc99cee947bf8,0xf},{0x7594c274d54ac957,0x8}},{{0xe130032664feeb38,0xf},{0x10b36dbf247fbbc4,0x0}}},{{{0x2b25c17854c1c390,0x10},{0x244ab2aae85fe507,0x5}},{{0x82df5622386a0f58,0x4},{0xc2004194e01e932,0x8}}},{{{0x314c9339e82cf4a2,0x10},{0xe20cb2911d6a55e0,0x8}},{{0xb26c2e7c4a242a69,0xb},{0x9f82169d741c4402,0x2}}}}};

//##############################
    //#THE SECRET COLLISION IS:
    	//# c0 = 0;
    	//# k0 = 9329
    	//# c1 = 1;
    	//# k1 = 9907
    //# THE KERNEL POINTS ARE
    	//#K0_x = 0x00000000000000089cf052142e207b96 + i * 0x00000000000000126398ae2027e385d7;
    	//#K1_x = 0x0000000000000008895d8b7f26eab747 + i * 0x00000000000000024637d6837e4da2a3;
//# Dont tell anyone!
//##############################

