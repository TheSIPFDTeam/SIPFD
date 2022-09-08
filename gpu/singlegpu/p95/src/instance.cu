#include "api.h"

/* +++ Fixed instance for p95 +++ */

#if RADIX == 64

__device__ proj_t const_E[2] = {{{{0x407e8a4ebc31086a,0x14f5b3d6},{0xed0000b2cddf67a6,0x27da44db}},{{0x268f66938fe84533,0xb043a65},{0x1cedbbf09b57d802,0x15e0a6c3}}},{{{0x1e0013ef9f76f5be,0x56756cf5},{0xf0abad27cc68ea26,0x538cc53b}},{{0xc0a6a83cafa64bdd,0x39bb7f17},{0x561c7e13daeb1f90,0x489ff805}}}};
__device__ proj_t const_A2[2] = {{{{0x407e8a4ebc31086a,0x14f5b3d6},{0xed0000b2cddf67a6,0x27da44db}},{{0x268f66938fe84533,0xb043a65},{0x1cedbbf09b57d802,0x15e0a6c3}}},{{{0x1e0013ef9f76f5be,0x56756cf5},{0xf0abad27cc68ea26,0x538cc53b}},{{0xc0a6a83cafa64bdd,0x39bb7f17},{0x561c7e13daeb1f90,0x489ff805}}}};
__device__ proj_t const_BASIS[2][3] = {{{{{0x5bd93f89dd19c259,0x532f4cdf},{0xdda80926b8c33c56,0x433c9dbf}},{{0x1e84e61c36f806d0,0x25e83d05},{0xf94c1cedbee2b2e5,0x455830b7}}},{{{0x7b0463647a553cd0,0x26f605a2},{0x251cc2ee080e7689,0x28278d55}},{{0x50842e8e59c3faaa,0x376cbc7c},{0xcbd50cf773389cdf,0x5325d483}}},{{{0xb21c66134e41d393,0x4cd05481},{0xf49353ec1e326a9b,0x2261e5a1}},{{0x7ec111269394744c,0x3503bcd5},{0xd00012b35a55c90f,0x5d158f2}}}},{{{{0x3abc044c53ecac94,0x14ed9130},{0x7862f2bbd3b708c5,0x5abbbbb}},{{0xb99ffe2d920ae658,0x70d9236},{0x55be28d3f189500e,0x224a8665}}},{{{0x9ab91f771bd2706e,0x13a5b50c},{0x2cd01cba2f4a4543,0x53909a9a}},{{0xdc2b6f9021cdc7b8,0x49de0d7},{0x9137e9fa448878da,0x5e80e93a}}},{{{0xb688ef6631f336bb,0x2b12518},{0x15f82b00c6ae925f,0x440fb061}},{{0x25f07fdde26da7fe,0x48a69f0a},{0xb90ea24b0665ff19,0x1a71711b}}}}};

#elif RADIX == 32
// arith
#else
#error "not implemented"
#endif

const proj_t h_E[2] = {{{{0x407e8a4ebc31086a,0x14f5b3d6},{0xed0000b2cddf67a6,0x27da44db}},{{0x268f66938fe84533,0xb043a65},{0x1cedbbf09b57d802,0x15e0a6c3}}},{{{0x1e0013ef9f76f5be,0x56756cf5},{0xf0abad27cc68ea26,0x538cc53b}},{{0xc0a6a83cafa64bdd,0x39bb7f17},{0x561c7e13daeb1f90,0x489ff805}}}};
const proj_t h_A2[2] = {{{{0x407e8a4ebc31086a,0x14f5b3d6},{0xed0000b2cddf67a6,0x27da44db}},{{0x268f66938fe84533,0xb043a65},{0x1cedbbf09b57d802,0x15e0a6c3}}},{{{0x1e0013ef9f76f5be,0x56756cf5},{0xf0abad27cc68ea26,0x538cc53b}},{{0xc0a6a83cafa64bdd,0x39bb7f17},{0x561c7e13daeb1f90,0x489ff805}}}};
const proj_t h_BASIS[2][3] = {{{{{0x5bd93f89dd19c259,0x532f4cdf},{0xdda80926b8c33c56,0x433c9dbf}},{{0x1e84e61c36f806d0,0x25e83d05},{0xf94c1cedbee2b2e5,0x455830b7}}},{{{0x7b0463647a553cd0,0x26f605a2},{0x251cc2ee080e7689,0x28278d55}},{{0x50842e8e59c3faaa,0x376cbc7c},{0xcbd50cf773389cdf,0x5325d483}}},{{{0xb21c66134e41d393,0x4cd05481},{0xf49353ec1e326a9b,0x2261e5a1}},{{0x7ec111269394744c,0x3503bcd5},{0xd00012b35a55c90f,0x5d158f2}}}},{{{{0x3abc044c53ecac94,0x14ed9130},{0x7862f2bbd3b708c5,0x5abbbbb}},{{0xb99ffe2d920ae658,0x70d9236},{0x55be28d3f189500e,0x224a8665}}},{{{0x9ab91f771bd2706e,0x13a5b50c},{0x2cd01cba2f4a4543,0x53909a9a}},{{0xdc2b6f9021cdc7b8,0x49de0d7},{0x9137e9fa448878da,0x5e80e93a}}},{{{0xb688ef6631f336bb,0x2b12518},{0x15f82b00c6ae925f,0x440fb061}},{{0x25f07fdde26da7fe,0x48a69f0a},{0xb90ea24b0665ff19,0x1a71711b}}}}};

//##############################
    //#THE SECRET COLLISION IS:
    	//# c0 = 0;
    	//# k0 = 1640318
    	//# c1 = 1;
    	//# k1 = 1128617
    //# THE KERNEL POINTS ARE
    	//#K0_x = 0x000000005755f3dda5075be67fc58a90 + i * 0x00000000146397e5545e6fa7e47de4c7;
    	//#K1_x = 0x00000000355c611ac353135dfc5f3efe + i * 0x00000000270dd71e04b1824fc7c9a6fd;
//# Dont tell anyone!
//##############################

