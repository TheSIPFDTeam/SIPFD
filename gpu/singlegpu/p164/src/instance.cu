#include "api.h"

/* +++ Fixed instance for p164 +++ */

#if RADIX == 64

__device__ proj_t const_E[2] = {{{{0xf1f7cb35a2866c81,0x98c58db34e8752b9,0xd0c9f990},{0xa80cbd40f65dbe23,0x9fc4f94053af98a7,0x38f5aa8b}},{{0x65cb217f158dea23,0x8141677232c53f9f,0xa41a0598},{0x6f5fcfe51228f348,0xfc77de1897dbbc3f,0x58a140359}}},{{{0x8433ecca6d6e052,0xa0ec3f022412900a,0x33c73a891},{0x37b39fb274df61dd,0x38fa1e20d16d8745,0x78d45bc70}},{{0xce3352938b98ef58,0x4c7db6e17e1775f4,0x651e8d9a1},{0x777db270eaabbcb9,0xc5ea07019871f057,0x6a90b3592}}}};
__device__ proj_t const_A2[2] = {{{{0xf1f7cb35a2866c81,0x98c58db34e8752b9,0xd0c9f990},{0xa80cbd40f65dbe23,0x9fc4f94053af98a7,0x38f5aa8b}},{{0x65cb217f158dea23,0x8141677232c53f9f,0xa41a0598},{0x6f5fcfe51228f348,0xfc77de1897dbbc3f,0x58a140359}}},{{{0x8433ecca6d6e052,0xa0ec3f022412900a,0x33c73a891},{0x37b39fb274df61dd,0x38fa1e20d16d8745,0x78d45bc70}},{{0xce3352938b98ef58,0x4c7db6e17e1775f4,0x651e8d9a1},{0x777db270eaabbcb9,0xc5ea07019871f057,0x6a90b3592}}}};
__device__ proj_t const_BASIS[2][3] = {{{{{0x6d8814658b80282c,0x8d266569e20fcfbd,0xba1b2be7},{0x592f51710665fb5b,0x62003e4cf0528e69,0x44f63ace}},{{0x962ec3a04bcba4e6,0x956352d042d69713,0x4fedd077e},{0x8c14d28804dfe7b4,0x1c4df7841bf91a4e,0xc981a287}}},{{{0xb57fb3a342ee6d11,0x5f592dd881001f89,0x6c8719afe},{0x494e0001020bb9a8,0x3b6494fdc9f7e917,0x493ef85b8}},{{0x87804540eb9a677f,0xa008ddd77085a62e,0x60078ca88},{0x442f18725749bb0b,0x743d4d3c06f20879,0x8593cceb9}}},{{{0xf3be254029956d31,0x8804dca0f811c927,0xd4175635},{0xfca219c5d4d07ebb,0x207f6bcbb47f32e8,0x2b2d7a4f2}},{{0x2c2564b763c4d496,0x4525062df1e74b76,0x5e16d7afc},{0xfb7a84b445b1a3c7,0x207a690df6692264,0x7059d4a4d}}}},{{{{0x513532d588a42eb1,0x66c54ff69b2a1c46,0x60afd2893},{0xba774f02d5be888c,0x8c08d308326bc1c5,0x783f40365}},{{0x9805888bef1e45a8,0x9d9266a839703a0,0x10753b684},{0xcebf2b79a7c7d278,0x80e6a10cfccceb5a,0x105f2a6ac}}},{{{0x3c9bc47de5961c90,0x3f8fbc4e13125f39,0x683191681},{0x3a1711fd49e442d5,0x627d1657a2d884dd,0x348716592}},{{0x2d6dc754d953a077,0x84f115b2773f408a,0x323a037e0},{0xea49139d060f2efb,0x94b3282f1f4b74e0,0x7840831fa}}},{{{0xb5aa58d265c8c095,0x6375f9d54fa0f863,0x8563495b6},{0x88286427742cd38b,0xf4a626b1bde50025,0x213d37226}},{{0x88ac83839cd600de,0x3bf4549d5b8fa423,0x70ce9fd73},{0x4ec65dba751187f4,0x639779f6e7868fb0,0x232f925bd}}}}};

#elif RADIX == 32
// arith
#else
#error "not implemented"
#endif

const proj_t h_E[2] = {{{{0xf1f7cb35a2866c81,0x98c58db34e8752b9,0xd0c9f990},{0xa80cbd40f65dbe23,0x9fc4f94053af98a7,0x38f5aa8b}},{{0x65cb217f158dea23,0x8141677232c53f9f,0xa41a0598},{0x6f5fcfe51228f348,0xfc77de1897dbbc3f,0x58a140359}}},{{{0x8433ecca6d6e052,0xa0ec3f022412900a,0x33c73a891},{0x37b39fb274df61dd,0x38fa1e20d16d8745,0x78d45bc70}},{{0xce3352938b98ef58,0x4c7db6e17e1775f4,0x651e8d9a1},{0x777db270eaabbcb9,0xc5ea07019871f057,0x6a90b3592}}}};
const proj_t h_A2[2] = {{{{0xf1f7cb35a2866c81,0x98c58db34e8752b9,0xd0c9f990},{0xa80cbd40f65dbe23,0x9fc4f94053af98a7,0x38f5aa8b}},{{0x65cb217f158dea23,0x8141677232c53f9f,0xa41a0598},{0x6f5fcfe51228f348,0xfc77de1897dbbc3f,0x58a140359}}},{{{0x8433ecca6d6e052,0xa0ec3f022412900a,0x33c73a891},{0x37b39fb274df61dd,0x38fa1e20d16d8745,0x78d45bc70}},{{0xce3352938b98ef58,0x4c7db6e17e1775f4,0x651e8d9a1},{0x777db270eaabbcb9,0xc5ea07019871f057,0x6a90b3592}}}};
const proj_t h_BASIS[2][3] = {{{{{0x6d8814658b80282c,0x8d266569e20fcfbd,0xba1b2be7},{0x592f51710665fb5b,0x62003e4cf0528e69,0x44f63ace}},{{0x962ec3a04bcba4e6,0x956352d042d69713,0x4fedd077e},{0x8c14d28804dfe7b4,0x1c4df7841bf91a4e,0xc981a287}}},{{{0xb57fb3a342ee6d11,0x5f592dd881001f89,0x6c8719afe},{0x494e0001020bb9a8,0x3b6494fdc9f7e917,0x493ef85b8}},{{0x87804540eb9a677f,0xa008ddd77085a62e,0x60078ca88},{0x442f18725749bb0b,0x743d4d3c06f20879,0x8593cceb9}}},{{{0xf3be254029956d31,0x8804dca0f811c927,0xd4175635},{0xfca219c5d4d07ebb,0x207f6bcbb47f32e8,0x2b2d7a4f2}},{{0x2c2564b763c4d496,0x4525062df1e74b76,0x5e16d7afc},{0xfb7a84b445b1a3c7,0x207a690df6692264,0x7059d4a4d}}}},{{{{0x513532d588a42eb1,0x66c54ff69b2a1c46,0x60afd2893},{0xba774f02d5be888c,0x8c08d308326bc1c5,0x783f40365}},{{0x9805888bef1e45a8,0x9d9266a839703a0,0x10753b684},{0xcebf2b79a7c7d278,0x80e6a10cfccceb5a,0x105f2a6ac}}},{{{0x3c9bc47de5961c90,0x3f8fbc4e13125f39,0x683191681},{0x3a1711fd49e442d5,0x627d1657a2d884dd,0x348716592}},{{0x2d6dc754d953a077,0x84f115b2773f408a,0x323a037e0},{0xea49139d060f2efb,0x94b3282f1f4b74e0,0x7840831fa}}},{{{0xb5aa58d265c8c095,0x6375f9d54fa0f863,0x8563495b6},{0x88286427742cd38b,0xf4a626b1bde50025,0x213d37226}},{{0x88ac83839cd600de,0x3bf4549d5b8fa423,0x70ce9fd73},{0x4ec65dba751187f4,0x639779f6e7868fb0,0x232f925bd}}}}};

//##############################
    //#THE SECRET COLLISION IS:
    	//# c0 = 0;
    	//# k0 = 9510516926080907440
    	//# c1 = 1;
    	//# k1 = 3894202106796621600
    //# THE KERNEL POINTS ARE
    	//#K0_x = 0x00000007250eec0ed42461a6df8fdf81e20de155c714fd5b + i * 0x0000000180235a0db38734ca0765e01de5b925883fb79e19;
    	//#K1_x = 0x0000000277a4230910a6896087600ce46e1bb69139e8b1f0 + i * 0x00000003c8f9e841d0c680cf8215d2dafe33d327c2d493dd;
//# Dont tell anyone!
//##############################

