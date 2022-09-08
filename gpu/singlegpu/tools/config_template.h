#ifndef _CONFIG_P<bitlength>_H_
#define _CONFIG_P<bitlength>_H_
#include <stdio.h>
#include <inttypes.h>

#define RADIX <radix>
#define LOG2RADIX <log2radix>
typedef uint<arch>_t limb_t;
#define NWORDS_FIELD <nwordsfield>
#define MASK 0x<mask32> // Mask for the last 32-bits word

static inline void TOHEX(limb_t x) { printf("%0<rephex>" <prix>, x); };
static inline void SPRINTF(char *input, limb_t x) { sprintf(input, <sprintf>); };

extern __constant__ limb_t __p[NWORDS_FIELD];
extern __constant__ limb_t __mu[NWORDS_FIELD];
extern __constant__ limb_t mont_one[NWORDS_FIELD];
extern __constant__ limb_t mont_minus_one[NWORDS_FIELD];
extern __constant__ limb_t bigone[NWORDS_FIELD];

extern __device__ limb_t P_MINUS_TWO[NWORDS_FIELD];
extern __device__ limb_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD];
extern __device__ limb_t P_MINUS_ONE_HALVES[NWORDS_FIELD];

extern __device__ limb_t BOUND2[NWORDS_FIELD];
extern __constant__ limb_t BOUND2_0[NWORDS_FIELD];
extern __constant__ limb_t BOUND2_1[NWORDS_FIELD];

extern __device__ limb_t BOUND3[NWORDS_FIELD];
extern __device__ limb_t BOUND3_0[NWORDS_FIELD];
extern __device__ limb_t BOUND3_1[NWORDS_FIELD];

extern const limb_t h_mont_one[NWORDS_FIELD];

#define NBITS_FIELD <bitlength> // Number of bits
#define NBYTES_FIELD <byteslength>  // Number of bytes

// Conversion macro from number of bits to number of bytes')
#define NBITS_TO_NBYTES(nbits) (((nbits)+7)/8)
// Conversion macro from number of bits to number of computer words')
#define NBITS_TO_NWORDS(nbits) (((nbits)+(sizeof(limb_t)*8)-1)/(sizeof(limb_t)*8))
// Conversion macro from number of bytes to number of computer words')
#define NBYTES_TO_NWORDS(nbytes) (((nbytes)+sizeof(limb_t)-1)/sizeof(limb_t))

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1')
#define COFACTOR <cofactor>
#define PC_DEPTH <pc_depth>
#define PC_STRATEGY_SIZE_0 <pc_strategy_size_0>
#define PC_STRATEGY_SIZE_1 <pc_strategy_size_1>

#define EXPONENT2 <e2>
#define EXPONENT2_BITS <e2>
#define EXPONENT2_MASK 0x<mask2>
#define EXP2_MASK_GPU 0x<expmk2>

#define EXP0 ((EXPONENT2 - (EXPONENT2 >> 1)) - 1)
#define EXP1 ((EXPONENT2 >> 1) - 1)
#define EXP20_BITS ((EXPONENT2_BITS - (EXPONENT2_BITS >> 1)) - 1)
#define EXP21_BITS ((EXPONENT2_BITS >> 1) - 1)
#define EBITS_MAX EXP0
#define LOG2OFE <log2ofe>

extern __device__ limb_t STRATEGY2[<e2min1>];
extern __device__ limb_t STRATEGY2_0[<ceildive2>];
extern __device__ limb_t STRATEGY2_1[<e2div21>];

extern __device__ limb_t STRATEGY2_REDUCED_0[<ceildive22>];
extern __device__ limb_t STRATEGY2_REDUCED_1[<e2div22>];
extern __device__ limb_t STRATEGY2_PC_0[<pc_strategy_size_0>];
extern __device__ limb_t STRATEGY2_PC_1[<pc_strategy_size_1>];

#define EXPONENT3 <e3>
#define EXPONENT3_BITS <log2e3>
#define EXPONENT3_MASK 0x<mask3>
#define EXP3_MASK_GPU 0x<expmk3>

extern __device__ uint32_t STRATEGY3[<e3min1>];
extern const uint32_t STRATEGY3_0[<ceildive3>];
extern const uint32_t STRATEGY3_1[<e3div21>];

#endif
