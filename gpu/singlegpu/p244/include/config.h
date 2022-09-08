#ifndef _CONFIG_P244_H_
#define _CONFIG_P244_H_
#include <stdio.h>
#include <inttypes.h>

#define RADIX 64
#define LOG2RADIX 6
typedef uint64_t limb_t;
#define NWORDS_FIELD 4
#define MASK 0xFFFFF // Mask for the last 32-bits word

static inline void TOHEX(limb_t x) { printf("%016" PRIx64, x); };
static inline void SPRINTF(char *input, limb_t x) { sprintf(input, "%" PRIx64, x); };

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

#define NBITS_FIELD 244 // Number of bits
#define NBYTES_FIELD 31  // Number of bytes

// Conversion macro from number of bits to number of bytes')
#define NBITS_TO_NBYTES(nbits) (((nbits)+7)/8)
// Conversion macro from number of bits to number of computer words')
#define NBITS_TO_NWORDS(nbits) (((nbits)+(sizeof(limb_t)*8)-1)/(sizeof(limb_t)*8))
// Conversion macro from number of bytes to number of computer words')
#define NBYTES_TO_NWORDS(nbytes) (((nbytes)+sizeof(limb_t)-1)/sizeof(limb_t))

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1')
#define COFACTOR 473
#define PC_DEPTH 20
#define PC_STRATEGY_SIZE_0 39
#define PC_STRATEGY_SIZE_1 39

#define EXPONENT2 122
#define EXPONENT2_BITS 122
#define EXPONENT2_MASK 0x3
#define EXP2_MASK_GPU 0x3FFFFFF

#define EXP0 ((EXPONENT2 - (EXPONENT2 >> 1)) - 1)
#define EXP1 ((EXPONENT2 >> 1) - 1)
#define EXP20_BITS ((EXPONENT2_BITS - (EXPONENT2_BITS >> 1)) - 1)
#define EXP21_BITS ((EXPONENT2_BITS >> 1) - 1)
#define EBITS_MAX EXP0
#define LOG2OFE 12

extern __device__ limb_t STRATEGY2[121];
extern __device__ limb_t STRATEGY2_0[60];
extern __device__ limb_t STRATEGY2_1[60];

extern __device__ limb_t STRATEGY2_REDUCED_0[59];
extern __device__ limb_t STRATEGY2_REDUCED_1[59];
extern __device__ limb_t STRATEGY2_PC_0[39];
extern __device__ limb_t STRATEGY2_PC_1[39];

#define EXPONENT3 71
#define EXPONENT3_BITS 113
#define EXPONENT3_MASK 0x1
#define EXP3_MASK_GPU 0x1FFFF

extern __device__ uint32_t STRATEGY3[70];
extern const uint32_t STRATEGY3_0[35];
extern const uint32_t STRATEGY3_1[34];

#endif

