#ifndef _CONFIG_P184_H_
#define _CONFIG_P184_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	3
	#define MASK			0xFFFFFFFFFFFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFD, 0xB7DEBAC72FFFFFFF, 0x00C167473A7BDA64 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x2DF7AEB1CBFFFFFF, 0x003059D1CE9EF699 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x5BEF5D6397FFFFFF, 0x0060B3A39D3DED32 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0000000000000000, 0x0000000010000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000400000000000, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000400000000000, 0x0000000000000000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0xDA671E0B00A70633, 0x000000000010088F, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x000006EF79077FBB, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x0000024FD3027FE9, 0x0000000000000000, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	6
	#define MASK			0xFFFFFF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0xFFFFFFFF, 0x2FFFFFFF, 0xB7DEBAC7, 0x3A7BDA64, 0x00C16747 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xCBFFFFFF, 0x2DF7AEB1, 0xCE9EF699, 0x003059D1 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFFFFFFFF, 0x97FFFFFF, 0x5BEF5D63, 0x9D3DED32, 0x0060B3A3 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00000000, 0x10000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x00000000, 0x00004000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x00000000, 0x00004000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x00A70633, 0xDA671E0B, 0x0010088F, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x79077FBB, 0x000006EF, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0xD3027FE9, 0x0000024F, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		184	// Number of bits
#define NBYTES_FIELD	23	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		193
#define PC_DEPTH		24

#define EXPONENT2		92
#define EXPONENT2_BITS	92
#define EXPONENT2_MASK	0xF
const digit_t STRATEGY2[91] = { 40, 21, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 17, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_0[45] = { 20, 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_1[45] = { 20, 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0[44] = { 19, 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 8, 4, 3, 2, 1, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1[44] = { 19, 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 8, 4, 3, 2, 1, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_[45] = { 19, 11, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 2, 2, 1, 1, 1, 2, 1, 1, 1, 8, 5, 2, 2, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[21] = { 9, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[21] = { 9, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_0[20] = { 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_1[20] = { 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};

#define EXPONENT3		53
#define EXPONENT3_BITS	85
#define EXPONENT3_MASK	0x1F
const digit_t STRATEGY3[52] = { 19, 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1};
const digit_t STRATEGY3_0[26] = { 9, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_1[25] = { 9, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 46
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 13

#endif
