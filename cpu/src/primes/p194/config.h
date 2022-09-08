#ifndef _CONFIG_P194_H_
#define _CONFIG_P194_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	4
	#define MASK			0x3		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFD, 0xF1A4E6CD3FFFFFFF, 0x85BD88797CF1E74B, 0x0000000000000003 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0xFC6939B34FFFFFFF, 0xE16F621E5F3C79D2 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0xF8D273669FFFFFFF, 0xC2DEC43CBE78F3A5, 0x0000000000000001 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0000000000000000, 0x0000000040000000, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000800000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000800000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0xEFA48D54DBA0A73B, 0x000000002DA861A4, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x0000BB41C3CA78B9, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x00003E6B41437D93, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	7
	#define MASK			0x3		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0xFFFFFFFF, 0x3FFFFFFF, 0xF1A4E6CD, 0x7CF1E74B, 0x85BD8879, 0x00000003 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFFFFFFFF, 0x4FFFFFFF, 0xFC6939B3, 0x5F3C79D2, 0xE16F621E };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFFFFFFFF, 0x9FFFFFFF, 0xF8D27366, 0xBE78F3A5, 0xC2DEC43C, 0x00000001 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00000000, 0x40000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x00000000, 0x00008000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x00000000, 0x00008000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0xDBA0A73B, 0xEFA48D54, 0x2DA861A4, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0xC3CA78B9, 0x0000BB41, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x41437D93, 0x00003E6B, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		194	// Number of bits
#define NBYTES_FIELD	25	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		79
#define PC_DEPTH		26

#define EXPONENT2		94
#define EXPONENT2_BITS	94
#define EXPONENT2_MASK	0x3F
const digit_t STRATEGY2[93] = { 38, 24, 15, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1, 17, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_0[46] = { 21, 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_1[46] = { 21, 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0[45] = { 20, 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1[45] = { 20, 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_[46] = { 20, 11, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 2, 2, 1, 1, 1, 2, 1, 1, 1, 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[22] = { 9, 5, 4, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[22] = { 9, 5, 4, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_0[19] = { 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_1[19] = { 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};

#define EXPONENT3		59
#define EXPONENT3_BITS	94
#define EXPONENT3_MASK	0x3F
const digit_t STRATEGY3[58] = { 21, 13, 8, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_0[29] = { 11, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_1[28] = { 10, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 4, 2, 1, 1, 1, 1, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 47
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 14

#endif
