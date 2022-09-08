#ifndef _CONFIG_P111_H_
#define _CONFIG_P111_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	2
	#define MASK			0x7FFFFFFFFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0x944FFFFFFFFFFFFD, 0x0000510E67901461 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x6513FFFFFFFFFFFF, 0x0000144399E40518 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xCA27FFFFFFFFFFFF, 0x0000288733C80A30 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0010000000000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000000004000000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000000004000000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x0006954FE21E3E81, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x000000000290D741, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x000000000290D741, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	4
	#define MASK			0x7FFF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0x944FFFFF, 0x67901461, 0x0000510E };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0x6513FFFF, 0x99E40518, 0x00001443 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0xCA27FFFF, 0x33C80A30, 0x00002887 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00100000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x04000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x04000000, 0x00000000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0xE21E3E81, 0x0006954F, 0x00000000, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x0290D741, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x0290D741, 0x00000000, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		111	// Number of bits
#define NBYTES_FIELD	14	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		197
#define PC_DEPTH		14

#define EXPONENT2		52
#define EXPONENT2_BITS	52
#define EXPONENT2_MASK	0xF
const digit_t STRATEGY2[51] = { 21, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_0[25] = { 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_1[25] = { 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0[24] = { 10, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1[24] = { 10, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_[25] = { 10, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[11] = { 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[11] = { 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_PC_0[10] = { 4, 3, 2, 1, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_1[10] = { 4, 3, 2, 1, 1, 1, 1, 2, 1, 1};

#define EXPONENT3		32
#define EXPONENT3_BITS	51
#define EXPONENT3_MASK	0x7
const digit_t STRATEGY3[31] = { 11, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_0[15] = { 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
const digit_t STRATEGY3_1[15] = { 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 26
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 8

#endif
