#ifndef _CONFIG_P118_H_
#define _CONFIG_P118_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	2
	#define MASK			0x3FFFFFFFFFFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0x82FFFFFFFFFFFFFD, 0x0027939F3C5BD1C1 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x60BFFFFFFFFFFFFF, 0x0009E4E7CF16F470 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xC17FFFFFFFFFFFFF, 0x0013C9CF9E2DE8E0 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0100000000000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000000010000000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000000010000000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x02153E468B91C6D1, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x0000000017179149, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x0000000017179149, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	4
	#define MASK			0x3FFFFF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0x82FFFFFF, 0x3C5BD1C1, 0x0027939F };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0x60BFFFFF, 0xCF16F470, 0x0009E4E7 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0xC17FFFFF, 0x9E2DE8E0, 0x0013C9CF };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x01000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x10000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x10000000, 0x00000000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x8B91C6D1, 0x02153E46, 0x00000000, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x17179149, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x17179149, 0x00000000, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		118	// Number of bits
#define NBYTES_FIELD	15	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		19
#define PC_DEPTH		15

#define EXPONENT2		56
#define EXPONENT2_BITS	56
#define EXPONENT2_MASK	0xFF
const digit_t STRATEGY2[55] = { 24, 15, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_0[27] = { 12, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_1[27] = { 12, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0[26] = { 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1[26] = { 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_[27] = { 12, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[12] = { 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[12] = { 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_PC_0[11] = { 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_PC_1[11] = { 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};

#define EXPONENT3		36
#define EXPONENT3_BITS	58
#define EXPONENT3_MASK	0x3
const digit_t STRATEGY3[35] = { 13, 8, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY3_0[17] = { 6, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
const digit_t STRATEGY3_1[17] = { 6, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 28
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 9

#endif
