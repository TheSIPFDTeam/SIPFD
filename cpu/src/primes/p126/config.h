#ifndef _CONFIG_P126_H_
#define _CONFIG_P126_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	2
	#define MASK			0x3FFFFFFFFFFFFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0x2FFFFFFFFFFFFFFD, 0x27F79AE995FD16CA };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x8BFFFFFFFFFFFFFF, 0x09FDE6BA657F45B2 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x17FFFFFFFFFFFFFF, 0x13FBCD74CAFE8B65 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x1000000000000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000000040000000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000000040000000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x02153E468B91C6D1, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x0000000017179149, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x0000000017179149, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	4
	#define MASK			0x3FFFFFFF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0x2FFFFFFF, 0x95FD16CA, 0x27F79AE9 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0x8BFFFFFF, 0x657F45B2, 0x09FDE6BA };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0x17FFFFFF, 0xCAFE8B65, 0x13FBCD74 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x10000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x40000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x40000000, 0x00000000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x8B91C6D1, 0x02153E46, 0x00000000, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x17179149, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x17179149, 0x00000000, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		126	// Number of bits
#define NBYTES_FIELD	16	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		307
#define PC_DEPTH		16

#define EXPONENT2		60
#define EXPONENT2_BITS	60
#define EXPONENT2_MASK	0xF
const digit_t STRATEGY2[59] = { 27, 15, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_0[29] = { 13, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};
const digit_t STRATEGY2_1[29] = { 13, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0[28] = { 13, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1[28] = { 13, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};
const digit_t STRATEGY2_[29] = { 13, 7, 4, 2, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[13] = { 5, 4, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[13] = { 5, 4, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_PC_0[12] = { 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};
const digit_t STRATEGY2_PC_1[12] = { 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};

#define EXPONENT3		36
#define EXPONENT3_BITS	58
#define EXPONENT3_MASK	0x3
const digit_t STRATEGY3[35] = { 13, 8, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY3_0[17] = { 6, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
const digit_t STRATEGY3_1[17] = { 6, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 30
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 9

#endif
