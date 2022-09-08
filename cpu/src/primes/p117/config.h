#ifndef _CONFIG_P117_H_
#define _CONFIG_P117_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	2
	#define MASK			0x1FFFFFFFFFFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0x383FFFFFFFFFFFFD, 0x0017AA3C6895382F };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xCE0FFFFFFFFFFFFF, 0x0005EA8F1A254E0B };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x9C1FFFFFFFFFFFFF, 0x000BD51E344A9C17 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0040000000000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000000008000000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000000008000000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x003B3FCEF3103289, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x0000000007B285C3, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x0000000007B285C3, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	4
	#define MASK			0x1FFFFF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0x383FFFFF, 0x6895382F, 0x0017AA3C };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0xCE0FFFFF, 0x1A254E0B, 0x0005EA8F };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0x9C1FFFFF, 0x344A9C17, 0x000BD51E };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00400000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x08000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x08000000, 0x00000000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0xF3103289, 0x003B3FCE, 0x00000000, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x07B285C3, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x07B285C3, 0x00000000, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		117	// Number of bits
#define NBYTES_FIELD	15	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		409
#define PC_DEPTH		15

#define EXPONENT2		54
#define EXPONENT2_BITS	54
#define EXPONENT2_MASK	0x3F
const digit_t STRATEGY2[53] = { 23, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_0[26] = { 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_1[26] = { 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0[25] = { 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1[25] = { 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_[26] = { 11, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 2, 2, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[12] = { 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[12] = { 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_PC_0[10] = { 4, 3, 2, 1, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_1[10] = { 4, 3, 2, 1, 1, 1, 1, 2, 1, 1};

#define EXPONENT3		34
#define EXPONENT3_BITS	54
#define EXPONENT3_MASK	0x3F
const digit_t STRATEGY3[33] = { 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY3_0[16] = { 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
const digit_t STRATEGY3_1[16] = { 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 27
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 9

#endif
