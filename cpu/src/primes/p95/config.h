#ifndef _CONFIG_P95_H_
#define _CONFIG_P95_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	2
	#define MASK			0x7FFFFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xA44F8FFFFFFFFFFD, 0x000000005EEDC896 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xA913E3FFFFFFFFFF, 0x0000000017BB7225 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x5227C7FFFFFFFFFF, 0x000000002F76E44B };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0000100000000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000400000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000400000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x000014CE6B167F31, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x000000000048FB79, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x000000000048FB79, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	3
	#define MASK			0x7FFFFFFF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0xA44F8FFF, 0x5EEDC896 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0xA913E3FF, 0x17BB7225 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0x5227C7FF, 0x2F76E44B };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00001000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x00400000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x00400000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x6B167F31, 0x000014CE, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x0048FB79, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x0048FB79, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		95	// Number of bits
#define NBYTES_FIELD	12	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		73
#define PC_DEPTH		12

#define EXPONENT2		44
#define EXPONENT2_BITS	44
#define EXPONENT2_MASK	0xF
const digit_t STRATEGY2[43] = { 18, 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_0[21] = { 9, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_1[21] = { 9, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0[20] = { 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1[20] = { 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_[21] = { 9, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[9] = { 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[9] = { 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_0[8] = { 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_1[8] = { 4, 2, 1, 1, 1, 2, 1, 1};

#define EXPONENT3		28
#define EXPONENT3_BITS	45
#define EXPONENT3_MASK	0x1F
const digit_t STRATEGY3[27] = { 9, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_0[13] = { 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY3_1[13] = { 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 22
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 7

#endif
