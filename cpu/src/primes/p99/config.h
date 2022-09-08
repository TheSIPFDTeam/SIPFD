#ifndef _CONFIG_P99_H_
#define _CONFIG_P99_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	2
	#define MASK			0x7FFFFFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xEB3D3FFFFFFFFFFD, 0x000000054D9CCB7B };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFACF4FFFFFFFFFFF, 0x00000001536732DE };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xF59E9FFFFFFFFFFF, 0x00000002A6CE65BD };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0000400000000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000800000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000800000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x0000BB41C3CA78B9, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x0000000000DAF26B, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x0000000000DAF26B, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	4
	#define MASK			0x7		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0xEB3D3FFF, 0x4D9CCB7B, 0x00000005 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFACF4FFF, 0x536732DE, 0x00000001 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0xF59E9FFF, 0xA6CE65BD, 0x00000002 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00004000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x00800000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x00800000, 0x00000000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0xC3CA78B9, 0x0000BB41, 0x00000000, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x00DAF26B, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x00DAF26B, 0x00000000, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		99	// Number of bits
#define NBYTES_FIELD	13	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		29
#define PC_DEPTH		13

#define EXPONENT2		46
#define EXPONENT2_BITS	46
#define EXPONENT2_MASK	0x3F
const digit_t STRATEGY2[45] = { 20, 12, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_0[22] = { 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_1[22] = { 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0[21] = { 9, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1[21] = { 9, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_[22] = { 9, 5, 4, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[10] = { 5, 2, 2, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[10] = { 5, 2, 2, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_PC_0[8] = { 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_1[8] = { 4, 2, 1, 1, 1, 2, 1, 1};

#define EXPONENT3		30
#define EXPONENT3_BITS	48
#define EXPONENT3_MASK	0xFF
const digit_t STRATEGY3[29] = { 11, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_0[14] = { 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
const digit_t STRATEGY3_1[14] = { 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 23
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 8

#endif
