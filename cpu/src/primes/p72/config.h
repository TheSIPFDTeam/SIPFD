#ifndef _CONFIG_P72_H_
#define _CONFIG_P72_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	2
	#define MASK			0xFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0x212361F3FFFFFFFD, 0x0000000000000092 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x8848D87CFFFFFFFF, 0x0000000000000024 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x1091B0F9FFFFFFFF, 0x0000000000000049 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0000000400000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000020000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000020000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x000000074E74F819, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x000000000002B3FB, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x000000000002B3FB, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	3
	#define MASK			0xFF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0x212361F3, 0x00000092 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0x8848D87C, 0x00000024 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0x1091B0F9, 0x00000049 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00000004, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x00020000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x00020000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x4E74F819, 0x00000007, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x0002B3FB, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x0002B3FB, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		72	// Number of bits
#define NBYTES_FIELD	9	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		5
#define PC_DEPTH		9

#define EXPONENT2		34
#define EXPONENT2_BITS	34
#define EXPONENT2_MASK	0x3
const digit_t STRATEGY2[33] = { 16, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_0[16] = { 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_1[16] = { 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0[15] = { 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1[15] = { 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_[16] = { 7, 4, 2, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[7] = { 3, 2, 1, 1, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[7] = { 3, 2, 1, 1, 1, 1, 1};
const digit_t STRATEGY2_PC_0[6] = { 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_PC_1[6] = { 3, 2, 1, 1, 1, 1};

#define EXPONENT3		22
#define EXPONENT3_BITS	35
#define EXPONENT3_MASK	0x7
const digit_t STRATEGY3[21] = { 8, 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_0[10] = { 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_1[10] = { 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 17
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 6

#endif
