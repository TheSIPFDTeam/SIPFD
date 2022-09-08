#ifndef _CONFIG_P224_H_
#define _CONFIG_P224_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	4
	#define MASK			0xFFFFFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFD, 0x6D1CFFFFFFFFFFFF, 0x2125834CACCFBE4F, 0x00000000B24428D6 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0xDB473FFFFFFFFFFF, 0x884960D32B33EF93, 0x000000002C910A35 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0xB68E7FFFFFFFFFFF, 0x1092C1A65667DF27, 0x000000005922146B };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0000000000000000, 0x0001000000000000, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0100000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0100000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x58DE83727119CD51, 0x00000DB6794B8C65, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x003B3FCEF3103289, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x003B3FCEF3103289, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	7
	#define MASK			0xFFFFFFFF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0xFFFFFFFF, 0xFFFFFFFF, 0x6D1CFFFF, 0xACCFBE4F, 0x2125834C, 0xB24428D6 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xDB473FFF, 0x2B33EF93, 0x884960D3, 0x2C910A35 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xB68E7FFF, 0x5667DF27, 0x1092C1A6, 0x5922146B };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00000000, 0x00000000, 0x00010000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x00000000, 0x01000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x00000000, 0x01000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x7119CD51, 0x58DE8372, 0x794B8C65, 0x00000DB6, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0xF3103289, 0x003B3FCE, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0xF3103289, 0x003B3FCE, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		224	// Number of bits
#define NBYTES_FIELD	28	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		13
#define PC_DEPTH		25

#define EXPONENT2		112
#define EXPONENT2_BITS	112
#define EXPONENT2_MASK	0xFF
const digit_t STRATEGY2[111] = { 48, 27, 16, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 21, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_0[55] = { 24, 15, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_1[55] = { 24, 15, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0[54] = { 24, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1[54] = { 24, 14, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_[55] = { 22, 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 9, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[26] = { 11, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 2, 2, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[26] = { 11, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 2, 2, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_PC_0[29] = { 13, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};
const digit_t STRATEGY2_PC_1[29] = { 13, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};

#define EXPONENT3		68
#define EXPONENT3_BITS	108
#define EXPONENT3_MASK	0xF
const digit_t STRATEGY3[67] = { 23, 17, 9, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 8, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_0[33] = { 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY3_1[33] = { 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 56
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 16

#endif
