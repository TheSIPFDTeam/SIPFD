#ifndef _CONFIG_CPU_H_
#define _CONFIG_CPU_H_

#include <stdio.h>

#include <inttypes.h>

//#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	2
	#define MASK			0x1F		// Mask for the last 64-bits word

//	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
//	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };
//
//	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xAC0E7A06FFFFFFFD, 0x0000000000000012 };
//	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xAB039E81BFFFFFFF, 0x0000000000000004 };
//	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x56073D037FFFFFFF, 0x0000000000000009 };
//
//	const digit_t BOUND2[NWORDS_FIELD] = { 0x0000000100000000, 0x0000000000000000 };
//	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000010000, 0x0000000000000000 };
//	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000010000, 0x0000000000000000 };
//
//	const digit_t BOUND3[NWORDS_FIELD] = { 0x00000000CFD41B91, 0x0000000000000000 };
//	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x000000000000E6A9, 0x0000000000000000 };
//	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x000000000000E6A9, 0x0000000000000000 };

//#elif defined(_x86_)
//	#define RADIX			32
//	#define LOG2RADIX		5
//	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
//	#define NWORDS_FIELD	3
//	#define MASK			0x1F		// Mask for the last 32-bits word
//
//	static inline void TOHEX(digit_t x) { printf("%08X", x); };
//	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };
//
//	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0xAC0E7A06, 0x00000012 };
//	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xBFFFFFFF, 0xAB039E81, 0x00000004 };
//	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x7FFFFFFF, 0x56073D03, 0x00000009 };
//
//	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00000001, 0x00000000 };
//	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x00010000, 0x00000000, 0x00000000 };
//	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x00010000, 0x00000000, 0x00000000 };
//
//	const digit_t BOUND3[NWORDS_FIELD] = { 0xCFD41B91, 0x00000000, 0x00000000 };
//	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x0000E6A9, 0x00000000, 0x00000000 };
//	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x0000E6A9, 0x00000000, 0x00000000 };
//
//#else
//	#error -- "Unsupported ARCHITECTURE"
//#endif

#define NBITS_FIELD		69	// Number of bits
#define NBYTES_FIELD	9	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		23
#define PC_DEPTH		13

#define EXPONENT2		32
#define EXPONENT2_BITS	32
#define EXPONENT2_MASK	0xFF

//const digit_t STRATEGY2[31] = { 15, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
//const digit_t STRATEGY2_0[15] = { 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
//const digit_t STRATEGY2_1[15] = { 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
//const digit_t STRATEGY2_REDUCED_0[14] = { 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
//const digit_t STRATEGY2_REDUCED_1[14] = { 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
//const digit_t STRATEGY2_[15] = { 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
//const digit_t STRATEGY2_REDUCED_0_[6] = { 3, 2, 1, 1, 1, 1};
//const digit_t STRATEGY2_REDUCED_1_[6] = { 3, 2, 1, 1, 1, 1};
//const digit_t STRATEGY2_PC_0[1] = { 1};
//const digit_t STRATEGY2_PC_1[1] = { 1};

#define EXPONENT3		20
#define EXPONENT3_BITS	32
#define EXPONENT3_MASK	0xFF
//const digit_t STRATEGY3[19] = { 7, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1};
//const digit_t STRATEGY3_0[9] = { 4, 2, 1, 1, 1, 1, 1, 1, 1};
//const digit_t STRATEGY3_1[9] = { 4, 2, 1, 1, 1, 1, 1, 1, 1};

#endif