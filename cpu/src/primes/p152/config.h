#ifndef _CONFIG_P152_H_
#define _CONFIG_P152_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	3
	#define MASK			0xFFFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFD, 0xC1DC7B0717B28FFF, 0x0000000000BE2EB2 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0xB0771EC1C5ECA3FF, 0x00000000002F8BAC };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x60EE3D838BD947FF, 0x00000000005F1759 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0000000000000000, 0x0000000000001000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000004000000000, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000004000000000, 0x0000000000000000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x275329FD09495753, 0x00000000000000A0, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x00000015EB5EE84B, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x000000074E74F819, 0x0000000000000000, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	5
	#define MASK			0xFFFFFF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0xFFFFFFFF, 0x17B28FFF, 0xC1DC7B07, 0x00BE2EB2 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xC5ECA3FF, 0xB0771EC1, 0x002F8BAC };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFFFFFFFF, 0x8BD947FF, 0x60EE3D83, 0x005F1759 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00000000, 0x00001000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x00000000, 0x00000040, 0x00000000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x09495753, 0x275329FD, 0x000000A0, 0x00000000, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0xEB5EE84B, 0x00000015, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x4E74F819, 0x00000007, 0x00000000, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		152	// Number of bits
#define NBYTES_FIELD	19	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		19
#define PC_DEPTH		21

#define EXPONENT2		76
#define EXPONENT2_BITS	76
#define EXPONENT2_MASK	0xF
const digit_t STRATEGY2[75] = { 33, 18, 10, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 15, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_0[37] = { 17, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_1[37] = { 17, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0[36] = { 16, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1[36] = { 16, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_[37] = { 16, 9, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[17] = { 7, 5, 2, 2, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[17] = { 7, 5, 2, 2, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_PC_0[15] = { 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_PC_1[15] = { 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};

#define EXPONENT3		45
#define EXPONENT3_BITS	72
#define EXPONENT3_MASK	0xFF
const digit_t STRATEGY3[44] = { 17, 9, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
const digit_t STRATEGY3_0[22] = { 8, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_1[21] = { 8, 5, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 38
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 11

#endif
