#ifndef _CONFIG_P107_H_
#define _CONFIG_P107_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	2
	#define MASK			0x7FFFFFFFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0x0AA3FFFFFFFFFFFD, 0x00000437A72CDB60 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0x02A8FFFFFFFFFFFF, 0x0000010DE9CB36D8 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0x0551FFFFFFFFFFFF, 0x0000021BD3966DB0 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0004000000000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000000002000000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000000002000000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x0006954FE21E3E81, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x000000000290D741, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x000000000290D741, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	4
	#define MASK			0x7FF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0x0AA3FFFF, 0xA72CDB60, 0x00000437 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0x02A8FFFF, 0xE9CB36D8, 0x0000010D };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0x0551FFFF, 0xD3966DB0, 0x0000021B };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00040000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x02000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x02000000, 0x00000000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0xE21E3E81, 0x0006954F, 0x00000000, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x0290D741, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x0290D741, 0x00000000, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		107	// Number of bits
#define NBYTES_FIELD	14	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		41
#define PC_DEPTH		14

#define EXPONENT2		50
#define EXPONENT2_BITS	50
#define EXPONENT2_MASK	0x3
const digit_t STRATEGY2[49] = { 22, 12, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 9, 6, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_0[24] = { 10, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_1[24] = { 10, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0[23] = { 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1[23] = { 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_[24] = { 9, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[11] = { 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[11] = { 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY2_PC_0[9] = { 4, 2, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_1[9] = { 4, 2, 2, 1, 1, 1, 2, 1, 1};

#define EXPONENT3		32
#define EXPONENT3_BITS	51
#define EXPONENT3_MASK	0x7
const digit_t STRATEGY3[31] = { 11, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_0[15] = { 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
const digit_t STRATEGY3_1[15] = { 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 25
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 8

#endif
