#ifndef _CONFIG_P164_H_
#define _CONFIG_P164_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	3
	#define MASK			0xFFFFFFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFD, 0xF0680FCA98D3FFFF, 0x000000085B8D5B04 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x3C1A03F2A634FFFF, 0x0000000216E356C1 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFFFFFFFFFF, 0x783407E54C69FFFF, 0x000000042DC6AD82 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0000000000000000, 0x0000000000040000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000020000000000, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000020000000000, 0x0000000000000000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x61EC79E5539411EB, 0x00000000000005A1, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x00000041C21CB8E1, 0x0000000000000000, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x00000015EB5EE84B, 0x0000000000000000, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	6
	#define MASK			0xF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0xFFFFFFFF, 0x98D3FFFF, 0xF0680FCA, 0x5B8D5B04, 0x00000008 };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xA634FFFF, 0x3C1A03F2, 0x16E356C1, 0x00000002 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFFFFFFFF, 0x4C69FFFF, 0x783407E5, 0x2DC6AD82, 0x00000004 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00000000, 0x00040000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x00000000, 0x00000200, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x00000000, 0x00000200, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x539411EB, 0x61EC79E5, 0x000005A1, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0xC21CB8E1, 0x00000041, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0xEB5EE84B, 0x00000015, 0x00000000, 0x00000000, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		164	// Number of bits
#define NBYTES_FIELD	21	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		95
#define PC_DEPTH		22

#define EXPONENT2		82
#define EXPONENT2_BITS	82
#define EXPONENT2_MASK	0x3
const digit_t STRATEGY2[81] = { 34, 21, 12, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 16, 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_0[40] = { 17, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_1[40] = { 17, 10, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0[39] = { 17, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1[39] = { 17, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_[40] = { 16, 9, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[19] = { 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[19] = { 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
const digit_t STRATEGY2_PC_0[17] = { 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_1[17] = { 9, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1};

#define EXPONENT3		47
#define EXPONENT3_BITS	75
#define EXPONENT3_MASK	0x7
const digit_t STRATEGY3[46] = { 17, 11, 6, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
const digit_t STRATEGY3_0[23] = { 9, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_1[22] = { 8, 6, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 41
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 12

#endif
