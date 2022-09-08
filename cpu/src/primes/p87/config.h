#ifndef _CONFIG_P87_H_
#define _CONFIG_P87_H_

#include <stdio.h>

#include <inttypes.h>

#if defined(_x64_)
	#define RADIX			64
	#define LOG2RADIX		6
	typedef uint64_t		digit_t;	// Unsigned 64-bit digit
	#define NWORDS_FIELD	2
	#define MASK			0x7FFFFF		// Mask for the last 64-bits word

	static inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0x667C50FFFFFFFFFD, 0x00000000005EC8CB };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xD99F143FFFFFFFFF, 0x000000000017B232 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xB33E287FFFFFFFFF, 0x00000000002F6465 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x0000010000000000, 0x0000000000000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x0000000000100000, 0x0000000000000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x0000000000100000, 0x0000000000000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0x0000024FD3027FE9, 0x0000000000000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x00000000001853D3, 0x0000000000000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x00000000001853D3, 0x0000000000000000 };

#elif defined(_x86_)
	#define RADIX			32
	#define LOG2RADIX		5
	typedef uint32_t		digit_t;	// Unsigned 32-bit digit
	#define NWORDS_FIELD	3
	#define MASK			0x7FFFFF		// Mask for the last 32-bits word

	static inline void TOHEX(digit_t x) { printf("%08X", x); };
	static inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };

	const digit_t P_MINUS_TWO[NWORDS_FIELD] = { 0xFFFFFFFD, 0x667C50FF, 0x005EC8CB };
	const digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = { 0xFFFFFFFF, 0xD99F143F, 0x0017B232 };
	const digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = { 0xFFFFFFFF, 0xB33E287F, 0x002F6465 };

	const digit_t BOUND2[NWORDS_FIELD] = { 0x00000000, 0x00000100, 0x00000000 };
	const digit_t BOUND2_0[NWORDS_FIELD] = { 0x00100000, 0x00000000, 0x00000000 };
	const digit_t BOUND2_1[NWORDS_FIELD] = { 0x00100000, 0x00000000, 0x00000000 };

	const digit_t BOUND3[NWORDS_FIELD] = { 0xD3027FE9, 0x0000024F, 0x00000000 };
	const digit_t BOUND3_0[NWORDS_FIELD] = { 0x001853D3, 0x00000000, 0x00000000 };
	const digit_t BOUND3_1[NWORDS_FIELD] = { 0x001853D3, 0x00000000, 0x00000000 };

#else
	#error -- "Unsupported ARCHITECTURE"
#endif

#define NBITS_FIELD		87	// Number of bits
#define NBYTES_FIELD	11	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		41
#define PC_DEPTH		11

#define EXPONENT2		40
#define EXPONENT2_BITS	40
#define EXPONENT2_MASK	0xFF
const digit_t STRATEGY2[39] = { 17, 9, 6, 4, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_0[19] = { 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_1[19] = { 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_0[18] = { 8, 4, 3, 2, 1, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1[18] = { 8, 4, 3, 2, 1, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_[19] = { 8, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1};
const digit_t STRATEGY2_REDUCED_0_[8] = { 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_REDUCED_1_[8] = { 4, 2, 1, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_0[7] = { 4, 2, 1, 1, 2, 1, 1};
const digit_t STRATEGY2_PC_1[7] = { 4, 2, 1, 1, 2, 1, 1};

#define EXPONENT3		26
#define EXPONENT3_BITS	42
#define EXPONENT3_MASK	0x3
const digit_t STRATEGY3[25] = { 9, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1};
const digit_t STRATEGY3_0[12] = { 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};
const digit_t STRATEGY3_1[12] = { 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1};

// Constants for PRTL
#define __FP_BITS__ 20
/// Size (in bytes) of data stored in a single vector
#define __DATA_SIZE_IN_BYTES__ 7

#endif
