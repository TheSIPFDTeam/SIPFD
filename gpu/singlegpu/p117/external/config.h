#ifndef _CONFIG_CPU_H_
#define _CONFIG_CPU_H_

#include <stdio.h>

#include <inttypes.h>

#define RADIX			64
#define LOG2RADIX		6
typedef uint64_t		digit_t;	// Unsigned 64-bit digit
#define NWORDS_FIELD	2
#define MASK			0x1FFFFFFFFFFFFF		// Mask for the last 64-bits word

#define NBITS_FIELD		117	// Number of bits
#define NBYTES_FIELD	15	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		409

#define EXPONENT2		54
#define EXPONENT2_BITS	54
#define EXPONENT2_MASK	0x3F

#define EXPONENT3		34
#define EXPONENT3_BITS	54
#define EXPONENT3_MASK	0x3F

#endif
