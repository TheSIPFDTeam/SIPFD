#ifndef _CONFIG_CPU_H_
#define _CONFIG_CPU_H_

#include <stdio.h>

#include <inttypes.h>

#define RADIX			64
#define LOG2RADIX		6
typedef uint64_t		digit_t;	// Unsigned 64-bit digit
#define NWORDS_FIELD	2
#define MASK			0xFF		// Mask for the last 64-bits word

#define NBITS_FIELD		72	// Number of bits
#define NBYTES_FIELD	9	// Number of bytes

#define NBITS_TO_NBYTES(nbits)		(((nbits)+7)/8)											// Conversion macro from number of bits to number of bytes
#define NBITS_TO_NWORDS(nbits)		(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))	// Conversion macro from number of bits to number of computer words
#define NBYTES_TO_NWORDS(nbytes)	(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))			// Conversion macro from number of bytes to number of computer words

// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1
#define COFACTOR		5
#define PC_DEPTH		14

#define EXPONENT2		34
#define EXPONENT2_BITS	34
#define EXPONENT2_MASK	0x3

#define EXPONENT3		22
#define EXPONENT3_BITS	35
#define EXPONENT3_MASK	0x7

#endif
