#importing sys module
import sys
# importing os module
import os
# importing click
import click
# importing math
import math

# The next variables are required for constructing isogenies
# ++++++++++ shortw (release version requires measured costs on mont)
p2 =  8.5   # xdbl
q2 =  5.7   # 2-isogeny evaluations
p3 = 18.9   # xtpl
q3 =  8.4   # 3-isogeny evaluations
p4 = 2*p2   # 2 xdbl
q4 = 1.5*q2 # 4-isogeny evaluations
# ++++++++++

lbracket = '{'
rbracket = '}'

ceildiv = lambda x,y: -(-x//y)
int2str = lambda x, string : string.format(x)
def fp2str(x : int, type : click.Choice(['x86', 'x64'])):
    s = {'x86':8, 'x64':16}[type]
    length = ceildiv(x.bit_length(), s * 4) * s
    X = int2str(x, '{:0%dX}' % length)
    Y = [ X[i:i+s] for i in range(0, length, s) ]
    Y.reverse()
    return '{ 0x' + ', 0x'.join(Y) + ' }'

def bound2str(x : int, p : int, type : click.Choice(['x86', 'x64'])):
    s = {'x86':8, 'x64':16}[type]
    length = ceildiv(p.bit_length(), s * 4) * s
    X = int2str(x, '{:0%dX}' % length)
    Y = [ X[i:i+s] for i in range(0, length, s) ]
    Y.reverse()
    return '{ 0x' + ', 0x'.join(Y) + ' }'

# Primality test
from random import randrange
def is_prime(n):
    """
    Miller-Rabin primality test.

    A return value of False means n is certainly not prime. A return value of
    True means n is very likely a prime.
    """
    if n!=int(n):
        return False
    n=int(n)
    #Miller-Rabin test for prime
    if n==0 or n==1 or n==4 or n==6 or n==8 or n==9:
        return False

    if n==2 or n==3 or n==5 or n==7:
        return True
    s = 0
    d = n-1
    while d%2==0:
        d>>=1
        s+=1
    assert(2**s * d == n-1)

    def trial_composite(a):
        if pow(a, d, n) == 1:
            return False
        for i in range(s):
            if pow(a, 2**i * d, n) == n-1:
                return False
        return True

    for i in range(128):        #number of trials
        a = randrange(2, n)
        if trial_composite(a):
            return False

    return True

# Optimal strategy
def strategy(n, p, q):
    S = { 1: [] }
    C = { 1: 0 }
    for i in range(2, n+1):
        b, cost = min(((b, C[i-b] + C[b] + b*p + (i-b)*q) for b in range(1,i)),
                      key=lambda t: t[1])
        S[i] = [b] + S[i-b] + S[b]
        C[i] = cost
    return S[n]

# Creating config.h file
def config(p : int, e2 : int, e3 : int, f : int, pc_depth : int):
    bitlength = p.bit_length()
    # Last 64-bits word
    if bitlength % 64 == 0:
        mask64 = '%X' % 0xFFFFFFFFFFFFFFFF
    else:
        mask64 = '%X' % (2**(bitlength % 64) - 1)
    # Last 32-bits word
    if bitlength % 32 == 0:
        mask32 = '%X' % 0xFFFFFFFF
    else:
        mask32 = '%X' % (2**(bitlength % 32) - 1)
    # Last byte
    log2e2 = int(2**e2 - 1).bit_length()
    if log2e2 % 8 == 0:
        mask2 = 0xFF
    else:
        mask2 = 2**(log2e2 % 8) - 1
    # Last byte
    log2e3 = int(3**e3 - 1).bit_length()
    if log2e3 % 8 == 0:
        mask3 = 0xFF
    else:
        mask3 = 2**(log2e3 % 8) - 1

    click.echo(f'#ifndef _CONFIG_P{bitlength}_H_')
    click.echo(f'#define _CONFIG_P{bitlength}_H_')
    click.echo('\n#include <stdio.h>')
    click.echo('\n#include <inttypes.h>')
    # ++++++++++++++++++++++++++++++++
    click.echo('\n#if defined(_x64_)')
    click.echo('\t#define RADIX\t\t\t64')
    click.echo('\t#define LOG2RADIX\t\t6')
    click.echo('\ttypedef uint64_t\t\tdigit_t;\t// Unsigned 64-bit digit')
    click.echo(f'\t#define NWORDS_FIELD\t{ceildiv(bitlength, 64)}')
    click.echo(f'\t#define MASK\t\t\t0x{mask64}\t\t// Mask for the last 64-bits word')

    click.echo('\n\tstatic inline void TOHEX(digit_t x) { printf("%016" PRIx64, x); };')
    click.echo('\tstatic inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%" PRIx64, x); };')

    click.echo(f'\n\tconst digit_t P_MINUS_TWO[NWORDS_FIELD] = {fp2str(p-2, "x64")};')
    click.echo(f'\tconst digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = {fp2str((p-3)//4, "x64")};')
    click.echo(f'\tconst digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = {fp2str((p-1)//2, "x64")};')

    click.echo(f'\n\tconst digit_t BOUND2[NWORDS_FIELD] = {bound2str(2**(e2), p, "x64")};')
    click.echo(f'\tconst digit_t BOUND2_0[NWORDS_FIELD] = {bound2str(2**(ceildiv(e2,2)), p, "x64")};')
    click.echo(f'\tconst digit_t BOUND2_1[NWORDS_FIELD] = {bound2str(2**(e2//2), p, "x64")};')

    click.echo(f'\n\tconst digit_t BOUND3[NWORDS_FIELD] = {bound2str(3**(e3), p, "x64")};')
    click.echo(f'\tconst digit_t BOUND3_0[NWORDS_FIELD] = {bound2str(3**(ceildiv(e3,2)), p, "x64")};')
    click.echo(f'\tconst digit_t BOUND3_1[NWORDS_FIELD] = {bound2str(3**(e3//2) , p, "x64")};')

    # ++++++++++++++++++++++++++++++++
    click.echo('\n#elif defined(_x86_)')
    click.echo('\t#define RADIX\t\t\t32')
    click.echo('\t#define LOG2RADIX\t\t5')
    click.echo('\ttypedef uint32_t\t\tdigit_t;\t// Unsigned 32-bit digit')
    click.echo(f'\t#define NWORDS_FIELD\t{ceildiv(bitlength, 32)}')
    click.echo(f'\t#define MASK\t\t\t0x{mask32}\t\t// Mask for the last 32-bits word')

    click.echo('\n\tstatic inline void TOHEX(digit_t x) { printf("%08X", x); };')
    click.echo('\tstatic inline void SPRINTF(char *input, digit_t x) { sprintf(input, "%X", x); };')

    click.echo(f'\n\tconst digit_t P_MINUS_TWO[NWORDS_FIELD] = {fp2str(p-2, "x86")};')
    click.echo(f'\tconst digit_t P_MINUS_THREE_QUARTERS[NWORDS_FIELD] = {fp2str((p-3)//4, "x86")};')
    click.echo(f'\tconst digit_t P_MINUS_ONE_HALVES[NWORDS_FIELD] = {fp2str((p-1)//2, "x86")};')

    click.echo(f'\n\tconst digit_t BOUND2[NWORDS_FIELD] = {bound2str(2**(e2), p, "x86")};')
    click.echo(f'\tconst digit_t BOUND2_0[NWORDS_FIELD] = {bound2str(2**(ceildiv(e2,2)), p, "x86")};')
    click.echo(f'\tconst digit_t BOUND2_1[NWORDS_FIELD] = {bound2str(2**(e2//2), p, "x86")};')

    click.echo(f'\n\tconst digit_t BOUND3[NWORDS_FIELD] = {bound2str(3**(e3), p, "x86")};')
    click.echo(f'\tconst digit_t BOUND3_0[NWORDS_FIELD] = {bound2str(3**(ceildiv(e3,2)), p, "x86")};')
    click.echo(f'\tconst digit_t BOUND3_1[NWORDS_FIELD] = {bound2str(3**(e3//2), p, "x86")};')

    # +++++++++++++++++++
    click.echo('\n#else')
    click.echo('\t#error -- "Unsupported ARCHITECTURE"')
    click.echo('#endif')

    # -------------------
    click.echo(f'\n#define NBITS_FIELD\t\t{bitlength}\t// Number of bits')
    click.echo(f'#define NBYTES_FIELD\t{ceildiv(bitlength, 8)}\t// Number of bytes')

    click.echo('\n#define NBITS_TO_NBYTES(nbits)\t\t(((nbits)+7)/8)\t\t\t\t\t\t\t\t\t\t\t// Conversion macro from number of bits to number of bytes')
    click.echo('#define NBITS_TO_NWORDS(nbits)\t\t(((nbits)+(sizeof(digit_t)*8)-1)/(sizeof(digit_t)*8))\t// Conversion macro from number of bits to number of computer words')
    click.echo('#define NBYTES_TO_NWORDS(nbytes)\t(((nbytes)+sizeof(digit_t)-1)/sizeof(digit_t))\t\t\t// Conversion macro from number of bytes to number of computer words')

    click.echo('\n// Prime characteristic p = 2^{EXPONENT2} * 3^{EXPONENT3} * COFACTOR - 1')
    click.echo(f'#define COFACTOR\t\t{f}')
    click.echo(f'#define PC_DEPTH\t\t{pc_depth}')

    click.echo(f'\n#define EXPONENT2\t\t{e2}')
    click.echo(f'#define EXPONENT2_BITS\t{e2}')
    click.echo(f'#define EXPONENT2_MASK\t{"0x%X" % mask2}')
    strategy2 = '{ ' + ', '.join(map(str, strategy(e2, p2, q2))) + '};'
    click.echo(f'const digit_t STRATEGY2[{e2 - 1}] = {strategy2}')
    strategy2_0 = '{ ' + ', '.join(map(str, strategy(ceildiv(e2,2), p2, q2))) + '};'
    click.echo(f'const digit_t STRATEGY2_0[{ceildiv(e2,2) - 1}] = {strategy2_0}')
    strategy2_1 = '{ ' + ', '.join(map(str, strategy(e2//2, p2, q2))) + '};'
    click.echo(f'const digit_t STRATEGY2_1[{e2//2 - 1}] = {strategy2_1}')
    strategy2_REDUCED_0 = '{ ' + ', '.join(map(str, strategy(ceildiv(e2,2)-1, p2, q2))) + '};'
    click.echo(f'const digit_t STRATEGY2_REDUCED_0[{ceildiv(e2,2) - 2}] = {strategy2_REDUCED_0}')
    strategy2_REDUCED_1 = '{ ' + ', '.join(map(str, strategy(e2//2-1, p2, q2))) + '};'
    click.echo(f'const digit_t STRATEGY2_REDUCED_1[{e2//2 - 2}] = {strategy2_REDUCED_1}')
    e4_floor = (e2-2)//2
    e4_ceil = ceildiv(e2-2,2)
    strategy2_ = '{ ' + ', '.join(map(str, strategy(e2//2, p4, q4))) + '};'
    click.echo(f'const digit_t STRATEGY2_[{e2//2 - 1}] = {strategy2_}')
    strategy2_REDUCED_0_ = '{ ' + ', '.join(map(str, strategy(e4_ceil//2, p4, q4))) + '};'
    click.echo(f'const digit_t STRATEGY2_REDUCED_0_[{e4_ceil//2 - 1}] = {strategy2_REDUCED_0_}')
    strategy2_REDUCED_1_ = '{ ' + ', '.join(map(str, strategy(e4_floor//2, p4, q4))) + '};'
    click.echo(f'const digit_t STRATEGY2_REDUCED_1_[{e4_floor//2 - 1}] = {strategy2_REDUCED_1_}')
    strategy2_pc_0 = '{ ' + ', '.join(map(str, strategy(ceildiv(e2,2)-1-pc_depth, p2, q2))) + '};'
    click.echo(f'const digit_t STRATEGY2_PC_0[{ceildiv(e2,2) - 2 - pc_depth}] = {strategy2_pc_0}')
    strategy2_pc_1 = '{ ' + ', '.join(map(str, strategy(e2//2-1-pc_depth, p2, q2))) + '};'
    click.echo(f'const digit_t STRATEGY2_PC_1[{e2//2 - 2 - pc_depth}] = {strategy2_pc_1}')

    click.echo(f'\n#define EXPONENT3\t\t{e3}')
    click.echo(f'#define EXPONENT3_BITS\t{log2e3}')
    click.echo(f'#define EXPONENT3_MASK\t{"0x%X" % mask3}')
    strategy3 = '{ ' + ', '.join(map(str, strategy(e3, p3, q3))) + '};'
    click.echo(f'const digit_t STRATEGY3[{e3 - 1}] = {strategy3}')
    strategy3_0 = '{ ' + ', '.join(map(str, strategy(ceildiv(e3,2), p3, q3))) + '};'
    click.echo(f'const digit_t STRATEGY3_0[{ceildiv(e3,2) - 1}] = {strategy3_0}')
    strategy3_1 = '{ ' + ', '.join(map(str, strategy(e3//2, p3, q3))) + '};'
    click.echo(f'const digit_t STRATEGY3_1[{e3//2 - 1}] = {strategy3_1}')

    click.echo('')
    click.echo('// Constants for PRTL')
    # e on Alice side is always bigger so we take e2
    click.echo(f'#define __FP_BITS__ {math.ceil(e2/2)}')
    click.echo('/// Size (in bytes) of data stored in a single vector')
    level = 6 #adapt this per prime according to w
    bit_vector_size = math.ceil(e2/2) - level + 1 + math.ceil(e2/2) + 1 + 16 # adapt this if length becomes bigger than 16
    click.echo(f'#define __DATA_SIZE_IN_BYTES__ {math.ceil(bit_vector_size/8)}')

    click.echo('\n#endif')

# Creating pX0X_api.h file
def api(p : int):
    bitlength = p.bit_length()
    click.echo(f'#ifndef _P{bitlength}_API_H_')
    click.echo(f'#define _P{bitlength}_API_H_')

    click.echo('\n#include <stdint.h>')
    click.echo('#include "config.h"')
    click.echo('#if defined(_shortw_)')
    click.echo('#include "../../shortw/api.h"')
    click.echo('#elif defined(_mont_)')
    click.echo('#include "../../mont/api.h"')
    click.echo('#endif')
    click.echo('\n// GF(p)')

    click.echo(f'void fiat_fp{bitlength}_mul(fp_t out1, const fp_t arg1, const fp_t arg2);')
    click.echo(f'void fiat_fp{bitlength}_square(fp_t out1, const fp_t arg1);')
    click.echo(f'void fiat_fp{bitlength}_add(fp_t out1, const fp_t arg1, const fp_t arg2);')
    click.echo(f'void fiat_fp{bitlength}_sub(fp_t out1, const fp_t arg1, const fp_t arg2);')
    click.echo(f'void fiat_fp{bitlength}_opp(fp_t out1, const fp_t arg1);')
    click.echo(f'void fiat_fp{bitlength}_from_montgomery(fp_t out1, const fp_t arg1);')
    click.echo(f'void fiat_fp{bitlength}_to_montgomery(fp_t out1, const fp_t arg1);')
    click.echo(f'void fiat_fp{bitlength}_nonzero(digit_t* out1, const digit_t arg1[NWORDS_FIELD]);')
    click.echo(f'void fiat_fp{bitlength}_to_bytes(uint8_t out1[NBYTES_FIELD], const digit_t arg1[NWORDS_FIELD]);')
    click.echo(f'void fiat_fp{bitlength}_from_bytes(digit_t out1[NWORDS_FIELD], const uint8_t arg1[NBYTES_FIELD]);')
    click.echo(f'void fiat_fp{bitlength}_set_one(fp_t out1);')

    click.echo(f'\nvoid fiat_fp{bitlength}_random(fp_t x);')
    click.echo(f'void fiat_fp{bitlength}_copy(fp_t b, const fp_t a);')
    click.echo(f'int fiat_fp{bitlength}_compare(const fp_t b, const fp_t a);')
    click.echo(f'int fiat_fp{bitlength}_iszero(const fp_t x);')
    click.echo(f'void fiat_fp{bitlength}_string(char x_string[2*NBYTES_FIELD + 1], const fp_t x);')
    click.echo(f'void fiat_fp{bitlength}_printf(const fp_t x);')
    click.echo(f'void fiat_fp{bitlength}_pow(fp_t c, const fp_t a, const fp_t e);')
    click.echo(f'void fiat_fp{bitlength}_inv(fp_t x);')

    click.echo('\n#endif')

def asm_api(p : int):
    bitlength = p.bit_length()
    click.echo(f'#ifndef _P{bitlength}_ASM_H_')
    click.echo(f'#define _P{bitlength}_ASM_H_')
    click.echo(f'\n')
    click.echo(f'#include <stdbool.h>')
    click.echo(f'#include <string.h>')
    click.echo(f'#include "config.h"')
    click.echo(f'#if defined(_shortw_)')
    click.echo(f'#include "../../shortw/api.h"')
    click.echo(f'#elif defined(_mont_)')
    click.echo(f'#include "../../mont/api.h"')
    click.echo(f'#endif')
    click.echo(f'\n')
    click.echo(f'extern const fp_t uintbig_1;')
    click.echo(f'extern const fp_t fp_0, fp_1, r_squared_mod_p;')
    click.echo(f'bool uintbig_add(fp_t x, fp_t const y, fp_t const z); /* returns carry */')
    click.echo(f'bool uintbig_sub(fp_t x, fp_t const y, fp_t const z); /* returns borrow */')
    click.echo(f'\n')
    click.echo(f'void fp_mul(fp_t out1, const fp_t arg1, const fp_t arg2);')
    click.echo(f'void fp_sqr(fp_t out1, const fp_t arg1);')
    click.echo(f'void fp_add(fp_t out1, const fp_t arg1, const fp_t arg2);')
    click.echo(f'void fp_sub(fp_t out1, const fp_t arg1, const fp_t arg2);')
    click.echo(f'void fp_random(fp_t x);')
    click.echo(f'void fp_copy(fp_t b, const fp_t a);')
    click.echo(f'void fp_pow(fp_t c, const fp_t a, const fp_t e);')
    click.echo(f'void fp_inv(fp_t x);')
    click.echo(f'\n')
    click.echo(f'static inline void from_montgomery(fp_t c, const fp_t a)')
    click.echo(f"{{fp_mul(c, a, uintbig_1);}}")
    click.echo(f'\n')
    click.echo(f'static inline void to_montgomery(fp_t c, const fp_t a)')
    click.echo(f"{{fp_mul(c, a, r_squared_mod_p);}}")
    click.echo(f'\n')
    click.echo(f'static inline void from_bytes(digit_t out1[NWORDS_FIELD], const uint8_t arg1[NBYTES_FIELD])')
    click.echo(f"{{")
    click.echo(f"\tfor (size_t i = 0; i < NBYTES_FIELD; i++)")
    click.echo(f"\t\tout1[i>>3] += arg1[i] << (i & 0x7);")
    click.echo(f"}}")
    click.echo(f'\n')
    click.echo(f'static inline void to_bytes(uint8_t out1[NBYTES_FIELD], const digit_t arg1[NWORDS_FIELD])')
    click.echo(f"{{")
    click.echo(f"\tfor (size_t i = 0; i < NBYTES_FIELD; i++)")
    click.echo(f"\t\tout1[i] = arg1[i>>3] & (0xFF << ( i & 0x7));")
    click.echo(f"}}")
    click.echo(f'\n')
    click.echo(f'static inline void fp_set_one(fp_t c)')
    click.echo(f"{{")
    click.echo(f'    for (size_t i = 0; i < NWORDS_FIELD; i++)')
    click.echo(f'        c[i] = fp_1[i];')
    click.echo(f"}}")
    click.echo(f'\n')
    click.echo(f'static inline void fp_set_zero(fp_t c)')
    click.echo(f"{{")
    click.echo(f'    for (size_t i = 0; i < NWORDS_FIELD; i++)')
    click.echo(f'        c[i] = 0;')
    click.echo(f"}}")
    click.echo(f'\n')
    click.echo(f'static inline void fp_neg(fp_t c, const fp_t a)')
    click.echo(f"{{fp_sub(c, fp_0, a);}}")
    click.echo(f'\n')
    click.echo(f'static inline void fp_nonzero(digit_t* c, const digit_t a[NWORDS_FIELD])')
    click.echo(f"{{")
    click.echo(f'    int i;')
    click.echo(f'    digit_t out = 0;')
    click.echo(f'    for (i = 0; i < NWORDS_FIELD; i++)')
    click.echo(f'        out = out | a[i];')
    click.echo(f'    *c = out;')
    click.echo(f"}}")
    click.echo(f'\n')
    click.echo(f'#endif')


# Creating pX0X.c file
def pX0X(p : int):
    bitlength = p.bit_length()
    click.echo(f'#include "p{bitlength}_api.h"')
    click.echo('#include "../../rng.h"')

    click.echo('\n// Namespace concerning GF(p)')
    click.echo(f'#if defined(_assembly_)')
    click.echo(f'\t#include "p{bitlength}asm_api.h"')
    click.echo(f'#else')
    click.echo(f'\t#define fp_mul fiat_fp{bitlength}_mul')
    click.echo(f'\t#define fp_sqr fiat_fp{bitlength}_square')
    click.echo(f'\t#define fp_add fiat_fp{bitlength}_add')
    click.echo(f'\t#define fp_sub fiat_fp{bitlength}_sub')
    click.echo(f'\t#define fp_neg fiat_fp{bitlength}_opp')
    click.echo(f'\t#define from_montgomery fiat_fp{bitlength}_from_montgomery')
    click.echo(f'\t#define to_montgomery fiat_fp{bitlength}_to_montgomery')
    click.echo(f'\t#define fp_nonzero fiat_fp{bitlength}_nonzero')
    click.echo(f'\t#define to_bytes fiat_fp{bitlength}_to_bytes')
    click.echo(f'\t#define from_bytes fiat_fp{bitlength}_from_bytes')
    click.echo(f'\t#define fp_set_one fiat_fp{bitlength}_set_one')
    click.echo(f'\t#define fp_random fiat_fp{bitlength}_random')
    click.echo(f'\t#define fp_copy fiat_fp{bitlength}_copy')
    click.echo(f'\t#define fp_compare fiat_fp{bitlength}_compare')
    click.echo(f'\t#define fp_iszero fiat_fp{bitlength}_iszero')
    click.echo(f'\t#define fp_string fiat_fp{bitlength}_string')
    click.echo(f'\t#define fp_printf fiat_fp{bitlength}_printf')
    click.echo(f'\t#define fp_pow fiat_fp{bitlength}_pow')
    click.echo(f'\t#define fp_inv fiat_fp{bitlength}_inv')
    click.echo(f'#endif')

    click.echo('\n// GF(p²) implementation')
    click.echo('#include "../../fpx.c"')

    click.echo('\n// Short Weierstrass model')
    click.echo('#if defined(_shortw_)')
    click.echo('// Curve arithmetic')
    click.echo('#include "../../shortw/curvemodel.c"')
    click.echo('// Utility functions')
    click.echo('#include "../../shortw/utils.c"')
    click.echo('// MITM')
    click.echo('#if defined(_mitm_)')
    click.echo('#include "../../shortw/mitm-basic.c"')
    click.echo('#include "../../shortw/mitm-dfs-2.c"')
    click.echo('#include "../../shortw/mitm-dfs-3.c"')
    click.echo('#endif')
    click.echo('// vOW GCS')
    click.echo('#if defined(_vowgcs_)')
    click.echo('#include "../../shortw/vowgcs.c"')
    click.echo('#endif')

    click.echo('\n// Montgomery model')
    click.echo('#elif defined(_mont_)')
    click.echo('// Curve arithmetic')
    click.echo('#include "../../mont/curvemodel.c"')
    click.echo('// Utility functions')
    click.echo('#include "../../mont/utils.c"')
    click.echo('// MITM')
    click.echo('#if defined(_mitm_)')
    click.echo('#include "../../mont/mitm-basic.c"')
    click.echo('#include "../../mont/mitm-dfs-2.c"')
    click.echo('#include "../../mont/mitm-dfs-3.c"')
    click.echo('#endif')
    click.echo('// vOW GCS')
    click.echo('#if defined(_vowgcs_)')
    click.echo('#include<omp.h>')
    click.echo('#include "../../mont/pcs_vect_bin.c"')
    click.echo('#include "../../mont/pcs_struct_PRTL.c"')
    click.echo('#include "../../mont/vowgcs.c"')
    click.echo('#endif')

    click.echo('\n#else')
    click.echo('#error -- "Unsupported Curve Model"')
    click.echo('#endif')

@click.command()
@click.option('--e2', type=int, help='Exponent of 2')
@click.option('--e3', type=int, help='Exponent of 3')
@click.option('--f', type=int, help='Cofactor')
@click.option('--pc', type=int, help='Precomputation depth')
def setup(e2, e3, f, pc):
    original_stdout = sys.stdout # Save a reference to the original standard output
    p = 2**e2 * 3**e3 * f - 1
    assert(is_prime(p))
    # path
    path = f'src/primes/p{p.bit_length()}'

    try:
        os.mkdir(path)
        click.echo(f'File created: \'{path}\'')
    except OSError as error:
        click.echo(f'{str(error).replace("Errno 17", "WARNING")}')

    with open(f'{path}/config.h', 'w') as file:
        sys.stdout = file
        config(p, e2, e3, f, pc)
        sys.stdout = original_stdout # Reset the standard output to its original value
    click.echo(f'File written: \'{path}/config.h\'')

    with open(f'{path}/p{p.bit_length()}_api.h', 'w') as file:
        sys.stdout = file
        api(p)
        sys.stdout = original_stdout # Reset the standard output to its original value
    click.echo(f'File written: \'{path}/p{p.bit_length()}_api.h\'')

    with open(f'{path}/p{p.bit_length()}asm_api.h', 'w') as file:
        sys.stdout = file
        asm_api(p)
        sys.stdout = original_stdout # Reset the standard output to its original value
    click.echo(f'File written: \'{path}/p{p.bit_length()}asm_api.h\'')

    with open(f'{path}/p{p.bit_length()}.c', 'w') as file:
        sys.stdout = file
        pX0X(p)
        sys.stdout = original_stdout # Reset the standard output to its original value
    click.echo(f'File written: \'{path}/p{p.bit_length()}.c\'')

    prime = '0x%X' % p
    with open(f'fiat-crypto-p{p.bit_length()}.log', 'w') as file:
        sys.stdout = file
        click.echo(f'../../GitHub/fiat-crypto/src/ExtractionOCaml/word_by_word_montgomery \'fp{p.bit_length()}\' \'64\' \'{prime}\' > src/p{p.bit_length()}/fp{p.bit_length()}_x64_.c')
        click.echo(f'../../GitHub/fiat-crypto/src/ExtractionOCaml/word_by_word_montgomery \'fp{p.bit_length()}\' \'32\' \'{prime}\' > src/p{p.bit_length()}/fp{p.bit_length()}_x86_.c')
        sys.stdout = original_stdout # Reset the standard output to its original value
    click.echo(f'File written: \'fiat-crypto-p{p.bit_length()}.log\'')


if __name__ == '__main__':
    setup()
