#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#importing sys module
import sys
# importing os module
import os
import re
import json
import optparse
import math

# The next variables are required for constructing isogenies
# ++++++++++ shortw
p2 =  8.5   # xdbl
q2 =  5.7   # 2-isogeny evaluations
p3 = 18.9   # xtpl
q3 =  8.4   # 3-isogeny evaluations
q4 = 2*p2   # 2 xdbl
q4 = 1.5*q2 # 4-isogeny evaluations
# ++++++++++

ceildiv = lambda x,y: -(-x//y)
int2str = lambda x, string : string.format(x)

def fp2str(x : int, arch):
    s = arch >> 2
    length = ceildiv(x.bit_length(), s * 4) * s
    X = int2str(x, '{:0%dX}' % length)
    Y = [ X[i:i+s] for i in range(0, length, s) ]
    Y.reverse()
    return '{ 0x' + ', 0x'.join(Y) + ' }'

def fp2str_(x : int, p, arch):
    s = arch >> 2
    length = ceildiv(p.bit_length(), s * 4) * s
    X = int2str(x, '{:0%dX}' % length)
    Y = [ X[i:i+s] for i in range(0, length, s) ]
    Y.reverse()
    return '{ 0x' + ', 0x'.join(Y) + ' }'

def bound2str(x : int, p : int, arch):
    s = arch >> 2
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

def xgcd(a,b):
    prevx, x = 1, 0;  prevy, y = 0, 1
    while b:
        q, r = divmod(a,b)
        x, prevx = prevx - q*x, x
        y, prevy = prevy - q*y, y
        a, b = b, r
    return a, prevx, prevy

def mont_params(p, nw, arch):
    r = (2**arch)**nw
    R = r % p
    neg_R = (-R) % p
    _, mu,_ = xgcd((-p) % r, r)
    mu = mu % r
    return mu, R, neg_R

# Creating config.h file
def config(p, e2, e3, f, pc_depth, arch):
    bitlength = p.bit_length()
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

    # Last 32-bits word
    if log2e2 % 32 == 0:
        mke2 = '%X' % 0xFFFFFFFF
    else:
        mke2 = '%X' % (2**(log2e2 % 32) - 1)
    
    if log2e3 % 32 == 0:
        mke3 = '%X' % 0xFFFFFFFF
    else:
        mke3 = '%X' % (2**(log2e3 % 32) - 1)

    try:
        with open('../tools/config_template.h', 'r') as fconf:
            content = fconf.read()
    except:
        print("Error: config template")
        sys.exit(1)

    content = re.sub('<bitlength>', str(bitlength), content)
    content = re.sub('<radix>', str(arch), content)
    content = re.sub('<log2radix>', str(int(math.log(arch, 2))), content)
    content = re.sub('<arch>', str(arch), content)
    content = re.sub('<nwordsfield>', str(ceildiv(bitlength, arch)), content)
    content = re.sub('<mask32>', str(mask32), content)
    
    if arch == 64:
        content = re.sub('<rephex>', str(arch >> 2), content)
        prix = 'PRIx64'
        sprintf = '"%" PRIx64, x'
    else:
        content = re.sub('<rephex>', str(arch >> 2) + 'X', content)
        prix = ''
        sprintf = '"%X", x'
    content = re.sub('<prix>', prix, content)
    content = re.sub('<sprintf>', sprintf, content)

    content = re.sub('<byteslength>', str(ceildiv(bitlength, 8)), content)
    content = re.sub('<cofactor>', str(f), content)
    content = re.sub('<pc_depth>', str(pc_depth), content)

    content = re.sub('<e2>', str(e2), content)
    content = re.sub('<mask2>', str(mask2), content)
    content = re.sub('<expmk2>', str(mke2), content)

    exp0 = (e2 - (e2 >> 1)) - 1
    exp1 = (e2 >> 1) - 1
    log_of_e = exp0

    log_of_e = math.ceil(math.log(log_of_e)/math.log(2)) * 2
    content = re.sub('<log2ofe>', str(log_of_e), content)

    content = re.sub('<e2min1>', str(e2 - 1), content)
    content = re.sub('<e2div21>', str(e2//2 - 1), content)
    content = re.sub('<ceildive2>', str(ceildiv(e2,2) - 1), content)
    content = re.sub('<e2div22>', str(e2//2 - 2), content)
    content = re.sub('<ceildive22>', str(ceildiv(e2,2) - 2), content)
    content = re.sub('<pc_strategy_size_0>', str(ceildiv(e2,2) - 2 - pc_depth), content)
    content = re.sub('<pc_strategy_size_1>', str(e2//2 - 2 - pc_depth), content)
    
    content = re.sub('<e3>', str(e3), content)
    content = re.sub('<log2e3>', str(log2e3), content)
    content = re.sub('<mask3>', str(mask3), content)
    content = re.sub('<expmk3>', str(mke3), content)

    content = re.sub('<e3min1>', str(e3 - 1), content)
    content = re.sub('<e3div21>', str(e3//2 - 1), content)
    content = re.sub('<ceildive3>', str(ceildiv(e3,2) - 1), content)
    
    print(content)

# Creating constants.cu file
def constants(p, e2, e3, f, pc_depth, arch):
    bitlength = p.bit_length()

    try:
        with open('../tools/constants.cu', 'r') as fconst:
            content = fconst.read()
    except:
        print("Error: config template")
        sys.exit(1)

    content = re.sub('<prime>', str(fp2str(p, arch)), content)
    mu, mont_one, mont_minus_one = mont_params(p, ceildiv(bitlength, arch), arch)
    content = re.sub('<mu>', str(fp2str(mu, arch)), content)
    content = re.sub('<montone>', str(fp2str(mont_one, arch)), content)
    content = re.sub('<mminusone>', str(fp2str(mont_minus_one, arch)), content)
    content = re.sub('<bigone>', str(fp2str_(1, p, arch)), content)
    
    content = re.sub('<p_minus_two>', str(fp2str(p-2, arch)), content)
    content = re.sub('<p_minus_three_q>', str(fp2str((p-3)//4, arch)), content)
    content = re.sub('<p_min_one_h>', str(fp2str((p-1)//2, arch)), content)
    content = re.sub('<bound2>', str(bound2str(2**(e2-1), p, arch)), content)
    content = re.sub('<bound20>', str(bound2str((2**(e2//2-1)) >> 1, p, arch)), content)
    content = re.sub('<bound21>', str(bound2str((2**(ceildiv(e2,2)-1)) >> 1, p, arch)), content)
    content = re.sub('<bound3>', str(bound2str(3**(e3-1), p, arch)), content)
    content = re.sub('<bound30>', str(bound2str(3**(e3//2-1), p, arch)), content)
    content = re.sub('<bound31>', str(bound2str(3**(ceildiv(e3,2)-1) , p, arch)), content)

    content = re.sub('<e2min1>', str(e2 - 1), content)
    strategy2 = '{ ' + ', '.join(map(str, strategy(e2, p2, q2))) + '};'
    content = re.sub('<strategy2>', strategy2, content)

    content = re.sub('<e2div21>', str(e2//2 - 1), content)
    strategy2_0 = '{ ' + ', '.join(map(str, strategy(e2//2, p2, q2))) + '};'
    content = re.sub('<strategy20>', strategy2_0, content)

    content = re.sub('<ceildive2>', str(ceildiv(e2,2) - 1), content)
    strategy2_1 = '{ ' + ', '.join(map(str, strategy(ceildiv(e2,2), p2, q2))) + '};'
    content = re.sub('<strategy21>', strategy2_1, content)

    content = re.sub('<e2div22>', str(e2//2 - 2), content)
    strategy2_REDUCED_0 = '{ ' + ', '.join(map(str, strategy(e2//2-1, p2, q2))) + '};'
    content = re.sub('<strategy2_red_0>', strategy2_REDUCED_0, content)
    
    content = re.sub('<ceildive22>', str(ceildiv(e2,2) - 2), content)
    strategy2_REDUCED_1 = '{ ' + ', '.join(map(str, strategy(ceildiv(e2,2)-1, p2, q2))) + '};'
    content = re.sub('<strategy2_red_1>', strategy2_REDUCED_1, content)
    
    content = re.sub('<pc_strategy_size_0>', str(ceildiv(e2,2) - 2 - pc_depth), content)
    strategy2_pc_0 = '{ ' + ', '.join(map(str, strategy(ceildiv(e2,2)-1-pc_depth, p2, q2))) + '};'
    content = re.sub('<strategy2_pc_0>', strategy2_pc_0, content)
    
    content = re.sub('<pc_strategy_size_1>', str(e2//2 - 2 - pc_depth), content)
    strategy2_pc_1 = '{ ' + ', '.join(map(str, strategy(e2//2-1-pc_depth, p2, q2))) + '};'
    content = re.sub('<strategy2_pc_1>', strategy2_pc_1, content)

    content = re.sub('<e3min1>', str(e3 - 1), content)
    strategy3 = '{ ' + ', '.join(map(str, strategy(e3, p3, q3))) + '};'
    content = re.sub('<strategy3>', strategy3, content)

    content = re.sub('<e3div21>', str(e3//2 - 1), content)
    strategy3_0 = '{ ' + ', '.join(map(str, strategy(e3//2, p3, q3))) + '};'
    content = re.sub('<strategy30>', strategy3_0, content)
    
    content = re.sub('<ceildive3>', str(ceildiv(e3,2) - 1), content)
    strategy3_1 = '{ ' + ', '.join(map(str, strategy(ceildiv(e3,2), p3, q3))) + '};'
    content = re.sub('<strategy31>', strategy3_1, content)
    
    print(content)

def get_data(path):
    try:
        with open(path, 'r') as fp:
            data = fp.read()
    except:
        print('Error: %s' % path)
        sys.exit(1)
    return data

# src/rng
def randombytes(p, arch):
    nw = ceildiv(p.bit_length(), arch)

    if arch == 64:
        if nw == 2:
            path = ('../tools/rng64_1.cu')
            rng = get_data(path)
        if nw == 3:
            path = ('../tools/rng64_2.cu')
            rng = get_data(path)

            path = ('../tools/rng64_1.cu')
            data = get_data(path)
            rng = re.sub('<rng1>', data, rng)

        if nw == 4:
            path = ('../tools/rng64_3.cu')
            rng = get_data(path)
            
            path = ('../tools/rng64_2.cu')
            data = get_data(path)
            rng = re.sub('<rng2>', data, rng)
            
            path = ('../tools/rng64_1.cu')
            data = get_data(path)
            rng = re.sub('<rng1>', data, rng)
    else:
        if nw <= 4:
            path = ('../tools/rng32_1.cu')
            rng = get_data(path)
        elif nw > 4 and nw <= 6:
            path = ('../tools/rng32_2.cu')
            rng = get_data(path)

            path = ('../tools/rng32_1.cu')
            data = get_data(path)
            rng = re.sub('<rng1>', data, rng)
        else:
            path = ('../tools/rng32_3.cu')
            rng = get_data(path)
            
            path = ('../tools/rng32_2.cu')
            data = get_data(path)
            rng = re.sub('<rng2>', data, rng)
            
            path = ('../tools/rng32_1.cu')
            data = get_data(path)
            rng = re.sub('<rng1>', data, rng)

    try:
        with open('../tools/rng_template.cu', 'r') as fp:
            content = fp.read()
    except:
        print('Error: tools/rng_template.cu')
        sys.exit(1)
    
    content = re.sub('<rng>', rng, content)
    print(content)

# vOW setup
def vow_setup(p, cells, blocks, threads, e2, cores):
    try:
        with open('../tools/vowgcs_setup.h', 'r') as fvow:
            content = fvow.read()
    except:
        print("Error: config vowgcs_setup.h")
        sys.exit(1)

    # NO Frobenius
    ebits0 = (e2 - (e2 >> 1)) - 1
    ebits1 = (e2 >> 1) - 1
    
    content = re.sub('<omegabits>', str(cells), content)
    omega = 1 << cells
    content = re.sub('<omega>', str(omega), content)
    beta = 10
    content = re.sub('<beta>', str(beta), content)
    betaXomega = beta * omega
    # Each thread has the task of reaching (BETA * OMEGA)/NUMBER_OF_THREADS
    betaXomega = math.ceil(betaXomega / (blocks * threads))
    content = re.sub('<betaXomega>', str(betaXomega), content)
    m = math.pow(2, ebits0) + math.pow(2, ebits1)
    theta = 2.25 * math.sqrt(omega / m)
    content = re.sub('<theta>', str(theta), content)
    maxtrail = math.ceil(10.0 / theta)
    content = re.sub('<maxtrail>', str(maxtrail), content)
    n = math.floor(math.log(1.0 / theta)/math.log(2))
    content = re.sub('<n>', str(n), content)
    Rbits = 4
    content = re.sub('<Rbits>', str(Rbits), content)
    distinguished = math.floor(theta * math.pow(2, n) * math.pow(2, Rbits))
    content = re.sub('<distinguished>', str(distinguished), content)
    maxprf = 1
    content = re.sub('<maxprf>', str(maxprf), content)
    trail_bits = math.ceil(math.log(maxtrail) / math.log(2))
    content = re.sub('<trailbits>', str(trail_bits), content)
    triplet_bytes = int((2 * (ebits0 + 1) - cells - n + trail_bits) / 8 + 1)
    content = re.sub('<tripletbytes>', str(triplet_bytes), content)
    
    content = re.sub('<numblocks>', str(blocks), content)
    content = re.sub('<numthreads>', str(threads), content)
    
    content = re.sub('<cores>', str(cores), content)
    
    print(content)

if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option("--arch", action="store", dest="arch")
    parser.add_option("-c", action="store_true", dest="do_config", default=False)
    parser.add_option("-r", action="store_true", dest="do_rng", default=False)
    parser.add_option("-s", action="store_true", dest="do_const", default=False)
    parser.add_option("-v", action="store_true", dest="do_vow", default=False)
    options, args = parser.parse_args()

    try:
        word_size = int(options.arch)
        assert(word_size == 64 or word_size == 32)
        with open('params.json') as fp:
            params = json.load(fp)
    except:
        print('Error: params.json')
        sys.exit(1)

    e2 = int(params['e2'])
    e3 = int(params['e3'])
    f = int(params['f'])
    cells = int(params['cells'])
    pc_depth = int(params['pc_depth'])
    cores = int(params['cores'])
    blocks = int(params['blocks'])
    threads = int(params['threads'])
    
    p = 2**e2 * 3**e3 * f - 1
    assert(is_prime(p))
    # Note: this version use just pre-computing
    assert(pc_depth != 0)

    if options.do_config: config(p, e2, e3, f, pc_depth, word_size)
    if options.do_rng: randombytes(p, word_size)
    if options.do_const: constants(p, e2, e3, f, pc_depth, word_size)
    if options.do_vow: vow_setup(p, cells, blocks, threads, e2, cores)

