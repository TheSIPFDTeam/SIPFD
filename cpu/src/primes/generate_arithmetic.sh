#!/bin/bash
bits=`python -c "import math;print(int(math.ceil(math.log($1,2))))"`
words=`python -c "import math;print(int(64*math.ceil(math.log($1,2)/64)))"`
mkdir -p p$bits

echo "p:=$1;" > tmp.magma
cat AsmMultCodegenerator.magma >> tmp.magma
magma tmp.magma
rm tmp.magma
cat uint$words.s > p$bits/fp${bits}_asm_.s
tail -n +4 fp$words.s | head -n +45 >> p$bits/fp${bits}_asm_.s
cat asm_${words}.s >> p$bits/fp${bits}_asm_.s
rm fp*.s
rm uint*.s
rm gmp*.s

word_by_word_montgomery fp$bits 64 $1 > p$bits/fp${bits}_x64_.c
word_by_word_montgomery fp$bits 32 $1 > p$bits/fp${bits}_x86_.c
