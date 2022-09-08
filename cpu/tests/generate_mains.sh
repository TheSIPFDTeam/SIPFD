#!/bin/bash

mkdir -p arith_tests
echo "#include \"../../src/primes/p$1/p$1.c\"" > arith_tests/p$1.c
echo "#include \"../arith_tests.c\"" >> arith_tests/p$1.c

mkdir -p mitm-basic
echo "#include \"../../src/primes/p$1/p$1.c\"" > mitm-basic/p$1.c
echo "#include \"../mitm-basic.c\"" >> mitm-basic/p$1.c

mkdir -p mitm-dfs
echo "#include \"../../src/primes/p$1/p$1.c\"" > mitm-dfs/p$1.c
echo "#include \"../mitm-dfs.c\"" >> mitm-dfs/p$1.c

mkdir -p mitm-dfs-memory-limited
echo "#include \"../../src/primes/p$1/p$1.c\"" > mitm-dfs-memory-limited/p$1.c
echo "#include \"../mitm-dfs-memory-limited.c\"" >> mitm-dfs-memory-limited/p$1.c

mkdir -p mitm-memory-limited
echo "#include \"../../src/primes/p$1/p$1.c\"" > mitm-memory-limited/p$1.c
echo "#include \"../mitm-memory-limited.c\"" >> mitm-memory-limited/p$1.c

mkdir -p vow_instance_generator
echo "#include \"../../src/primes/p$1/p$1.c\"" > vow_instance_generator/p$1.c
echo "#include \"../vow_instance_generator.c\"" >> vow_instance_generator/p$1.c

mkdir -p vowgcs
echo "#include \"../../src/primes/p$1/p$1.c\"" > vowgcs/p$1.c
echo "#include \"../vowgcs.c\"" >> vowgcs/p$1.c

mkdir -p vowrigged
echo "#include \"../../src/primes/p$1/p$1.c\"" > vowrigged/p$1.c
echo "#include \"fixed_instance_p$1.h\"" >> vowrigged/p$1.c
echo "#include \"../vowrigged.c\"" >> vowrigged/p$1.c

