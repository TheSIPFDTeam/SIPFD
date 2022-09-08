## Compiling
```bash
make ALGORITHM=[mitm/vowgcs/none] MODEL=[mont/shortw] ARITH=[asm/fiat] testsXX

Options:
	ALGORITHM:	Either meet-in-the-middle or golden collision search
	MODEL:		Short Weierstrass or Montgomery curve models	
	ARITH:		Optimized assembly-code arithmetic (64-bit only) or generic fiat crypto arithmetic
	testsXX:	Replace XX by the bitlength of the primes (e.j. `tests69`). See src/primes/ for a list of available primes
```

## Implemented algorithms
- mitm-basic: basic meet-in-the-middle without tree structure
- mitm-dfs: meet-in-the-middle exploiting the tree structure in a depth-dirst search
- mitm-dfs-memory-limited: attack carried out in batches for finite memory
- vow-gcs: van Oorschot & Wiener (vOW) Golden Collision Search (GCS)
- vow-rigged: the GCS attack on a fixed predefined instance (located in tests/vowrigged/fixed_instance_pXX.h)

## Requirements

Any gcc compiler version.
Python3 is used for generating new `config.h` files.
Additionally, `openssl` and `openmp` library for the C-code compilation.

```bash
# OPENMP
sudo apt install libomp-dev
```

## Example Runs
Benchmarking the assembly arithmetic
```bash
make ALGORITHM=none MODEL=mont ARITH=asm tests69
SIPFD69/arith_tests_x64_ 
```

Running the mitm-basic attack on Bob's side (l=3) and 3^2 cores
```bash
make ALGORITHM=mitm MODEL=mont ARITH=asm tests72
SIPFD72/mitm-basic_x64_ -s Alice -c 2
```

Running the mitm-dfs-memory-limmited attack on Alice's side (l=2) with 2^3 cores and 2^7 memory
```bash
make ALGORITHM=mitm MODEL=mont ARITH=asm tests82
SIPFD72/mitm-dfs-memory-limited_x64_  -s Alice -c 3 -w 7
```

Running the vOW Golden Collision Search on Alice's side (l=1) with 8 cores and 2^18 memory
```bash
make ALGORITHM=vowgcs MODEL=mont ARITH=asm tests111
SIPFD111/vow-gcs_x64_ -s Alice -c 8 -w 18
```

Generating a new fixed instance for vow-rigged
```bash 
make ALGORITHM=instance_generator MODEL=mont ARITH=asm tests130
SIPFD130/instance_generator_x64_
```

## Precomputation
For the vOW attack, precomputation can be toggled with the -p flag. The precomputation depth can be adjusted by editing and rerunning src/setupall.sh
