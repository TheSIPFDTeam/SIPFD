#include <cooperative_groups.h>
#include "api.h"
#include "vowgcs_setup.h"
#include "fp.h"

namespace cg = cooperative_groups;

extern "C" __global__ void bench_fp_add(uint64_t *cc, double *times, proj_t *out,
        curandStatePhilox4_32_10_t *state, int clockrate) {
    int j, t = threadIdx.x + blockIdx.x * blockDim.x;
    fp_t t1;
    fp_t t2;
    fp_t t3;
    uint64_t start, stop;

    fp_random(t1, state);
    fp_random(t2, state);
    fp_random(t3, state);

    start = clock64();
    for (j = 0; j < RUNS_BENCH; j++) {
        fp_add(t1, t2, t3);
        fp_add(t2, t3, t1);
        fp_add(t3, t1, t2);
    }
    stop = clock64();
    
    j = t & 31;
    t = t >> 5;
    if (j == 0) {
        fp_copy(out[t][0][0], t3);
        cc[t] = stop - start;
        // microsec
        times[t] = (stop - start ) * 1000 / clockrate;
    }
}

extern "C" __global__ void bench_fp_mul(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate) {
    int j, t = threadIdx.x + blockIdx.x * blockDim.x;
    fp_t t1;
    fp_t t2;
    fp_t t3;
    uint64_t start, stop;

    fp_random(t1, state);
    fp_random(t2, state);
    fp_random(t3, state);

    start = clock64();
    for (j = 0; j < RUNS_BENCH; j++) {
        fp_mul(t1, t2, t3);
        fp_mul(t2, t3, t1);
        fp_mul(t3, t1, t2);
    }
    stop = clock64();

    j = t & 31;
    t = t >> 5;
    if (j == 0) {
        fp_copy(out[t][0][0], t3);
        cc[t] = stop - start;
        // microsec
        times[t] = (stop - start) * 1000 / clockrate;
    }
}

extern "C" __global__ void bench_fp2_add(uint64_t *cc, double *times,
        curandStatePhilox4_32_10_t *state, int clockrate) {
    int j, t = threadIdx.x + blockIdx.x * blockDim.x;
    fp2_t t1;
    fp2_t t2;
    fp2_t t3;
    fp2_t t4;
    uint64_t start, stop;

    fp2_random(t2, state);
    fp2_random(t3, state);

    start = clock64();
    for (j = 0; j < RUNS_BENCH; j++) {
        fp2_add(t1, t2, t3);
        fp2_add(t2, t3, t1);
        fp2_add(t3, t1, t2);
    }
    stop = clock64();
    
    j = t & 31;
    t = t >> 5;
    if (j == 0) {
        fp2_copy(t4, t3);
        cc[t] = stop - start;
        times[t] = (stop - start) * 1000 / clockrate;
    }
}

extern "C" __global__ void bench_fp2_mul(uint64_t *cc, double *times,
        curandStatePhilox4_32_10_t *state, int clockrate) {
    int j, t = threadIdx.x + blockIdx.x * blockDim.x;
    fp2_t t1;
    fp2_t t2;
    fp2_t t3;
    fp2_t t4;
    uint64_t start, stop;

    fp2_random(t2, state);
    fp2_random(t3, state);

    start = clock64();
    for (j = 0; j < RUNS_BENCH; j++) {
        fp2_mul(t1, t2, t3);
        fp2_mul(t2, t3, t1);
        fp2_mul(t3, t1, t2);
    }
    stop = clock64();

    j = t & 31;
    t = t >> 5;
    if (j == 0) {
        fp2_copy(t4, t3);
        // microsec
        cc[t] = stop - start;
        times[t] = (stop - start) * 1000 / clockrate;
    }
}

extern "C" __global__ void bench_fp2_sqr(uint64_t *cc, double *times,
        curandStatePhilox4_32_10_t *state, int clockrate) {
    int j, t = threadIdx.x + blockIdx.x * blockDim.x;
    fp2_t t1;
    fp2_t t2;
    fp2_t t3;
    uint64_t start, stop;

    fp2_random(t2, state);

    start = clock64();
    for (j = 0; j < RUNS_BENCH; j++) {
        fp2_sqr(t1, t2);
        fp2_sqr(t2, t1);
    }
    stop = clock64();

    j = t & 31;
    t = t >> 5;
    if (j == 0) {
        fp2_copy(t3, t2);
        // microsec
        cc[t] = stop - start;
        times[t] = (stop - start) * 1000 / clockrate;
    }
}

extern "C" __global__ void bench_xadd(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate) {
    int j, t = threadIdx.x + blockIdx.x * blockDim.x;
    proj_t t1;
    proj_t t2;
    proj_t t3;
    proj_t t4;
    uint64_t start, stop;

    fp2_random(t1[0], state);
    fp2_random(t1[1], state);
    fp2_random(t2[0], state);
    fp2_random(t2[1], state);
    fp2_random(t3[0], state);
    fp2_random(t3[1], state);

    start = clock64();
    for (j = 0; j < RUNS_BENCH; j++) {
        xadd(t4, t1, t2, t3);
        xadd(t1, t4, t3, t2);
        xadd(t3, t2, t1, t4);
        xadd(t2, t3, t4, t1);
    }
    stop = clock64();

    j = t & 31;
    t = t >> 5;
    if (j == 0) {
        proj_copy(out[t], t2);
        // microsec
        cc[t] = stop - start;
        times[t] = (stop - start) * 1000 / clockrate;
    }
}

extern "C" __global__ void bench_xdbl(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate) {
    int j, t = threadIdx.x + blockIdx.x * blockDim.x;
    proj_t t1;
    proj_t t2;
    proj_t t3;
    uint64_t start = 0, stop = 0;
    
    fp2_random(t1[0], state);
    fp2_random(t1[1], state);
    fp2_random(t2[0], state);
    fp2_random(t2[1], state);
    fp2_random(t3[0], state);
    fp2_random(t3[1], state);

    start = clock64();
    for (j = 0; j < RUNS_BENCH; j++) {
        xdbl(t3, t1, t2);
        xdbl(t1, t2, t3);
        xdbl(t2, t3, t1);
    }
    stop = clock64();

    j = t & 31;
    t = t >> 5;
    if (j == 0) {
        proj_copy(out[t], t2);
        // microsec
        cc[t] = stop - start;
        times[t] = (stop - start) * 1000 / clockrate;
    }
}

extern "C" __global__ void bench_xtpl(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate) {
    int j, t = threadIdx.x + blockIdx.x * blockDim.x;
    proj_t t1;
    proj_t t2;
    proj_t t3;
    uint64_t start = 0, stop = 0;

    fp2_random(t1[0], state);
    fp2_random(t1[1], state);
    fp2_random(t2[0], state);
    fp2_random(t2[1], state);
    fp2_random(t3[0], state);
    fp2_random(t3[1], state);

    start = clock64();
    for (j = 0; j < RUNS_BENCH; j++) {
        xtpl(t3, t1, t2);
        xtpl(t1, t2, t3);
        xtpl(t2, t3, t1);
    }
    stop = clock64();
    
    j = t & 31;
    t = t >> 5;
    if (j == 0) {
        proj_copy(out[t], t2);
        // microsec
        cc[t] = stop - start;
        times[t] = (stop - start) * 1000 / clockrate;
    }
}

extern "C" __global__ void bench_ladder3pt(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate) {
    int j, t = threadIdx.x + blockIdx.x * blockDim.x;
    fp_t a = {0};
    proj_t t1;
    proj_t t2;
    proj_t t3;
    proj_t t4;
    proj_t t5;
    uint64_t start = 0, stop = 0;

    random_mod_A(a, state);
    fp2_random(t1[0], state);
    fp2_random(t1[1], state);
    fp2_random(t2[0], state);
    fp2_random(t2[1], state);
    fp2_random(t3[0], state);
    fp2_random(t3[1], state);
    fp2_random(t4[0], state);
    fp2_random(t4[1], state);
    fp2_random(t5[0], state);
    fp2_random(t5[1], state);

    start = clock64();
    for (j = 0; j < RUNS_BENCH_S; j++) {
        ladder3pt(t1, a[0], t2, t3, t4, t5, (EXPONENT2-2)>>1);
        ladder3pt(t2, a[0], t3, t4, t5, t1, (EXPONENT2-2)>>1);
        ladder3pt(t3, a[0], t4, t5, t1, t2, (EXPONENT2-2)>>1);
        ladder3pt(t4, a[0], t5, t1, t2, t3, (EXPONENT2-2)>>1);
        ladder3pt(t5, a[0], t1, t2, t3, t4, (EXPONENT2-2)>>1);
    }
    stop = clock64();
    
    j = t & 31;
    t = t >> 5;
    if (j == 0) {
        proj_copy(out[t], t5);
        // microsec
        cc[t] = stop - start;
        times[t] = (stop - start) * 1000 / clockrate;
    }
}

__device__ void prf(limb_t S[2][EXP0], fp2_t *P0, fp2_t *Q0, fp2_t *PQ0, fp2_t *E0, fp2_t *Z0,
        fp2_t *P1, fp2_t *Q1, fp2_t *PQ1, fp2_t *E1, fp2_t *Z1, limb_t *expo, limb_t *ebits,
        uint64_t seed, uint64_t *cc, double *times, int t, curandStatePhilox4_32_10_t *state, int clockrate)
{
    int i;
    uint64_t start = 0, stop = 0;
    point_t x = {0};
    fp2_t j;
    ctx_t context;
    
    context.NONCE = 0x2F74AD2924062C22 + seed;

    fp2_random(j, state);
    _gn_(&x, j, context.NONCE);

    start = clock64();

    for (i = 0; i < RUNS_BENCH_S; i++)
        _fn_(&x, j, &x, &context, S, P0, Q0, PQ0, E0, Z0, P1, Q1, PQ1, E1, Z1, expo, ebits);

    stop = clock64();

    i = t & 31;
    t = t >> 5;
    if (i == 0) {
        // microsec
        cc[t] = stop - start;
        times[t] = (double)(stop - start) * 1000 / clockrate;
    }
}

extern "C" __global__ void bench_prf(fp2_t *P0, fp2_t *Q0, fp2_t *PQ0, fp2_t *E0, fp2_t *Z0,
        fp2_t *P1, fp2_t *Q1, fp2_t *PQ1, fp2_t *E1, fp2_t *Z1, curandStatePhilox4_32_10_t *state,
        uint64_t seed, uint64_t *cc, double *times, int clockrate)
{
    cg::grid_group g = cg::this_grid();

    int t = g.thread_rank();
    int i;
    __shared__ limb_t expo[2];
    __shared__ limb_t ebits[2];
    __shared__ limb_t strategy_reduced[2][EXP0];

    if (threadIdx.x == 0) {
        expo[0] = EXP0;
        expo[1] = EXP1;
        ebits[0] = EXP20_BITS;
        ebits[1] = EXP21_BITS;

        for (i = 0; i < PC_STRATEGY_SIZE_0; i++)
            strategy_reduced[0][i] = STRATEGY2_PC_0[i];
        
        for (i = 0; i < PC_STRATEGY_SIZE_1; i++)
            strategy_reduced[1][i] = STRATEGY2_PC_1[i];
    }

    g.sync();

    prf(strategy_reduced, P0, Q0, PQ0, E0, Z0, P1, Q1, PQ1, E1, Z1, expo, ebits, seed, cc, times, t, state, clockrate);
}

extern "C" __global__ void bench_xisog_2e_2nd(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate) {
    int i;
    cg::grid_group g = cg::this_grid();
    __shared__ limb_t strategy_reduced[2][EXP0];
    int t = g.thread_rank();
    
    proj_t E0, EA2;
    proj_t PA, QA, PQA, RA;
    fp_t a = {0};
    uint64_t start = 0, stop = 0;

    if (threadIdx.x == 0) {
        for (i = 0; i < EXP0; i++) {
            strategy_reduced[0][i] = STRATEGY2_REDUCED_0[i];
        }

        for (i = 0; i < EXP1; i++) {
            strategy_reduced[1][i] = STRATEGY2_REDUCED_1[i];
        }
    }

    set_initial_curve(E0);
    init_basis(PA, QA, PQA, E0, state);
    xmul(PA, COFACTOR, PA, E0);
    xmul(QA, COFACTOR, QA, E0);
    xmul(PQA, COFACTOR, PQA, E0);

    g.sync();

    start = clock64();
    for (i = 0; i < RUNS_BENCH_S; i++) {
        // derive: degree-(2^e2) isogeny
        random_mod_A(a, state);
        ladder3pt_long(RA, a, PA, QA, PQA, E0);
        xisog_2e_2nd(EA2, RA, E0, strategy_reduced[0], EXP0);
    }
    stop = clock64();
    
    i = t & 31;
    t = t >> 5;
    if (i == 0) {
        proj_copy(out[t], EA2);
        // microsec
        cc[t] = stop - start;
        times[t] = (stop - start) * 1000 / clockrate;
    }
}

