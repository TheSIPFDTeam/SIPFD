#include <time.h>
#include "api.h"
#include "fp.h"
#include "../external/precomp.h"

#ifdef _vowgcs_
#include "vowgcs_setup.h"
#endif

extern "C" __global__ void tests(curandStatePhilox4_32_10_t *state);
extern "C" __global__ void bench_fp_add(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate);
extern "C" __global__ void bench_fp_mul(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate);

extern "C" __global__ void bench_fp2_add(uint64_t *cc,double *times, 
        curandStatePhilox4_32_10_t *state, int clockrate);
extern "C" __global__ void bench_fp2_mul(uint64_t *cc, double *times,
        curandStatePhilox4_32_10_t *state, int clockrate);
extern "C" __global__ void bench_fp2_sqr(uint64_t *cc, double *times,
        curandStatePhilox4_32_10_t *state, int clockrate);

extern "C" __global__ void bench_xadd(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate);
extern "C" __global__ void bench_xdbl(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate);
extern "C" __global__ void bench_xtpl(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate);

extern "C" __global__ void bench_ladder3pt(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate);

extern "C" __global__ void bench_prf(fp2_t *P0, fp2_t *Q0, fp2_t *PQ0, fp2_t *E0, fp2_t *Z0,
        fp2_t *P1, fp2_t *Q1, fp2_t *PQ1, fp2_t *E1, fp2_t Z1, curandStatePhilox4_32_10_t *state,
        uint64_t seed, uint64_t *cc, double *times, int clockrate);

extern "C" __global__ void bench_xisog_2e_2nd(uint64_t *cc, double *times, proj_t *out, 
        curandStatePhilox4_32_10_t *state, int clockrate);

#define gpuchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/******************************************************************************/
void arith_test() {
    uint32_t blocks = 1;
    uint32_t threads = 1;
    uint64_t seed;
 
    curandStatePhilox4_32_10_t *PHILOXStates;

    srand(time(0));
    seed = rand();

    cudaMalloc((void **)&PHILOXStates, (blocks * threads) * sizeof(curandStatePhilox4_32_10_t));

    setup_rand<<<blocks, threads>>>(PHILOXStates, seed);
    // Launch the tests in GPU
    tests<<<blocks, threads>>>(PHILOXStates);

    cudaDeviceSynchronize();
    cudaFree(PHILOXStates);
}

/******************************************************************************/
void stat(uint64_t *cc, double *times, uint64_t warps, uint32_t runs, limb_t ops, int clockrate, int big) {
    int i;
    uint64_t a_cc = 0;
    double a_time = 0;
    double avg_cc;

    for (i = 0; i < warps; i++) {
        a_cc += cc[i];
        a_time += times[i];
    }

    if (ops) { 
        avg_cc = (double)a_cc/runs/warps/ops;
        a_time = a_time/runs/warps/ops; 
    }
    else {
        avg_cc = (double)a_cc/runs/warps;
        a_time = a_time/runs/warps;
    }

    if (big)
        printf("%.6f clock cycles (%f ms)\n", avg_cc, a_time / 1000);
    else
        printf("%.6f clock cycles (%f us)\n", avg_cc, a_time);
}

void arith_bench() {
    uint32_t blocks = 72;
    uint32_t threads = 64;
    // A result by warp
    uint32_t size = (blocks * threads) >> 5;
    uint64_t seed;

    printf("GPU: %d | Blocks: %d | Threads: %d\n", GPUID, blocks, threads);
    cudaSetDevice(GPUID);
    
    uint64_t *d_cc, *h_cc;
    double *d_times, *h_times;
    curandStatePhilox4_32_10_t *PHILOXStates;
    proj_t *out;

    cudaDeviceProp prop;

    srand(time(0));
    seed = rand();

    cudaMalloc((void **)&PHILOXStates, (blocks * threads) * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc((void**)&d_times, size * sizeof(double));
    cudaMalloc((void**)&d_cc, size * sizeof(uint64_t));
    
    cudaMalloc((void**)&out, size * sizeof(proj_t));
    
    h_cc = (uint64_t *)malloc(size * sizeof(uint64_t));
    h_times = (double *)malloc(size * sizeof(double));

    cudaGetDeviceProperties(&prop, GPUID);
    printf("GPU: %d | ClockRate: %d\n", GPUID, prop.clockRate);

    setup_rand<<<blocks, threads>>>(PHILOXStates, seed);
    
    cudaDeviceSynchronize();
    
    printf("Benchmarks:\n");
    // -------------------------------------------------------------------------
    printf("fp_add: ");

    bench_fp_add<<<blocks, threads>>>(d_cc, d_times, out, PHILOXStates, prop.clockRate); 
    
    cudaMemcpy(h_times, d_times, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cc, d_cc, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    stat(h_cc, h_times, size, RUNS_BENCH, 3, prop.clockRate, 0);
    // -------------------------------------------------------------------------
    printf("fp_mul: ");

    bench_fp_mul<<<blocks, threads>>>(d_cc, d_times, out, PHILOXStates, prop.clockRate); 
    
    cudaMemcpy(h_times, d_times, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cc, d_cc, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    stat(h_cc, h_times, size, RUNS_BENCH, 3, prop.clockRate, 0);

    // -------------------------------------------------------------------------
    printf("fp2_add: ");

    bench_fp2_add<<<blocks, threads>>>(d_cc, d_times, PHILOXStates, prop.clockRate); 
    
    cudaMemcpy(h_times, d_times, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cc, d_cc, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    stat(h_cc, h_times, size, RUNS_BENCH, 3, prop.clockRate, 0);
    // -------------------------------------------------------------------------
    printf("fp2_mul: ");

    bench_fp2_mul<<<blocks, threads>>>(d_cc, d_times, PHILOXStates, prop.clockRate); 
    
    cudaMemcpy(h_times, d_times, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cc, d_cc, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    stat(h_cc, h_times, size, RUNS_BENCH, 3, prop.clockRate, 0);
    // -------------------------------------------------------------------------
    printf("fp2_sqr: ");

    bench_fp2_sqr<<<blocks, threads>>>(d_cc, d_times, PHILOXStates, prop.clockRate); 
    
    cudaMemcpy(h_times, d_times, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cc, d_cc, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    stat(h_cc, h_times, size, RUNS_BENCH, 2, prop.clockRate, 0);
    // -------------------------------------------------------------------------
    printf("xadd: ");

    bench_xadd<<<blocks, threads>>>(d_cc, d_times, out, PHILOXStates, prop.clockRate); 
    
    cudaMemcpy(h_times, d_times, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cc, d_cc, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    stat(h_cc, h_times, size, RUNS_BENCH, 4, prop.clockRate, 0);
    // -------------------------------------------------------------------------

    printf("xdbl: ");
    
    bench_xdbl<<<blocks, threads>>>(d_cc, d_times, out, PHILOXStates, prop.clockRate);

    cudaMemcpy(h_times, d_times, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cc, d_cc, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    stat(h_cc, h_times, size, RUNS_BENCH, 3, prop.clockRate, 0);
    // -------------------------------------------------------------------------

    printf("xtpl: ");
    
    bench_xtpl<<<blocks, threads>>>(d_cc, d_times, out, PHILOXStates, prop.clockRate);

    cudaMemcpy(h_times, d_times, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cc, d_cc, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    stat(h_cc, h_times, size, RUNS_BENCH, 3, prop.clockRate, 0);
    // -------------------------------------------------------------------------

    printf("prf: ");
    uint64_t pc_size = (uint64_t)1 << PC_DEPTH;

    fp2_t *h_P0, *h_Q0, *h_PQ0, *h_E0, *h_Z0;
    fp2_t *h_P1, *h_Q1, *h_PQ1, *h_E1, *h_Z1;

    h_P0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    h_Q0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    h_PQ0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    h_E0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    h_Z0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    h_P1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    h_Q1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    h_PQ1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    h_E1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    h_Z1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));

    fp2_t *P0, *Q0, *PQ0, *E0, *Z0, *P1, *Q1, *PQ1, *E1, *Z1;

    cudaMalloc((void **)&P0, pc_size * sizeof(fp2_t));
    cudaMalloc((void **)&Q0, pc_size * sizeof(fp2_t));
    cudaMalloc((void **)&PQ0, pc_size * sizeof(fp2_t));
    cudaMalloc((void **)&E0, pc_size * sizeof(fp2_t));
    cudaMalloc((void **)&Z0, pc_size * sizeof(fp2_t));
    cudaMalloc((void **)&P1, pc_size * sizeof(fp2_t));
    cudaMalloc((void **)&Q1, pc_size * sizeof(fp2_t));
    cudaMalloc((void **)&PQ1, pc_size * sizeof(fp2_t));
    cudaMalloc((void **)&E1, pc_size * sizeof(fp2_t));
    cudaMalloc((void **)&Z1, pc_size * sizeof(fp2_t));

    cudaMemcpy(P0, h_P0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Q0, h_P0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(PQ0, h_P0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(E0, h_P0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Z0, h_P0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(P1, h_P0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Q1, h_P0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(PQ1, h_P0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(E1, h_P0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Z1, h_P0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice);
    
    void *args1[] = {(void *)&P0, (void *)&Q0, (void *)&PQ0, (void *)&E0, (void *)&Z0,
        (void *)&P1, (void *)&Q1, (void *)&PQ1, (void *)&E1, (void *)&Z1, 
        (void *)&PHILOXStates, (void *)&seed, (void *)&d_cc, (void *)&d_times, (void *)&prop.clockRate};
    dim3 dimGrid1(blocks, 1, 1), dimBlock1(threads, 1, 1);
    size_t smem = sizeof(limb_t) * (1 << 7);

    gpuchk(cudaLaunchCooperativeKernel((void *)bench_prf, dimGrid1, dimBlock1, args1, smem, NULL));

    cudaMemcpy(h_times, d_times, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cc, d_cc, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    stat(h_cc, h_times, size, RUNS_BENCH_S, 0, prop.clockRate, 1);
    
    cudaFree(P0);
    cudaFree(Q0);
    cudaFree(PQ0);
    cudaFree(E0);
    cudaFree(Z0);
    cudaFree(P1);
    cudaFree(Q1);
    cudaFree(PQ1);
    cudaFree(E1);
    cudaFree(Z1);
    
    free(h_P0);
    free(h_Q0);
    free(h_PQ0);
    free(h_E0);
    free(h_Z0);
    free(h_P1);
    free(h_Q1);
    free(h_PQ1);
    free(h_E1);
    free(h_Z1);
  
  // -------------------------------------------------------------------------

//    printf("ladder3pt: ");
//    
//    bench_ladder3pt<<<blocks, threads>>>(d_cc, d_times, out, PHILOXStates, prop.clockRate);
//
//    cudaMemcpy(h_times, d_times, size * sizeof(double), cudaMemcpyDeviceToHost);
//    cudaMemcpy(h_cc, d_cc, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
//    
//    stat(h_cc, h_times, size, RUNS_BENCH_S, 5, prop.clockRate, 1);
  // -------------------------------------------------------------------------

//    printf("xisog_2e_2nd: ");
//    // cudaLaunchCooperativeKernel
//    bench_xisog_2e_2nd<<<blocks, threads>>>(d_cc, d_times, out, PHILOXStates, prop.clockRate);
//    
//    cudaMemcpy(h_times, d_times, size * sizeof(double), cudaMemcpyDeviceToHost);
//    cudaMemcpy(h_cc, d_cc, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
//
//    stat(h_cc, h_times, size, RUNS_BENCH_S, 0, prop.clockRate, 1);
  // -------------------------------------------------------------------------
  
    cudaFree(PHILOXStates);
    cudaFree(d_cc);
    cudaFree(d_times);
    free(h_cc);
    free(h_times);
    cudaFree(out);
}

int main(int argc, char **argv) {
    printf("+++++++++++++++++++++++++++++++++\n");
    printf("\t\t{RADIX}\t\t: %d\n", RADIX);
    printf("log\u2082(p)\t\t{NBITS_FIELD}\t: %d\n", NBITS_FIELD);
    printf("log\u2082(p) / 8\t{NBYTES_FIELD}\t: %d\n", NBYTES_FIELD);
    printf("log\u2082(p) / %d\t{NWORDS_FIELD}\t: %d\n", RADIX, NWORDS_FIELD);
    printf("+++++++++++++++++++++++++++++++++\n");

    assert(argc > 1 && argv[1][0] == '-');

    /* Tests */
    if (argv[1][1] == 't') {
        arith_test();
    }

    /* Benchs */
    if (argv[1][1] == 'b') {
        //benchs_xisog();
        arith_bench();
    }

    return 0;
}
