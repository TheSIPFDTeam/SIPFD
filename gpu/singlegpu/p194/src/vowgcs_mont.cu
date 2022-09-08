#include <cooperative_groups.h>
#include <stdio.h>
#include <omp.h>
#include "api.h"
#include "vowgcs_setup.h"
#include "../external/precomp.h"

namespace cg = cooperative_groups;

#define gpuchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

extern "C" __global__ void sipfd_vow(point_t *golden, uint8_t *finished, uint8_t *hashtable,
        limb_t seed, limb_t seq, ctx_t *context, int *prf_counter,
        fp2_t *P0, fp2_t *Q0, fp2_t *PQ0, fp2_t *E0, fp2_t *Z0,
        fp2_t *P1, fp2_t *Q1, fp2_t *PQ1, fp2_t *E1, fp2_t *Z1,
        curandStatePhilox4_32_10_t *state, curandStatePhilox4_32_10_t *gstate) {

    cg::grid_group g = cg::this_grid();
    int counter = 0;
    uint64_t i;
    int t = g.thread_rank();

    __shared__ limb_t expo[2];
    __shared__ limb_t ebits[2];
    __shared__ limb_t strategy_reduced[2][EXP0];

    if (threadIdx.x == 0) {
        expo[0] = EXP0;
        expo[1] = EXP1;
        ebits[0] = EXP20_BITS;
        ebits[1] = EXP21_BITS;

        for (counter = 0; counter < PC_STRATEGY_SIZE_0; counter++)
            strategy_reduced[0][counter] = STRATEGY2_PC_0[counter];

        for (counter = 0; counter < PC_STRATEGY_SIZE_1; counter++)
            strategy_reduced[1][counter] = STRATEGY2_PC_1[counter];
    }

    init_rand(state, seed, 0);

    if (t == 0) {
        // get the NONCE based on seed and sequence 
        init_rand(gstate, seed + seq, seq);
    }
    
    g.sync();
    counter = 0;
    while(!*finished) {
        counter += 1;
        
        // At th beginning the hash-table must be empty; that is, the trail length  of each
        // element in the hashtable must be equal to 0.
        for(i = t + 1; i < OMEGA + 1; i+=g.size())
            hashtable[i*TRIPLETBYTES - 1] = 0;

        if (t == 0) {
            context->NONCE = curand(gstate);
            context->NONCE ^= ((uint64_t)curand(gstate) << 32);
            //printf("//[#%d]\t_fn_() with nonce 0x%lX\n", counter, context->NONCE);
            *prf_counter = counter;
        }
#if NFNC
        // catch If counter is less than NFUNCTIONS when we are calculating timings 
        if (counter == NFUNCTIONS)
            break;
#endif
        g.sync();

        vowgcs(golden, finished, hashtable, strategy_reduced, P0, Q0, PQ0, E0, Z0, 
                P1, Q1, PQ1, E1, Z1, context, expo, ebits, state);

        // In order to each thread can read the variable finished at the same time
        g.sync();
    }
}

int main (int argc, char **argv) {
    printf("// +++++++++++++++++++++++++++++++++\n");
    printf("// Framework:\n");
    printf("e_2 := %d;\ne_3 := %d;\nf := %d;\n", EXPONENT2, EXPONENT3, COFACTOR);
    printf("p := 2^e_2 * 3^e_3 * f - 1;\n");
    printf("fp2<i> := GF(p^2);\n");
    printf("P<t>   := PolynomialRing(fp2);\n");
    printf("assert(i^2 eq -1);\n// +++++++++++++++++++++++++++++++++\n");

    /* ----------------------------- GPU ----------------------------- */
    printf("// GPU ID: %d\n", GPUID);
    cudaSetDevice(GPUID);

    ctx_t *context, *h_context;
    curandStatePhilox4_32_10_t *global_PHILOXStates;
    curandStatePhilox4_32_10_t *PHILOXStates;

    srand(time(0));
    
    cudaMalloc((void **)&PHILOXStates, (NUM_BLOCKS * NUM_THREADS) * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc((void **)&global_PHILOXStates, sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc((void **)&context, sizeof(ctx_t));
    cudaMallocHost((void **)&h_context, sizeof(ctx_t));

    printf("//\t\tcores:\t%d\n", (int)NUM_BLOCKS * NUM_THREADS);
    printf("//\t\tside:\t%s\n", "Alice/Alicia");
    printf("//\t\tdegree:\t%d^%d\n", DEG, EXP0 + EXP1);
    printf("// vOW GCS setup:\n");
    printf("//\t\tω:\t\t2^%d\t\t(cells of memory)\n", (int)OMEGABITS);
    printf("//\t\tβ:\t\t%d\n", (int)BETA);
    printf("//\t\tβω / cores:\t%d\t\t(distinguished points per each PRF)\n", (int)BETAXOMEGA);
    printf("//\t\tθ:\t\t%f\t(portion of distinguished points)\n", THETA);
    printf("//\t\t10 / θ:\t\t%d\t\t(maximum trail length)\n\n", (int)MAXTRAIL);
    printf("//\t\ts:\t\t%d\t\t(bytes per triplet)\n", TRIPLETBYTES);
    printf("//\t\tMemory Used:\t%f\t(GB)\n\n",(float)OMEGA * TRIPLETBYTES / pow(1024,3));
    printf("//\tApproximating θ = 1/(R*2^n) with n=%d and 1/R to the nearest 1/2^%d-th: θ = %f \n\n", 
        N, RBITS, 1/(pow(2, N)*pow(2,RBITS)/DISTINGUISHED));
    printf("//\tGenerating precomputation table of size %f GB ...\n", 32*NWORDS_FIELD*RADIX*pow(DEG, PC_DEPTH-32));

    uint8_t finished;
    point_t golden[2];
    limb_t seed, seq;
 
    fp2_t *P0, *Q0, *PQ0, *E0, *Z0;
    fp2_t *P1, *Q1, *PQ1, *E1, *Z1;

    fp2_t *d_P0, *d_Q0, *d_PQ0, *d_E0, *d_Z0;
    fp2_t *d_P1, *d_Q1, *d_PQ1, *d_E1, *d_Z1;
    
    uint8_t *d_finished;
    point_t *d_golden;
    uint8_t *d_hashtable;

    int prf_counter, *d_counter;
    uint64_t pc_size = (uint64_t)1 << PC_DEPTH;

    cudaStream_t stream;
    dim3 dimGrid(NUM_BLOCKS, 1, 1), dimBlock(NUM_THREADS, 1, 1);
    size_t smem = sizeof(limb_t) * (1 << 7);

    float time = 0;
    cudaEvent_t start, stop;

    P0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    Q0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    PQ0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    E0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    Z0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    P1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    Q1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    PQ1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    E1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    Z1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));

    if (P0 == NULL || Q0 == NULL || PQ0 == NULL || E0 == NULL || Z0 == NULL || P1 == NULL || Q1 == NULL || PQ1 == NULL || E1 == NULL || Z1 == NULL) {
        printf("Error: malloc P0 ... Z1\n");
    }

    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuchk(cudaMalloc((void **)&d_P0, pc_size * sizeof(fp2_t)));
    gpuchk(cudaMalloc((void **)&d_Q0, pc_size * sizeof(fp2_t)));
    gpuchk(cudaMalloc((void **)&d_PQ0, pc_size * sizeof(fp2_t)));
    gpuchk(cudaMalloc((void **)&d_E0, pc_size * sizeof(fp2_t)));
    gpuchk(cudaMalloc((void **)&d_Z0, pc_size * sizeof(fp2_t)));
    gpuchk(cudaMalloc((void **)&d_P1, pc_size * sizeof(fp2_t)));
    gpuchk(cudaMalloc((void **)&d_Q1, pc_size * sizeof(fp2_t)));
    gpuchk(cudaMalloc((void **)&d_PQ1, pc_size * sizeof(fp2_t)));
    gpuchk(cudaMalloc((void **)&d_E1, pc_size * sizeof(fp2_t)));
    gpuchk(cudaMalloc((void **)&d_Z1, pc_size * sizeof(fp2_t)));

    randombytes(&seed, sizeof(limb_t));
    randombytes(&seq, sizeof(limb_t));

    // Check that there are enough bits in the scalar
    if (OMEGABITS + RBITS + N > EXP20_BITS)
    {
        printf("Error: there are not enoght bits in the scalar\n");
        return -1;    
    }
    
    // Auxiliary variables
    fp2_t *root_P0, *root_Q0, *root_PQ0, *root_E0, *root_Z0;
    fp2_t *root_P1, *root_Q1, *root_PQ1, *root_E1, *root_Z1;
   
    root_P0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    root_Q0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    root_PQ0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    root_E0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    root_Z0 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    root_P1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    root_Q1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    root_PQ1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    root_E1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));
    root_Z1 = (fp2_t *)malloc(pc_size * sizeof(fp2_t));

    if (root_P0 == NULL || root_Q0 == NULL || root_PQ0 == NULL || root_E0 == NULL || root_Z0 == NULL || 
            root_P1 == NULL || root_Q1 == NULL || root_PQ1 == NULL || root_E1 == NULL || root_Z1 == NULL) {
        printf("Error: malloc root_P0 ... root_Z1\n");
    }

    // Computing log(cores)
    int logcores = 0;
    while ( (1 << (logcores + 1)) <= CORES )
        logcores++;

    if (logcores > PC_DEPTH)
    {
        printf("Error: Number of threads exceed precomputation depth\n");
        return -1;
    }

    // Compute the first part of the trees sequentialy
    precompute(root_P0, root_Q0, root_PQ0, root_E0, root_Z0, 0, h_BASIS[0], h_E[0], EXP0, 0, logcores);
    precompute(root_P1, root_Q1, root_PQ1, root_E1, root_Z1, 0, h_BASIS[1], h_E[1], EXP1, 0, logcores);

    // Compute the remaining parts in parallel
    uint64_t path;
    int k, i;
    proj_t basis[3], curve;
    omp_set_num_threads(CORES);
    #pragma omp parallel shared(root_P0, root_Q0, root_PQ0, root_E0, root_Z0, root_P1, root_Q1, root_PQ1, root_E1, root_Z1, logcores) \
        private(i, k, path, basis, curve)
    {
        k = omp_get_thread_num();
        path = 0;
        for(i = 0; i < logcores; i++)
        {
            path <<= 1;
            path += (k >> i) & 1;
        }

        fp2_copy(basis[0][0], root_P0[k]);
        fp2_copy(basis[1][0], root_Q0[k]);
        fp2_copy(basis[2][0], root_PQ0[k]);
        fp2_copy(basis[0][1], root_Z0[k]);
        fp2_copy(basis[1][1], root_Z0[k]);
        fp2_copy(basis[2][1], root_Z0[k]);
        fp2_copy(curve[0], root_E0[k]);
        fp2_copy(curve[1], root_Z0[k]);

        precompute(P0, Q0, PQ0, E0, Z0, path, basis, curve, EXP0, logcores, PC_DEPTH);

        fp2_copy(basis[0][0], root_P1[k]);
        fp2_copy(basis[1][0], root_Q1[k]);
        fp2_copy(basis[2][0], root_PQ1[k]);
        fp2_copy(basis[0][1], root_Z1[k]);
        fp2_copy(basis[1][1], root_Z1[k]);
        fp2_copy(basis[2][1], root_Z1[k]);
        fp2_copy(curve[0], root_E1[k]);
        fp2_copy(curve[1], root_Z1[k]);

        precompute(P1, Q1, PQ1, E1, Z1, path, basis, curve, EXP1, logcores, PC_DEPTH);
    }

    free(root_P0);
    free(root_Q0);
    free(root_PQ0);
    free(root_E0);
    free(root_Z0);
    free(root_P1);
    free(root_Q1);
    free(root_PQ1);
    free(root_E1);
    free(root_Z1);

    printf("//\tPrecomputation complete\n\n");

    cudaMemcpyAsync(d_P0, P0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Q0, Q0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_PQ0,PQ0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_E0, E0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Z0, Z0, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_P1, P1, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Q1, Q1, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_PQ1,PQ1, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_E1, E1, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_Z1, Z1, pc_size * sizeof(fp2_t), cudaMemcpyHostToDevice, stream);

    // vOW
    cudaMalloc((void **)&d_finished, sizeof(uint8_t));
    cudaMalloc((void **)&d_golden, 2 * sizeof(point_t));
    cudaMalloc((void **)&d_counter, sizeof(int));
    gpuchk(cudaMalloc((void **)&d_hashtable, ((uint64_t)OMEGA * TRIPLETBYTES) * sizeof(uint8_t)));

    cudaMemset(d_finished, 0, sizeof(uint8_t));

    void *args[] = {(void *)&d_golden, (void *)&d_finished, (void *)&d_hashtable,
        (void *)&seed, (void *)&seq, (void *)&context, (void *)&d_counter,
        (void *)&d_P0, (void *)&d_Q0, (void *)&d_PQ0, (void *)&d_E0, (void *)&d_Z0,
        (void *)&d_P1, (void *)&d_Q1, (void *)&d_PQ1, (void *)&d_E1, (void *)&d_Z1,
        (void *)&PHILOXStates, (void *)&global_PHILOXStates};

    cudaEventRecord(start, 0);

    gpuchk(cudaLaunchCooperativeKernel((void *)sipfd_vow, dimGrid, dimBlock, args, smem, stream));
    
    cudaEventRecord(stop, 0);
    cudaStreamSynchronize(stream);
    cudaEventSynchronize(stop);
    time = 0;
    cudaEventElapsedTime(&time, start, stop);
    
    cudaMemcpy(&prf_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&finished, d_finished, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_context, context, sizeof(ctx_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(golden, d_golden, 2 * sizeof(point_t), cudaMemcpyDeviceToHost);

#ifdef NFNC
    printf("//prf: %d\n", prf_counter);
    printf("//function: %f hours\n", time/NFUNCTIONS/3600000);
#else
    printf("//prf: %d\n", prf_counter);
    printf("//vOW: %f hours\n", time/3600000);
    // Printing solution
    collision_printf<<<1, 1>>>(d_golden, context);
    cudaDeviceSynchronize();
#endif

    free(P0);
    free(Q0);
    free(PQ0);
    free(E0);
    free(Z0);
    free(P1);
    free(Q1);
    free(PQ1);
    free(E1);
    free(Z1);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_hashtable);
    cudaFree(d_counter);
    cudaFree(PHILOXStates);
    cudaFree(global_PHILOXStates);
    cudaFree(context);
    cudaFree(h_context);
    cudaFree(d_finished);
    cudaFree(d_golden);
    cudaFree(d_P0);
    cudaFree(d_Q0);
    cudaFree(d_PQ0);
    cudaFree(d_E0);
    cudaFree(d_Z0);
    cudaFree(d_P1);
    cudaFree(d_Q1);
    cudaFree(d_PQ1);
    cudaFree(d_E1);
    cudaFree(d_Z1);

    cudaStreamDestroy(stream);
    
    return 0;
}
