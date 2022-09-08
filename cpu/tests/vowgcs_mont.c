#include <omp.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

void usage_vowgcs(void)
{
    printf("usage:\n"
           "\tHelp text when the program is executed without arguments or with an invalid argument configuration.\n"
           "\tTo view the help text run with option -h. Otherwise, the program accepts two options: -w, and -c.\n"
           "\t-s: option label comes from side, either [Alice/Alicia] or [Bob/Beto]\n"
           "\t-c: option label comes from cores. Integer i determines i cores\n"
           "\t-w: option label comes from omega. Integer i determines 2ⁱ cells of memory\n"
           "\t-b: beta\n"
           "\t-f: stop after i functions\n"
           "\t-p: toggle on precomputation. To change precomputation depth, edit the --pc flags in src/setupall.sh and run it.\n");
}

int main (int argc, char **argv)
{
  	struct timeval tv1;
	struct timeval tv2;
	unsigned long long int time, time1, time2;
  	gettimeofday(&tv1,NULL);


    char data[101];
    char entity[101];
    char cells[101] = "0";
    int opt;
	digit_t nr_functions = -1;
    int pc_depth = 0;
	int beta = 10;


    memset(data, 0, 101);
    memset(entity, 0, 101);
    while ((opt = getopt(argc, argv, "hc:s:w:f:pb:")) != -1) {
        switch (opt) {
            case 'h':
                usage_vowgcs();
                return 0;
            case 's': // can accept 0 or 1 parameters
                if (optarg[0] == '-') { //not an optarg of ours...
                    optind--;
                    printf("option: -s, no argument (another option follows)\n");
                    return -1;
                }
                strncpy(entity, optarg, 100);
                //printf("option: -s, argument: %s\n", entity);
                break;
            case 'w': // can accept 0 or 1 parameters
                if (optarg[0] == '-') { //not an optarg of ours...
                    optind--;
                    printf("option: -w, no argument (another option follows)\n");
                    return -1;
                }
                strncpy(cells, optarg, 100);
                //printf("option: -w, argument: %s\n", entity);
                break;
            case 'c': // can accept 0 or 1 parameters
                if (optarg[0] == '-') { //not an optarg of ours...
                    optind--;
                    printf("option: -c, no argument (another option follows)\n");
                    return -1;
                }
                strncpy(data, optarg, 100);
                //printf("option: -c, argument: %s\n", data);
                break;
            case 'f': // can accept 0 or 1 parameters
                if (optarg[0] == '-') { //not an optarg of ours...
                    optind--;
                    printf("option: -, no argument (another option follows)\n");
                    return -1;
                }

                //strncpy(data, optarg, 100);
    			nr_functions = (digit_t)strtol(optarg, NULL, 10);
                //printf("option: -f, argument: %s\n", data);
            case 'b': // can accept 0 or 1 parameters
                if (optarg[0] == '-') { //not an optarg of ours...
                    optind--;
                    printf("option: -, no argument (another option follows)\n");
                    return -1;
                }

                //strncpy(data, optarg, 100);
    			beta = (digit_t)strtol(optarg, NULL, 10);
                //printf("option: -f, argument: %s\n", data);
            case 'p': // can accept 0 or 1 parameters
                pc_depth = PC_DEPTH;
                //printf("option: -c, argument: %s\n", data);
                break;
            case ':': //this happens if we got an option which expects an arg without any optarg.
                if(optopt == 's' || optopt == 'c') {//lets allow a '-d' without its optarg
                    printf("option: -%c, no argument\n", optopt);
                    break;
                }
                //otherwise fall through to the default handler
            default: //covers ':' '?' for missing value,  '-h' for help, etc.
                printf("on error you get: opt=%c, optopt=%c opterr=%d\n", opt, optopt, opterr);
                usage_vowgcs();
                return -1;
        }
    }

    digit_t deg;

    if ((strcmp(entity, "Alice") == 0) || (strcmp(entity, "Alicia") == 0))
        deg = 2;
    else if ((strcmp(entity, "Bob") == 0) || (strcmp(entity, "Beto") == 0))
        deg = 3;
    else if ( (strcmp(entity, "\0") == 0) )
        deg = 2;
    else
    {
        printf("option: -s, no valid argument %s.\n", entity);
        usage_vowgcs();
        return -1;
    }

    printf("// +++++++++++++++++++++++++++++++++\n");
    printf("// Framework:\n");
    //printf("clear;\n");
    printf("e_2 := %d;\ne_3 := %d;\nf := %d;\n", EXPONENT2, EXPONENT3, COFACTOR);
    printf("p := 2^e_2 * 3^e_3 * f - 1;\n");
    printf("fp2<i> := GF(p^2);\n");
    printf("P<t>   := PolynomialRing(fp2);\n");
    printf("assert(i^2 eq -1);\n// +++++++++++++++++++++++++++++++++\n");

    // MITM setup
    int i;
    digit_t cores = (digit_t)strtol(data, NULL, 10);

    ctx_mitm_t context = {0};
    init_context_mitm(&context, deg);
    random_instance(&context, deg);
    (&context)->pc_depth = pc_depth;

    // When using 2-isogenies in Montgomery the backtracking trick MUST be used or attack will fail
    if (deg == 2)
        undo_2isog(&context);

    // vOW GCS setup
    ctx_vow_t ctx = {0};
    (&ctx)->omegabits = (digit_t)strtol(cells, NULL, 10);   // omega = 2^omegabits
    (&ctx)->omega = (digit_t)pow(2, (&ctx)->omegabits);     // Limit: memory cells
    (&ctx)->omega_minus_one = (&ctx)->omega - 1;            // 2^{omega_bits} - 1
    (&ctx)->beta = beta;
    (&ctx)->betaXomega = (&ctx)->beta * (&ctx)->omega;      // (beta x omega) distinguished points per each PRF
    double M = (double)(pow(2, (&context)->ebits[0])+pow(2, (&context)->ebits[1]));
    (&ctx)->theta = 2.25 * sqrt((double)(&ctx)->omega / M); //2.25 x (omega / 2N) portion of distinguished point
    (&ctx)->maxtrail = (digit_t)ceil(10.0 / (&ctx)->theta); // 10 / theta
    (&ctx)->trail_bits = ceil(log((&ctx)->maxtrail)/log(2));
    // We write 1/theta = R*2^n for 0 <= R < 2, and approximate 1/R to the nearest 1/(2^Rbtis)-th
    (&ctx)->n = floor(log(1.0 / (&ctx)->theta)/log(2));
    (&ctx)->Rbits = 4;
    (&ctx)->distinguished = floor((&ctx)->theta*pow(2, (&ctx)->n)*pow(2, (&ctx)->Rbits));
    (&ctx)->triplet_bytes = (2*((&context)->ebits_max+1) - (&ctx)->omegabits - (&ctx)->n + (&ctx)->trail_bits)/8 + 1;

    // Each thread has the task of reaching (BETA * OMEGA)/NUMBER_OF_THREADS
    (&ctx)->betaXomega  = (digit_t)ceil((double)(&ctx)->betaXomega /(double)cores);

    // Concerning number of cores (metrics: number of collisions)
    (&ctx)->cores = cores;
    (&ctx)->address = (linkedlist_t **)malloc(cores * sizeof(linkedlist_t *));      // Each core has its own list of pointers
    // Upper bound of different collisions
    for( i = 0; i < (&ctx)->cores; i++)
        (&ctx)->address[i] = (linkedlist_t *)malloc(2 * (&ctx)->omega * sizeof(linkedlist_t));
    (&ctx)->collisions = (linkedlist_t **)malloc(cores * sizeof(linkedlist_t *));      // Each core has its own list of pointers
    (&ctx)->index = (digit_t *)malloc(cores * sizeof(digit_t));                     // Current number of different collisions
    (&ctx)->runtime_collision = (digit_t *)malloc(cores * sizeof(digit_t));         // Number of all collisions
    (&ctx)->runtime_different = (digit_t *)malloc(cores * sizeof(digit_t));         // Number of all different collisions
    (&ctx)->runtime = (digit_t *)malloc(cores * sizeof(digit_t));                   // Number of function evaluations _fn_()
    (&ctx)->heuristic = nr_functions == -1 ? 1 : 2;
    (&ctx)->maxprf = nr_functions-1;


    if (cores==0 || (&ctx)->omegabits==0){
        printf("options -c and -w must be specified\n");
        usage_vowgcs();
        return -1;
    }

    printf("//\t\t\tcores:\t%d\n", (int)(&ctx)->cores);
    printf("//\t\t\tside:\t%s\n", entity);
    printf("//\t\t\tdegree:\t%d^%d\n", (int)deg, (int)((&context)->e[0] + (&context)->e[1]));
    printf("// vOW GCS setup:\n");
    printf("//\t\t\tω:\t\t2^%d\t\t(cells of memory)\n", (int)(&ctx)->omegabits);
    printf("//\t\t\tβ:\t\t%d\n", (int)(&ctx)->beta);
    printf("//\t\t\tβω / cores:\t%d\t\t(distinguished points per each PRF)\n", (int)(&ctx)->betaXomega);
    printf("//\t\t\tθ:\t\t%f\t(portion of distinguished points)\n", (&ctx)->theta);
    printf("//\t\t\t10 / θ:\t\t%d\t\t(maximum trail length)\n", (int)(&ctx)->maxtrail);
    printf("//\t\t\ts:\t\t%d\t\t(bytes per triplet)\n\n", (int)(&ctx)->triplet_bytes);
    //printf("//\t\t\tMemory Used:\t%f\t(GB)\n\n",(float)(&ctx)->omega * sizeof(vowgcs_t)/pow(1024,3));
    printf("//\t\t\tMemory Used:\t%f\t(GB)\n\n",(float)(&ctx)->omega * (&ctx)->triplet_bytes/pow(1024,3));
    printf("//\tApproximating θ = 1/(R*2^n) with n=%d and 1/R to the nearest 1/2^%d-th: θ = %f \n\n", (int)(&ctx)->n, (int)(&ctx)->Rbits, 1/(pow(2, (int)(&ctx)->n)*pow(2,(int)(&ctx)->Rbits)/(int)(&ctx)->distinguished));

    // Check that there are enough bits in the scalar
    assert((&ctx)->omegabits+(&ctx)->Rbits+(&ctx)->n <= ( ((&context)->ebits[0]+(&context)->ebits[1]) >> 1 ) );
	#ifdef STRUCT_PRTL
	int level = compute_optimal_level((&ctx)->omega);
	#endif


    // Precomputation Setup
    if(pc_depth > 0)
    {
        if (strcmp(entity, "Bob") == 0)
        {
            printf("Error: precomputation only implemented for Alice side\n");
            exit(-1);
        }
        printf("//\tGenerating precomputation table of size %f GB ...\n", 20*NWORDS_FIELD*RADIX*pow(deg, pc_depth-33));
        (&context)->pc_table = (fp2_t ***)malloc(2*sizeof(fp2_t**));
        fp2_t ***ROOTS = (fp2_t ***)malloc(2*sizeof(fp2_t**));
        for(int i=0; i < 2; i++)
        {
            (&context)->pc_table[i] = (fp2_t **)malloc(5*sizeof(fp2_t*));
            ROOTS[i] = (fp2_t **)malloc(5*sizeof(fp2_t*));
            for( int j=0; j < 5; j++)
            {
                (&context)->pc_table[i][j] = (fp2_t *)malloc((1 << pc_depth)*sizeof(fp2_t));
                ROOTS[i][j] = (fp2_t *)malloc((1 << pc_depth)*sizeof(fp2_t));
            }
        }

        // Computing log(cores)
        int logcores = 0;
        while ( (1 << (logcores+1)) <= cores )
            logcores++;
        assert(logcores < pc_depth); //Number of cores should not exceed precomputation depth

        // Compute the first part of the trees sequentialy
        precompute(ROOTS[0], 0, (&context)->BASIS[0], (&context)->E[0], (&context)->e[0], 0, logcores);
        precompute(ROOTS[1], 0, (&context)->BASIS[1], (&context)->E[1], (&context)->e[1], 0, logcores);

        // Compute the remaining parts in parallel
        uint64_t path;
        int k;
        proj_t basis[3], curve;
        omp_set_num_threads(cores);
        #pragma omp parallel shared(logcores, ROOTS) private(i, k, path, basis, curve)
        {
            k = omp_get_thread_num();
            path = 0;
            for(i = 0; i < logcores; i++)
            {
                path <<= 1;
                path += (k >> i) & 1;
            }

            fp2_copy(basis[0][0], ROOTS[0][0][k]);
            fp2_copy(basis[1][0], ROOTS[0][1][k]);
            fp2_copy(basis[2][0], ROOTS[0][2][k]);
            fp2_copy(basis[0][1], ROOTS[0][4][k]);
            fp2_copy(basis[1][1], ROOTS[0][4][k]);
            fp2_copy(basis[2][1], ROOTS[0][4][k]);
            fp2_copy(curve[0], ROOTS[0][3][k]);
            fp2_copy(curve[1], ROOTS[0][4][k]);
            precompute((&context)->pc_table[0], path, basis, curve, (&context)->e[0], logcores, pc_depth);

            fp2_copy(basis[0][0], ROOTS[1][0][k]);
            fp2_copy(basis[1][0], ROOTS[1][1][k]);
            fp2_copy(basis[2][0], ROOTS[1][2][k]);
            fp2_copy(basis[0][1], ROOTS[1][4][k]);
            fp2_copy(basis[1][1], ROOTS[1][4][k]);
            fp2_copy(basis[2][1], ROOTS[1][4][k]);
            fp2_copy(curve[0], ROOTS[1][3][k]);
            fp2_copy(curve[1], ROOTS[1][4][k]);
            precompute((&context)->pc_table[1], path, basis, curve, (&context)->e[1], logcores, pc_depth);
        }


        for(int i=0; i < 2; i++)
        {
            for( int j=0; j < 5; j++)
                free(ROOTS[i][j]);
            free(ROOTS[i]);
        }
        free(ROOTS);

        printf("//\tPrecomputation complete\n\n");
    }

#ifdef STRUCT_PRTL
    	// Memory allocation corresponding with the PRTL
    	struct_init_PRTL(level, (&ctx)->n, (&ctx)->omega);
#endif
#ifdef STRUCT_HASH_TABLE

    // Memory allocation corresponding with the hashtable
    uint8_t hashtable[(&ctx)->omega*(&ctx)->triplet_bytes];
#endif

    uint8_t finished = 0;
    point_t golden[2];
    double clockcycles = 0;
    int prf_counter = 0;
    do {
        prf_counter += 1;
#ifdef STRUCT_PRTL
        struct_change_prf();
#endif
#ifdef STRUCT_HASH_TABLE
        // At th beginning the hash-table must be empty; that is, the dirty bit (in last word)
        //   of each element in the hashtable must be equal to 0.
        for(i = 1; i < (&ctx)->omega + 1; i++)
            hashtable[i*(&ctx)->triplet_bytes - 1] = 0;
#endif

        // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        // At the beginning the the linked list of different collisions reached must be empty
        for(i = 0; i < (&ctx)->cores; i++)
        {
            // Each thread has its own list of memory addresses
            (&ctx)->index[i] = 0;           // Memory address is set as the first element of a global list
            (&ctx)->collisions[i] = NULL;   // Linked list of different collisions reached is set to NULL
        }
        // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        randombytes(&(&context)->NONCE, (int)sizeof(digit_t));	// NONCE is randomly sampled
        printf("\r\x1b[K");printf("//[#%d]\t_fn_() with nonce 0x", prf_counter);TOHEX((&context)->NONCE);fflush(stdout);
        omp_set_num_threads((&ctx)->cores);
#ifdef STRUCT_HASH_TABLE
#pragma omp parallel shared(hashtable,golden,finished,ctx,prf_counter) private(i) reduction(+:clockcycles)
        {
            i = omp_get_thread_num();
            clockcycles += vowgcs(golden, &finished, hashtable, context, ctx, i);
        }
#else
#pragma omp parallel shared(golden,finished,ctx,prf_counter) private(i) reduction(+:clockcycles)
        {
            i = omp_get_thread_num();
            clockcycles += vowgcs(golden, &finished, context, ctx, i, prf_counter);
        }
#endif
        if(((&ctx)->heuristic == 2) && (prf_counter > (&ctx)->maxprf))
            break;
    } while(finished == 0);
    printf("\n\n");

    // Checking if solution was found
    if ((&ctx)->heuristic != 2)
        assert(finished == 1);

    // Printing heuristics
    digit_t runtime_collision = 0, runtime_different = 0, runtime = 0;
    for(i = 0; i < (&ctx)->cores; i++) {
        runtime_collision += (&ctx)->runtime_collision[i];
        runtime_different += (&ctx)->runtime_different[i];
        runtime += (&ctx)->runtime[i];
    }
    printf("//\t#(prf):\t\t\t\t\t%d\n", (int)prf_counter);
    printf("//\t#(collisions):\t\t\t%3.02fω\t(average per function)\n", (float)runtime_collision/(float)(&ctx)->omega/(float)prf_counter);
    printf("//\t#(different collisions):\t%3.02fω\t(average per function)\n", (float)runtime_different/(float)(&ctx)->omega/(float)prf_counter);
    printf("//\t#(function evaluations):\t%3.02f\t(in log2 base)\n\n", log((float)runtime)/log(2));
    printf("//\tclock cycles:\t%2.3f\t(in log2 base)\n", log((float)clockcycles)/log(2));

    // Printing solution
    if ((&ctx)->heuristic != 2)
        collision_printf(golden, context);

    free((&ctx)->runtime);
    free((&ctx)->runtime_different);
    free((&ctx)->runtime_collision);
    free((&ctx)->index);
    free((&ctx)->collisions);
    for( i = 0; i < (&ctx)->cores; i++)
        free((&ctx)->address[i]);
    free((&ctx)->address);

#ifdef STRUCT_PRTL
      // Free PRTL structure
    	struct_free_PRTL();
#endif

if(pc_depth > 0)
{
    for(int i=0; i < 2; i++)
    {
        for( int j=0; j < 5; j++)
            free((&context)->pc_table[i][j]);
        free((&context)->pc_table[i]);
    }
    free((&context)->pc_table);
}

	gettimeofday(&tv2, NULL);
				time1=(tv1.tv_sec) * 1000000 + tv1.tv_usec;
				time2 = (tv2.tv_sec) * 1000000 + tv2.tv_usec;
				time = time2 - time1;
	printf("Time: %llu * 10^-6 seconds\n", time);

    return 0;
}
