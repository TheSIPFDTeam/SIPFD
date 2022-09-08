#include <omp.h>
#include <unistd.h>
#include <string.h>

void usage_vowgcs(void)
{
    printf("usage:\n"
           "\tHelp text when the program is executed without arguments or with an invalid argument configuration.\n"
           "\tTo view the help text run with option -h. Otherwise, the program accepts two options: -w, and -c.\n"
           "\t-s: option label comes from side, either [Alice/Alicia] or [Bob/Beto]\n"
           "\t-c: option label comes from cores. Integer i determines i cores\n"
           "\t-w: option label comes from omega. Integer i determines 2ⁱ cells of memory\n");
}

int main (int argc, char **argv)
{
    char data[101];
    char entity[101];
    char cells[101] = "0";
    int opt;

    memset(data, 0, 101);
    memset(entity, 0, 101);
    while ((opt = getopt(argc, argv, "hc:s:w:")) != -1) {
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

#if defined(_mont_)
    // When using 2-isogenies in Montgomery the backtracking trick MUST be used or attack will fail
    if (deg == 2)
        undo_2isog(&context);
#endif

    // vOW GCS setup
    ctx_vow_t ctx = {0};
    (&ctx)->omegabits = (digit_t)strtol(cells, NULL, 10);   // omega = 2^omegabits
    (&ctx)->omega = (digit_t)pow(2, (&ctx)->omegabits);     // Limit: memory cells
    (&ctx)->omega_minus_one = (&ctx)->omega - 1;            // 2^{omega_bits} - 1
    (&ctx)->beta = 10;
    (&ctx)->betaXomega = (&ctx)->beta * (&ctx)->omega;      // (beta x omega) distinguished points per each PRF
    double M = (double)(pow(2, (&context)->ebits[0])+pow(2, (&context)->ebits[1]));
    (&ctx)->theta = 2.25 * sqrt((double)(&ctx)->omega / M); //2.25 x (omega / 2N) portion of distinguished point
    (&ctx)->maxtrail = (digit_t)ceil(10.0 / (&ctx)->theta); // 10 / theta
    double Rbits = log(1.0 / (&ctx)->theta)/log(2);
    (&ctx)->distinguished = (digit_t)ceil(pow(2, 32-Rbits));// Distinguished property <---- larger values than 32?
    (&ctx)->maxprf=1;                                       // Maximum number of PRF

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
    (&ctx)->heuristic = 1;
    (&ctx)->maxprf = 1;


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
    printf("//\t\t\t10 / θ:\t\t%d\t\t(maximum trail length)\n\n", (int)(&ctx)->maxtrail);
    printf("//\t\t\tMemory Used:\t%f\t(GB)\n\n",(float)(&ctx)->omega * sizeof(vowgcs_t)/pow(1024,3));

    // Memory allocation corresponding with the hashtable
    vowgcs_t *hashtable = (vowgcs_t *)malloc((&ctx)->omega * sizeof(vowgcs_t));
    uint8_t finished = 0;
    point_t golden[2];
    double clockcycles = 0;
    int prf_counter = 0;
    do
    {
        prf_counter += 1;
        // At th beginning the hash-table must be empty; that is, the trail length  of each
        // element in the hashtable must be equal to 0.
        for(i = 0; i < (&ctx)->omega; i++)
            (&hashtable[i])->length = 0;

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
        printf("//[#%d]\t_fn_() with nonce 0x", prf_counter);TOHEX((&context)->NONCE);fflush(stdout);printf("\r\x1b[K");
        omp_set_num_threads((&ctx)->cores);
#pragma omp parallel shared(hashtable,golden,finished,ctx) private(i) reduction(+:clockcycles)
        {
            i = omp_get_thread_num();
            clockcycles += vowgcs(golden, &finished, hashtable, context, ctx, i);
        }
        if(((&ctx)->heuristic == 2) && (prf_counter > (&ctx)->maxprf))
            break;
    } while(finished == 0);

    // Checking if solution was found
    if ((&ctx)->heuristic != 2)
        assert(finished == 1);

    // Printing heuristics
    digit_t runtime_collision = 0, runtime_different = 0, runtime = 0;
    for(i = 0; i < (&ctx)->cores; i++)
    {
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

    free(hashtable);
    free((&ctx)->runtime);
    free((&ctx)->runtime_different);
    free((&ctx)->runtime_collision);
    free((&ctx)->index);
    free((&ctx)->collisions);
    for( i = 0; i < (&ctx)->cores; i++)
        free((&ctx)->address[i]);
    free((&ctx)->address);
    return 0;
}
