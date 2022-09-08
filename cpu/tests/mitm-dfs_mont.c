#include <omp.h>
#include <unistd.h>
#include <string.h>

void usage_mitm_dfs(void)
{
    printf("usage:\n"
           "\tHelp text when the program is executed without arguments or with an invalid argument configuration.\n"
           "\tTo view the help text run with option -h. Otherwise, the program accepts two options: -w, and -c.\n"
           "\t-s: option label comes from side, either [Alice/Alicia] or [Bob/Beto]\n"
           "\t-c: option label comes from cores. Integer i determines (d + 1)dⁱ cores\n"
           "\t-f: stop after i operations\n");
}

int main (int argc, char **argv)
{
  	// DO NOT DELETE
	struct timeval tv1;
	struct timeval tv2;
	unsigned long long int time, time1, time2;
  	gettimeofday(&tv1,NULL);

    char data[101];
    char entity[101];
    int opt;
	digit_t nr_operations = -1;

    memset(data, 0, 101);
    memset(entity, 0, 101);
    while ((opt = getopt(argc, argv, "hc:s:f:")) != -1) {
        switch (opt) {
            case 'h':
                usage_mitm_dfs();
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
            case 'c': // can accept 0 or 1 parameters
                if (optarg[0] == '-') { //not an optarg of ours...
                    optind--;
                    printf("option: -c, no argument (another option follows)\n");
                    return -1;
                }
                strncpy(data, optarg, 100);
                //printf("option: -c, argument: %s\n", data);
                break;
			case 'f': // accepts 1 parameter
                if (optarg[0] == '-') { //not an optarg of ours...
                    optind--;
                    printf("option: -f, no argument (another option follows)\n");
                    return -1;
                }
    			nr_operations = (digit_t)strtol(optarg, NULL, 10);
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
                usage_mitm_dfs();
                return -1;
        }
    }
    digit_t deg;

    if ((strcmp(entity, "Alice") == 0) || (strcmp(entity, "Alicia") == 0))
        deg = 2;
    else if ((strcmp(entity, "Bob") == 0) || (strcmp(entity, "Beto") == 0))
        deg = 3;
    else
    {
        printf("option: -s, no valid argument %s.\n", entity);
        usage_mitm_dfs();
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
    uint64_t j;
    uint64_t table_size;
    digit_t logcores = (digit_t)strtol(data, NULL, 10);
    ctx_mitm_t context = {0};
    init_context_mitm(&context, deg);
    random_instance(&context, deg);
    (&context)->c = 0;

    // DFS setup
    ctx_dfs_t ctx = {0};
    init_context_dfs(&ctx, (&context)->deg);
    (&ctx)->depth = logcores;
    (&ctx)->c = (&context)->c;
    
    // When using 2-isogenies in Montgomery the backtracking trick MUST be used or attack will fail
    if (deg == 2)
    {
        undo_2isog(&context);
        (&ctx)->e[0] -= 1;
        (&ctx)->e[1] -= 1;
    }
    
    // THIS OVERRIDE UNDOES THE FROBENIUS IN CASE IT WAS DEFINED.
    // REMOVE THIS ONCE FROBENIUS IS IMPLEMENTED IN MITM.
    #ifdef FROBENIUS
    (&context)->ebits[0] += 1;
    #endif

    runtime_per_thread(&context, logcores);
    (&ctx)->cores = (&context)->cores;
    table_size = (&context)->runtime[(&ctx)->c];
    (&ctx)->runtime[0] = (&context)->runtime[0];
    (&ctx)->runtime[1] = (&context)->runtime[1];
    
    mitm_t **LEAVES;
    proj_t **ROOTS;
    uint64_t *PATHS;
    LEAVES = (mitm_t **)malloc( (&ctx)->cores * sizeof(mitm_t *) );
    ROOTS = (proj_t **)malloc( (&ctx)->cores * sizeof(proj_t *) );
    PATHS = (uint64_t *)malloc( (&ctx)->cores * sizeof(uint64_t) );
    for(i = 0; i < (&ctx)->cores; i++)
    {
        LEAVES[i] = (mitm_t *) malloc(table_size * sizeof(mitm_t));
        ROOTS[i] = (proj_t *) malloc(4 * sizeof(proj_t));
    }

    (&ctx)->get_roots(ROOTS, PATHS, (&context)->BASIS[(&context)->c], (&context)->E[(&context)->c], ctx);

    printf("//\t\t\tcores:\t%d\n", (&ctx)->cores);
    printf("//\t\t\tside:\t%s\n", entity);
    printf("//\t\t\tdegree:\t%d^%d\n", (int)deg, (int)((&context)->e[0] + (&context)->e[1]));
    printf("//\t\t\tMemory Used:%f GB\n\n",(float)(table_size * sizeof(mitm_t)+4 * sizeof(proj_t)+sizeof(fp_t)+2*sizeof(proj_t *))*(&ctx)->cores/pow(1024,3));
    
    ticks clockcycles_0, clockcycles_1;
    double clockcycles = 0;

    omp_set_num_threads((&ctx)->cores);
#pragma omp parallel shared(LEAVES) private(i,j,clockcycles_0,clockcycles_1) reduction(+:clockcycles)
    {
        j = 0;
        i = omp_get_thread_num();
        clockcycles_0 = getticks();
        (&ctx)->left_mitm_side_dfs(LEAVES[i], &j, PATHS[i], ROOTS[i], ctx, (&ctx)->depth + 1);
        assert(table_size == j);
        hashtable_quicksort(LEAVES[i], 0, table_size - 1);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
    }

    // Collision is expressed as a pair of two points
    (&ctx)->c ^= 1;
    (&ctx)->get_roots(ROOTS, PATHS, (&context)->BASIS[(&context)->c ^ 1], (&context)->E[(&context)->c ^ 1], ctx);
    (&ctx)->c ^= 1;
    uint8_t finished = 0;
    point_t collision[2];
#pragma omp parallel shared(collision,finished) private(i,clockcycles_0,clockcycles_1) reduction(+:clockcycles)
    {
        i = omp_get_thread_num();
        clockcycles_0 = getticks();
        (&ctx)->right_mitm_side_dfs(collision, &finished, LEAVES, PATHS[i], ROOTS[i], ctx, (&ctx)->depth + 1);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
    }

    // Printing solution
	if (nr_operations == -1)
    	assert(finished == 1);
    
	(&ctx)->from_dfs_to_collision(collision, context);
    collision_printf(collision, context);
    printf("//\tclock cycles:\t%2.3f\t(in log2 base)\n", log((float)clockcycles)/log(2));

    for(i = 0; i < (&ctx)->cores; i++)
    {
        free(LEAVES[i]);
        free(ROOTS[i]);
    }

    free(LEAVES);
    free(ROOTS);
    free(PATHS);
	
	// DO NOT DELETE
	gettimeofday(&tv2, NULL);
				time1 =(tv1.tv_sec) * 1000000 + tv1.tv_usec;
				time2 = (tv2.tv_sec) * 1000000 + tv2.tv_usec;
				time = time2 - time1;
	printf("Time: %llu * 10^-6 seconds\n", time);
    return 0;
}

