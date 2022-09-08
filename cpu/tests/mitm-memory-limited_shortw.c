#include <omp.h>
#include <unistd.h>
#include <string.h>

void runtime_per_thread_memory_limited(ctx_mitm_t *context, int cores, int cells)
{
    if(cells < 1 || cells > context->ebits[0])
    {
        printf("Error: Memory out of range; choose 0 < w <= %lu\n",context->ebits[0]);
        exit(-1);
    }
    // Assuming runtime is deg^{e/2-1}... Recall, if e is odd, then the output must be multiplied by deg
    fp_t t0 = {0}, t1 = {0}, t2 = {0}, t3 = {0};
    t0[0] = context->deg;
    to_montgomery(t1, t0);
    fp_inv(t1);     // 1/deg
    fp_set_one(t0); // 1
    int i;
    for (i = 0; i < cores; i++)
        fp_mul(t0, t0, t1); // 1/deg^{i+1}

    // At this point t0 = 1 / (deg ^ cores)
    to_montgomery(t2, context->bound[1]);
    fp_mul(t1, t2, t0); // bound[1] / deg^{cores}
    from_montgomery((context->runtime)[1], t1);

    fp_copy(t2, t0);
    t3[0] = context->deg;
    to_montgomery(t1, t3); // deg
    fp_set_one(t0); // 1
    for (i = 0; i < cells; i++)
        fp_mul(t0, t0, t1); // deg^{i+1}

    // At this point t0 = deg ^ cells
    fp_mul(t1, t2, t0); // deg^w / deg^{cores}
    from_montgomery((context->runtime)[0], t1);

    context->cores = (int)pow((int)context->deg, cores);
}

void usage_mitm_basic(void)
{
    printf("usage:\n"
           "\tHelp text when the program is executed without arguments or with an invalid argument configuration.\n"
           "\tTo view the help text run with option -h. Otherwise, the program accepts two options: -w, and -c.\n"
           "\t-s: option label comes from side, either [Alice/Alicia] or [Bob/Beto]\n"
           "\t-c: option label comes from cores. Integer i determines 2ⁱ cores\n"
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
                usage_mitm_basic();
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
                usage_mitm_basic();
                return -1;
        }
    }
    digit_t deg;

    if ((strcmp(entity, "Alice") == 0) || (strcmp(entity, "Alicia") == 0))
        deg = 2;
    else
    {
        printf("option: -s, no valid argument %s (only works with deg=2 [Alice]).\n", entity);
        usage_mitm_basic();
        return -1;
    }

    if (strcmp(cells, "0") == 0)
    {
        printf("option: -w, no valid argument (must provide w > 0).\n");
        usage_mitm_basic();
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
    int64_t table_size;
    digit_t logcores = (digit_t)strtol(data, NULL, 10);
    digit_t omegabits= (digit_t)strtol(cells, NULL, 10);
    ctx_mitm_t context = {0};
    init_context_mitm(&context, deg);
    random_instance(&context, deg);
    (&context)->c = 0;

    runtime_per_thread_memory_limited(&context, logcores, omegabits);
    table_size = (&context)->runtime[(&context)->c][0] * ((&context)->deg+1);

    mitm_t **LEAVES;
    LEAVES = (mitm_t **)malloc( (&context)->cores * sizeof(mitm_t *) );
    for(i = 0; i < (&context)->cores; i++)
        LEAVES[i] = (mitm_t *)malloc( table_size * sizeof(mitm_t) );

    printf("//\t\t\tcores:\t%d\n", (&context)->cores);
    printf("//\t\t\tside:\t%s\n", entity);
    printf("//\t\t\tdegree:\t%d^%d\n", (int)deg, (int)((&context)->e[0] + (&context)->e[1]));
    printf("//\t\t\tω:\t2^%d\t\t(cells of memory)\n", (int)omegabits);
    printf("//\t\t\tMemory Used:%f GB\n\n",(float)(table_size * sizeof(mitm_t)+sizeof(mitm_t *))*(&context)->cores/pow(1024,3));

    ticks clockcycles_0, clockcycles_1;
    double clockcycles = 0;

    uint8_t finished = 0;
    int repetitions = 1 << ((&context)->e[0] - omegabits);
    point_t collision[2] = {0};
    int j;
    for (j = 0; j < repetitions; j++)
    {
        omp_set_num_threads((&context)->cores);
#pragma omp parallel shared(LEAVES) private(i, clockcycles_0, clockcycles_1) reduction(+:clockcycles)
        {
            i = omp_get_thread_num();
            clockcycles += left_mitm_side_basic(LEAVES[i], context, i + j*(&context)->cores);

            clockcycles_0 = getticks();
            hashtable_quicksort(LEAVES[i], 0, table_size - 1);
            clockcycles_1 = getticks();
            clockcycles += elapsed(clockcycles_1, clockcycles_0);

        }

        // Collision is expressed as a pair of two points
        finished = 0;
        memset(collision, 0, sizeof(point_t));
#pragma omp parallel shared(collision, finished) private(i) reduction(+:clockcycles)
        {
            i = omp_get_thread_num();
            clockcycles += right_mitm_side_basic(collision, &finished, LEAVES, context, i);
        }

        if (finished)
            break;
    }

    // Printing solution
    //collision[1].b = 4;
    assert(finished == 1);
    collision_printf(collision, context);
    printf("//\tclock cycles:\t%2.3f\t(in log2 base)\n", log((float)clockcycles)/log(2));
    for(i = 0; i < (&context)->cores; i++)
        free(LEAVES[i]);

    free(LEAVES);
    return 0;
}
