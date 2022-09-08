#include <omp.h>
#include <unistd.h>
#include <string.h>

#if defined(_shortw_)
#error "Instance generator only supported for montgomery"
#else

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

    ctx_mitm_t context = {0};
    int side, pq, xz, ri, word;
    digit_t e, e0, e1;
    fp_t random1, random2;
    uint64_t K0, K1;
    proj_t R0, R1;

    e0 = EXPONENT2 - (EXPONENT2 >> 1) - 1;
    e1 = (EXPONENT2 >> 1) - 1;
    e = e0 + e1;

    // Build random keys
    fp_random(random1);
    fp_random(random2);
    K0 = random1[0] & ((1 << e0) - 1);
    K1 = random2[0] & ((1 << e1) - 1);

    // Get 'random' instance
    init_context_mitm(&context, 2);
    random_instance_rigged(&context, e, K0, K1);

    // Compute kernel points
    ladder3pt(R0, K0, context.BASIS[0][0], context.BASIS[0][1], context.BASIS[0][2], context.A2[0], e0);
    ladder3pt(R1, K1, context.BASIS[1][0], context.BASIS[1][1], context.BASIS[1][2], context.A2[1], e1);
    fp2_inv(R0[1]);
    fp2_mul(R0[0], R0[0], R0[1]);
    fp2_inv(R1[1]);
    fp2_mul(R1[0], R1[0], R1[1]);

    printf("/* +++ Fixed instance for p%d +++ */\n\n", NBITS_FIELD);
    printf("proj_t E[2] = ");
    printf("{");
    for(side = 0; side < 2; side++){
        printf("{");
        for(xz = 0; xz < 2; xz ++){
            printf("{");
            for(ri = 0; ri < 2; ri++){
                printf("{");
                for(word = 0; word < NWORDS_FIELD; word++)
                {
                    printf("0x%lx,", context.E[side][xz][ri][word]);
                }
                printf("\b},");
            }
            printf("\b},");
        }
        printf("\b},");
    }
    printf("\b};\n\n");

    printf("proj_t notE[2] = ");
    printf("{");
    for(side = 0; side < 2; side++){
        printf("{");
        for(xz = 0; xz < 2; xz ++){
            printf("{");
            for(ri = 0; ri < 2; ri++){
                printf("{");
                for(word = 0; word < NWORDS_FIELD; word++)
                {
                    printf("0x%lx,", context.notE[side][xz][ri][word]);
                }
                printf("\b},");
            }
            printf("\b},");
        }
        printf("\b},");
    }
    printf("\b};\n\n");

    printf("proj_t BASIS[2][3] = ");
    printf("{");
    for(side = 0; side < 2; side++){
        printf("{");
        for(pq = 0; pq < 3; pq ++)
        {
            printf("{");
            for(xz = 0; xz < 2; xz ++){
                printf("{");
                for(ri = 0; ri < 2; ri++){
                    printf("{");
                    for(word = 0; word < NWORDS_FIELD; word++)
                    {
                        printf("0x%lx,", context.BASIS[side][pq][xz][ri][word]);
                    }
                    printf("\b},");
                }
                printf("\b},");
            }
            printf("\b},");
        }
        printf("\b},");
    }
    printf("\b};\n");

    printf("\n\n//##############################\n\
    //#THE SECRET COLLISION IS:\n\
    \t//# c0 = 0;\n\
    \t//# k0 = %lu\n", K0);
    printf("\
    \t//# c1 = 1;\n\
    \t//# k1 = %lu\n", K1);
    printf("\
    //# THE KERNEL POINTS ARE\n\
    \t//#K0_x = ");fp2_printf(R0[0]);
    printf("\
    \t//#K1_x = ");fp2_printf(R1[0]);
    printf("//# Dont tell anyone!\n");
    printf("//##############################\n");

/*
    point_t golden[2];
    golden[0].c = 0;
    golden[1].c = 1;
    golden[0].k = K0;
    golden[1].k = K1;
    printf("// +++++++++++++++++++++++++++++++++\n");
    printf("// Framework:\n");
    printf("e_2 := %d;\ne_3 := %d;\nf := %d;\n", EXPONENT2, EXPONENT3, COFACTOR);
    printf("p := 2^e_2 * 3^e_3 * f - 1;\n");
    printf("fp2<i> := GF(p^2);\n");
    printf("P<t>   := PolynomialRing(fp2);\n");
    printf("assert(i^2 eq -1);\n// +++++++++++++++++++++++++++++++++\n");
    collision_printf(golden, context);
*/

    return 0;
}

#endif