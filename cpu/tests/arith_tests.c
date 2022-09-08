#define RUNS 100

#if defined(_shortw_)
#include "arith_tests_shortw.c"
#elif defined(_mont_)
#include "arith_tests_mont.c"
#endif

int main()
{
    printf("+++++++++++++++++++++++++++++++++\n");
    printf("\t\t{RADIX}\t\t: %d\n", RADIX);
    printf("log\u2082(p)\t\t{NBITS_FIELD}\t: %d\n", NBITS_FIELD);
    printf("log\u2082(p) / 8\t{NBYTES_FIELD}\t: %d\n", NBYTES_FIELD);
    printf("log\u2082(p) / %d\t{NWORDS_FIELD}\t: %d\n", RADIX, NWORDS_FIELD);
    printf("+++++++++++++++++++++++++++++++++\n");

    printf("\n");
    test_fp();
    printf("\n");
    test_fp2();
    printf("\n");
    test_curve();
    printf("\n");
    test_isogenies();
    printf("\n");
    test_prf(2);
    test_prf(3);

    return 0;
}