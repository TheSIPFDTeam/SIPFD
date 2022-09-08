#include "api.h"
#include "fp.h"

__device__ void test_fp(curandStatePhilox4_32_10_t *state)
{
    fp_t a, b, c, d, e, g = {0};
    int i;
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("Testing fp_mul(): ");
    for(i = 0; i < RUNS; i++)
    {
        //fp_set_one(a);
        fp_copy(a, mont_one);
        fp_random(b, state);
        //printf("\na:= 0x%08X%08X%08X\n", a[2],a[1],a[0]);
        //printf("b:= 0x%08X%08X%08X\n", b[2],b[1],b[0]);
        fp_mul(c, a, b);
        //printf("c:= 0x%08X%08X%08X\n", c[2],c[1],c[0]);
        assert(fp_compare(b, c) == 0);
        fp_mul(c, b, a);
        assert(fp_compare(b, c) == 0);
    }
    printf("OK\n");

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("Testing fp_sqr(): ");
    for(i = 0; i < RUNS; i++)
    {
        fp_random(a, state);
        fp_random(b, state);
        fp_mul(c, b, b);
        fp_sqr(a, b);
        assert(fp_compare(a, c) == 0);
    }
    printf("OK\n");

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("Testing fp_add(): ");
    for(i = 0; i < RUNS; i++)
    {
        fp_random(a, state);
        fp_random(b, state);
        fp_random(c, state);
        fp_add(d, a, c);
        fp_add(e, b, c);
        assert(fp_compare(d, e) != 0);
        assert(fp_iszero(g));
        fp_add(a, d, g);
        assert(fp_compare(a, d) == 0);
        fp_add(b, g, e);
        assert(fp_compare(b, e) == 0);
    }
    printf("OK\n");

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("Testing fp_sub(): ");
    for(i = 0; i < RUNS; i++)
    {
        fp_random(a, state);
        fp_random(b, state);
        fp_random(c, state);
        fp_sub(d, a, c);
        fp_sub(e, b, c);
        assert(fp_compare(d, e) != 0);
        assert(fp_iszero(g));
        fp_sub(a, d, g);
        assert(fp_compare(a, d) == 0);
        fp_sub(b, g, e);
        assert(fp_compare(b, e) != 0);
        fp_neg(b, b);
        assert(fp_compare(b, e) == 0);
        fp_sub(a, b, b);
        assert(fp_iszero(a));
        assert(fp_compare(a, g) == 0);
    }
    printf("OK\n");

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("Testing fp_inv(): ");
    for(i = 0; i < RUNS; i++)
    {
        //fp_set_one(a);
        fp_copy(a, mont_one);
        fp_random(b, state);
        fp_copy(c, b);
        fp_inv(c);
        fp_mul(c, c, b);
        assert(fp_compare(c, a) == 0);
    }
    printf("OK\n");
}

__device__ void test_fp2(curandStatePhilox4_32_10_t *state)
{
    fp2_t a, b, c, d, e, g = {0};

    int i;
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("Testing fp2_mul(): ");
    for(i = 0; i < RUNS; i++)
    {
        fp2_set_one(a);
        fp2_random(b, state);
        fp2_mul(c, a, b);
        assert(fp2_compare(b, c) == 0);
        fp2_mul(c, b, a);
        assert(fp2_compare(b, c) == 0);
    }
    printf("OK\n");
    
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("Testing fp2_sqr(): ");
    for(i = 0; i < RUNS; i++)
    {
        fp2_random(a, state);
        fp2_random(b, state);
        fp2_mul(c, b, b);
        fp2_sqr(a, b);
        assert(fp2_compare(a, c) == 0);
    }
    printf("OK\n");

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("Testing fp2_add(): ");
    for(i = 0; i < RUNS; i++)
    {
        fp2_random(a, state);
        fp2_random(b, state);
        fp2_random(c, state);
        fp2_add(d, a, c);
        fp2_add(e, b, c);
        assert(fp2_compare(d, e) != 0);
        assert(fp2_iszero(g));
        fp2_add(a, d, g);
        assert(fp2_compare(a, d) == 0);
        fp2_add(b, g, e);
        assert(fp2_compare(b, e) == 0);
    }
    printf("OK\n");

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("Testing fp2_sub(): ");
    for(i = 0; i < RUNS; i++)
    {
        fp2_random(a, state);
        fp2_random(b, state);
        fp2_random(c, state);
        fp2_sub(d, a, c);
        fp2_sub(e, b, c);
        assert(fp2_compare(d, e) != 0);
        assert(fp2_iszero(g));
        fp2_sub(a, d, g);
        assert(fp2_compare(a, d) == 0);
        fp2_sub(b, g, e);
        assert(fp2_compare(b, e) != 0);
        fp2_neg(b, b);
        assert(fp2_compare(b, e) == 0);
        fp2_sub(a, b, b);
        assert(fp2_iszero(a));
        assert(fp2_compare(a, g) == 0);
    }
    printf("OK\n");

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("Testing fp2_inv(): ");
    for(i = 0; i < RUNS; i++)
    {
        fp2_set_one(a);
        fp2_random(b, state);
        fp2_copy(c, b);
        fp2_inv(c);
        fp2_mul(c, c, b);
        assert(fp2_compare(c, a) == 0);
    }
    printf("OK\n");

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("Testing fp2_issquare(): ");
    for(i = 0; i < RUNS; i++)
    {
        fp2_random(a, state);
        fp2_sqr(b, a);
        assert(fp2_issquare(c, b));
        fp2_sqr(d, c);
        assert(fp2_compare(d, b) == 0);
    }
    printf("OK\n");
}

__device__ void test_curve(curandStatePhilox4_32_10_t *state)
{
    int i;
    proj_t A, P, Q, PQ, t1, t2;
    set_initial_curve(A);

    printf("Testing x-only arithmetic: ");
    for(i = 0; i < RUNS; i++)
    {
        init_basis(P, Q, PQ, A, state);
        // Check P + Q + (P-Q) = 2P
        xadd(t1, P, Q, PQ);
        xdbl(t2, Q, A);
        xadd(t1, t1, PQ, t2);
        xdbl(t2, P, A);
        assert(proj_isequal(t1, t2));
        //Check 2P+P = 3P
        xadd(t1, t1, P, P);
        xtpl(t2, P, A);
        assert(proj_isequal(t1, t2));
        //Check 3P + 2P = 5P
        xdbl(t1, P, A);
        xadd(t1, t2, t1, P);
        xmul(t2, 5, P, A);
        assert(proj_isequal(t1, t2));

	// Check order of P
        assert(!isinfinity(P));xmul(P, COFACTOR, P, A);
        assert(!isinfinity(P));xdble(P, EXPONENT2, P, A);
        assert(!isinfinity(P));xtple(P, EXPONENT3, P, A);
        assert(isinfinity(P));

	// Check order of Q
        assert(!isinfinity(Q));xmul(Q, COFACTOR, Q, A);
        assert(!isinfinity(Q));xdble(Q, EXPONENT2, Q, A);
        assert(!isinfinity(Q));xtple(Q, EXPONENT3, Q, A);
        assert(isinfinity(Q));

	// Check order of P-Q
        assert(!isinfinity(PQ));xmul(PQ, COFACTOR, PQ, A);
        assert(!isinfinity(PQ));xdble(PQ, EXPONENT2, PQ, A);
        assert(!isinfinity(PQ));xtple(PQ, EXPONENT3, PQ, A);
        assert(isinfinity(PQ));
    }
    printf("OK\n");
}

__device__ void test_isogenies(curandStatePhilox4_32_10_t *state)
{
    int i;
    proj_t E0, EA2, EA3, EB2, EB3, EAB2, EAB3, EBA2, TMP;
    set_initial_curve(E0);

    proj_t PA, QA, PQA, PB, QB, PQB, RA, RB;
    proj_t A[3], B[3];
    fp_t a = {0}, b = {0};
    fp2_t jAB, jBA;

    init_basis(PA, QA, PQA, E0, state);
    xmul(PA, COFACTOR, PA, E0);
    xmul(QA, COFACTOR, QA, E0);
    xmul(PQA, COFACTOR, PQA, E0);

    // E0[3^e3] = <PB, QB>
    xdble(PB, EXPONENT2, PA, E0);
    xdble(QB, EXPONENT2, QA, E0);
    xdble(PQB, EXPONENT2, PQA, E0);
    // E0[2^e2] = <PA, QA>
    xtple(PA, EXPONENT3, PA, E0);
    xtple(QA, EXPONENT3, QA, E0);
    xtple(PQA, EXPONENT3, PQA, E0);
    
    // Check that (0,0) is in <QA>
    proj_copy(TMP, QA);
    for(i = 0; i < EXPONENT2-1; i++)
    	xdbl(TMP, TMP, E0);
    assert(fp2_iszero(TMP[0]));

    printf("Testing x-only isogenies: ");
    for(i = 0; i < RUNS; i++) {
        proj_copy(A[0], PB);
        proj_copy(A[1], QB);
        proj_copy(A[2], PQB);
        proj_copy(B[0], PA);
        proj_copy(B[1], QA);
        proj_copy(B[2], PQA);

        // keygen: degree-(2^e2) isogeny
        random_mod_A(a, state);
        ladder3pt_long(RA, a, PA, QA, PQA, E0);
        xisog_2e_1st(EA2, A, RA, E0, STRATEGY2, EXPONENT2);
        change_curvemodel(EA3, EA2);
        assert(!isinfinity(A[0]));
        xtple(TMP, EXPONENT3, A[0], EA3);
        assert(isinfinity(TMP));
        assert(!isinfinity(A[1]));
        xtple(TMP, EXPONENT3, A[1], EA3);
        assert(isinfinity(TMP));
        assert(!isinfinity(A[2]));
        xtple(TMP, EXPONENT3, A[2], EA3);
        assert(isinfinity(TMP));
        // keygen: degree-(3^e3) isogeny
        random_mod_B(b, state);
        ladder3pt_long(RB, b, PB, QB, PQB, E0);
        xisog_3e_1st(EB3, B, RB, E0, STRATEGY3, EXPONENT3);
        change_curvemodel(EB2, EB3);
        assert(!isinfinity(B[0]));
        xdble(TMP, EXPONENT2, B[0], EB2);
        assert(isinfinity(TMP));
        assert(!isinfinity(B[1]));
        xdble(TMP, EXPONENT2, B[1], EB2);
        assert(isinfinity(TMP));
        assert(!isinfinity(B[2]));
        xdble(TMP, EXPONENT2, B[2], EB2);
        assert(isinfinity(TMP));
        // derive: degree-(2^e2) isogeny
        ladder3pt_long(RA, a, B[0], B[1], B[2], EB2);
        xisog_2e_2nd(EBA2, RA, EB2, STRATEGY2, EXPONENT2);
        j_invariant(jBA, EBA2);
        // derive: degree-(3^e3) isogeny
        ladder3pt_long(RB, b, A[0], A[1], A[2], EA2);
        xisog_3e_2nd(EAB3, RB, EA3, STRATEGY3, EXPONENT3);
        change_curvemodel(EAB2, EAB3);
        j_invariant(jAB, EAB2);
        
        assert(fp2_compare(jAB, jBA) == 0);
    }
    printf("OK\n");
}

extern "C" __global__ void tests(curandStatePhilox4_32_10_t *state) {
    test_fp(state);
    test_fp2(state);
    test_curve(state);
    test_isogenies(state);
}
