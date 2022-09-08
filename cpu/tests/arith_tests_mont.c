void test_fp(void)
{
    ticks clockcycles_0, clockcycles_1;
    double clockcycles = 0;
    fp_t a, b, c, d, e, g = {0};
    int i;
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing fp_mul()", 100 * i / (int)RUNS);
        fflush(stdout);

        printf("\r\x1b[K");
        fp_set_one(a);
        fp_random(b);
        clockcycles_0 = getticks();
        fp_mul(c, a, b);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
        assert(fp_compare(b, c) == 0);
        fp_mul(c, b, a);
        assert(fp_compare(b, c) == 0);
    }
    printf("fp_mul()\t%f Mcc\n", clockcycles/(1000000*RUNS));
    clockcycles = 0;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing fp_sqr()", 100 * i / (int)RUNS);
        fflush(stdout);

        printf("\r\x1b[K");
        fp_random(a);
        fp_random(b);
        fp_mul(c, b, b);
        clockcycles_0 = getticks();
        fp_sqr(a, b);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
        assert(fp_compare(a, c) == 0);
    }
    printf("fp_sqr()\t%f Mcc\n", clockcycles/(1000000*RUNS));
    clockcycles = 0;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing fp_add()", 100 * i / (int)RUNS);
        fflush(stdout);

        printf("\r\x1b[K");
        fp_random(a);
        fp_random(b);
        fp_random(c);
        fp_add(d, a, c);
        clockcycles_0 = getticks();
        fp_add(e, b, c);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
        assert(fp_compare(d, e) != 0);
        assert(fp_iszero(g));
        fp_add(a, d, g);
        assert(fp_compare(a, d) == 0);
        fp_add(b, g, e);
        assert(fp_compare(b, e) == 0);
    }
    printf("fp_add()\t%f Mcc\n", clockcycles/(1000000*RUNS));
    clockcycles = 0;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing fp_sub()", 100 * i / (int)RUNS);
        fflush(stdout);

        printf("\r\x1b[K");
        fp_random(a);
        fp_random(b);
        fp_random(c);
        fp_sub(d, a, c);
        clockcycles_0 = getticks();
        fp_sub(e, b, c);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
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
    printf("fp_sub()\t%f Mcc\n", clockcycles/(1000000*RUNS));
    clockcycles = 0;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing fp_inv()", 100 * i / (int)RUNS);
        fflush(stdout);

        printf("\r\x1b[K");
        fp_set_one(a);
        fp_random(b);
        fp_copy(c, b);
        clockcycles_0 = getticks();
        fp_inv(c);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
        fp_mul(c, c, b);
        assert(fp_compare(c, a) == 0);
    }
    printf("fp_inv()\t%f Mcc\n", clockcycles/(1000000*RUNS));
    clockcycles = 0;
}

void test_fp2(void)
{
    ticks clockcycles_0, clockcycles_1;
    double clockcycles = 0;
    fp2_t a, b, c, d, e, g = {0};

    int i;
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing fp2_mul()", 100 * i / (int)RUNS);
        fflush(stdout);

        printf("\r\x1b[K");
        fp2_set_one(a);
        fp2_random(b);
        clockcycles_0 = getticks();
        fp2_mul(c, a, b);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
        assert(fp2_compare(b, c) == 0);
        fp2_mul(c, b, a);
        assert(fp2_compare(b, c) == 0);
    }
    printf("fp2_mul()\t%f Mcc\n", clockcycles/(1000000*RUNS));
    clockcycles = 0;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing fp2_sqr()", 100 * i / (int)RUNS);
        fflush(stdout);

        printf("\r\x1b[K");
        fp2_random(a);
        fp2_random(b);
        fp2_mul(c, b, b);
        clockcycles_0 = getticks();
        fp2_sqr(a, b);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
        assert(fp2_compare(a, c) == 0);
    }
    printf("fp2_sqr()\t%f Mcc\n", clockcycles/(1000000*RUNS));
    clockcycles = 0;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing fp2_add()", 100 * i / (int)RUNS);
        fflush(stdout);

        printf("\r\x1b[K");
        fp2_random(a);
        fp2_random(b);
        fp2_random(c);
        clockcycles_0 = getticks();
        fp2_add(d, a, c);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
        fp2_add(e, b, c);
        assert(fp2_compare(d, e) != 0);
        assert(fp2_iszero(g));
        fp2_add(a, d, g);
        assert(fp2_compare(a, d) == 0);
        fp2_add(b, g, e);
        assert(fp2_compare(b, e) == 0);
    }
    printf("fp2_add()\t%f Mcc\n", clockcycles/(1000000*RUNS));
    clockcycles = 0;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing fp2_sub()", 100 * i / (int)RUNS);
        fflush(stdout);

        printf("\r\x1b[K");
        fp2_random(a);
        fp2_random(b);
        fp2_random(c);
        fp2_sub(d, a, c);
        clockcycles_0 = getticks();
        fp2_sub(e, b, c);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
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
    printf("fp2_sub()\t%f Mcc\n", clockcycles/(1000000*RUNS));
    clockcycles = 0;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing fp2_inv()", 100 * i / (int)RUNS);
        fflush(stdout);

        printf("\r\x1b[K");
        fp2_set_one(a);
        fp2_random(b);
        fp2_copy(c, b);
        clockcycles_0 = getticks();
        fp2_inv(c);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
        fp2_mul(c, c, b);
        assert(fp2_compare(c, a) == 0);
    }
    printf("fp2_inv()\t%f Mcc\n", clockcycles/(1000000*RUNS));
    clockcycles = 0;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing fp2_issquare()", 100 * i / (int)RUNS);
        fflush(stdout);
        printf("\r\x1b[K");
        fp2_random(a);
        fp2_sqr(b, a);
        clockcycles_0 = getticks();
        assert(fp2_issquare(c, b));
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
        fp2_sqr(d, c);
        assert(fp2_compare(d, b) == 0);
    }
    printf("fp2_issqr()\t%f Mcc\n", clockcycles/(1000000*RUNS));
    clockcycles = 0;
}

void test_curve()
{
    ticks clockcycles_0, clockcycles_1;
    double clockcycles_add = 0, clockcycles_dbl = 0, clockcycles_ladder = 0, clockcycles_llader = 0;
    int i;
    proj_t A, P, Q, PQ, t1, t2;
    set_initial_curve(A);
    fp_t k;
    uint64_t m;

    fp_random(k);
    m = k[0];

    for(i = 0; i < RUNS; i++)
    {
        printf("[%3d%%] Testing x-only arithmetic", 100 * i / (int)RUNS);
        fflush(stdout);
        printf("\r\x1b[K");

        init_basis(P, Q, PQ, A);
        
        // Check P + Q + (P-Q) = 2P
        xadd(t1, P, Q, PQ);
        xdbl(t2, Q, A);
        clockcycles_0 = getticks();
        xadd(t1, t1, PQ, t2);
        clockcycles_1 = getticks();
        clockcycles_add += elapsed(clockcycles_1, clockcycles_0);
        clockcycles_0 = getticks();
        xdbl(t2, P, A);
        clockcycles_1 = getticks();
        clockcycles_dbl += elapsed(clockcycles_1, clockcycles_0);
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

        clockcycles_0 = getticks();
        ladder3pt(P, m, P, Q, PQ, A, (EXPONENT2-2)>>1);
        clockcycles_1 = getticks();
        clockcycles_ladder += elapsed(clockcycles_1, clockcycles_0);

        clockcycles_0 = getticks();
        ladder3pt_long(P, k, P, Q, PQ, A);
        clockcycles_1 = getticks();
        clockcycles_llader += elapsed(clockcycles_1, clockcycles_0);
    }
    printf("xadd()\t\t%f Mcc\n", clockcycles_add/(1000000*RUNS));
    printf("xdbl()\t\t%f Mcc\n", clockcycles_dbl/(1000000*RUNS));
    printf("ladder3pt()\t%f Mcc\n", clockcycles_ladder/(1000000*RUNS));
    printf("ladder3pt_long\t%f Mcc\n", clockcycles_llader/(1000000*RUNS));
}

void test_isogenies()
{
    ticks clockcycles_0, clockcycles_1;
    static double clockcycles[4] = {0};
    int i;
    proj_t E0, EA2, EA3, EB2, EB3, EAB2, EAB3, EBA2, TMP;
    set_initial_curve(E0);

    proj_t PA, QA, PQA, PB, QB, PQB, RA, RB;
    proj_t A[3], B[3];
    fp_t a = {0}, b = {0};
    fp2_t jAB, jBA;

    init_basis(PA, QA, PQA, E0);
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

    for(i = 0; i < RUNS; i++) {
        printf("[%3d%%] Testing x-only isogenies", 100 * i / (int) RUNS);
        fflush(stdout);
        printf("\r\x1b[K");

        proj_copy(A[0], PB);
        proj_copy(A[1], QB);
        proj_copy(A[2], PQB);
        proj_copy(B[0], PA);
        proj_copy(B[1], QA);
        proj_copy(B[2], PQA);

        // keygen: degree-(2^e2) isogeny
        random_mod_A(a);
        ladder3pt_long(RA, a, PA, QA, PQA, E0);
        clockcycles_0 = getticks();
        xisog_2e_1st_(EA2, A, RA, E0, STRATEGY2_, EXPONENT2);
        clockcycles_1 = getticks();
        clockcycles[0] += elapsed(clockcycles_1, clockcycles_0);
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
        random_mod_B(b);
        ladder3pt_long(RB, b, PB, QB, PQB, E0);
        clockcycles_0 = getticks();
        xisog_3e_1st(EB3, B, RB, E0, STRATEGY3, EXPONENT3);
        clockcycles_1 = getticks();
        clockcycles[2] += elapsed(clockcycles_1, clockcycles_0);
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
        clockcycles_0 = getticks();
        xisog_2e_2nd_(EBA2, RA, EB2, STRATEGY2_, EXPONENT2);
        clockcycles_1 = getticks();
        clockcycles[1] += elapsed(clockcycles_1, clockcycles_0);
        j_invariant(jBA, EBA2);
        // derive: degree-(3^e3) isogeny
        ladder3pt_long(RB, b, A[0], A[1], A[2], EA2);
        clockcycles_0 = getticks();
        xisog_3e_2nd(EAB3, RB, EA3, STRATEGY3, EXPONENT3);
        clockcycles_1 = getticks();
        clockcycles[3] += elapsed(clockcycles_1, clockcycles_0);
        change_curvemodel(EAB2, EAB3);
        j_invariant(jAB, EAB2);
        
        assert(fp2_compare(jAB, jBA) == 0);
    }
    printf("xisog2_1st()\t%f Mcc\n", clockcycles[0]/(1000000*RUNS));
    printf("xisog2_2nd()\t%f Mcc\n", clockcycles[1]/(1000000*RUNS));
    printf("xisog3_1st()\t%f Mcc\n", clockcycles[2]/(1000000*RUNS));
    printf("xisog3_2nd()\t%f Mcc\n", clockcycles[3]/(1000000*RUNS));
}

void test_prf(int deg)
{
    ticks clockcycles_0, clockcycles_1;
    double clockcycles = 0;

    int i;
    point_t  x = {0};
    ctx_mitm_t  context;
    fp2_t j;
    
    // Test 2-isogenies
    init_context_mitm(&context, deg);
    (&context)->NONCE = 7;
    set_initial_curve((&context)->E[0]);
    random_instance((&context), deg);

    fp2_random(j);
    _gn_(&x, j, (&context)->NONCE, deg, (&context)->bound[1], (&context)->ebits_max);
    for(i = 0; i < RUNS; i++) {
        printf("[%3d%%] Testing _fn_(), %d-isogeny case", 100 * i / (int) RUNS, deg);
        fflush(stdout);
        printf("\r\x1b[K");
        clockcycles_0 = getticks();
        _fn_(&x, j, x, context);
        clockcycles_1 = getticks();
        clockcycles += elapsed(clockcycles_1, clockcycles_0);
    }
    printf("_fn_(%d)\t\t%f Mcc\n", deg, clockcycles/(1000000*RUNS));
}
