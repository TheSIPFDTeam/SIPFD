#include <openssl/md5.h>
#include <math.h>
#include "../stats.h"

void rsh(fp_t x)
{
    /* Right shift (>> 1) of a fp_t element */
    int i;
    for (i=0; i < (NWORDS_FIELD-1); i++) {
        x[i] >>= 1;
        x[i]  ^= (x[i+1] & 1)<<(RADIX-1);
    }
    x[NWORDS_FIELD-1] >>= 1;
}

void lsh(fp_t x)
{
    /* Left shift (>> 1) of a fp_t element */
    int i;
    for (i=(NWORDS_FIELD-1); i > 0; i--)
    {
        x[i] <<= 1;
        x[i]  ^= (x[i-1]>>(RADIX-1)) & 1;
    }
    x[0] <<= 1;
}

void strrev(char str1[], int index, int size)
{
    /* Reverse of a string */
    char temp;

    temp = str1[index];
    str1[index] = str1[size - index];
    str1[size - index] = temp;

    if (index == size / 2)
    {
        return;
    }
    strrev(str1, index + 1, size);
}

void _h_(proj_t G, const proj_t P[3], const proj_t A, const point_t g, const digit_t deg, const fp_t bound)
{
    /* --------------------------------------------------------------------------------- *
     * Function h : {0, 1, ..., d} x |[0, d^{e/2-1} - 1]| --> E[deg^e] \ E[deg^{e-1}]
     * --------------------------------------------------------------------------------- */
    assert(((&g)->b & 0x3) <= deg);
    if(((&g)->b & 0x3) == deg)
    {
        proj_t R, S;
        xdbladd(R, S, P[0], P[2], P[1], A);	// R <- [2]P and S <- [2]P - Q

        if(deg == 3)
        {
            xadd(S, P[0], S, P[2], A);		// [3]P - Q
            xadd(R, R, P[0], P[0], A);		// [3]P
        }
        ladder3pt(G, (&g)->k, P[1], R, S, A);	// G <- [k]([deg]P) + Q
    }
    else
    {
        int i;
        fp_t m;
        fp_copy(m, (&g)->k);
        for(i = 1; i <= ((&g)->b & 0x3); i++)
            fp_add(m, m, bound); // [i * deg^(e - 1)] + m

        ladder3pt(G, m, P[0], P[1], P[2], A);	// G <- P + [b * deg^(e - 1) + m]Q
    }
}

void _gn_(point_t *g, const fp2_t jinv, const digit_t NONCE, const digit_t deg, const ctx_mitm_t context)
{
    /* ------------------------------------------------------------------------------------------------- *
     * Function gn : fp2_t --> {0,1} x {0, 1, ..., d} x |[0, d^{e/2-1} ]|
     * ------------------------------------------------------------------------------------------------- */
    uint8_t emask = ((uint8_t)1 << (((&context)->ebits_max  + 2) % 8)) - 1;

    char raw_data[4 * 2 * NBYTES_FIELD + 1], current[2 * NBYTES_FIELD + 1];
    digit_t TYPE = 1;
    raw_data[0] = '\0';
    current[0] = '\0';
    SPRINTF(current, TYPE);
    current[strlen(current)] = '\0';
    strcat(raw_data, current);
    raw_data[strlen(raw_data)] = '\0';

    // Adding the real part of the j-invariant
    current[0] = '\0';
    fp_string(current, jinv[0]);
    strcat(raw_data, current);
    raw_data[strlen(raw_data)] = '\0';

    // Adding the imaginary part of the j-invariant
    current[0] = '\0';
    fp_string(current, jinv[1]);
    strcat(raw_data, current);
    raw_data[strlen(raw_data)] = '\0';

    // Appending the NONCE to raw_data, different NONCE determines PRF
    current[0] = '\0';
    SPRINTF(current, NONCE);
    current[strlen(current)] = '\0';
    strcat(raw_data, current);
    raw_data[strlen(raw_data)] = '\0';

    digit_t local_N = strlen(raw_data);

    // Reduction algorithm: g_n
    digit_t local_counter = 0;
    unsigned char v_md5[MD5_DIGEST_LENGTH];

    fp_t zero = {0};
    fp_copy(g->k, zero);   // In order to avoid TRASH
    g->b = 0;
    do
    {
        // In this part we append the counter to (j-invariant || NONCE)
        raw_data[local_N] = '\0';
        current[0] = '\0';
        SPRINTF(current, local_counter);
        current[strlen(current)] = '\0';
        strcat(raw_data, current);
        raw_data[strlen(raw_data)] = '\0';

        // Now, we compute MD5(1||j-invariant||NONCE||counter)
        MD5((unsigned  char*)raw_data, strlen(raw_data), v_md5);
        local_counter += 1;

        // We save the (e + 2) least significant bit of MD5(j-invariant||NONCE||counter)
        strrev((char *)v_md5, 0, MD5_DIGEST_LENGTH - 1);
        v_md5[NBITS_TO_NBYTES((&context)->ebits_max + 2) - 1] &= emask;
        memcpy(g->k, (digit_t *)v_md5, sizeof(uint8_t) * NBITS_TO_NBYTES((&context)->ebits_max + 2));

        g->b = (g->k)[0] & 0x07;
        rsh(g->k);
        rsh(g->k);
        rsh(g->k);
        // Now, g->k < deg^(e - 1) and g->b := c||b
    } while( ((digit_t)(0x3 & g->b) > deg) || (fp_compare(g->k, (&context)->bound[g->b >> 2]) >= 0) );
}

void _fn_(point_t *y, fp2_t j, const point_t x, const ctx_mitm_t context, const digit_t *S[2])
{
    /* -------------------------------------------------------------------------------------------- *
     *  Pseudo Random Function fn : S --> S, where S = {0,1} x {0, 1, ..., d} x |[0, d^{e/2 - 1} ]|
     * -------------------------------------------------------------------------------------------- */
    uint8_t c = ((&x)->b >> 2) & 0x1;

    // Mapping x into a projective Weierstrass x-coordinate point
    proj_t C, X;
    _h_(X, (&context)->BASIS[c], (&context)->E[c], x, (&context)->deg, (&context)->bound[c]);
    /**/
#if defined(_TEST_)
    proj_t T;
    proj_copy(T, X);
    if ((&context)->deg == 2)
    {
        assert(!isinfinity(T));xdble(T, (&context)->e[c] - 1, T, (&context)->E[c]);
        assert(!isinfinity(T));xdbl(T, T, (&context)->E[c]);assert(isinfinity(T));
    }
    else
    {
        assert(!isinfinity(T));xtple(T, (&context)->e[c] - 1, T, (&context)->E[c]);
        assert(!isinfinity(T));xtpl(T, T, (&context)->E[c]);assert(isinfinity(T));
    }
#endif
    /**/
    // Mapping a projective Weierstrass x-coordinate point into a supersingular elliptic curve
    (*(&context)->xisoge_2nd)(C, X, (&context)->E[c], S[c], (&context)->e[c]);
    /**/
#if defined(_TEST_)
    proj_t P, Q, PQ;
    init_basis(P, Q, PQ, C);

    assert(!isinfinity(P));xmul(P, COFACTOR, P, C);
    assert(!isinfinity(P));xdble(P, EXPONENT2, P, C);
    assert(!isinfinity(P));xtple(P, EXPONENT3, P, C);
    assert(isinfinity(P));

    assert(!isinfinity(Q));xmul(Q, COFACTOR, Q, C);
    assert(!isinfinity(Q));xdble(Q, EXPONENT2, Q, C);
    assert(!isinfinity(Q));xtple(Q, EXPONENT3, Q, C);
    assert(isinfinity(Q));

    assert(!isinfinity(PQ));xmul(PQ, COFACTOR, PQ, C);
    assert(!isinfinity(PQ));xdble(PQ, EXPONENT2, PQ, C);
    assert(!isinfinity(PQ));xtple(PQ, EXPONENT3, PQ, C);
    assert(isinfinity(PQ));
#endif
    /**/

    // Mapping a supersingular elliptic curve into an element of fp2_t (j-invariant)
    j_invariant(j, C);	// The j-invariant is stored in order to be used for a collision detection.

    // Mapping an element of fp2_t (j-invariant) into an element of {0,1} x {0, 1, ..., d} x |[ 0, d^(e / 2 - 1) ]|
    _gn_(y, j, (&context)->NONCE, (&context)->deg, context);
    assert(fp_compare(y->k, (&context)->bound[y->b >> 2]) < 0);
}

void init_context_mitm(ctx_mitm_t *context, digit_t deg)
{
    if (deg == 2)
    {
        context->deg = 2;
        fp_copy(context->size, BOUND2);
        context->strategy = (digit_t *)STRATEGY2;
        context->not_e = EXPONENT3;
        context->e[0] = EXPONENT2 - (EXPONENT2 >> 1);
        context->e[1] = EXPONENT2 >> 1;
        context->ebits[0] = EXPONENT2_BITS - (EXPONENT2_BITS >> 1);
        context->ebits[1] = EXPONENT2_BITS >> 1;
        context->ebits_max = context->ebits[0];
        context->xmul_deg = &xdbl;
        context->xmul_notdeg = &xtpl;
        context->xmule_deg = &xdble;
        context->xmule_notdeg = &xtple;
        context->xeval = &xeval_2;
        context->xisog = &xisog_2;
        context->xisoge_1st = &xisog_2e_1st;
        context->xisoge_2nd = &xisog_2e_2nd;
        fp_copy(context->bound[0], BOUND2_0);
        rsh(context->bound[0]);
        fp_copy(context->bound[1], BOUND2_1);
        rsh(context->bound[1]);
        context->S[0] = (digit_t *)STRATEGY2_0;
        context->S[1] = (digit_t *)STRATEGY2_1;
    }
    else if (deg == 3)
    {
        context->deg = 3;
        fp_copy(context->size, BOUND3);
        context->strategy = (digit_t *)STRATEGY3;
        context->not_e = EXPONENT2;
        context->e[0] = EXPONENT3 - (EXPONENT3 >> 1);
        context->e[1] = EXPONENT3 >> 1;
        context->ebits[0] = EXPONENT3_BITS - (EXPONENT3_BITS >> 1);
        context->ebits[1] = EXPONENT3_BITS >> 1;
        context->ebits_max = context->ebits[0];
        context->xmul_deg = &xtpl;
        context->xmul_notdeg = &xdbl;
        context->xmule_deg = &xtple;
        context->xmule_notdeg = &xdble;
        context->xeval = &xeval_3;
        context->xisog = &xisog_3;
        context->xisoge_1st = &xisog_3e_1st;
        context->xisoge_2nd = &xisog_3e_2nd;
        fp_set_one(context->bound[0]);
        fp_add(context->bound[1], context->bound[0], context->bound[0]);
        fp_add(context->bound[1], context->bound[1], context->bound[0]);
        fp_inv(context->bound[1]);
        fp_mul(context->bound[0], context->bound[1], BOUND3_0);
        fp_mul(context->bound[1], context->bound[1], BOUND3_1);
        context->S[0] = (digit_t *)STRATEGY3_0;
        context->S[1] = (digit_t *)STRATEGY3_1;
    }
    else
        assert((deg == 2) || (deg == 3));
}

void random_instance(ctx_mitm_t *context, digit_t deg)
{
    proj_t E, P, Q, PQ;
    set_initial_curve(E);       // E : y^2 = x^3 + x
    init_basis(P, Q, PQ, E);    // P, Q, and PQ are random points, so we can take as random kernel a multiple of P
    xdble(P, EXPONENT2, P, E);
    xtple(P, EXPONENT3, P, E);
#if defined(_TEST_)
    assert(!isinfinity(P));xmul(Q, COFACTOR, P, E);assert(isinfinity(Q));
#endif
    // f-isogeny (f+1) different choices
    xisog_f(E, P, E);

    // +++ Generating basis
    proj_copy(context->E[0], E);
    init_basis(P, Q, PQ, E);
    xmul(P, COFACTOR, P, E);
    xmul(Q, COFACTOR, Q, E);
    xmul(PQ, COFACTOR, PQ, E);
    // E[3^EXPONENT3] = <PB, QB> and  PQB := PB -QB
    xdble(context->PB, EXPONENT2, P, E);
    xdble(context->QB, EXPONENT2, Q, E);
    xdble(context->PQB, EXPONENT2, PQ, E);
    // E[2^EXPONENT2] = <PA, QA> and  PQA := PA -QA
    xtple(context->PA, EXPONENT3, P, E);
    xtple(context->QA, EXPONENT3, Q, E);
    xtple(context->PQA, EXPONENT3, PQ, E);

#if defined(_TEST_)
    // Checking order 2^EXPONENT2
    assert(!isinfinity(context->PA)); xdble(P, EXPONENT2 - 1, context->PA, E);
    assert(!isinfinity(P)); xdbl(P, P, E); assert(isinfinity(P));
    assert(!isinfinity(context->QA)); xdble(Q, EXPONENT2 - 1, context->QA, E);
    assert(!isinfinity(Q)); xdbl(Q, Q, E); assert(isinfinity(Q));
    assert(!isinfinity(context->PQA)); xdble(PQ, EXPONENT2 - 1, context->PQA, E);
    assert(!isinfinity(PQ)); xdbl(PQ, PQ, E); assert(isinfinity(PQ));
    // Checking order 3^EXPONENT3
    assert(!isinfinity(context->PB)); xtple(P, EXPONENT3 - 1, context->PB, E);
    assert(!isinfinity(P)); xtpl(P, P, E); assert(isinfinity(P));
    assert(!isinfinity(context->QB)); xtple(Q, EXPONENT3 - 1, context->QB, E);
    assert(!isinfinity(Q)); xtpl(Q, Q, E); assert(isinfinity(Q));
    assert(!isinfinity(context->PQB)); xtple(PQ, EXPONENT3 - 1, context->PQB, E);
    assert(!isinfinity(PQ)); xtpl(PQ, PQ, E); assert(isinfinity(PQ));
#endif

    point_t  x = {0};
    fp2_t j;
    fp2_random(j);
    if (deg == 2)
    {
        proj_copy(context->BASIS[0][0], context->PA);
        proj_copy(context->BASIS[0][1], context->QA);
        proj_copy(context->BASIS[0][2], context->PQA);
        assert(context->e[0] + context->e[1] == EXPONENT2);
    }
    else
    {
        proj_copy(context->BASIS[0][0], context->PB);
        proj_copy(context->BASIS[0][1], context->QB);
        proj_copy(context->BASIS[0][2], context->PQB);
        assert(context->e[0] + context->e[1] == EXPONENT3);
    }

    do{
        fp_random((&x)->k);
        (&x)->b = (&x)->k[0] & 3;
        rsh((&x)->k);
        rsh((&x)->k);
        rsh((&x)->k);
        for(int i = 1; i < NWORDS_FIELD; i++)
            (&x)->k[i] = 0;
        do{
            rsh((&x)->k);
        } while( fp_compare((&x)->k, context->size) >= 0 );
    } while( ((digit_t)(0x3 & (&x)->b) > deg) );
    _h_(P, context->BASIS[0], context->E[0], x, deg, context->size);
    (*(context->xisoge_2nd))(E, P, context->E[0], context->strategy, context->e[0] + context->e[1]);

    // +++ Generating basis
    proj_copy(context->E[1], E);
    init_basis(P, Q, PQ, E);
    xmul(P, COFACTOR, P, E);
    xmul(Q, COFACTOR, Q, E);
    xmul(PQ, COFACTOR, PQ, E);
    context->xmule_notdeg(context->BASIS[1][0], context->not_e, P, E);
    context->xmule_notdeg(context->BASIS[1][1], context->not_e, Q, E);
    context->xmule_notdeg(context->BASIS[1][2], context->not_e, PQ, E);

#if defined(_TEST_)
    // Checking order 2^EXPONENT2
    assert(!isinfinity(context->BASIS[1][0])); context->xmule_deg(P, context->e[0] + context->e[1] - 1, context->BASIS[1][0], E);
    assert(!isinfinity(P)); context->xmul_deg(P, P, E); assert(isinfinity(P));
    assert(!isinfinity(context->BASIS[1][1])); context->xmule_deg(Q, context->e[0] + context->e[1] - 1, context->BASIS[1][1], E);
    assert(!isinfinity(Q)); context->xmul_deg(Q, Q, E); assert(isinfinity(Q));
    assert(!isinfinity(context->BASIS[1][2])); context->xmule_deg(PQ, context->e[0] + context->e[1] - 1, context->BASIS[1][2], E);
    assert(!isinfinity(PQ)); context->xmul_deg(PQ, PQ, E); assert(isinfinity(PQ));
#endif

    // Reducing to the right order point
    // +++ Torsion: deg^e[0]
    context->xmule_deg(context->BASIS[0][0], context->e[1], context->BASIS[0][0], context->E[0]);
    context->xmule_deg(context->BASIS[0][1], context->e[1], context->BASIS[0][1], context->E[0]);
    context->xmule_deg(context->BASIS[0][2], context->e[1], context->BASIS[0][2], context->E[0]);
    // +++ Torsion: deg^e[1]
    context->xmule_deg(context->BASIS[1][0], context->e[0], context->BASIS[1][0], context->E[1]);
    context->xmule_deg(context->BASIS[1][1], context->e[0], context->BASIS[1][1], context->E[1]);
    context->xmule_deg(context->BASIS[1][2], context->e[0], context->BASIS[1][2], context->E[1]);
}

void collision_printf(const point_t collision[2], const ctx_mitm_t context)
{
    assert((((&collision[0])->b & 0x4) >> 2) == 0);
    assert((((&collision[1])->b & 0x4) >> 2) == 1);
    
    proj_t K0, K1;
    printf("// Side corresponding to E0\n");
    printf("A0 := ");fp2_printf((&context)->E[0][0]);
    printf("B0 := ");fp2_printf((&context)->E[0][1]);
    printf("E0 := EllipticCurve(t^3 + A0 * t + B0);\n");
    _h_(K0, (&context)->BASIS[0], (&context)->E[0], collision[0], (&context)->deg, (&context)->bound[0]);
    printf("XK0 := ");fp2_printf(K0[0]);
    printf("ZK0 := ");fp2_printf(K0[1]);
    printf("_, K0 := IsPoint(E0, XK0/ZK0);\n");

    printf("// Side corresponding to E1\n");
    printf("A1 := ");fp2_printf((&context)->E[1][0]);
    printf("B1 := ");fp2_printf((&context)->E[1][1]);
    printf("E1 := EllipticCurve(t^3 + A1 * t + B1);\n");
    _h_(K1, (&context)->BASIS[1], (&context)->E[1], collision[1], (&context)->deg, (&context)->bound[1]);
    printf("XK1 := ");fp2_printf(K1[0]);
    printf("ZK1 := ");fp2_printf(K1[1]);
    printf("_, K1 := IsPoint(E1, XK1/ZK1);\n\n");

    printf("// Verifying the correctness of the collision\n");
    printf("E0_K0 := ISOGENY(K0, E0, %d, %d);\n", (int)(&context)->deg, (int)(&context)->e[0]);
    printf("E1_K1 := ISOGENY(K1, E1, %d, %d);\n", (int)(&context)->deg, (int)(&context)->e[1]);
    printf("jInvariant(E0_K0) eq jInvariant(E1_K1);\n");

    printf("\n// To verify the solution, copy and paste the content of src/shortw/isogeny.mag following by the above output,"
           " into the online magma calculator at http://magma.maths.usyd.edu.au/calc/.\n");
}
