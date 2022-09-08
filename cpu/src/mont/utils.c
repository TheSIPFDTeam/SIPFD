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

void _h_(proj_t G, const proj_t P[3], const proj_t A2, const point_t g, const digit_t deg, const digit_t e)
{
    /* --------------------------------------------------------------------------------- *
     * Function h : |[0, d^{e/2} - 1]| --> E[deg^e] \ E[deg^{e-1}]
     * --------------------------------------------------------------------------------- */
    ladder3pt(G, (&g)->k, P[0], P[1], P[2], A2, e);	// G <- P + [b * deg^(e - 1) + m]Q
}

void _gn_(point_t *g, const fp2_t jinv, const digit_t NONCE, const digit_t deg, const fp_t bound, const digit_t ebits)
{
    /* ------------------------------------------------------------------------------------------------- *
     * Function gn : fp2_t --> {0,1} x {0, 1, ..., d} x |[0, d^{e/2-1} ]|
     * ------------------------------------------------------------------------------------------------- */

    //Dummy hash for test

    #ifdef FROBENIUS
    g->k = (jinv[0][0] ^ (NONCE>>1)) & (((digit_t)1 << ebits) -1 );
    g->c = (jinv[0][1] ^ NONCE) & 1;
    #else
    g->k = (jinv[0][0] ^ jinv[1][0] ^ (NONCE>>1)) & (((digit_t)1 << ebits) -1 );
    g->c = (jinv[0][1] ^ NONCE) & 1;
    #endif

    
/*
    digit_t emask = ((digit_t)1 << ((ebits + 1) % 8)) - 1;

    char raw_data[4 * 2 * NBYTES_FIELD + 1], current[2 * NBYTES_FIELD + 1];
    raw_data[0] = '\0';

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
    g->c = 0;
    do
    {
        // In this part we append the counter to (j-invariant || NONCE)
        raw_data[local_N] = '\0';
        current[0] = '\0';
        SPRINTF(current, local_counter);
        current[strlen(current)] = '\0';
        strcat(raw_data, current);
        raw_data[strlen(raw_data)] = '\0';

        // Now, we compute MD5(j-invariant||NONCE||counter)
        MD5((unsigned  char*)raw_data, strlen(raw_data), v_md5);
        local_counter += 1;

        // We save the (e + 1) least significant bit of MD5(j-invariant||NONCE||counter)
        strrev((char *)v_md5, 0, MD5_DIGEST_LENGTH - 1);
        v_md5[NBITS_TO_NBYTES(ebits + 2) - 1] &= emask;
        memcpy(g->k, (digit_t *)v_md5, sizeof(uint8_t) * NBITS_TO_NBYTES(ebits + 1));

        g->c = (g->k)[0] & 0x1;
        rsh(g->k);
        // Now, g->k < 2^(ebits) and g->c is 0 or 1
    } while( (fp_compare(g->k, bound) >= 0) ); // Loop may only repeat if deg=3
    */
}

void _fn_(point_t *y, fp2_t j, const point_t x, const ctx_mitm_t context)
{
    /* -------------------------------------------------------------------------------------------- *
     *  Pseudo Random Function fn : S --> S, where S = {0,1} x {0, 1, ..., d} x |[0, d^{e/2 - 1} ]|
     * -------------------------------------------------------------------------------------------- */
    uint8_t c = (&x)->c;

    // Mapping x into a projective Weierstrass x-coordinate point
    proj_t C2, X;
    if( (&context)->pc_depth == 0 )
    {
        _h_(X, (&context)->BASIS[c], (&context)->A2[c], x, (&context)->deg, (&context)->ebits[c]);
        (*(&context)->xisoge_2nd)(C2, X, (&context)->E[c], (&context)->S[c], (&context)->e[c]);
    }
    else
    {
        proj_t P, Q, PQ, E;
        fp2_copy(P[0], (&context)->pc_table[c][0][x.k & ((1 << (&context)->pc_depth) - 1)]);
        fp2_copy(Q[0], (&context)->pc_table[c][1][x.k & ((1 << (&context)->pc_depth) - 1)]);
        fp2_copy(PQ[0], (&context)->pc_table[c][2][x.k & ((1 << (&context)->pc_depth) - 1)]);
        fp2_copy(E[0], (&context)->pc_table[c][3][x.k & ((1 << (&context)->pc_depth) - 1)]);
        fp2_copy(P[1], (&context)->pc_table[c][4][x.k & ((1 << (&context)->pc_depth) - 1)]);
        fp2_copy(Q[1], P[1]);
        fp2_copy(PQ[1], P[1]);
        fp2_copy(E[1], P[1]);
        ladder3pt(X, (&x)->k >> (&context)->pc_depth, P, Q, PQ, E, (&context)->ebits[c] - (&context)->pc_depth);
        (*(&context)->xisoge_2nd)(C2, X, E, (&context)->S_PC[c], (&context)->e[c] - (&context)->pc_depth);
    }    

    // Mapping a supersingular elliptic curve into an element of fp2_t (j-invariant)
    j_invariant(j, C2);	// The j-invariant is stored in order to be used for a collision detection.

    // Mapping an element of fp2_t (j-invariant) into an element of {0,1} x {0, 1, ..., d} x |[ 0, d^(e / 2 - 1) ]|
    _gn_(y, j, (&context)->NONCE, (&context)->deg, (&context)->bound[1], (&context)->ebits_max);
}

void init_context_mitm(ctx_mitm_t *context, digit_t deg)
{
    if (deg == 2)
    {
        context->deg = 2;
        context->pc_depth = 0;
        fp_copy(context->size, BOUND2);
        context->strategy = (digit_t *)STRATEGY2;
        context->not_e = EXPONENT3;
        context->e[0] = EXPONENT2 - (EXPONENT2 >> 1);
        context->e[1] = EXPONENT2 >> 1;
        context->ebits[0] = EXPONENT2_BITS - (EXPONENT2_BITS >> 1);
        context->ebits[1] = EXPONENT2_BITS >> 1;
        #ifdef FROBENIUS
        context->ebits_max = context->ebits[1];
        #else
        context->ebits_max = context->ebits[0];
        #endif
        context->xmul_deg = &xdbl;
        context->xmul_notdeg = &xtpl;
        context->xmule_deg = &xdble;
        context->xmule_notdeg = &xtple;
        context->xeval = &xeval_2;
        context->xisog = &xisog_2;
        context->xisoge_1st = &xisog_2e_1st;
        context->xisoge_2nd = &xisog_2e_2nd;
        fp_copy(context->bound[0], BOUND2_0);
        fp_copy(context->bound[1], BOUND2_1);
        context->S[0] = (digit_t *)STRATEGY2_0;
        context->S[1] = (digit_t *)STRATEGY2_1;
    }
    else if (deg == 3)
    {
        #ifdef FROBENIUS
        printf("Error: Frobenius not implemented for deg=3. Turn off Frobenius in mont/api.h\n");
        exit(-1);
        #endif
        context->deg = 3;
        context->pc_depth = 0;
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
        fp_copy(context->bound[0], BOUND3_0);
        fp_copy(context->bound[1], BOUND3_1);
        context->S[0] = (digit_t *)STRATEGY3_0;
        context->S[1] = (digit_t *)STRATEGY3_1;
    }
    else
        assert((deg == 2) || (deg == 3));
}

void random_instance(ctx_mitm_t *context, digit_t deg)
{
    /* Generates a random instance starting from E_6 if FROBENIUS is on or a random curve if FROBENIUS is off*/

    proj_t E, P, Q, PQ;
    fp_t k;

    set_initial_curve(E);       // E : y^2 = x^3 + 6x^2 + x corresponding to A2 = 2

    #ifndef FROBENIUS
    // Randomize initial curve
    init_basis(P, Q, PQ, E);
    xmul(P, COFACTOR, P, E);
    xmul(Q, COFACTOR, Q, E);
    xmul(PQ, COFACTOR, PQ, E);
    xtple(P, EXPONENT3, P, E);
    xtple(Q, EXPONENT3, Q, E);
    xtple(PQ, EXPONENT3, PQ, E);
    fp_random(k);
    ladder3pt_long(P, k, P, Q, PQ, E);
    xisog_2e_2nd(E, P, E, STRATEGY2, EXPONENT2);
    #endif

    // +++ Generating basis
    proj_copy(context->E[0], E);
    change_curvemodel(context->notE[0], E);
    proj_copy(context->A2[0], E);
    init_basis(P, Q, PQ, context->E[0]);
    xmul(P, COFACTOR, P, context->E[0]);
    xmul(Q, COFACTOR, Q, context->E[0]);
    xmul(PQ, COFACTOR, PQ, context->E[0]);
    // E[3^EXPONENT3] = <PB, QB> and  PQB := PB -QB
    xdble(context->PB, EXPONENT2, P, context->E[0]);
    xdble(context->QB, EXPONENT2, Q, context->E[0]);
    xdble(context->PQB, EXPONENT2, PQ, context->E[0]);
    // E[2^EXPONENT2] = <PA, QA> and  PQA := PA -QA
    xtple(context->PA, EXPONENT3, P, context->notE[0]);
    xtple(context->QA, EXPONENT3, Q, context->notE[0]);
    xtple(context->PQA, EXPONENT3, PQ, context->notE[0]);
    

#if defined(_TEST_)
    // Checking order 2^EXPONENT2
    assert(!isinfinity(context->PA)); xdble(P, EXPONENT2 - 1, context->PA, context->E[0]);
    assert(!isinfinity(P)); xdbl(P, P, context->E[0]); assert(isinfinity(P));
    assert(!isinfinity(context->QA)); xdble(Q, EXPONENT2 - 1, context->QA, context->E[0]);
    assert(!isinfinity(Q)); xdbl(Q, Q, context->E[0]); assert(isinfinity(Q));
    assert(!isinfinity(context->PQA)); xdble(PQ, EXPONENT2 - 1, context->PQA, context->E[0]);
    assert(!isinfinity(PQ)); xdbl(PQ, PQ, context->E[0]); assert(isinfinity(PQ));
    // Checking order 3^EXPONENT3
    assert(!isinfinity(context->PB)); xtple(P, EXPONENT3 - 1, context->PB, context->notE[0]);
    assert(!isinfinity(P)); xtpl(P, P, context->notE[0]); assert(isinfinity(P));
    assert(!isinfinity(context->QB)); xtple(Q, EXPONENT3 - 1, context->QB, context->notE[0]);
    assert(!isinfinity(Q)); xtpl(Q, Q, context->notE[0]); assert(isinfinity(Q));
    assert(!isinfinity(context->PQB)); xtple(PQ, EXPONENT3 - 1, context->PQB, context->notE[0]);
    assert(!isinfinity(PQ)); xtpl(PQ, PQ, context->notE[0]); assert(isinfinity(PQ));
#endif

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
        proj_copy(context->E[0], context->notE[0]);
        proj_copy(context->notE[0], context->A2[0]);
        assert(context->e[0] + context->e[1] == EXPONENT3);
    }

    fp_random(k);
    ladder3pt_long(P, k, context->BASIS[0][0], context->BASIS[0][1], context->BASIS[0][2], context->A2[0]);
    (*(context->xisoge_2nd))(E, P, context->E[0], context->strategy, context->e[0] + context->e[1]);
 

    // Curve E is now given either by A2 or A3 depending on deg
    init_basis(P, Q, PQ, E);
    proj_copy(context->E[1], E);
    change_curvemodel(context->notE[1], E);
    if (deg == 2)
    	proj_copy(context->A2[1], context->E[1]);
    else
    	proj_copy(context->A2[1], context->notE[1]);
    
    // +++ Generating basis
    init_basis(P, Q, PQ, context->A2[1]);
    xmul(P, COFACTOR, P, context->A2[1]);
    xmul(Q, COFACTOR, Q, context->A2[1]);
    xmul(PQ, COFACTOR, PQ, context->A2[1]);
    context->xmule_notdeg(context->BASIS[1][0], context->not_e, P, context->notE[1]);
    context->xmule_notdeg(context->BASIS[1][1], context->not_e, Q, context->notE[1]);
    context->xmule_notdeg(context->BASIS[1][2], context->not_e, PQ, context->notE[1]);
    
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
    context->xmule_deg(context->BASIS[1][0], context->e[0], context->BASIS[1][0], E);
    context->xmule_deg(context->BASIS[1][1], context->e[0], context->BASIS[1][1], E);
    context->xmule_deg(context->BASIS[1][2], context->e[0], context->BASIS[1][2], E);

    #ifdef FROBENIUS
    // Overwrite Frobenius base
    if(deg == 2)
        frobenius_basis(context->BASIS[0][0], context->BASIS[0][1], context->BASIS[0][2], context->A2[0], context->e[0]);
    #endif
}

void random_instance_rigged(ctx_mitm_t *context, const digit_t e, const uint64_t K0, const uint64_t K1)
{
    /*Generates a 'random' instance with key K and degree 2^e. ALWAYS starts from E_6*/
    
    digit_t i;
    proj_t E, notE, P, Q, PQ, R, R2;
    fp2_t A;
    uint64_t K1minus1;

    context->deg = 2;
    context->e[0] = e - (e >> 1);
    context->e[1] = e >> 1;
    context->ebits[0] = context->e[0];
    context->ebits[1] = context->e[1];
    context->ebits_max = context->e[0];

    set_initial_curve(E);                // E : y^2 = x^3 + 6x^2 + x corresponding to A2 = 2

    #ifndef FROBENIUS
    fp_t k;
    // Randomize initial curve
    init_basis(P, Q, PQ, E);
    xmul(P, COFACTOR, P, E);
    xmul(Q, COFACTOR, Q, E);
    xmul(PQ, COFACTOR, PQ, E);
    xtple(P, EXPONENT3, P, E);
    xtple(Q, EXPONENT3, Q, E);
    xtple(PQ, EXPONENT3, PQ, E);
    fp_random(k);
    ladder3pt_long(P, k, P, Q, PQ, E);
    xisog_2e_2nd(E, P, E, STRATEGY2, EXPONENT2);
    #endif

    proj_copy(context->E[0], E);         // note for A=6 we have E = notE = A2 = 2
    change_curvemodel(context->notE[0], E);         
    proj_copy(context->A2[0], E);

    // +++ Generating basis
    #ifdef FROBENIUS
    frobenius_basis(P, Q, PQ, E, e);
    #else
    init_basis(P, Q, PQ, context->E[0]);
    xmul(P, COFACTOR, P, context->E[0]);
    xmul(Q, COFACTOR, Q, context->E[0]);
    xmul(PQ, COFACTOR, PQ, context->E[0]);
    xdble(P, EXPONENT2-e, P, context->E[0]);
    xdble(Q, EXPONENT2-e, Q, context->E[0]);
    xdble(PQ, EXPONENT2-e, PQ, context->E[0]);
    xtple(P, EXPONENT3, P, context->notE[0]);
    xtple(Q, EXPONENT3, Q, context->notE[0]);
    xtple(PQ, EXPONENT3, PQ, context->notE[0]);
    #endif

    proj_copy(context->BASIS[0][0], P);
    proj_copy(context->BASIS[0][1], Q);
    proj_copy(context->BASIS[0][2], PQ);

    // Doing the isogeny
    ladder3pt(R, K0, context->BASIS[0][0], context->BASIS[0][1], context->BASIS[0][2], context->E[0], context->e[0]);
    for(i = 0; i < e; i++){
        xdble(R2, e - i - 1, R, E);
        xisog_2(E, R2);
        xeval_2(R, R, R2);
        xeval_2(Q, Q, R2);
    }


    // Alternate coefficients
    change_curvemodel(notE, E);
    coeff(A, E);

    // Find points of order 4 and 2
    fp2_t P2, P4;
    init_basis(P, R, PQ, E);
    xmul(P, COFACTOR, P, E);
    xdble(P, EXPONENT2 - 2, P, E);
    xtple(P, EXPONENT3, P, notE);
    fp2_copy(P4, P[1]);
    fp2_inv(P4);
    fp2_mul(P4, P4, P[0]);
    xdbl(P, P, E);
    fp2_copy(P2, P[1]);
    fp2_inv(P2);
    fp2_mul(P2, P2, P[0]);

    
    // Isomorphism that exchanges P and Q
    fp2_inv(Q[1]);
    fp2_mul(Q[0], Q[0], Q[1]);
    fp2_sub(Q[0], Q[0], P2);
    fp2_sub(Q[1], P4, P2);
    fp2_add(E[0], A, P2);
    fp2_add(E[0], E[0], P2);
    fp2_add(E[0], E[0], P2);
    fp2_add(E[1], Q[1], Q[1]);
    fp2_add(E[0], E[0], E[1]);
    fp2_add(E[1], E[1], E[1]);
    change_curvemodel(notE, E);
    coeff(A, E);
    proj_copy(P, Q);

    // Find new basis fixing P
    init_basis(R, Q, PQ, E);
    xmul(Q, COFACTOR, Q, E);
    xdble(Q, EXPONENT2 - e, Q, E);
    xtple(Q, EXPONENT3, Q, notE);
    make_affine(P, P, A);
    make_affine(Q, Q, A);
    difference_point(PQ, P, Q, A);
    // Randomize denominators
    fp2_random(P[1]);
    fp2_random(Q[1]);
    fp2_mul(P[0], P[0], P[1]);
    fp2_mul(Q[0], Q[0], Q[1]);

    // Replace P by P+K1*Q
    ladder3pt(R, K1, P, Q, PQ, E, context->e[1]);
    K1minus1 = K1 - 1;
    ladder3pt(PQ, K1minus1, P, Q, PQ, E, context->e[1]);
    proj_copy(P, R);

    // Replace Q by -Q
    xadd(PQ, P, Q, PQ);

    // Finally, save everything
    proj_copy(context->E[1], E);
    proj_copy(context->A2[1], E);
    proj_copy(context->notE[1], notE);
    proj_copy(context->BASIS[1][0], P);
    proj_copy(context->BASIS[1][1], Q);
    proj_copy(context->BASIS[1][2], PQ);

    // Reducing to the right order point
    // +++ Torsion: deg^e[0]
    xdble(context->BASIS[0][0], context->e[1], context->BASIS[0][0], context->E[0]);
    xdble(context->BASIS[0][1], context->e[1], context->BASIS[0][1], context->E[0]);
    xdble(context->BASIS[0][2], context->e[1], context->BASIS[0][2], context->E[0]);
    // +++ Torsion: deg^e[1]
    xdble(context->BASIS[1][0], context->e[0], context->BASIS[1][0], E);
    xdble(context->BASIS[1][1], context->e[0], context->BASIS[1][1], E);
    xdble(context->BASIS[1][2], context->e[0], context->BASIS[1][2], E);
    }

void undo_2isog(ctx_mitm_t *context)
{
    // For isogenies of degree 2^e, the last two steps can be undone
    
    assert(context->deg == 2);
    
    // Compute the 4-isogenous curve
    fp2_copy(context->A2[1][1], context->E[1][0]);                   // C24' = A24
    fp2_sub(context->A2[1][0], context->E[1][0], context->E[1][1]);  // A24' = A24-C24
    
    proj_copy(context->E[1], context->A2[1]);
    change_curvemodel(context->notE[1], context->E[1]);
    
    // Reduce exponents by one each
    context->e[0] -= 1;
    context->e[1] -= 1;
    #ifdef FROBENIUS
    context->ebits[0] -= 2;
    #else
    context->ebits[0] -= 1;
    #endif
    context->ebits[1] -= 1;
    context->ebits_max -= 1;
    context->S[0] = (digit_t *)STRATEGY2_REDUCED_0;
    context->S[1] = (digit_t *)STRATEGY2_REDUCED_1;
    context->S_PC[0] = (digit_t *)STRATEGY2_PC_0;
    context->S_PC[1] = (digit_t *)STRATEGY2_PC_1;
    rsh(context->bound[0]);
    rsh(context->bound[1]);
    
    // Redo right basis
    init_basis(context->BASIS[1][0], context->BASIS[1][1], context->BASIS[1][2], context->A2[1]);
    xmul(context->BASIS[1][0], COFACTOR, context->BASIS[1][0], context->A2[1]);
    xmul(context->BASIS[1][1], COFACTOR, context->BASIS[1][1], context->A2[1]);
    xmul(context->BASIS[1][2], COFACTOR, context->BASIS[1][2], context->A2[1]);
    context->xmule_notdeg(context->BASIS[1][0], context->not_e, context->BASIS[1][0], context->notE[1]);
    context->xmule_notdeg(context->BASIS[1][1], context->not_e, context->BASIS[1][1], context->notE[1]);
    context->xmule_notdeg(context->BASIS[1][2], context->not_e, context->BASIS[1][2], context->notE[1]);
    context->xmule_deg(context->BASIS[1][0], context->e[0]+2, context->BASIS[1][0], context->A2[1]);
    context->xmule_deg(context->BASIS[1][1], context->e[0]+2, context->BASIS[1][1], context->A2[1]);
    context->xmule_deg(context->BASIS[1][2], context->e[0]+2, context->BASIS[1][2], context->A2[1]);
    
    // Reduce order of left basis
    context->xmul_deg(context->BASIS[0][0], context->BASIS[0][0], context->E[0]);
    context->xmul_deg(context->BASIS[0][1], context->BASIS[0][1], context->E[0]);
    context->xmul_deg(context->BASIS[0][2], context->BASIS[0][2], context->E[0]);
}

void collision_printf(point_t collision[2], const ctx_mitm_t context)
{
    assert((&collision[0])->c == 0);
    assert((&collision[1])->c == 1);

    // Note: all k's are saved with ebits_max bits. When points on one side have one less bit, the highest
    // bit is always disregarded in the calculations but it may be saved with any value. The following depuration
    // check ensures that the extra bit is set to zero at the very end for consistency when printing the result.
    collision[0].k &= ((1 << (&context)->ebits[0]) - 1);
    collision[1].k &= ((1 << (&context)->ebits[1]) - 1);
    fp2_t A0 = {0}, A1 = {0}, j0, j1;
    proj_t K0, K1, C0, C1;

    // Did we end up with complex conjugates?
    _h_(K0, (&context)->BASIS[0], (&context)->A2[0], collision[0], (&context)->deg, (&context)->ebits[0]);
    (*(&context)->xisoge_2nd)(C0, K0, (&context)->E[0], (&context)->S[0], (&context)->e[0]);
    j_invariant(j0, C0);
    _h_(K1, (&context)->BASIS[1], (&context)->A2[1], collision[1], (&context)->deg, (&context)->ebits[1]);
    (*(&context)->xisoge_2nd)(C1, K1, (&context)->E[1], (&context)->S[1], (&context)->e[1]);
    j_invariant(j1, C1);
    if( j0[1][0] != j1[1][0] )
    {
        collision[0].k = (1 << (&context)->e[0]) - collision[0].k;
        _h_(K0, (&context)->BASIS[0], (&context)->A2[0], collision[0], (&context)->deg, (&context)->ebits[0]+1);
    }

    printf("//k0 = %lu\n//k1 = %lu\n", (collision[0].k), (collision[1].k));
    printf("// Side corresponding to E0\n");
    coeff(A0, (&context)->A2[0]);
    printf("A0 := ");fp2_printf(A0);
    printf("E0 := EllipticCurve(t^3 + A0 * t^2 + t);\n");
    printf("XK0 := ");fp2_printf(K0[0]);
    printf("ZK0 := ");fp2_printf(K0[1]);
    printf("_, K0 := IsPoint(E0, XK0/ZK0);\n");

    printf("// Side corresponding to E1\n");
    coeff(A1, (&context)->A2[1]);
    printf("A1 := ");fp2_printf(A1);
    printf("E1 := EllipticCurve(t^3 + A1 * t^2 + t);\n");
    printf("XK1 := ");fp2_printf(K1[0]);
    printf("ZK1 := ");fp2_printf(K1[1]);
    printf("_, K1 := IsPoint(E1, XK1/ZK1);\n\n");

    printf("// Verifying the correctness of the collision\n");
    printf("E0_K0 := ISOGENY(K0, E0, %d, %d);\n", (int)(&context)->deg, (int)(&context)->e[0]);
    printf("E1_K1 := ISOGENY(K1, E1, %d, %d);\n", (int)(&context)->deg, (int)(&context)->e[1]);
    printf("jInvariant(E0_K0) eq jInvariant(E1_K1);\n");

    printf("\n// To verify the solution, copy and paste the content of src/mont/isogeny.mag following by the above output,"
           " into the online magma calculator at http://magma.maths.usyd.edu.au/calc/.\n");
}
