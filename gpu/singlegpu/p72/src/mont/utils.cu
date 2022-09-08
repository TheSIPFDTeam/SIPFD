#include "api.h"
#include "fp.h"

void randombytes(void *x, size_t l)
{
    static int fd = -1;
    ssize_t n, i;
    if (fd < 0 && 0 > (fd = open("/dev/urandom", O_RDONLY)))
        exit(1);
    for (i = 0; i < l; i += n)
        if (0 >= (n = read(fd, (char *) x + i, l - i)))
            exit(2);
}

__device__ void rsh(fp_t x)
{
    /* Right shift (>> 1) of a fp_t element */
    int i;
    for (i=0; i < (NWORDS_FIELD-1); i++) {
        x[i] >>= 1;
        x[i]  ^= (x[i+1] & 1)<<(RADIX-1);
    }
    x[NWORDS_FIELD-1] >>= 1;
}

__device__ void lsh(fp_t x)
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

__device__ void get_char(char *out, limb_t e, int *j, int nbytes) {
    int i;

    for (i = 0; i < nbytes; i++) {
        out[*j] = (char)e & 0xFF;
        e = e >> 8;
        *j = *j + 1;
    }
}

__device__ void fp_string(char *out, fp_t a, int *j) {
    int i;

    for (i = 0; i < NWORDS_FIELD - 1; i++) {
        get_char(out, a[i], j, sizeof(limb_t));
    }

    // get bytes from last word
    get_char(out, a[i], j, (NBYTES_FIELD - (NWORDS_FIELD - 1) * sizeof(limb_t)));
}

/* --------------------------------------------------------------------------------- *
 * Function h : |[0, d^{e/2} - 1]| --> E[deg^e] \ E[deg^{e-1}]
 * --------------------------------------------------------------------------------- */
__device__ void _h_(proj_t G, proj_t P[3], proj_t A2, point_t *g, limb_t e) {
    ladder3pt(G, g->k, P[0], P[1], P[2], A2, e); // G <- P + [b * deg^(e - 1) + m]Q
}

/* ------------------------------------------------------------------------------------------------- *
 * Function gn : fp2_t --> {0,1} x {0, 1, ..., d} x |[0, d^{e/2-1} ]|
 * ------------------------------------------------------------------------------------------------- */
__device__ void _gn_(point_t *g, fp2_t jinv, uint64_t NONCE) {
    /* Dummy hash for tests */
   
#if RADIX == 64
    g->k = ((NONCE >> 1) ^ jinv[0][0] ^ jinv[1][0]) & (((uint64_t)1 << EBITS_MAX) - 1);
    g->c = (jinv[0][1] ^ NONCE) & 1;
#elif RADIX == 32
    uint64_t aux0 = ((uint64_t)jinv[0][1] << 32) ^ jinv[0][0];
    uint64_t aux1 = ((uint64_t)jinv[1][1] << 32) ^ jinv[1][0];
    g->k = ((NONCE >> 1) ^ aux0 ^ aux1) & (((uint64_t)1 << EBITS_MAX) - 1);
    g->c = (aux0 ^ NONCE) & 1;
#else
#error "Not Implemented"
#endif

}

/* -------------------------------------------------------------------------------------------- *
 *  Pseudo Random Function fn : S --> S, where S = {0,1} x {0, 1, ..., d} x |[0, d^{e/2 - 1} ]|
 * -------------------------------------------------------------------------------------------- */
__device__ void _fn_(point_t *y, fp2_t j, point_t *x, ctx_t *context, 
        limb_t S[2][EXP0], fp2_t *P0, fp2_t *Q0, fp2_t *PQ0, fp2_t *E0, fp2_t *Z0,
        fp2_t *P1, fp2_t *Q1, fp2_t *PQ1, fp2_t *E1, fp2_t *Z1, limb_t *expo, limb_t *ebits)
{
    uint8_t c = x->c;

    // Mapping x into a projective Weierstrass x-coordinate point
    proj_t C2, X;
    proj_t P, Q, PQ, E;
    uint64_t index = x->k & ((1 << PC_DEPTH) - 1);

    if (c) {
        fp2_copy(P[0], P1[index]);
        fp2_copy(Q[0], Q1[index]);
        fp2_copy(PQ[0], PQ1[index]);
        fp2_copy(E[0], E1[index]);
        fp2_copy(P[1], Z1[index]);
        fp2_copy(Q[1], P[1]);
        fp2_copy(PQ[1], P[1]);
        fp2_copy(E[1], P[1]);
    }
    else {
        fp2_copy(P[0], P0[index]);
        fp2_copy(Q[0], Q0[index]);
        fp2_copy(PQ[0], PQ0[index]);
        fp2_copy(E[0], E0[index]);
        fp2_copy(P[1], Z0[index]);
        fp2_copy(Q[1], P[1]);
        fp2_copy(PQ[1], P[1]);
        fp2_copy(E[1], P[1]);
    }
    
    ladder3pt(X, x->k >> PC_DEPTH, P, Q, PQ, E, ebits[c] - PC_DEPTH);

    // Mapping a projective Weierstrass x-coordinate point into a supersingular elliptic curve
    xisog_2e_2nd(C2, X, E, S[c], expo[c] - PC_DEPTH);
    
    // Mapping a supersingular elliptic curve into an element of fp2_t (j-invariant)
    j_invariant(j, C2); // The j-invariant is stored in order to be used for a collision detection.
    // Mapping an element of fp2_t (j-invariant) into an element of {0,1} x {0, 1, ..., d} x |[ 0, d^(e / 2 - 1) ]|
    _gn_(y, j, context->NONCE); // DEG = 2
}

#if RADIX == 64
__device__ void TOHEX_GPU(uint64_t x) { printf("%016" PRIx64, x); };
#elif RADIX == 32
__device__ void TOHEX_GPU(uint32_t x) { printf("%08X", x); };
#else
#error "Not Implemented"
#endif

__device__ void fp2_printf(fp2_t x) {
    int i;
    fp2_t tmp = {0};
    from_montgomery(tmp[0], x[0]);
    from_montgomery(tmp[1], x[1]);
    printf("0x");
    for(i = NWORDS_FIELD-1; i > -1; i--)
        TOHEX_GPU(tmp[0][i]);
    printf(" + i * 0x");
    for(i = NWORDS_FIELD-1; i > -1; i--)
        TOHEX_GPU(tmp[1][i]);
    printf(";\n");
}

extern "C" __global__ void collision_printf(point_t collision[2], ctx_t *context) {
    assert(collision[0].c == 0);
    assert(collision[1].c == 1);

    fp2_t A0 = {0}, A1 = {0}, j0 = {0}, j1 = {0};
    proj_t K0, K1, C0 = {0}, C1 = {0};

    // Note: all k's are saved with ebits[1] bits. When points on the left side have one less bit, the highest
    // bit is always disregarded in the calculations but it may be saved with any value. The following depuration
    // check ensures that the extra bit is set to zero at the very end for consistency when printing the result.
    collision[0].k &= (((uint64_t)1 << EXP20_BITS) - 1);

    // Did we end up with complex conjugates?
    _h_(K0, const_BASIS[0], const_A2[0], &collision[0], EXP20_BITS);
    xisog_2e_2nd(C0, K0, const_E[0], STRATEGY2_REDUCED_0, EXP0);
    j_invariant(j0, C0);
    _h_(K1, const_BASIS[1], const_A2[1], &collision[1], EXP21_BITS);
    xisog_2e_2nd(C1, K1, const_E[1], STRATEGY2_REDUCED_1, EXP1);
    j_invariant(j1, C1);
    if( j0[1][0] != j1[1][0] )
    {
        collision[0].k = (((uint64_t)1) << EXP0) - collision[0].k;
        _h_(K0, const_BASIS[0], const_A2[0], &collision[0], EXP20_BITS + 1);
    }

    printf("//k0 = %lu\n//k1 = %lu\n", (collision[0].k), (collision[1].k));
    printf("// Side corresponding to E0\n");
    coeff(A0, const_A2[0]);
    printf("A0 := ");fp2_printf(A0);
    printf("E0 := EllipticCurve(t^3 + A0 * t^2 + t);\n");
    printf("XK0 := ");fp2_printf(K0[0]);
    printf("ZK0 := ");fp2_printf(K0[1]);
    printf("_, K0 := IsPoint(E0, XK0/ZK0);\n");

    printf("// Side corresponding to E1\n");
    coeff(A1, const_A2[1]);
    printf("A1 := ");fp2_printf(A1);
    printf("E1 := EllipticCurve(t^3 + A1 * t^2 + t);\n");
    printf("XK1 := ");fp2_printf(K1[0]);
    printf("ZK1 := ");fp2_printf(K1[1]);
    printf("_, K1 := IsPoint(E1, XK1/ZK1);\n\n");

    printf("// Verifying the correctness of the collision\n");
    printf("E0_K0 := ISOGENY(K0, E0, %d, %d);\n", 2, EXP0);
    printf("E1_K1 := ISOGENY(K1, E1, %d, %d);\n", 2, EXP1);
    printf("jInvariant(E0_K0) eq jInvariant(E1_K1);\n");

    printf("\n// To verify the solution, copy and paste the content of src/isogeny.mag following by the above output,"
           " into the online magma calculator at http://magma.maths.usyd.edu.au/calc/.\n");
}

