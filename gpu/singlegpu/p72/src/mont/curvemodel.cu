#include <inttypes.h>
#include "api.h"

__device__ void change_curvemodel(proj_t Q, proj_t P)
{
    /* ---------------------------------------------------------------------- *
     * Switches between constant domains A2 = (A+2C:4C) and A3 = (A+2C:A-2C)
     * ---------------------------------------------------------------------- */
    fp2_copy(Q[0], P[0]);
    fp2_sub(Q[1], P[0], P[1]);
}

__device__ void get_A(proj_t A2, proj_t P, proj_t Q, proj_t PQ)
{
    /* ---------------------------------------------------------------------- *
     * Projective get_A() version of SIKE implementation
     * input : Projective x-coordinate Montgoery points P, Q, (P-Q)
     * output: Montgomery coefficient A2=(A+2C:4C) of the curve they belong to
     * ---------------------------------------------------------------------- */
    
    // To check and improve it
    fp2_t XPXQ, ZPZQ, t, s, t0, t1;
    fp2_add(t0, P[0], P[1]);	// XP + ZP
    fp2_add(t1, Q[0], Q[1]);	// XQ + ZQ

    fp2_mul(t, t0, t1);		    // (XP + ZP) * (XQ + ZQ)
    fp2_mul(XPXQ, P[0], Q[0]);	// XP * XQ
    fp2_mul(ZPZQ, P[1], Q[1]);	// ZP * ZQ

    fp2_sub(t, t, XPXQ);
    fp2_sub(t, t, ZPZQ);		// XPZQ + ZPXQ
    fp2_sub(s, XPXQ, ZPZQ);		// XPXQ - ZPZQ

    fp2_mul(t0, t,  PQ[0]);		// (XPZQ + ZPXQ) * XPQ
    fp2_mul(t1, s,  PQ[1]);		// (XPXQ - ZPZQ) * ZPQ
    fp2_add(t0, t0, t1);		// (XPZQ + ZPXQ) * XPQ + (XPXQ - ZPZQ) * ZPQ
    fp2_sqr(t0, t0);		    // [(XPZQ + ZPXQ) * XPQ + (XPXQ - ZPZQ) * ZPQ] ^ 2

    fp2_mul(t1, t, PQ[1]);		// (XPZQ + ZPXQ) * ZPQ
    fp2_mul(s, ZPZQ, PQ[0]);	// ZPZQ * XPQ
    fp2_add(t1, t1, s);		    // (XPZQ + ZPXQ) * ZPQ + ZPZQ * XPQ
    fp2_mul(s, XPXQ, PQ[0]);	// (XPXQ) * XPQ
    fp2_add(s, s, s);		    // 2 * [(XPXQ) * XPQ]
    fp2_add(s, s, s);		    // 4 * [(XPXQ) * XPQ]
    fp2_mul(t1, t1, s);		    // [(XPZQ + ZPXQ) * ZPQ + ZPZQ * XPQ] * (4 * [(XPXQ) * XPQ])

    fp2_mul(t, ZPZQ, PQ[1]);	// ZPZQ * ZPQ

    fp2_sub(XPXQ, t0, t1);		// [(XPZQ + ZPXQ) * XPQ + (XPXQ - ZPZQ) * ZPQ] ^ 2 - [(XPZQ + ZPXQ) * ZPQ + ZPZQ * XPQ] * (4 * [(XPXQ) * XPQ])
    fp2_mul(ZPZQ, s, t);		// (4 * [(XPXQ) * XPQ]) * (ZPZQ * ZPQ)

    // Recall, we requir (A + 2C : 4C) instead of (A : C)
    fp2_add(A2[1], ZPZQ, ZPZQ);  // 2C
    fp2_add(A2[0], XPXQ, A2[1]);  // A + 2C
    fp2_add(A2[1], A2[1], A2[1]);  // 4C
}

__device__ void coeff(fp2_t A, proj_t A2)
{
    /* Affine Montgomery coefficient computation: from (A+2C:4C) to A/C */
    fp2_t t;
    fp2_add(t, A2[0], A2[0]);	// (2 * A24)
    fp2_sub(t, t, A2[1]);	// (2 * A24) - C24

    fp2_copy(A, A2[1]);
    fp2_inv(A);		    // 1 / (C24)
    fp2_add(t, t, t);	// 4*A = 2[(2 * A24) - C24]
    fp2_mul(A, t, A);	// A/C = 2[(2 * A24) - C24] / C24
}

__device__ void j_invariant(fp2_t j, proj_t A2)
{
    /* j-invariant computation for montgommery coefficient A2=(A+2C:4C) */
    fp2_t t0, t1;

    proj_t A;
    fp2_add(A[0], A2[0], A2[0]);
    fp2_sub(A[0], A[0], A2[1]);
    fp2_add(A[0], A[0], A[0]);
    fp2_copy(A[1], A2[1]);

    fp2_sqr(t1, A[1]);
    fp2_sqr(j, A[0]);
    fp2_add(t0, t1, t1);
    fp2_sub(t0, j, t0);
    fp2_sub(t0, t0, t1);
    fp2_sub(j, t0, t1);
    fp2_sqr(t1, t1);
    fp2_mul(j, j, t1);
    fp2_add(t0, t0, t0);
    fp2_add(t0, t0, t0);
    fp2_sqr(t1, t0);
    fp2_mul(t0, t0, t1);
    fp2_add(t0, t0, t0);
    fp2_add(t0, t0, t0);
    fp2_inv(j);
    fp2_mul(j, t0, j);
}

__device__ void set_initial_curve(proj_t A2)
{
    /* ----------------------------------------------------------------------------- *
     * set_initial_curve()
     * output: the Montgomery curve constant A2=(A+2C:4C)=(2:1)
     * ----------------------------------------------------------------------------- */
    fp2_set_one(A2[1]);          // 1
    fp2_add(A2[0], A2[1], A2[1]);  // 2
}

__device__ int isinfinity(proj_t P)
{
    /* ----------------------------------------------------------------------------- *
     * isinfinity()
     * inputs: the projective Weierstrass x-coordinate point x(P)=P[0]/P[1];
     * output:
     *       1 if ZP == 0, or
     *       0 if ZP != 0
     * ----------------------------------------------------------------------------- */
    return fp2_iszero(P[1]);
}

__device__ int proj_isequal(proj_t P, proj_t Q)
{
    /* ----------------------------------------------------------------------------- *
     * proj_isequal()
     * inputs: the projective Weierstrass x-coordinate points x(P)=P[0]/P[1] and
     *         x(Q)=Q[0]/Q[1];
     * output:
     *         1 if XP*ZQ == ZP*XQ, or
     *         0 if XP*ZQ != ZP*XQ
     * ----------------------------------------------------------------------------- */
    fp2_t XPZQ, ZPXQ;
    fp2_mul(XPZQ, P[0], Q[1]);
    fp2_mul(ZPXQ, P[1], Q[0]);
    return (int)(0 == fp2_compare(XPZQ, ZPXQ));
}

__device__ void proj_copy(proj_t Q, proj_t P)
{
    /* ----------------------------------------------------------------------------- *
     * proj_copy()
     * inputs: a projective Weierstrass x-coordinate point P;
     * output: a copy of the projective Weierstrass x-coordinate point
     * ----------------------------------------------------------------------------- */
    fp2_copy(Q[0], P[0]);
    fp2_copy(Q[1], P[1]);
}

__device__ void xdbl(proj_t Q, proj_t P, proj_t A2)
{
    /* ----------------------------------------------------------------------------- *
     * Differential point doubling given the montgomery coefficient A2 = (A+2C:4C)
     * ----------------------------------------------------------------------------- */
     
    fp2_t t_0, t_1;
    fp2_sub(t_0, P[0],P[1]);
    fp2_add(t_1, P[0],P[1]);

    fp2_sqr(t_0, t_0);
    fp2_sqr(t_1, t_1);

    fp2_mul(Q[1], A2[1], t_0);
    fp2_mul(Q[0], Q[1], t_1);

    fp2_sub(t_1, t_1, t_0);
    fp2_mul(t_0, A2[0], t_1);
    fp2_add(Q[1], Q[1], t_0);
    fp2_mul(Q[1], Q[1], t_1);
}

__device__ void xadd(proj_t R, proj_t P, proj_t Q, proj_t PQ)
{
    /* ----------------------------------------------------------------------------- *
     * Differential point addition of P and Q given PQ = P-Q
     * ----------------------------------------------------------------------------- */
     
    fp2_t t0, t1, t2, t3;

    fp2_add(t0, P[0], P[1]);
    fp2_sub(t1, Q[0], Q[1]);
    fp2_mul(t1, t1, t0);
    fp2_sub(t0, P[0], P[1]);
    fp2_add(t2, Q[0], Q[1]);
    fp2_mul(t2, t2, t0);
    fp2_add(t3, t1, t2);
    fp2_sqr(t3, t3);
    fp2_sub(t2, t1, t2);
    fp2_sqr(t2, t2);
    fp2_copy(t1, PQ[0]);
    fp2_mul(R[0], PQ[1], t3);
    fp2_mul(R[1], t1, t2);
}	// 6A + 2S + 4M


__device__ void xtpl(proj_t Q, proj_t P, proj_t A3)
{
    /* ----------------------------------------------------------------------------- *
     * Differential point tripling given the montgomery coefficient A3 = (A+2C:A-2C)
     * ----------------------------------------------------------------------------- */
     
    fp2_t t0, t1, t2, t3, t4;
    fp2_sub(t0, P[0], P[1]);
    fp2_sqr(t2, t0);
    fp2_add(t1, P[0], P[1]);
    fp2_sqr(t3, t1);
    fp2_add(t4, t1, t0);
    fp2_sub(t0, t1, t0);
    fp2_sqr(t1, t4);
    fp2_sub(t1, t1, t3);
    fp2_sub(t1, t1, t2);
    fp2_mul(Q[0], t3, A3[0]);
    fp2_mul(t3, Q[0], t3);
    fp2_mul(Q[1], t2, A3[1]);
    fp2_mul(t2, t2, Q[1]);
    fp2_sub(t3, t2, t3);
    fp2_sub(t2, Q[0], Q[1]);
    fp2_mul(t1, t2, t1);
    fp2_add(t2, t3, t1);
    fp2_sqr(t2, t2);
    fp2_mul(Q[0], t2, t4);
    fp2_sub(t1, t3, t1);
    fp2_sqr(t1, t1);
    fp2_mul(Q[1], t1, t0);
}

__device__ void xdbladd(proj_t R, proj_t S, proj_t P, proj_t Q, proj_t PQ, proj_t A2)
{
    /* ----------------------------------------------------------------------------- *
     * Computes x([2]P) and x(P + Q) given x(P-Q) and the Montgommery
     * coefficient A2 = (A+2C:4C)
     * ----------------------------------------------------------------------------- */
     proj_t tmp;
     xadd(tmp, P, Q, PQ);
     xdbl(R, P, A2);
     proj_copy(S, tmp);
}

/* ----------------------------------------------------------------------------- *
 * Computes x(P + [m]Q) given the Montgommery coefficient A2 = (A+2C:4C)
 * ----------------------------------------------------------------------------- */
__device__ void ladder3pt(proj_t R, uint64_t m, proj_t P, proj_t Q, proj_t PQ, proj_t A2, limb_t e)
{
    proj_t X0, X1, X2;
    proj_copy(X0, Q);
    proj_copy(X1, P);
    proj_copy(X2, PQ);

    limb_t j;
    limb_t flag = 1;

    for(j = 0; j < e; j++)
    {
        if( (flag & m) != 0 )
            xdbladd(X0, X1, X0, X1, X2, A2);
        else
            xdbladd(X0, X2, X0, X2, X1, A2);

        flag <<= 1;
    }
    proj_copy(R, X1);
}

__device__ void ladder3pt_long(proj_t R, fp_t m, proj_t P, proj_t Q, proj_t PQ, proj_t A2)
{
    proj_t X0, X1, X2;
    proj_copy(X0, Q);
    proj_copy(X1, P);
    proj_copy(X2, PQ);

    limb_t i, j;
    limb_t flag = 1;

    for (i = 0; i < NWORDS_FIELD; i++)
    {
        flag = 1;
        for(j = 0; j < RADIX; j++)
        {
            if( (flag & m[i]) != 0 )
                xdbladd(X0, X1, X0, X1, X2, A2);
            else
                xdbladd(X0, X2, X0, X2, X1, A2);

            flag <<= 1;
        }
    }
    proj_copy(R, X1);
}

__device__ void xdble(proj_t Q, limb_t e, proj_t P, proj_t A2)
{
    /* ----------------------------------------------------------------------------- *
     * xdble()
     * inputs: an integer number 0 <= e <= e2, the projective Montgomery x-coordinate
     *         point x(P), and the Montomery coefficient A2=(A+2C:4C)
     * output: the projective Montgomery x-coordinate point x([2^e]P)
     * ----------------------------------------------------------------------------- */
    int i;
    proj_t T;
    proj_copy(T, P);
    for(i = 0; i < e; i++)
        xdbl(T, T, A2);

    proj_copy(Q, T);
}

__device__ void xtple(proj_t Q, limb_t e, proj_t P, proj_t A3)
{
    /* ----------------------------------------------------------------------------- *
     * xtple()
     * inputs: an integer number 0 <= e <= e3, the projective Montgomery x-coordinate
     *         point x(P), and the Montgomery coefficient A3=(A+2C:A-2C)
     * output: the projective Montgomery x-coordinate point x([3^e]P)
     * ----------------------------------------------------------------------------- */
    int i;
    proj_t T;
    proj_copy(T, P);
    for(i = 0; i < e; i++)
        xtpl(T, T, A3);

    proj_copy(Q, T);
}

__device__ void xmul(proj_t Q, limb_t k, proj_t P, proj_t A2)
{
    /* ----------------------------------------------------------------------------- *
     * xmul()
     * inputs: a projective Montgomery x-coordinate point P, the Montgomery
     * coefficient A2=(A+2C:4C), and an integer number 0 <= k < 2^64;
     * output: the projective Montgomery x-coordinate point [k]P
     * ----------------------------------------------------------------------------- */
    if (k == 1)
    {
        proj_copy(Q, P);
        return ;
    }
    proj_t R[2];

    // Initial 3-tuple of points
    proj_copy(R[0], P);    // P
    xdbl(R[1], P, A2);      // [2]P

    // Bit-reverse of k
    limb_t l = k, tmp = 0;
    while(l > 1)
    {
        tmp = (tmp << 1) ^ (l & 0x1);
        l >>= 1;
    }

    while(tmp > 0)
    {
    	xdbladd(R[(tmp & 0x1)], R[(tmp & 0x1) ^ 0x1], R[tmp & 0x1], R[(tmp & 0x1) ^ 0x1], P, A2);
        tmp >>= 1;
    }
    proj_copy(Q, R[0]);
}   // Cost ~ 2*Ceil[log_2(k)]*(4M + 2S)


__device__ void random_affine_point(proj_t P, fp2_t A, curandStatePhilox4_32_10_t *state)
{
    /* ---------------------------------------------------------------------- *
    random_point()
    input : the Montgommery coefficient A2 = (A+2C:4C)
    output: a random affine point (x,y) of the Montomery curve
    E : y^2 = x^3 + Ax^2 + x
    * ---------------------------------------------------------------------- */
    fp2_t tmp;
    
    while(1)
    {
    	fp2_random(P[0], state);
	fp2_sqr(tmp, P[0]);
	fp2_mul(P[1], A, tmp);
	fp2_add(P[1], P[1], P[0]);
	fp2_mul(tmp, tmp, P[0]);
	fp2_add(P[1], P[1], tmp);

        if(fp2_issquare(P[1], P[1]) == 1)
            break;
    } 
}

__device__ void difference_point(proj_t PQ, proj_t P, proj_t Q, fp2_t A)
{
    /* ---------------------------------------------------------------------- *
     * difference_point()
     * input : the affine Montgomery points (x(P), y(P)) and (x(Q), y(Q)) and
     * Monetgomery coefficient A
     * output: the projective Montgomery x-coordinate (x(P-Q):z(P-Q))
     * ---------------------------------------------------------------------- */
    fp2_t lambda, tmp;
    
    fp2_sub(lambda, P[0], Q[0]);
    fp2_inv(lambda);
    fp2_add(tmp, P[1], Q[1]);
    fp2_mul(lambda, lambda, tmp);  //lambda = (Py+Qy)/(Px-Qx)
    
    fp2_sqr(tmp, lambda);
    fp2_sub(tmp, tmp, P[0]);
    fp2_sub(tmp, tmp, Q[0]);
    fp2_sub(PQ[0], tmp, A);  // (P-Q)x = lambda^2 - Px - Qx - A    
    fp2_set_one(PQ[1]);
}

__device__ int isfull_order(proj_t P2, proj_t P3, proj_t P, proj_t A2)
{
    /* ---------------------------------------------------------------------- *
     * isfull_order()
     * input : a projective x-coordainte point P, and the
     *         Montgomery coefficient A2=(A+2C:4C)
     * output: returns 1 if the point P has full order [(2^e2) * (3^e3) * f] or
     *         0 if it doesn't have full order, and writes the projective Montgomery
     *         x-coordinate points x([(p+1)/2]P) and x([(p+1)/3]P);
     * ---------------------------------------------------------------------- */
    proj_t Q, Pf, A3;
    
    change_curvemodel(A3, A2); // Projective constant for point tripling
    
    proj_copy(Q, P);
    xdble(Q, EXPONENT2 - 1, Q, A2);	// [2^(e2 - 1)]P
    xtple(Q, EXPONENT3 - 1, Q, A3);	// [3^(e3 - 1)][2^(e2 - 1)]P

    xdbl(Pf, Q, A2);		// [3^(e3 - 1)][2^e2]P
    xtpl(Pf, Pf, A3);	// [3^e3][2^e2]P

    xmul(Q, COFACTOR, Q, A2);	// [3^(e3 - 1)][2^(e2 - 1)][f]P
    xtpl(P2, Q, A3);		// [3^e3][2^(e2 - 1)][f]P
    xdbl(P3, Q, A2);		// [3^(e3 - 1)][2^e2][f]P

    return (int)(isinfinity(P2) != 1) && (int)(isinfinity(P3) != 1) && ((int)(isinfinity(Pf) != 1) || (COFACTOR == 1));
}

__device__ void init_basis(proj_t P, proj_t Q, proj_t PQ, proj_t A2, curandStatePhilox4_32_10_t *state)
{
    /* ---------------------------------------------------------------------- *
     * full_torsion_points()
     * input : the Montgomery coefficient A2=(A+2C:4C)
     * output: the projective Montomgery x-coordinate full order generators x(P),
     * x(Q), and x(P-Q);
     * ---------------------------------------------------------------------- */
    proj_t P2, P3, Q2, Q3;
    fp2_t tmp, A;
    
    // Generating the affine Montgomery constant A
    fp2_add(tmp, A2[0], A2[0]);
    fp2_sub(tmp, tmp, A2[1]);
    fp2_add(tmp, tmp, tmp);
    fp2_copy(A, A2[1]);
    fp2_inv(A);
    fp2_mul(A, A, tmp);
    
    while( 1 )
    {
        random_affine_point(P, A, state);
        random_affine_point(Q, A, state);
        difference_point(PQ, P, Q, A);
        fp2_set_one(P[1]);
        fp2_set_one(Q[1]);

        if (isfull_order(P2, P3, P, A2) && isfull_order(Q2, Q3, Q, A2))
        {
            if( (proj_isequal(P2, Q2) != 1) && (proj_isequal(P3, Q3) != 1) )
            {
            	if( fp2_iszero(Q2[0]) )
                	break;
            }
        }
    }

}

/* ---------------------------------------------------------------------- *
 * xisog_2()
 * input : a projective Montgomery x-coordinate order-2 point P,
 * output: Coefficient A2=(A24plus:C24) of the codomain curve
 * ---------------------------------------------------------------------- */
__device__ void xisog_2(proj_t C, proj_t P)
{
     
     fp2_sqr(C[0], P[0]);
     fp2_sqr(C[1], P[1]);
     fp2_sub(C[0], C[1], C[0]);
} // 2S + 1A

/* ---------------------------------------------------------------------- *
 * xeval_2()
 * input : a projective Montgomery x-coordinate point Q,
 *         a projective Montgomery x-coordinate order-2 point P,
 * output: the image of Q under a 2-isogeny with kernel generated
 *         by P
* ---------------------------------------------------------------------- */
__device__ void xeval_2(proj_t R, proj_t Q, proj_t P)
{
    fp2_t t0, t1, t2, t3;
    
    fp2_add(t0, P[0], P[1]);
    fp2_sub(t1, P[0], P[1]);
    fp2_add(t2, Q[0], Q[1]);
    fp2_sub(t3, Q[0], Q[1]);
    fp2_mul(t0, t0, t3);
    fp2_mul(t1, t1, t2);
    fp2_add(t2, t0, t1);
    fp2_sub(t3, t0, t1);
    fp2_mul(R[0], Q[0], t2);
    fp2_mul(R[1], Q[1], t3);
} // 4M + 6A


//__device__ void xisog_2e_1st(proj_t C, proj_t W[3], const proj_t P, const proj_t A2, const uint32_t *S2, const uint32_t e)
//{
//    /* ---------------------------------------------------------------------- *
//     *  xisog_2e_1st()
//     *  Input : three projective Montgomery x-coordinate points: W[1], W[2], W[3],
//     *          Montgomery x-coordinate order-2^e point x(P)=P[0]/P[1], 
//     *          Montgomery coefficient A2=(A+2C:4C), a strategy S2,
//     *          and a positive integer e;
//     *  Output: Overwrites to W the image of the three inputs points under a 2^e-isogeny
//     *          with kernel generated by x(P) and stores coefficient A2=(A+2C:4C)
//     *		of the codomain curve to C
//     * ---------------------------------------------------------------------- */
//    uint8_t log2_of_e, tmp;
//    for(tmp = e, log2_of_e = 0; tmp > 0; tmp>>=1, ++log2_of_e);
//    log2_of_e *= 2; // In order to ensure each splits is at most of size log2_of_e
//
//    proj_t SPLITTING_POINTS[EXPONENT2];
//    //proj_t SPLITTING_POINTS[log2_of_e];
//    // TODO: Check this for GPUs
//    proj_copy(SPLITTING_POINTS[0], P);
//    proj_copy(C, A2);
//
//    int strategy = 0,    // Current element of the strategy to be used
//    local_i, local_j;
//
//    int BLOCK = 0,       // BLOCK is used for determined when a point has order l
//    current = 0;         // At the beginning we have only one point in each split
//
//    //int XDBLs[log2_of_e]; // The current number of doublings performed
//    // TODO: Check this for GPUs
//    int XDBLs[EXPONENT2]; // The current number of doublings performed
//
//    for(local_j = 0; local_j < (e - 1); local_j++)
//    {
//        while (BLOCK != (e -  1 - local_j) )
//        {
//            // A new split will be added
//            current += 1;
//            // We set the seed of the new split to be computed and saved
//            xdble(SPLITTING_POINTS[current], S2[strategy], SPLITTING_POINTS[current - 1], C);
//            XDBLs[current] = S2[strategy];  // The number of doublings performed is saved
//            BLOCK += S2[strategy];          // BLOCK is increased by the number of doublings performed
//            strategy += 1;                  // Next, we move to the next element of the strategy
//        }
//
//        // At this point, our kernel has order 2. Therefore, we can construct a 2-isogeny
//        xisog_2(C, SPLITTING_POINTS[current]);
//        // Pushing points through 2-isogeny
//        for(local_i = 0; local_i < current; local_i++)
//            xeval_2(SPLITTING_POINTS[local_i], SPLITTING_POINTS[local_i], SPLITTING_POINTS[current]);
//
//        xeval_2(W[0], W[0], SPLITTING_POINTS[current]);
//        xeval_2(W[1], W[1], SPLITTING_POINTS[current]);
//        xeval_2(W[2], W[2], SPLITTING_POINTS[current]);
//
//        BLOCK -= XDBLs[current];   // BLOCK is decreased by the last number of doublings performed
//        XDBLs[current] = 0;        // The last element in the splits are removed
//        current -= 1;              // The number of splits is decreased by one
//    }
//
//    // At this point, our kernel has order 2. Therefore, we can construct a 2-isogeny
//    xisog_2(C, SPLITTING_POINTS[current]);
//    // Pushing points through 2-isogeny
//
//    xeval_2(W[0], W[0], SPLITTING_POINTS[current]);
//    xeval_2(W[1], W[1], SPLITTING_POINTS[current]);
//    xeval_2(W[2], W[2], SPLITTING_POINTS[current]);
//}

__device__ void xisog_2e_1st(proj_t C, proj_t W[3], proj_t P, proj_t A2, limb_t *S2, limb_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_2e_1st()
     *  Output: Overwrites to W the image of the three inputs points under a 2^e-isogeny
     *          with kernel generated by x(P) and stores coefficient A2=(A+2C:4C)
     *		of the codomain curve to C
     * ---------------------------------------------------------------------- */
    limb_t local_j;
    proj_t P2, Q;

    proj_copy(C, A2);
    proj_copy(Q, P);

    for(local_j = 0; local_j < e; local_j++)
    {
        xdble(P2, e - local_j - 1, Q, C);
        xisog_2(C, P2);
        xeval_2(W[0], W[0], P2);
        xeval_2(W[1], W[1], P2);
        xeval_2(W[2], W[2], P2);
        xeval_2(Q, Q, P2);
    }

}

__device__ void full_xisog_2e_2nd(proj_t C, proj_t P, proj_t A2, limb_t *S2, limb_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_2e_2nd()
     *  Input : a Montogmery projective x-coordinate order-2^e point x(P)=P[0]/P[1],
     *          the Montgomery coefficient A2=(A+2C:4C), and a strategy S2
     *  Output: the coefficient A2=(A+2C:4C) of the image curve of a 2^e-isogeny
     *		 with kernel generated by P
     * ---------------------------------------------------------------------- */
    limb_t local_j;
    proj_t P2, Q;

    proj_copy(C, A2);
    proj_copy(Q, P);

    for(local_j = 0; local_j < e; local_j++)
    {
        xdble(P2, e - local_j - 1, Q, C);
        // At this point, our kernel has order 2. Therefore, we can construct a 2-isogeny
        xisog_2(C, P2);
        // Pushing points through 2-isogeny
        xeval_2(Q, Q, P2);
    }
}

__device__ void xisog_2e_2nd(proj_t C, proj_t P, proj_t A2, limb_t *S2, limb_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_2e_2nd()
     *  Input : a Montogmery projective x-coordinate order-2^e point x(P)=P[0]/P[1],
     *          the Montgomery coefficient A2=(A+2C:4C), and a strategy S2
     *  Output: the coefficient A2=(A+2C:4C) of the image curve of a 2^e-isogeny
     *		 with kernel generated by P
     * ---------------------------------------------------------------------- */
    proj_t SPLITTING_POINTS[LOG2OFE];
    //proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A2);

    limb_t strategy = 0,    // Current element of the strategy to be used
    local_i, local_j;

    limb_t BLOCK = 0,       // BLOCK is used for determined when a point has order l
    current = 0;         // At the beginning we have only one point in each split

    //int XDBLs[log2_of_e]; // The current number of doublings performed
    int XDBLs[LOG2OFE]; // The current number of doublings performed

    for(local_j = 0; local_j < (e - 1); local_j++)
    {
        while (BLOCK != (e -  1 - local_j) )
        {
            // A new split will be added
            current += 1;
            // We set the seed of the new split to be computed and saved
            xdble(SPLITTING_POINTS[current], S2[strategy], SPLITTING_POINTS[current - 1], C);
            XDBLs[current] = S2[strategy];  // The number of doublings performed is saved
            BLOCK += S2[strategy];          // BLOCK is increased by the number of doublings performed
            strategy += 1;                  // Next, we move to the next element of the strategy
        }

        // At this point, our kernel has order 2. Therefore, we can construct a 2-isogeny
        xisog_2(C, SPLITTING_POINTS[current]);
        // Pushing points through 2-isogeny
        for(local_i = 0; local_i < current; local_i++)
            xeval_2(SPLITTING_POINTS[local_i], SPLITTING_POINTS[local_i], SPLITTING_POINTS[current]);

        BLOCK -= XDBLs[current];   // BLOCK is decreased by the last number of doublings performed
        XDBLs[current] = 0;        // The last element in the splits are removed
        current -= 1;              // The number of splits is decreased by one
    }

    // At this point, our kernel has order 2. Therefore, we can construct a 2-isogeny
    xisog_2(C, SPLITTING_POINTS[current]);
}

__device__ void xisog_3(proj_t C, proj_t K, proj_t P)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_3()
     *  input : a projective Montgomery x-coordinate order-3 point x(P)=P[0]/P[1]
     *  output: Montgomery coefficient A3=(A+2C:A-2C) of a 3-isogenous curve
     *          and auxiliary constants K[0]=x(P)-z(P), K[1]=x(P)+z(P)
     * ---------------------------------------------------------------------- */
    fp2_t t0, t1, t2, t3, t4;
    
    fp2_sub(K[0], P[0], P[1]);
    fp2_sqr(t0, K[0]);
    fp2_add(K[1], P[0], P[1]);
    fp2_sqr(t1, K[1]);
    fp2_add(t2, t0, t1);
    fp2_add(t3, K[0], K[1]);
    fp2_sqr(t3, t3);
    fp2_sub(t3, t3, t2);
    fp2_add(t2, t1, t3);
    fp2_add(t3, t3, t0);
    fp2_add(t4, t3, t0);
    fp2_add(t4, t4, t4);
    fp2_add(t4, t1, t4);
    fp2_mul(C[1], t2, t4);
    fp2_add(t4, t1, t2);
    fp2_add(t4, t4, t4);
    fp2_add(t4, t0, t4);
    fp2_mul(C[0], t3, t4);
}   // 2M + 3S + 13A

__device__ void xeval_3(proj_t R, proj_t Q, proj_t K)
{
    /* ---------------------------------------------------------------------- *
     *  xeval_3()
     *  input : a projective Montgomery x-coordinate point x(Q)=XQ/ZQ,
     *          auxiliary constants K[0]=x(P)-z(P), K[1]=x(P)+z(P)
     *  output: The image of x(Q) under a 3-isogeny with kernel generated
     *          by P
     * ---------------------------------------------------------------------- */
    fp2_t t0, t1, t2;
    
    fp2_add(t0, Q[0], Q[1]);
    fp2_sub(t1, Q[0], Q[1]);
    fp2_mul(t0, K[0], t0);
    fp2_mul(t1, K[1], t1);
    fp2_add(t2, t0, t1);
    fp2_sub(t0, t1, t0);
    fp2_sqr(t2, t2);
    fp2_sqr(t0, t0);
    fp2_mul(R[0], Q[0], t2);
    fp2_mul(R[1], Q[1], t0);
}   // 4M + 2S + 4A

__device__ void xisog_3e_1st(proj_t C, proj_t W[3], proj_t P, proj_t A3, uint32_t *S3, uint32_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_3e_1st()
     *  input : three projective Weierstrass x-coordinate points W[0], W[1], W[2]
     *          and a Montgomery x-coordinate order-3^e point x(P)=P[0]/P[1] in a 
     *          curve with Montgomery coefficient A3=(A+2C:A-2C), 
     *          and a strategy S3
     *  output: Overwrites the image of the points W under the isogeny generated
     *		by P and writes the codomain coefficient A3=(A+2C:A-2C) to C
     * ---------------------------------------------------------------------- */
    uint8_t log2_of_e, tmp;
    for(tmp = e, log2_of_e = 0; tmp > 0; tmp>>=1, ++log2_of_e);
    log2_of_e *= 2; // In order to ensure each splits is at most of size log2_of_e

    proj_t K;
    // TODO: Check this for GPUs
    proj_t SPLITTING_POINTS[EXPONENT3];
    //proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A3);

    int strategy = 0,      // Current element of the strategy to be used
    local_i, local_j;

    int BLOCK = 0,                 // BLOCK is used for determined when a point has order l
    current = 0;         // At the beginning we have only one point in each split

    //int XTPLs[log2_of_e]; // The current number of triplings performed
    // TODO: Check this for GPUs
    int XTPLs[EXPONENT3]; // The current number of triplings performed

    for(local_j = 0; local_j < (e - 1); local_j++)
    {
        while (BLOCK != (e -  1 - local_j) )
        {
            // A new split will be added
            current += 1;

            // We set the seed of the new split to be computed and saved
            xtple(SPLITTING_POINTS[current], S3[strategy], SPLITTING_POINTS[current - 1], C);
            XTPLs[current] = S3[strategy];  // The number of triplings performed is saved
            BLOCK += S3[strategy];          // BLOCK is increased by the number of triplings performed
            strategy += 1;                  // Next, we move to the next element of the strategy
        }

        // At this point, our kernel has order 3. Therefore, we can construct a 3-isogeny
        xisog_3(C, K, SPLITTING_POINTS[current]);
        // Pushing points through 3-isogeny
        for(local_i = 0; local_i < current; local_i++)
            xeval_3(SPLITTING_POINTS[local_i], SPLITTING_POINTS[local_i],K);

        xeval_3(W[0], W[0], K);
        xeval_3(W[1], W[1], K);
        xeval_3(W[2], W[2], K);

        BLOCK -= XTPLs[current];   // BLOCK is decreased by the last number of triplings performed
        XTPLs[current] = 0;        // The last element in the splits are removed
        current -= 1;              // The number of splits is decreased by one
    }

    // At this point, our kernel has order 3. Therefore, we can construct a 3-isogeny
    xisog_3(C, K, SPLITTING_POINTS[current]);
    // Pushing points through 3-isogeny
    xeval_3(W[0], W[0], K);
    xeval_3(W[1], W[1], K);
    xeval_3(W[2], W[2], K);
}

__device__ void xisog_3e_2nd(proj_t C, proj_t P, proj_t A3, uint32_t *S3, uint32_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_3e_2nd()
     *  input : a Montogmery projective x-coordinate order-3^e point x(P)=P[0]/P[1],
     *          the Montgomery coefficient A3=(A+2C:A-2C), and a strategy S3
     *  output: the coefficient A3=(A+2C:A-2C) of the image curve of a 3^e-isogeny
     *           with kernel generated by P
     * ---------------------------------------------------------------------- */
    uint8_t log2_of_e, tmp;
    for(tmp = e, log2_of_e = 0; tmp > 0; tmp>>=1, ++log2_of_e);
    log2_of_e *= 2; // In order to ensure each splits is at most of size log2_of_e

    proj_t K;
    // TODO: Check this for GPUs
    proj_t SPLITTING_POINTS[EXPONENT3];
    //proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A3);

    int strategy = 0,      // Current element of the strategy to be used
    local_i, local_j;

    int BLOCK = 0,                 // BLOCK is used for determined when a point has order l
    current = 0;         // At the beginning we have only one point in each split

    //int XTPLs[log2_of_e]; // The current number of triplings performed
    // TODO: Check this for GPUs
    int XTPLs[EXPONENT3]; // The current number of triplings performed

    for(local_j = 0; local_j < (e - 1); local_j++)
    {
        while (BLOCK != (e -  1 - local_j) )
        {
            // A new split will be added
            current += 1;

            // We set the seed of the new split to be computed and saved
            xtple(SPLITTING_POINTS[current], S3[strategy], SPLITTING_POINTS[current - 1], C);
            XTPLs[current] = S3[strategy];  // The number of triplings performed is saved
            BLOCK += S3[strategy];          // BLOCK is increased by the number of triplings performed
            strategy += 1;                  // Next, we move to the next element of the strategy
        }

        // At this point, our kernel has order 3. Therefore, we can construct a 3-isogeny
        xisog_3(C, K, SPLITTING_POINTS[current]);
        // Pushing points through 3-isogeny
        for(local_i = 0; local_i < current; local_i++)
            xeval_3(SPLITTING_POINTS[local_i], SPLITTING_POINTS[local_i], K);

        BLOCK -= XTPLs[current];   // BLOCK is decreased by the last number of triplings performed
        XTPLs[current] = 0;        // The last element in the splits are removed
        current -= 1;              // The number of splits is decreased by one
    }

    // At this point, our kernel has order 3. Therefore, we can construct a 3-isogeny
    xisog_3(C, K, SPLITTING_POINTS[current]);
}

void xisog_f_p69(proj_t C, proj_t P, const proj_t A)
{
    printf("xisog_f not implemented\n");
}
