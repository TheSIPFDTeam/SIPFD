#include "api.h"

/* ----------------------------------------------------------------------------- *
 * proj_copy()
 * inputs: a projective Weierstrass x-coordinate point P;
 * output: a copy of the projective Weierstrass x-coordinate point
 * ----------------------------------------------------------------------------- */
void proj_copy(proj_t Q, const proj_t P)
{
    fp2_copy(Q[0], P[0]);
    fp2_copy(Q[1], P[1]);
}

/* ----------------------------------------------------------------------------- *
 * Differential point doubling given the montgomery coefficient A2 = (A+2C:4C)
 * ----------------------------------------------------------------------------- */
void xdbl(proj_t Q, const proj_t P, proj_t const A2)
{
     
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

/* ----------------------------------------------------------------------------- *
 * Differential point addition of P and Q given PQ = P-Q
 * ----------------------------------------------------------------------------- */
void xadd(proj_t R, const proj_t P, proj_t const Q, proj_t const PQ)
{
     
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

/* ----------------------------------------------------------------------------- *
 * xdble()
 * inputs: an integer number 0 <= e <= e2, the projective Montgomery x-coordinate
 *         point x(P), and the Montomery coefficient A2=(A+2C:4C)
 * output: the projective Montgomery x-coordinate point x([2^e]P)
 * ----------------------------------------------------------------------------- */
void xdble(proj_t Q, const digit_t e, const proj_t P, const proj_t A2)
{
    int i;
    proj_t T;
    proj_copy(T, P);
    for(i = 0; i < e; i++)
        xdbl(T, T, A2);

    proj_copy(Q, T);
}

/* ---------------------------------------------------------------------- *
 * xisog_2()
 * input : a projective Montgomery x-coordinate order-2 point P,
 * output: Coefficient A2=(A24plus:C24) of the codomain curve
 * ---------------------------------------------------------------------- */
void xisog_2(proj_t C, proj_t P)
{
     
     fp2_sqr(C[0], P[0]);
     fp2_sqr(C[1], P[1]);
     fp2_sub(C[0], C[1], C[0]);
}	// 2S + 1A

/* ---------------------------------------------------------------------- *
 * xeval_2()
 * input : a projective Montgomery x-coordinate point Q,
 *         a projective Montgomery x-coordinate order-2 point P,
 * output: the image of Q under a 2-isogeny with kernel generated
 *         by P
* ---------------------------------------------------------------------- */
void xeval_2(proj_t R, const proj_t Q, const proj_t P)
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
}	// 4M + 6A

void xisog_f_p69(proj_t C, proj_t P, const proj_t A)
{
    	printf("xisog_f not implemented\n");
}
