void change_curvemodel(proj_t Q, const proj_t P)
{
    /* ---------------------------------------------------------------------- *
     * Switches between constant domains A2 = (A+2C:4C) and A3 = (A+2C:A-2C)
     * ---------------------------------------------------------------------- */
    fp2_copy(Q[0], P[0]);
    fp2_sub(Q[1], P[0], P[1]);
}

void get_A(proj_t A2, proj_t const P, proj_t const Q, proj_t const PQ)
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

void coeff(fp2_t A, proj_t const A2)
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

void j_invariant(fp2_t j, proj_t A2)
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

void set_initial_curve(proj_t A2)
{
    /* ----------------------------------------------------------------------------- *
     * set_initial_curve()
     * output: the Montgomery curve constant A2=(A+2C:4C)=(2:1)
     * ----------------------------------------------------------------------------- */
    fp2_set_one(A2[1]);          // 1
    fp2_add(A2[0], A2[1], A2[1]);  // 2
}

int isinfinity(const proj_t P)
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

int proj_isequal(const proj_t P, const proj_t Q)
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

void proj_copy(proj_t Q, const proj_t P)
{
    /* ----------------------------------------------------------------------------- *
     * proj_copy()
     * inputs: a projective Weierstrass x-coordinate point P;
     * output: a copy of the projective Weierstrass x-coordinate point
     * ----------------------------------------------------------------------------- */
    fp2_copy(Q[0], P[0]);
    fp2_copy(Q[1], P[1]);
}

void xdbl(proj_t Q, const proj_t P, proj_t const A2)
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

void xadd(proj_t R, const proj_t P, proj_t const Q, proj_t const PQ)
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


void xtpl(proj_t Q, const proj_t P, proj_t const A3)
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

void xdbladd(proj_t R, proj_t S, proj_t const P, proj_t const Q, proj_t const PQ, proj_t const A2)
{
    /* ----------------------------------------------------------------------------- *
     * Computes x([2]P) and x(P + Q) given x(P-Q) and the Montgommery
     * coefficient A2 = (A+2C:4C)
     * ----------------------------------------------------------------------------- */
     proj_t tmp;
     xadd(tmp, P, Q, PQ);
     xdbl(R, P, A2);
     proj_copy(S, tmp);
    
    /*
    fp2_t t0, t1, t2, rx, rz, sx, sz;
    // ---
    fp2_add(t0, P[0], P[1]);
    fp2_sub(t1, P[0], P[1]);
    fp2_sqr(rx, t0);
    fp2_sub(t2, Q[0], Q[1]);
    fp2_add(sx, Q[0], Q[1]);
    fp2_mul(t0, t0, t2);
    fp2_sqr(rz, t1);
    // ---
    fp2_mul(t1, t1, sx);
    fp2_sub(t2, rx, rz);
    fp2_mul(rx, rx, rz);
    fp2_mul(sx, a24, t2);
    fp2_sub(sz, t0, t1);
    fp2_add(rz, sx, rz);
    fp2_add(sx, t0, t1);
    // ---
    fp2_sqr(sz, sz);
    fp2_sqr(sx, sx);
    fp2_mul(sz, PQ[0], sz);
    fp2_mul(S[0], PQ[1], sx);
    fp2_mul(R[1], rz, t2);
    fp2_copy(S[1], sz);
    fp2_copy(R[0], rx);
    */
}

void ladder3pt(proj_t R, uint64_t const m, proj_t const P, proj_t const Q, proj_t const PQ, proj_t const A2, digit_t const e)
{
    /* ----------------------------------------------------------------------------- *
     * Computes x(P + [m]Q) given the Montgommery coefficient A2 = (A+2C:4C)
     * ----------------------------------------------------------------------------- */
     
    proj_t X0, X1, X2;
    proj_copy(X0, Q);
    proj_copy(X1, P);
    proj_copy(X2, PQ);


    int j;
    digit_t flag;
    flag = 1;
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

void ladder3pt_long(proj_t R, fp_t const m, proj_t const P, proj_t const Q, proj_t const PQ, proj_t const A2)
{
    /* ----------------------------------------------------------------------------- *
     * Computes x(P + [m]Q) given the Montgommery coefficient A2 = (A+2C:4C)
     * ----------------------------------------------------------------------------- */
     
    proj_t X0, X1, X2;
    proj_copy(X0, Q);
    proj_copy(X1, P);
    proj_copy(X2, PQ);


    int i, j;
    digit_t flag;
    for(i = 0; i < NWORDS_FIELD; i++)
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

void xdble(proj_t Q, const digit_t e, const proj_t P, const proj_t A2)
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

void xtple(proj_t Q, const digit_t e, const proj_t P, const proj_t A3)
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

void xmul(proj_t Q, const digit_t k, const proj_t P, const proj_t A2)
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
    digit_t l = k, tmp = 0;
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

void make_affine(proj_t R, const proj_t P, const fp2_t A){
    /* ---------------------------------------------------------------------- *
    Given the affine montfommery coefficient, transforms a
    projective point P = (x:z) to an affine point R = (x, y)
    * ---------------------------------------------------------------------- */
   fp2_t x, t1, t2;
   fp2_copy(t1, P[1]);
   fp2_inv(t1);
   fp2_mul(x, P[0], t1);
   fp2_mul(t1, A, x);
   fp2_sqr(t2, x);
   fp2_add(R[0], t1, t2);
   fp2_mul(R[0], R[0], x);
   fp2_add(R[0], R[0], x);
   fp2_issquare(R[1], R[0]);
   fp2_copy(R[0], x);
}


void random_affine_point(proj_t P, const fp2_t A)
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
    	fp2_random(P[0]);
	fp2_sqr(tmp, P[0]);
	fp2_mul(P[1], A, tmp);
	fp2_add(P[1], P[1], P[0]);
	fp2_mul(tmp, tmp, P[0]);
	fp2_add(P[1], P[1], tmp);

        if(fp2_issquare(P[1], P[1]) == 1)
            break;
    } 
}

void difference_point(proj_t PQ, proj_t P, proj_t Q, const fp2_t A)
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

int isfull_order(proj_t P2, proj_t P3, const proj_t P, const proj_t A2)
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

void init_basis(proj_t P, proj_t Q, proj_t PQ, const proj_t A2)
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
        random_affine_point(P, A);
        random_affine_point(Q, A);
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

void frobenius_basis(proj_t P, proj_t Q, proj_t PQ, const proj_t A2, const int e)
{
    /* ---------------------------------------------------------------------- *
     * input : the Montgomery coefficient A2=(A+2C:4C) and exponent e
     * output: the projective x(P), x(Q), and x(P-Q) for a basis of the 2^e-torsion
     *         such that P is in E[pi+1] and Q in E[pi-1]
     * ---------------------------------------------------------------------- */
    proj_t P2, Q2, P0, Q0, A3;
    fp2_t tmp, A;
    int i;

    // Frobenius basis does not exist for e = e2
    assert(e < EXPONENT2);

    // Generating the affine Montgomery constant A
    fp2_add(tmp, A2[0], A2[0]);
    fp2_sub(tmp, tmp, A2[1]);
    fp2_add(tmp, tmp, tmp);
    fp2_copy(A, A2[1]);
    fp2_inv(A);
    fp2_mul(A, A, tmp);
    
    while( 1 )
    {
        while( 1 )
        {
            fp2_random(P[0]);
            fp2_random(Q[0]);

            for(i = 0; i < NWORDS_FIELD; i++)
            {
                P[0][1][i] = 0;
                Q[0][1][i] = 0;
            }

            fp2_add(tmp, P[0], A);
            fp2_mul(tmp, tmp, P[0]);
            fp2_mul(tmp, tmp, P[0]);
            fp2_add(tmp, tmp, P[0]);
            fp2_issquare(P[1], tmp);

            fp2_add(tmp, Q[0], A);
            fp2_mul(tmp, tmp, Q[0]);
            fp2_mul(tmp, tmp, Q[0]);
            fp2_add(tmp, tmp, Q[0]);
            fp2_issquare(Q[1], tmp);

            if( fp_iszero(P[1][0]) && fp_iszero(Q[1][1]) )
                break;
        }

        difference_point(PQ, P, Q, A);
        fp2_set_one(P[1]);
        fp2_set_one(Q[1]);
        change_curvemodel(A3, A2);

        xtple(P, EXPONENT3, P, A3);
        xtple(Q, EXPONENT3, Q, A3);
        xtple(PQ, EXPONENT3, PQ, A3);

        xmul(P, COFACTOR, P, A2);
        xmul(Q, COFACTOR, Q, A2);
        xmul(PQ, COFACTOR, PQ, A2);

        xdble(P, EXPONENT2 - e - 1, P, A2);
        xdble(Q, EXPONENT2 - e - 1, Q, A2);
        xdble(PQ, EXPONENT2 - e - 1, PQ, A2);
        
        xdble(P2, e-1, P, A2);
        xdble(Q2, e-1, Q, A2);
        xdbl(P0, P2, A2);
        xdbl(Q0, Q2, A2);

        if(!isinfinity(P2) && !isinfinity(Q2) && isinfinity(P0) && isinfinity(Q0) && fp2_iszero(Q2[0]) && !fp2_iszero(P2[0]) )
            break;
    }
}

void xisog_2(proj_t C, proj_t P)
{
    /* ---------------------------------------------------------------------- *
     * xisog_2()
     * input : a projective Montgomery x-coordinate order-2 point P,
     * output: Coefficient A2=(A24plus:C24) of the codomain curve
     * ---------------------------------------------------------------------- */
     
     fp2_sqr(C[0], P[0]);
     fp2_sqr(C[1], P[1]);
     fp2_sub(C[0], C[1], C[0]);
}	// 2S + 1A

void xeval_2(proj_t R, const proj_t Q, const proj_t P)
{
    /* ---------------------------------------------------------------------- *
     * xeval_2()
     * input : a projective Montgomery x-coordinate point Q,
     *         a projective Montgomery x-coordinate order-2 point P,
     * output: the image of Q under a 2-isogeny with kernel generated
     *         by P
    * ---------------------------------------------------------------------- */
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

void xisog_2e_1st(proj_t C, proj_t W[3], const proj_t P, const proj_t A2, const digit_t *S2, const digit_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_2e_1st()
     *  Input : three projective Montgomery x-coordinate points: W[1], W[2], W[3],
     *          Montgomery x-coordinate order-2^e point x(P)=P[0]/P[1], 
     *          Montgomery coefficient A2=(A+2C:4C), a strategy S2,
     *          and a positive integer e;
     *  Output: Overwrites to W the image of the three inputs points under a 2^e-isogeny
     *          with kernel generated by x(P) and stores coefficient A2=(A+2C:4C)
     *		    of the codomain curve to C
     * ---------------------------------------------------------------------- */
    uint8_t log2_of_e, tmp;
    for(tmp = e, log2_of_e = 0; tmp > 0; tmp>>=1, ++log2_of_e);
    log2_of_e *= 2; // In order to ensure each splits is at most size log2_of_e

    proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A2);

    int strategy = 0,    // Current element of the strategy to be used
    local_i, local_j;

    int BLOCK = 0,       // BLOCK is used for determined when a point has order l
    current = 0;         // At the beginning we have only one point in each split

    int XDBLs[log2_of_e]; // The current number of doublings performed

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

        xeval_2(W[0], W[0], SPLITTING_POINTS[current]);
        xeval_2(W[1], W[1], SPLITTING_POINTS[current]);
        xeval_2(W[2], W[2], SPLITTING_POINTS[current]);

        BLOCK -= XDBLs[current];   // BLOCK is decreased by the last number of doublings performed
        XDBLs[current] = 0;        // The last element in the splits are removed
        current -= 1;              // The number of splits is decreased by one
    }

    // At this point, our kernel has order 2. Therefore, we can construct a 2-isogeny
    xisog_2(C, SPLITTING_POINTS[current]);
    // Pushing points through 2-isogeny
    xeval_2(W[0], W[0], SPLITTING_POINTS[current]);
    xeval_2(W[1], W[1], SPLITTING_POINTS[current]);
    xeval_2(W[2], W[2], SPLITTING_POINTS[current]);
}

void xisog_2e_2nd(proj_t C, const proj_t P, const proj_t A2, const digit_t *S2, const digit_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_2e_2nd()
     *  Input : a Montogmery projective x-coordinate order-2^e point x(P)=P[0]/P[1],
     *          the Montgomery coefficient A2=(A+2C:4C), and a strategy S2
     *  Output: the coefficient A2=(A+2C:4C) of the image curve of a 2^e-isogeny
     *		    with kernel generated by P
     * ---------------------------------------------------------------------- */
    uint8_t log2_of_e, tmp;
    for(tmp = e, log2_of_e = 0; tmp > 0; tmp>>=1, ++log2_of_e);
    log2_of_e *= 2; // In order to ensure each splits is at most size log2_of_e

    proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A2);

    int strategy = 0,    // Current element of the strategy to be used
    local_i, local_j;

    int BLOCK = 0,       // BLOCK is used for determined when a point has order l
    current = 0;         // At the beginning we have only one point in each split

    int XDBLs[log2_of_e]; // The current number of doublings performed

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

void xisog_4(proj_t C, fp2_t K[3], proj_t P)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_4()
     *  input : a projective Montgomery x-coordinate order-4 point x(P)=P[0]/P[1]
     *  output: Montgomery coefficient A3=(A+2C:4C) of a 4-isogenous curve
     *          and auxiliary constants K[0]=x(P)-z(P), K[1]=x(P)+z(P), K[2] = z(P)^2
     * ---------------------------------------------------------------------- */
    fp2_sub(K[1], P[0], P[1]);
    fp2_add(K[2], P[0], P[1]);
    fp2_sqr(K[0], P[1]);
    fp2_add(K[0], K[0], K[0]);
    fp2_sqr(C[1], K[0]);
    fp2_add(K[0], K[0], K[0]);
    fp2_sqr(C[0], P[0]);
    fp2_add(C[0], C[0], C[0]);
    fp2_sqr(C[0], C[0]);
}   // 4S + 5a

void xeval_4(proj_t R, const proj_t Q, const fp2_t K[3])
{
    /* ---------------------------------------------------------------------- *
     *  xeval_4()
     *  input : a projective Montgomery x-coordinate point x(Q)=XQ/ZQ,
     *          auxiliary constants K[0]=x(P)-z(P), K[1]=x(P)+z(P), K[2]=z(P)^2
     *  output: The image of x(Q) under a 4-isogeny with kernel generated
     *          by P
     * ---------------------------------------------------------------------- */
    fp2_t t0, t1;

    fp2_add(t0, Q[0], Q[1]);
    fp2_sub(t1, Q[0], Q[1]);
    fp2_mul(R[0], t0, K[1]);
    fp2_mul(R[1], t1, K[2]);
    fp2_mul(t0, t0, t1);
    fp2_mul(t0, t0, K[0]);
    fp2_add(t1, R[0], R[1]);
    fp2_sub(R[1], R[0], R[1]);
    fp2_sqr(t1, t1);
    fp2_sqr(R[1], R[1]);
    fp2_add(R[0], t0, t1);
    fp2_sub(t0, R[1], t0);
    fp2_mul(R[0], R[0], t1);
    fp2_mul(R[1], R[1], t0);
} // 6M + 2S + 6a

void xisog_2e_1st_(proj_t C, proj_t W[3], const proj_t P, const proj_t A2, const digit_t *S2, const digit_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_2e_1st_()
     *  Input : three projective Montgomery x-coordinate points: W[1], W[2], W[3],
     *          Montgomery x-coordinate order-2^e point x(P)=P[0]/P[1],
     *          Montgomery coefficient A2=(A+2C:4C), a strategy S2,
     *          and a positive integer e;
     *  Output: Overwrites to W the image of the three inputs points under a 2^e-isogeny
     *          with kernel generated by x(P) and stores coefficient A2=(A+2C:4C)
     *		    of the codomain curve to C
     * Note: This function use 4-isogenies
     * ---------------------------------------------------------------------- */
    uint8_t log2_of_e, tmp;
    digit_t e_half = e>>1;
    for(tmp = e_half, log2_of_e = 0; tmp > 0; tmp>>=1, ++log2_of_e);
    log2_of_e *= 2; // In order to ensure each splits is at most size log2_of_e

    fp2_t K[3];
    proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A2);

    int strategy = 0,    // Current element of the strategy to be used
    local_i, local_j;

    int BLOCK = 0,       // BLOCK is used for determined when a point has order l
    current = 0;         // At the beginning we have only one point in each split

    int XDBLs[log2_of_e]; // The current number of doublings performed

    if (e % 2 == 1)
    {
        // Same as current SIDH code from the NIST competition
        xdble(SPLITTING_POINTS[1], e - 1, SPLITTING_POINTS[0], C);
        xisog_2(C, SPLITTING_POINTS[1]);
        xeval_2(W[0], W[0], SPLITTING_POINTS[1]);
        xeval_2(W[1], W[1], SPLITTING_POINTS[1]);
        xeval_2(W[2], W[2], SPLITTING_POINTS[1]);
        xeval_2(SPLITTING_POINTS[0], SPLITTING_POINTS[0], SPLITTING_POINTS[1]);
    }

    for(local_j = 0; local_j < (e_half - 1); local_j++)
    {
        while (BLOCK != (e_half -  1 - local_j) )
        {
            // A new split will be added
            current += 1;
            // We set the seed of the new split to be computed and saved
            xdble(SPLITTING_POINTS[current], 2*S2[strategy], SPLITTING_POINTS[current - 1], C);
            XDBLs[current] = S2[strategy];  // The number of doublings performed is saved
            BLOCK += S2[strategy];          // BLOCK is increased by the number of doublings performed
            strategy += 1;                  // Next, we move to the next element of the strategy
        }

        // At this point, our kernel has order 4. Therefore, we can construct a 4-isogeny
        xisog_4(C, K, SPLITTING_POINTS[current]);
        // Pushing points through 4-isogeny
        for(local_i = 0; local_i < current; local_i++)
            xeval_4(SPLITTING_POINTS[local_i], SPLITTING_POINTS[local_i], K);

        xeval_4(W[0], W[0], K);
        xeval_4(W[1], W[1], K);
        xeval_4(W[2], W[2], K);

        BLOCK -= XDBLs[current];   // BLOCK is decreased by the last number of doublings performed
        XDBLs[current] = 0;        // The last element in the splits are removed
        current -= 1;              // The number of splits is decreased by one
    }

    // At this point, our kernel has order 4. Therefore, we can construct a 4-isogeny
    xisog_4(C, K, SPLITTING_POINTS[current]);
    // Pushing points through 4-isogeny
    xeval_4(W[0], W[0], K);
    xeval_4(W[1], W[1], K);
    xeval_4(W[2], W[2], K);
}

void xisog_2e_2nd_(proj_t C, const proj_t P, const proj_t A2, const digit_t *S2, const digit_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_2e_2nd_()
     *  Input : a Montogmery projective x-coordinate order-2^e point x(P)=P[0]/P[1],
     *          the Montgomery coefficient A2=(A+2C:4C), and a strategy S2
     *  Output: the coefficient A2=(A+2C:4C) of the image curve of a 2^e-isogeny
     *		    with kernel generated by P
     * Note: This function use 4-isogenies
     * ---------------------------------------------------------------------- */
    uint8_t log2_of_e, tmp;
    digit_t e_half = e>>1;
    for(tmp = e_half, log2_of_e = 0; tmp > 0; tmp>>=1, ++log2_of_e);
    log2_of_e *= 2; // In order to ensure each splits is at most size log2_of_e

    fp2_t K[3];
    proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A2);

    int strategy = 0,    // Current element of the strategy to be used
    local_i, local_j;

    int BLOCK = 0,       // BLOCK is used for determined when a point has order l
    current = 0;         // At the beginning we have only one point in each split

    int XDBLs[log2_of_e]; // The current number of doublings performed

    if (e % 2 == 1)
    {
        // Same as current SIDH code from the NIST competition
        xdble(SPLITTING_POINTS[1], e - 1, SPLITTING_POINTS[0], C);
        xisog_2(C, SPLITTING_POINTS[1]);
        xeval_2(SPLITTING_POINTS[0], SPLITTING_POINTS[0], SPLITTING_POINTS[1]);
    }

    for(local_j = 0; local_j < (e_half - 1); local_j++)
    {
        while (BLOCK != (e_half -  1 - local_j) )
        {
            // A new split will be added
            current += 1;
            // We set the seed of the new split to be computed and saved
            xdble(SPLITTING_POINTS[current], 2*S2[strategy], SPLITTING_POINTS[current - 1], C);
            XDBLs[current] = S2[strategy];  // The number of doublings performed is saved
            BLOCK += S2[strategy];          // BLOCK is increased by the number of doublings performed
            strategy += 1;                  // Next, we move to the next element of the strategy
        }

        // At this point, our kernel has order 4. Therefore, we can construct a 4-isogeny
        xisog_4(C, K, SPLITTING_POINTS[current]);
        // Pushing points through 4-isogeny
        for(local_i = 0; local_i < current; local_i++)
            xeval_4(SPLITTING_POINTS[local_i], SPLITTING_POINTS[local_i], K);

        BLOCK -= XDBLs[current];   // BLOCK is decreased by the last number of doublings performed
        XDBLs[current] = 0;        // The last element in the splits are removed
        current -= 1;              // The number of splits is decreased by one
    }

    // At this point, our kernel has order 4. Therefore, we can construct a 4-isogeny
    xisog_4(C, K, SPLITTING_POINTS[current]);
}

void xisog_3(proj_t C, proj_t K, proj_t P)
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

void xeval_3(proj_t R, const proj_t Q, const proj_t K)
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

void xisog_3e_1st(proj_t C, proj_t W[3], const proj_t P, const proj_t A3, const digit_t *S3, const digit_t e)
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
    proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A3);

    int strategy = 0,      // Current element of the strategy to be used
    local_i, local_j;

    int BLOCK = 0,                 // BLOCK is used for determined when a point has order l
    current = 0;         // At the beginning we have only one point in each split

    int XTPLs[log2_of_e]; // The current number of triplings performed

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

void xisog_3e_2nd(proj_t C, const proj_t P, const proj_t A3, const digit_t *S3, const digit_t e)
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
    proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A3);

    int strategy = 0,      // Current element of the strategy to be used
    local_i, local_j;

    int BLOCK = 0,                 // BLOCK is used for determined when a point has order l
    current = 0;         // At the beginning we have only one point in each split

    int XTPLs[log2_of_e]; // The current number of triplings performed

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
