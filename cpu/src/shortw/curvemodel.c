void change_curvemodel(proj_t Q, const proj_t P)
{
    fp2_t t = {0};
    fp2_copy(Q[1], P[1]);   // Z
    fp2_add(t, Q[1], Q[1]); // 2Z
    fp2_sub(Q[0], P[0], t); // X - 2Z
}

void get_A(proj_t B, proj_t const P, proj_t const Q, proj_t const PQ)
{
    //fp2_t t = {0};
}

void j_invariant(fp2_t j, proj_t A)
{
    /* ----------------------------------------------------------------------------- *
     * j_invariant()
     * input : the short Weierstrass curve constants A[0] := A, and A[1] := B;
     * output: the j-invariant of E, which is equal to
     *                                                                  4A^3
     *                                                       1728 x ------------
     *                                                              4A^3 + 27B^2
     * ----------------------------------------------------------------------------- */
    fp2_t b2_27, a3_4, t;

    fp2_sqr(b2_27, A[1]);       // B^2
    fp2_add(t, b2_27, b2_27);   // 2(B^2)
    fp2_add(j, t, t);           // 4(B^2)
    fp2_add(j, j, j);           // 8(B^2)
    fp2_add(a3_4, j, j);        // 16(B^2)
    fp2_add(j, a3_4, j);        // 24(B^2)
    fp2_add(t, t, j);           // 26(B^2)
    fp2_add(b2_27, b2_27, t);   // 27(B^2)

    fp2_sqr(a3_4, A[0]);        // A^2
    fp2_mul(a3_4, A[0], a3_4);  // A^3
    fp2_add(a3_4, a3_4, a3_4);  // 2(A^3)
    fp2_add(a3_4, a3_4, a3_4);  // 4(A^3)

    fp2_add(t, b2_27, a3_4);    // [4(A^3) + 27(B^2)]
    fp2_copy(j, t);
    fp2_inv(j);                 // 1 / [4(A^3) + 27(B^2)]
    fp2_mul(t, j, a3_4);        // [4(A^3)] / [4(A^3) + 27(B^2)]

    fp2_add(t, t, t);           //    2([4(A^3)] / [4(A^3) + 27(B^2)])
    fp2_add(t, t, t);           //    4([4(A^3)] / [4(A^3) + 27(B^2)])
    fp2_add(t, t, t);           //    8([4(A^3)] / [4(A^3) + 27(B^2)])
    fp2_add(t, t, t);           //   16([4(A^3)] / [4(A^3) + 27(B^2)])
    fp2_add(t, t, t);           //   32([4(A^3)] / [4(A^3) + 27(B^2)])
    fp2_add(t, t, t);           //   64([4(A^3)] / [4(A^3) + 27(B^2)])
    fp2_add(j, t, t);           //  128([4(A^3)] / [4(A^3) + 27(B^2)])
    fp2_add(j, j, t);           //  192([4(A^3)] / [4(A^3) + 27(B^2)])
    fp2_add(t, j, j);           //  384([4(A^3)] / [4(A^3) + 27(B^2)])
    fp2_add(t, t, t);           //  768([4(A^3)] / [4(A^3) + 27(B^2)])
    fp2_add(t, t, t);           // 1536([4(A^3)] / [4(A^3) + 27(B^2)])
    fp2_add(j, t, j);           // 1728([4(A^3)] / [4(A^3) + 27(B^2)])
}   // 1I + 2M + 2S + 21a

void set_initial_curve(proj_t A)
{
    /* ----------------------------------------------------------------------------- *
     * set_initial_curve()
     * output: the short Weierstrass curve constants A[0] := A, and A[1] := B;
     * ----------------------------------------------------------------------------- */
    fp2_t tmp = {0};
    fp2_set_one(A[0]);      // A = 1
    fp2_copy(A[1], tmp);    // B = 0
}

int isinfinity(const proj_t P)
{
    /* ----------------------------------------------------------------------------- *
     * isinfinity()
     * inputs: the projective Weierstrass x-coordinate point x(P)=XP/ZP;
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
     * inputs: the projective Weierstrass x-coordinate points x(P)=XP/ZP and
     *         x(Q)=XQ/XQ;
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
     * inputs: a projective Weierstrass x-coordinate point x(P)=XP/ZP;
     * output: a copy of the projective Weierstrass x-coordinate point x(P)
     * ----------------------------------------------------------------------------- */
    fp2_copy(Q[0], P[0]);
    fp2_copy(Q[1], P[1]);
}

void xdbl(proj_t Q, const proj_t P, const proj_t A)
{
    /* ----------------------------------------------------------------------------- *
     * xdbl()
     * inputs: the projective Weierstrass x-coordinate point x(P), and the short
     *         Weierstrass curve constants A[0] := A, and A[1] := B;
     * output: the projective Weierstrass x-coordinate point x([2]P)
     * ----------------------------------------------------------------------------- */
    if (isinfinity(P) == 1)
        proj_copy(Q, P);
    else
    {
        fp2_t XX, ZZ, aZZ, b2, b4, t;
        fp2_add(b2, A[1], A[1]);    // 2B
        fp2_add(b4, b2, b2);        // 4B

        fp2_sqr(XX, P[0]);          // XP ^ 2
        fp2_sqr(ZZ, P[1]);          // ZP ^ 2
        fp2_add(t, P[0], P[1]);     // XP + ZP
        fp2_sqr(t, t);              //   (XP + ZP) ^ 2
        fp2_sub(t, t, XX);          //   (XP + ZP) ^ 2 - (XP ^ 2)
        fp2_sub(t, t, ZZ);          //   (XP + ZP) ^ 2 - (XP ^ 2) - (ZP ^ 2)
        fp2_add(t, t, t);           // 2((XP + ZP) ^ 2 - (XP ^ 2) - (ZP ^ 2))
        fp2_mul(aZZ, A[0], ZZ);     // A * (ZP ^ 2)

        fp2_mul(Q[1], b2, t);       // (2B) * (2((XP + ZP) ^ 2 - (XP ^ 2) - (ZP ^ 2)))
        fp2_mul(Q[1], Q[1], ZZ);    // (2B) * (2((XP + ZP) ^ 2 - (XP ^ 2) - (ZP ^ 2))) * (ZP ^ 2)
        fp2_sub(Q[0], XX, aZZ);     //  (XP ^ 2) - (A * ZP ^ 2)
        fp2_sqr(Q[0], Q[0]);        // [(XP ^ 2) - (A * ZP ^ 2)]^2
        fp2_sub(Q[0], Q[0], Q[1]);  // [(XP ^ 2) - (A * ZP ^ 2)]^2 - [(2B) * (2((XP + ZP) ^ 2 - (XP ^ 2) - (ZP ^ 2))) * (ZP ^ 2)]

        fp2_sqr(ZZ, ZZ);            // (ZP ^ 4)
        fp2_mul(ZZ, b4, ZZ);        // (4B * ZP ^ 4)
        fp2_add(Q[1], XX, aZZ);     //  XP ^ 2 + (A * ZP ^ 2)
        fp2_mul(Q[1], Q[1], t);     // [XP ^ 2 + (A * ZP ^ 2)] * [2((XP + ZP) ^ 2 - (XP ^ 2) - (ZP ^ 2))]
        fp2_add(Q[1], Q[1], ZZ);    // [XP ^ 2 + (A * ZP ^ 2)] * [2((XP + ZP) ^ 2 - (XP ^ 2) - (ZP ^ 2))] + (4B * ZP ^ 4)
    }
}   // 5M + 5S + 10a

void xadd(proj_t R, const proj_t P, const proj_t Q, const proj_t PQ, const proj_t A)
{
    /* ----------------------------------------------------------------------------- *
     * xadd()
     * inputs: the projective Weierstrass x-coordinate points x(P), x(Q), and x(P - Q),
     *         and the short Weierstrass curve constants A[0] := A, and A[1] := B;
     * output: the projective Weierstrass x-coordinate point x(P+Q);
     * ----------------------------------------------------------------------------- */
    if (isinfinity(PQ) == 1)
        xdbl(R, P, A);
    else if (isinfinity(Q) == 1)
        proj_copy(R, P);
    else if (isinfinity(P) == 1)
        proj_copy(R, Q);
    else
    {
        fp2_t XPXQ, ZPZQ, XPZQ, ZPXQ, t0, t1;

        fp2_mul(XPXQ, P[0], Q[0]);  // XP * XQ
        fp2_mul(ZPZQ, P[1], Q[1]);  // ZP * ZQ
        fp2_mul(XPZQ, P[0], Q[1]);  // XP * ZQ
        fp2_mul(ZPXQ, P[1], Q[0]);  // ZP * XQ

        fp2_mul(t0, A[0], ZPZQ);    //            A * ZP * ZQ
        fp2_sub(t0, XPXQ, t0);      //  XP * XQ - A * ZP * ZQ
        fp2_sqr(t0, t0);            // (XP * XQ - A * ZP * ZQ)^2

        fp2_add(t1, ZPXQ, XPZQ);    //   ZP * XQ + XP * ZQ
        fp2_mul(t1, t1, ZPZQ);      //  (ZP * XQ + XP * ZQ) * (ZP * ZQ)
        fp2_mul(t1, A[1], t1);      //  (ZP * XQ + XP * ZQ) * (ZP * ZQ) * B
        fp2_add(t1, t1, t1);        // 2(ZP * XQ + XP * ZQ) * (ZP * ZQ) * B
        fp2_add(t1, t1, t1);        // 4(ZP * XQ + XP * ZQ) * (ZP * ZQ) * B

        fp2_sub(t0, t0, t1);        // (XP * XQ - B * ZP * ZQ)^2 - 4(ZP * XQ + XP * ZQ) * (ZP * ZQ) * B

        fp2_sub(t1, ZPXQ, XPZQ);    //  ZP * XQ - XP * ZQ
        fp2_sqr(t1, t1);            // (ZP * XQ - XP * ZQ)^2

        // The next two lines are required for allowing
        // P <- P + Q, Q <- P + Q, PQ <- P + Q where PQ = P - Q
        fp2_copy(XPXQ, PQ[0]);      // PQ[0] := X_{P-Q}
        fp2_copy(ZPZQ, PQ[1]);      // PQ[1] := Z_{P-Q}

        fp2_mul(R[0], ZPZQ, t0);   // ZPQ * [(XP * XQ - A * ZP * ZQ)^2 - 4(ZP * XQ + XP * ZQ) * (ZP * ZQ) * B]
        fp2_mul(R[1], XPXQ, t1);   // XPQ * (ZP * XQ - XP * ZQ)^2
    }
}   // 9M + 2S + 6a

void xtpl(proj_t Q, const proj_t P, const proj_t A)
{
    /* ----------------------------------------------------------------------------- *
     * xtpl()
     * inputs: the projective Weierstrass x-coordinate point x(P), and the short
     *         Weierstrass curve constants A[0] := A, and A[1] := B;
     * output: the projective Weierstrass x-coordinate point x([3]P)
     * ----------------------------------------------------------------------------- */
    proj_t P2;
    xdbl(P2, P, A);
    xadd(Q, P2, P, P, A);
}   // 14M + 7S + 16a

void xdbladd(proj_t R, proj_t S, const proj_t P, const proj_t Q, const proj_t PQ, const proj_t A)
{
    /* ----------------------------------------------------------------------------- *
     * xdbladd()
     * inputs: the projective Weierstrass x-coordinate points x(P), x(Q), and x(P - Q),
     *         and the short Weierstrass curve constants A[0] := A, and A[1] := B;
     * output: the projective Weierstrass x-coordinate points x([2]P) and x(P+Q);
     * ----------------------------------------------------------------------------- */
    xadd(S, P, Q, PQ, A);
    xdbl(R, P, A);
}   // 14M + 7S + 14a

void ladder3pt(proj_t R, const fp_t m, const proj_t P, const proj_t Q, const proj_t PQ, const proj_t A)
{
    /* ----------------------------------------------------------------------------- *
     * Ladder3pt()
     * inputs: an integer number 0 <= m <= (p+1), the projective Weierstrass x-coordinate
     *         points x(P), x(Q), and x(P - Q), and the short Weierstrass curve constants
     *         A[0] := A, and A[1] := B;
     * output: the projective Weierstrass x-coordinate point x(P + [m]Q);
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
                xdbladd(X0, X1, X0, X1, X2, A);
            else
                xdbladd(X0, X2, X0, X2, X1, A);

            flag <<= 1;
        }
    }
    proj_copy(R, X1);
}

void xdble(proj_t Q, const digit_t e, const proj_t P, const proj_t A)
{
    /* ----------------------------------------------------------------------------- *
     * xdble()
     * inputs: an integer number 0 <= e <= e2, the projective Weierstrass x-coordinate
     *         point x(P), and the short Weierstrass curve constants A[0] := A, and
     *         A[1] := B;
     * output: the projective Weierstrass x-coordinate point x([2^e]P)
     * ----------------------------------------------------------------------------- */
    int i;
    proj_t T;
    proj_copy(T, P);
    for(i = 0; i < e; i++)
        xdbl(T, T, A);

    proj_copy(Q, T);
}

void xtple(proj_t Q, const digit_t e, const proj_t P, const proj_t A)
{
    /* ----------------------------------------------------------------------------- *
     * xtple()
     * inputs: an integer number 0 <= e <= e3, the projective Weierstrass x-coordinate
     *         point x(P), and the short Weierstrass curve constants A[0] := A, and
     *         A[1] := B;
     * output: the projective Weierstrass x-coordinate point x([3^e]P)
     * ----------------------------------------------------------------------------- */
    int i;
    proj_t T;
    proj_copy(T, P);
    for(i = 0; i < e; i++)
        xtpl(T, T, A);

    proj_copy(Q, T);
}

void xmul(proj_t Q, const digit_t k, const proj_t P, const proj_t A)
{
    /* ----------------------------------------------------------------------------- *
     * xmul()
     * inputs: a projective Weierstrass x-coordinate point x(P), the short Weierstrass
     *  curve constants A[0] := A, and A[1] := B, and an integer number
     *  0 <= k < 2^64;
     * output: the projective Weierstrass x-coordinate point x([k]P)
     * ----------------------------------------------------------------------------- */
    if (k == 1)
    {
        proj_copy(Q, P);
        return ;
    }
    proj_t R[2];

    // Initial 3-tuple of points
    proj_copy(R[0], P);    // P
    xdbl(R[1], P, A);      // [2]P

    // Bit-reverse of k
    digit_t l = k, tmp = 0;
    while(l > 1)
    {
        tmp = (tmp << 1) ^ (l & 0x1);
        l >>= 1;
    }

    while(tmp > 0)
    {
        xadd(R[(tmp & 0x1) ^ 0x1], R[(tmp & 0x1) ^ 0x1], R[tmp & 0x1], P, A);
        xdbl(R[(tmp & 0x1)], R[(tmp & 0x1)], A);

        tmp >>= 1;
    }
    proj_copy(Q, R[0]);
}   // Cost ~ 2*Ceil[log_2(k)]*(4M + 2S)

void random_affine_point(proj_t P, const proj_t A)
{
    /* ---------------------------------------------------------------------- *
    random_point()
    input : the Weierstrass curve constants A[0] := A, and A[1] := B;
    output: a random affine point of the Weierstrass curve
    E : y^2 = x^3 + Ax^ + B
    * ---------------------------------------------------------------------- */
    fp2_t tmp;

    while(1)
    {
        fp2_random(P[0]);			// x <- random element of F_{p^2}

        fp2_sqr(tmp, P[0]);         // x^2
        fp2_add(tmp, tmp, A[0]);	// x^2 + A
        fp2_mul(tmp, P[0], tmp);	// x^3 + Ax
        fp2_add(tmp, tmp, A[1]);	// x^3 + Ax + B

        if(fp2_issquare(P[1], tmp) == 1)
            break;
    } // Cost per iteration : 1M + 1S + 2a
}     // ~ 2M + 2S + 4a

void difference_point(proj_t PQ, proj_t P, proj_t Q)
{
    /* ---------------------------------------------------------------------- *
     * difference_point()
     * input : the Weierstrass curve constants A[0] := A, and A[1] := B;
     * output: the projective Weierstrass x-coordinate points x(P), x(Q), and
     *         x(P - Q).
     * ---------------------------------------------------------------------- */
    fp2_add(PQ[0], P[1], Q[1]);		// (y(P) + y(Q))
    fp2_sub(PQ[1], P[0], Q[0]);		// (x(P) - x(Q))
    fp2_inv(PQ[1]);		            // 1 / (x(P) - x(Q))
    fp2_mul(PQ[0], PQ[1], PQ[0]);	// (y(P) + y(Q)) / (x(P) - x(Q))
    fp2_sqr(PQ[0], PQ[0]);		    // [(y(P) + y(Q)) / (x(P) - x(Q))] ^ 2

    fp2_sub(PQ[0], PQ[0], P[0]);		// [(y(P) + y(Q)) / (x(P) - x(Q))] ^ 2 - x(P)
    fp2_sub(PQ[0], PQ[0], Q[0]);		// [(y(P) + y(Q)) / (x(P) - x(Q))] ^ 2 - (x(P) + x(Q))

    fp2_set_one(P[1]);			// (x(P) : 1)
    fp2_set_one(Q[1]);			// (x(Q) : 1)
    fp2_set_one(PQ[1]);			// (x(P-Q) : 1)
}   // 1I + 1M + 1S + 4a

int isfull_order(proj_t P2, proj_t P3, const proj_t P, const proj_t A)
{
    /* ---------------------------------------------------------------------- *
     * isfull_order()
     * input : a projective Weierstrass x-coordinate point x(P), and the
     *         Weierstrass curve constants A[0] := A, and A[1] := B;
     * output: (1 if the point P has full order [(2^e2) * (3^e3) * f] or
     *         0 if it doesn't have full order), and the projective Weierstrass
     *         x-coordinate points x([(p+1)/2]P) and x([(p+1)/3]P);
     * ---------------------------------------------------------------------- */
    proj_t Q, Pf;

    proj_copy(Q, P);
    xdble(Q, EXPONENT2 - 1, Q, A);	// [2^(e2 - 1)][f]P
    xtple(Q, EXPONENT3 - 1, Q, A);	// [3^(e3 - 1)][2^(e2 - 1)]P

    xdbl(Pf, Q, A);		// [3^(e3 - 1)][2^e2]P
    xtpl(Pf, Pf, A);	// [3^e3][2^e2]P

    xmul(Q, COFACTOR, Q, A);	// [3^(e3 - 1)][2^(e2 - 1)][f]P
    xtpl(P2, Q, A);		// [3^e3][2^(e2 - 1)][f]P
    xdbl(P3, Q, A);		// [3^(e3 - 1)][2^e2][f]P

    return (int)(isinfinity(P2) != 1) && (int)(isinfinity(P3) != 1) && ((int)(isinfinity(Pf) != 1) || (COFACTOR == 1));
}

void init_basis(proj_t P, proj_t Q, proj_t PQ, const proj_t A)
{
    /* ---------------------------------------------------------------------- *
     * full_torsion_points()
     * input : the Weierstrass curve constants A[0] := A, and A[1] := B;
     * output: the projective Weierstrass x-coordinate full order points x(P),
     * x(Q), and x(P-Q);
     * ---------------------------------------------------------------------- */
    proj_t P2, P3, Q2, Q3;
    while( 1 )
    {
        random_affine_point(P, A);
        random_affine_point(Q, A);
        difference_point(PQ, P, Q);

        if (isfull_order(P2, P3, P, A) && isfull_order(Q2, Q3, Q, A))
        {
            if( (proj_isequal(P2, Q2) != 1) && (proj_isequal(P3, Q3) != 1) )
                break;
        }
    }
}

void xisog_2(proj_t C, fp2_t V, proj_t P, const proj_t A)
    {
    /* ---------------------------------------------------------------------- *
     * xisog_2()
     * input : a projective Weierstrass x-coordinate order-2 point x(P)=XP/ZP,
     *         and the short Weierstrass curve constants A[0] := A and A[1] := B;
     * output: 2-isogenous Weierstrass curve
     * ---------------------------------------------------------------------- */
    fp2_t XX, ZZ, ZZZ, aZZ, bZZZ;

    // The next two lines are required to ensure the isomorphism iota
    // belongs to F_{p^2} where (iota * phi_P) is the 2-isogeny
    // NOTE. iota requires the sqrt of ZP, then we set the as follows:
    fp2_mul(P[0], P[0], P[1]);      // XP <- XP * ZP
    fp2_sqr(P[1], P[1]);            // XP <- ZP ^ 2

    fp2_sqr(XX, P[0]);          // XP^2
    fp2_sqr(ZZ, P[1]);          // ZP^2
    fp2_mul(ZZZ, ZZ, P[1]);     // ZP^3
    fp2_mul(aZZ, A[0], ZZ);     // A * ZP^2
    fp2_mul(bZZZ, A[1], ZZZ);   // B * ZP^3

    fp2_add(V, XX, XX);         // 2(XP^2)
    fp2_add(V, V, XX);          // 3(XP^2)
    fp2_add(V, V, aZZ);         // V := 3(XP^2) + (A * ZP^2)

    fp2_add(C[1], V, V);        // 2 * V
    fp2_add(C[0], C[1], C[1]);  // 4 * V
    fp2_add(C[0], C[0], V);     // 5 * V
    fp2_add(C[1], C[0], C[1]);  // 7 * V
    fp2_mul(C[1], C[1], P[0]);  // 7 * V * XP

    fp2_sub(C[0], aZZ, C[0]);   // (A * ZP^2) - (5 * V)
    fp2_sub(C[1], bZZZ, C[1]);  // (B * ZP^3) - (7 * V * XP)
}   // 5M + 3S + 9a

void xeval_2(proj_t R, const proj_t Q, const proj_t P, const fp2_t V)
{
    /* ---------------------------------------------------------------------- *
     * xeval_2()
     * input : a projective Weierstrass x-coordinate point x(Q)=XQ/ZQ,
     *         a projective Weierstrass x-coordinate order-2 point x(P)=XP/ZP,
     *         and an element v of F_p (output of xisog_2)
     * output: the image of x(Q) under a 2-isogeny with kernel generated
     *         by x(P)
    * ---------------------------------------------------------------------- */
    fp2_t XZP, XPZ, ZZ, ZV, Z;
    fp2_copy(Z, Q[1]);

    fp2_sqr(ZZ, Z);
    fp2_mul(ZV, ZZ, V);         // Z^2 * V
    fp2_mul(XZP, Q[0], P[1]);   // X * ZP
    fp2_mul(XPZ, Q[1], P[0]);   // XP * Z

    fp2_sub(R[1], XZP, XPZ);    //  X * ZP - XP * Z
    fp2_mul(R[0], R[1], XZP);   // (X * ZP - XP * Z) * (X * ZP)
    fp2_add(R[0], R[0], ZV);    // (X * ZP - XP * Z) * (X * ZP) + (Z^2 * V)
    fp2_mul(R[1], R[1], Z);     // (X * ZP - XP * Z) * Z
}   // 5M + 1S + 2a

void xisog_2e_1st(proj_t C, proj_t W[3], const proj_t P, const proj_t A, const digit_t *S2, const digit_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_2e_1st()
     *  Input : three projective Weierstrass x-coordinate points, a projective
     *          Weierstrass x-coordinate order-2^e point x(P)=XP/ZP, the short
     *          Weierstrass curve constants A[0] := A, and A[1] := B, an strategy,
     *          and a positive integer e;
     *  Output: the image of the three inputs points under a 2^e-isogeny
     *          with kernel generated by x(P)
     * ---------------------------------------------------------------------- */
    uint8_t log2_of_e, tmp;
    for(tmp = e, log2_of_e = 0; tmp > 0; tmp>>=1, ++log2_of_e);
    log2_of_e *= 2; // In order to ensure each splits is at most of size log2_of_e

    fp2_t v;
    proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A);

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
        xisog_2(C, v, SPLITTING_POINTS[current], C);
        // Pushing points through 2-isogeny
        for(local_i = 0; local_i < current; local_i++)
            xeval_2(SPLITTING_POINTS[local_i], SPLITTING_POINTS[local_i], SPLITTING_POINTS[current], v);

        xeval_2(W[0], W[0], SPLITTING_POINTS[current], v);
        xeval_2(W[1], W[1], SPLITTING_POINTS[current], v);
        xeval_2(W[2], W[2], SPLITTING_POINTS[current], v);

        BLOCK -= XDBLs[current];   // BLOCK is decreased by the last number of doublings performed
        XDBLs[current] = 0;        // The last element in the splits are removed
        current -= 1;              // The number of splits is decreased by one
    }

    // At this point, our kernel has order 2. Therefore, we can construct a 2-isogeny
    xisog_2(C, v, SPLITTING_POINTS[current], C);
    // Pushing points through 2-isogeny
    xeval_2(W[0], W[0], SPLITTING_POINTS[current], v);
    xeval_2(W[1], W[1], SPLITTING_POINTS[current], v);
    xeval_2(W[2], W[2], SPLITTING_POINTS[current], v);
}

void xisog_2e_2nd(proj_t C, const proj_t P, const proj_t A, const digit_t *S2, const digit_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_2e_2nd()
     *  Input : a Weierstrass projective x-coordinate order-2^e point x(P)=XP/ZP,
     *          the short Weierstrass curve constants A[0] := A, and A[1] := B, an
     *          strategy, and a positive integer e;
     *  Output: the image curve of a 2^e-isogeny with kernel generated
     *          by x(P)
     * ---------------------------------------------------------------------- */
    uint8_t log2_of_e, tmp;
    for(tmp = e, log2_of_e = 0; tmp > 0; tmp>>=1, ++log2_of_e);
    log2_of_e *= 2; // In order to ensure each splits is at most of size log2_of_e

    fp2_t v;
    proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A);

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
        xisog_2(C, v, SPLITTING_POINTS[current], C);
        // Pushing points through 2-isogeny
        for(local_i = 0; local_i < current; local_i++)
            xeval_2(SPLITTING_POINTS[local_i], SPLITTING_POINTS[local_i], SPLITTING_POINTS[current], v);

        BLOCK -= XDBLs[current];   // BLOCK is decreased by the last number of doublings performed
        XDBLs[current] = 0;        // The last element in the splits are removed
        current -= 1;              // The number of splits is decreased by one
    }

    // At this point, our kernel has order 2. Therefore, we can construct a 2-isogeny
    xisog_2(C, v, SPLITTING_POINTS[current], C);
}

void xisog_3(proj_t C, proj_t UV, proj_t P, const proj_t A)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_3()
     *  input : a projective Weierstrass x-coordinate order-3 point x(P)=XP/ZP,
     *          and the short Weierstrass curve constants A[0] := A and A[1] := B;
     *  output: 3-isogenous Weierstrass curve
     * ---------------------------------------------------------------------- */
    fp2_t XX, ZZ, ZZZ, aZZ, bZZZ, tmp;

    // The next two lines are required to ensure the isomorphism iota 
    // belongs to F_{p^2} where (iota * phi_P) is the 3-isogeny
    // NOTE. iota requires the sqrt of ZP, then we set the as follows:
    fp2_mul(P[0], P[0], P[1]);      // XP <- XP * ZP
    fp2_sqr(P[1], P[1]);            // XP <- ZP ^ 2

    fp2_sqr(XX, P[0]);              // XP^2
    fp2_sqr(ZZ, P[1]);              // ZP^2
    fp2_mul(ZZZ, ZZ, P[1]);         // ZP^3
    fp2_mul(aZZ, A[0], ZZ);         // A * ZP^2
    fp2_mul(bZZZ, A[1], ZZZ);       // B * ZP^3

    fp2_add(UV[1], XX, XX);         // 2(XP^2)
    fp2_add(UV[1], UV[1], XX);      // 3(XP^2)
    fp2_add(UV[1], UV[1], aZZ);     // 3(XP^2) + (A * ZP^2)
    fp2_add(UV[1], UV[1], UV[1]);   // V := 2[3(XP^2) + (A * ZP^2)]

    fp2_add(UV[0], XX, aZZ);        // XP^2 + (A * ZP^2)
    fp2_mul(UV[0], P[0], UV[0]);    // XP^3 + (A * ZP^2 * XP)
    fp2_add(UV[0], UV[0], bZZZ);    // XP^3 + (A * ZP^2 * XP) + (B * ZP^3)
    fp2_add(UV[0], UV[0], UV[0]);   // 2[XP^3 + (A * ZP^2 * XP) + (B * ZP^3)]
    fp2_add(UV[0], UV[0], UV[0]);   // U := 4[XP^3 + (A * ZP^2 * XP) + (B * ZP^3)]

    fp2_add(C[0], UV[1], UV[1]);    // 2 * V
    fp2_add(C[0], C[0], C[0]);      // 4 * V
    fp2_add(C[0], C[0], UV[1]);     // 5 * V

    fp2_mul(C[1], UV[1], P[0]);     // V * XP
    fp2_add(C[1], C[1], UV[0]);     // V * XP + U
    fp2_add(tmp, C[1], C[1]);       // 2(V * XP + U)
    fp2_add(C[1], tmp, C[1]);       // 3(V * XP + U)
    fp2_add(tmp, tmp, tmp);         // 4(V * XP + U)
    fp2_add(C[1], tmp, C[1]);       // 7(V * XP + U)

    fp2_sub(C[0], aZZ, C[0]);       // (A * ZP^2) - (5 * V)
    fp2_sub(C[1], bZZZ, C[1]);      // (B * ZP^3) - 7 * (V * XP + U)

}   // 6M + 3S + 18a

void xeval_3(proj_t R, const proj_t Q, const proj_t P, const proj_t UV)
{
    /* ---------------------------------------------------------------------- *
     *  xeval_3()
     *  input : a projective Montgomery x-coordinate point x(Q)=XQ/ZQ,
     *          a projective Montgomery x-coordinate order-3 point x(P)=XP/ZP,
     *          and a projective element uv (output of xisog_3)
     *  output: the image of x(Q) under a 3-isogeny with kernel generated
     *          by x(P)
     * ---------------------------------------------------------------------- */
    fp2_t XZP, XPZ, ZZ, Z, t0, t1, t2;
    fp2_copy(Z, Q[1]);

    fp2_sqr(ZZ, Z);
    fp2_mul(XZP, Q[0], P[1]);   // X * ZP
    fp2_mul(XPZ, Q[1], P[0]);   // XP * Z

    fp2_sub(R[1], XZP, XPZ);    //  X * ZP - XP * Z
    fp2_mul(t0, UV[0], Z);      // U * Z
    fp2_mul(t1, UV[1], R[1]);   // (X * ZP - XP * Z) * V

    fp2_sqr(R[1], R[1]);        //        (X * ZP - XP * Z)^2
    fp2_add(t2, t0, t1);        //       [(X * ZP - XP * Z) * V + U * Z]
    fp2_mul(t2, ZZ, t2);        // Z^2 * [(X * ZP - XP * Z) * V + U * Z])
    fp2_mul(R[0], R[1], XZP);   //                                          [(X * ZP - XP * Z)^2] * (X * ZP)
    fp2_add(R[0], R[0], t2);    // Z^2 * [(X * ZP - XP * Z) * V + U * Z]) + [(X * ZP - XP * Z)^2] * (X * ZP)
    fp2_mul(R[1], R[1], Z);     // [(X * ZP - XP * Z)^2] * Z

}   // 7M + 2S + 3a

void xisog_3e_1st(proj_t C, proj_t W[3], const proj_t P, const proj_t A, const digit_t *S3, const digit_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_3e_1st()
     *  input : three projective Weierstrass x-coordinate points, a projective
     *          Weierstrass x-coordinate order-3^e point x(P)=XP/ZP, the short
     *          Weierstrass curve constants A[0] := A, and A[1] := B, an strategy,
     *          and a positive integer e;
     *  output: the image of the three inputs points under a 3^e-isogeny
     *          with kernel generated by x(P)
     * ---------------------------------------------------------------------- */
    uint8_t log2_of_e, tmp;
    for(tmp = e, log2_of_e = 0; tmp > 0; tmp>>=1, ++log2_of_e);
    log2_of_e *= 2; // In order to ensure each splits is at most of size log2_of_e

    proj_t uv;
    proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A);

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
        xisog_3(C, uv, SPLITTING_POINTS[current], C);
        // Pushing points through 3-isogeny
        for(local_i = 0; local_i < current; local_i++)
            xeval_3(SPLITTING_POINTS[local_i], SPLITTING_POINTS[local_i], SPLITTING_POINTS[current], uv);

        xeval_3(W[0], W[0], SPLITTING_POINTS[current], uv);
        xeval_3(W[1], W[1], SPLITTING_POINTS[current], uv);
        xeval_3(W[2], W[2], SPLITTING_POINTS[current], uv);

        BLOCK -= XTPLs[current];   // BLOCK is decreased by the last number of triplings performed
        XTPLs[current] = 0;        // The last element in the splits are removed
        current -= 1;              // The number of splits is decreased by one
    }

    // At this point, our kernel has order 3. Therefore, we can construct a 3-isogeny
    xisog_3(C, uv, SPLITTING_POINTS[current], C);
    // Pushing points through 3-isogeny
    xeval_3(W[0], W[0], SPLITTING_POINTS[current], uv);
    xeval_3(W[1], W[1], SPLITTING_POINTS[current], uv);
    xeval_3(W[2], W[2], SPLITTING_POINTS[current], uv);
}

void xisog_3e_2nd(proj_t C, const proj_t P, const proj_t A, const digit_t *S3, const digit_t e)
{
    /* ---------------------------------------------------------------------- *
     *  xisog_3e_2nd()
     *  input : three projective Weierstrass x-coordinate points, a projective
     *          Weierstrass x-coordinate order-3^e point x(P)=XP/ZP, the short
     *          Weierstrass curve constants A[0] := A, and A[1] := B, an strategy,
     *          and a positive integer e;
     *  output: the image curve of a 3^e-isogeny with kernel generated
     *          by x(P)
     * ---------------------------------------------------------------------- */
    uint8_t log2_of_e, tmp;
    for(tmp = e, log2_of_e = 0; tmp > 0; tmp>>=1, ++log2_of_e);
    log2_of_e *= 2; // In order to ensure each splits is at most of size log2_of_e

    proj_t uv;
    proj_t SPLITTING_POINTS[log2_of_e];
    proj_copy(SPLITTING_POINTS[0], P);
    proj_copy(C, A);

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
        xisog_3(C, uv, SPLITTING_POINTS[current], C);
        // Pushing points through 3-isogeny
        for(local_i = 0; local_i < current; local_i++)
            xeval_3(SPLITTING_POINTS[local_i], SPLITTING_POINTS[local_i], SPLITTING_POINTS[current], uv);

        BLOCK -= XTPLs[current];   // BLOCK is decreased by the last number of triplings performed
        XTPLs[current] = 0;        // The last element in the splits are removed
        current -= 1;              // The number of splits is decreased by one
    }

    // At this point, our kernel has order 3. Therefore, we can construct a 3-isogeny
    xisog_3(C, uv, SPLITTING_POINTS[current], C);
}

void xisog_f(proj_t C, proj_t P, const proj_t A)
{
    /* ---------------------------------------------------------------------- *
     *  xISOG_f()
     *  input : a projective Weierstrass x-coordinate order-f point x(P)=XP/ZP,
     *          and the short Weierstrass curve constants A[0] := A and A[1] := B;
     *  output: the image curve of a f-isogeny with kernel generated by
     *          x(P)
     * ---------------------------------------------------------------------- */
    if (COFACTOR == 1)
    {
        proj_copy(C, A);
        return ;
    }
    proj_t R, S, T;

    fp2_t UX, VX, UZ, VZ, U, V, TMP;
    fp2_t XX, ZZ, ZZZ, aZZ, bZZZ;

    // u = UX/UZ and v = VX/VZ

    // ...............................................................
    // R := P
    proj_copy(R, P);
    // The next two lines are required to ensure the isomorphism iota
    // belongs to F_{p^2} where (iota * phi_P) is the 3-isogeny
    // NOTE. iota requires the sqrt of ZP, then we set the as follows:

    fp2_mul(R[0], R[0], R[1]);      // XR <- XR * ZR
    fp2_sqr(R[1], R[1]);            // XR <- ZR ^ 2

    fp2_sqr(XX, R[0]);              // XR^2
    fp2_sqr(VZ, R[1]);              // ZR^2
    fp2_mul(UZ, VZ, R[1]);          // ZR^3
    fp2_mul(aZZ, A[0], VZ);         // A * ZR^2
    fp2_mul(bZZZ, A[1], UZ);        // B * ZR^3

    // V
    fp2_add(VX, XX, XX);    // 2(XP^2)
    fp2_add(VX, VX, XX);    // 3(XP^2)
    fp2_add(VX, VX, aZZ);   //   3(XP^2) + (A * ZP^2)

    // U
    fp2_add(UX, XX, aZZ);    //   XP^2 + (A * ZP^2)
    fp2_mul(UX, R[0], UX);    //   XP^3 + (A * ZP^2 * XP)
    fp2_add(UX, UX, bZZZ);   //   XP^3 + (A * ZP^2 * XP) + (B * ZP^3)
    fp2_add(UX, UX, UX);     // 2[XP^3 + (A * ZP^2 * XP) + (B * ZP^3)]

    // V*X + U
    fp2_mul(TMP, VX, R[0]);  // V * XR
    fp2_add(UX, TMP, UX);     // U := V * XR + U

    digit_t i;
    xdbl(S, R, A);

    for(i = 1; i < ((digit_t)COFACTOR >> 1); i++)
    {
        // The next two lines are required to ensure the isomorphism iota
        // belongs to F_{p^2} where (iota * phi_P) is the 3-isogeny
        // NOTE. iota requires the sqrt of ZP, then we set the as follows:

        fp2_mul(S[0], S[0], S[1]);      // XR <- XR * ZR
        fp2_sqr(S[1], S[1]);            // XR <- ZR ^ 2

        fp2_sqr(XX, S[0]);              // XR^2
        fp2_sqr(ZZ, S[1]);              // ZR^2
        fp2_mul(ZZZ, ZZ, S[1]);         // ZR^3
        fp2_mul(aZZ, A[0], ZZ);         // A * ZR^2
        fp2_mul(bZZZ, A[1], ZZZ);       // B * ZR^3

        // V
        fp2_add(V, XX, XX);     // 2(XP^2)
        fp2_add(V, V, XX);      // 3(XP^2)
        fp2_add(V, V, aZZ);     // 3(XP^2) + (A * ZP^2)

        // U
        fp2_add(U, XX, aZZ);    //   XP^2 + (A * ZP^2)
        fp2_mul(U, S[0], U);    //   XP^3 + (A * ZP^2 * XP)
        fp2_add(U, U, bZZZ);    //   XP^3 + (A * ZP^2 * XP) + (B * ZP^3)
        fp2_add(U, U, U);       // 2[XP^3 + (A * ZP^2 * XP) + (B * ZP^3)]

        // V*X + U
        fp2_mul(TMP, V, S[0]);  // V * XR
        fp2_add(U, TMP, U);     // U := V * XR + U

        // VX := (VX * ZZ) + (V * VZ)
        fp2_mul(V, V, VZ);      //                   (V * VZ)
        fp2_mul(TMP, VX, ZZ);   //       (VX * ZZ)
        fp2_add(VX, TMP, V);    // VX := (VX * ZZ) + (V * VZ)
        fp2_mul(VZ, VZ, ZZ);    // VZ := (VZ * ZZ)

        // UX := (UX * ZZZ) + (U * UZ)
        fp2_mul(U, U, UZ);      //                    (U * UZ)
        fp2_mul(TMP, UX, ZZZ);  //       (UX * ZZZ)
        fp2_add(UX, TMP, U);    // UX := (UX * ZZZ) + (U * UZ)
        fp2_mul(UZ, UZ, ZZZ);   // UZ := (UZ * ZZZ)

        // Next projective Weierstrass x-coordinate point x([i+2]P)
        xadd(T, S, P, R, A);

        proj_copy(R, S);
        proj_copy(S, T);
    };

    // 5 * VX
    fp2_add(VX, VX, VX);      // VX := 2 * VX
    fp2_add(TMP, VX, VX);       // 2 * VX
    fp2_add(TMP, TMP, TMP);     // 4 * VX
    fp2_add(VX, TMP, VX);       // 5 * VX

    // 7 * UX
    fp2_add(UX, UX, UX);      // UX := 2 * UX
    fp2_add(TMP, UX, UX);       // 2 * UX
    fp2_add(UX, TMP, UX);       // 3 * UX
    fp2_add(TMP, TMP, TMP);     // 4 * UX
    fp2_add(UX, TMP, UX);       // 7 * UX

    // Weierstrass constants A' and B'
    fp2_mul(aZZ, A[0], VZ);     // A * VZ
    fp2_mul(bZZZ, A[1], UZ);    // B * UZ
    fp2_sub(C[0], aZZ, VX);     // (A * VZ) - (5 * VX)
    fp2_sub(C[1], bZZZ, UX);    // (B * UZ) - (7 * UX)

}