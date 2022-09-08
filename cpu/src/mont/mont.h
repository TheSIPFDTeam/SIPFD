#ifndef _MONT_H_
#define _MONT_H_

void set_initial_curve(proj_t A);                   // projective Weierstrass curve (A : B) = (1,0)
int isinfinity(const proj_t P);                     // (X : 0) for any quadratic field element X
void proj_copy(proj_t Q, const proj_t P);           // stores P in Q
int proj_isequal(const proj_t P, const proj_t Q);   // check if two projective points are equal

void random_affine_point(proj_t P, const fp2_t A);                          // Random affine point of the Weierstrass curve E : y^2 = x^3 + Ax + B
void difference_point(proj_t PQ, proj_t P, proj_t Q, const fp2_t A);        // x(P - Q) given the affine points P and Q
int isfull_order(proj_t P2, proj_t P3, const proj_t P, const proj_t A2);    // full-order check
void init_basis(proj_t P, proj_t Q, proj_t PQ, const proj_t A2);            // random basis generators

void j_invariant(fp2_t j, proj_t A2);

void xadd(proj_t R, const proj_t P, const proj_t Q, const proj_t PQ);
void xdbl(proj_t Q, const proj_t P, const proj_t A2);
void xdble(proj_t Q, digit_t e, const proj_t P, const proj_t A2);
void xtpl(proj_t Q, const proj_t P, const proj_t A3);
void xtple(proj_t Q, digit_t e, const proj_t P, const proj_t A3);
void xdbladd(proj_t R, proj_t S, const proj_t P, const proj_t Q, const proj_t PQ, proj_t const A2);
void ladder3pt(proj_t R, const uint64_t m, const proj_t P, const proj_t Q, const proj_t PQ, proj_t const A2, digit_t const e);
void ladder3pt_long(proj_t R, const fp_t m, const proj_t P, const proj_t Q, const proj_t PQ, proj_t const A2);
void xmul(proj_t Q, const digit_t k, const proj_t P, proj_t const A2);

// x-only isogenies
void xisog_2(proj_t C, proj_t P);			                // 2-isogeny construction
void xeval_2(proj_t R, const proj_t Q, const proj_t P);		// 2-isogeny evaluation
// 2^e-isogeny construction + three evaluations
void xisog_2e_1st(proj_t C, proj_t W[3], const proj_t P, const proj_t A2, const digit_t *S2, const digit_t e);
// 2^e-isogeny construction
void xisog_2e_2nd(proj_t C, const proj_t P, const proj_t A2, const digit_t *S2, const digit_t e);

void xisog_4(proj_t C, fp2_t K[3], proj_t P);			    // 4-isogeny construction
void xeval_4(proj_t R, const proj_t Q, const fp2_t K[3]);	// 4-isogeny evaluation
// 2^e-isogeny construction + three evaluations by using 4-isogenies
void xisog_2e_1st_(proj_t C, proj_t W[3], const proj_t P, const proj_t A2, const digit_t *S2, const digit_t e);
// 2^e-isogeny construction by using 4-isogenies
void xisog_2e_2nd_(proj_t C, const proj_t P, const proj_t A2, const digit_t *S2, const digit_t e);

void xisog_3(proj_t C, proj_t K, proj_t P);                 // 3-isogeny construction
void xeval_3(proj_t R, const proj_t Q, const proj_t K);     // 3-isogeny evaluation
// 3^e-isogeny construction + three evaluations
void xisog_3e_1st(proj_t C, proj_t W[3], const proj_t P, const proj_t A3, const digit_t *S3, const digit_t e);
// 3^e-isogeny construction
void xisog_3e_2nd(proj_t C, const proj_t P, const proj_t A3, const digit_t *S3, const digit_t e);

void xisog_f(proj_t C, proj_t P, const proj_t A);           // f-isogeny construction
#endif
