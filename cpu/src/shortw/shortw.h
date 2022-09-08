#ifndef _SHORTW_H_
#define _SHORTW_H_

void set_initial_curve(proj_t A);                   // projective Weierstrass curve (A : B) = (1,0)
int isinfinity(const proj_t P);                     // (X : 0) for any quadratic field element X
void proj_copy(proj_t Q, const proj_t P);           // stores P in Q
int proj_isequal(const proj_t P, const proj_t Q);   // check if two projective points are equal

void random_affine_point(proj_t P, const proj_t A);                     // Random affine point of the Weierstrass curve E : y^2 = x^3 + Ax + B
void difference_point(proj_t PQ, proj_t P, proj_t Q);                   // x(P - Q) given the affine points P and Q
int isfull_order(proj_t P2, proj_t P3, const proj_t P, const proj_t A); // full-order check
void init_basis(proj_t P, proj_t Q, proj_t PQ, const proj_t A);         // random basis generators

void j_invariant(fp2_t j, proj_t A);

void xadd(proj_t R, const proj_t P, const proj_t Q, const proj_t PQ, const proj_t A);
void xdbl(proj_t Q, const proj_t P, const proj_t A);
void xdble(proj_t Q, digit_t e, const proj_t P, const proj_t A);
void xtpl(proj_t Q, const proj_t P, const proj_t A);
void xtple(proj_t Q, digit_t e, const proj_t P, const proj_t A);
void xdbladd(proj_t R, proj_t S, const proj_t P, const proj_t Q, const proj_t PQ, const proj_t A);
void ladder3pt(proj_t R, const fp_t m, const proj_t P, const proj_t Q, const proj_t PQ, const proj_t A);
void xmul(proj_t Q, const digit_t k, const proj_t P, const proj_t A);

// x-only isogenies
void xisog_2(proj_t C, fp2_t V, proj_t P, const proj_t A);			        // 2-isogeny construction
void xeval_2(proj_t R, const proj_t Q, const proj_t P, const fp2_t V);		// 2-isogeny evaluation
// 2^e-isogeny construction + three evaluations
void xisog_2e_1st(proj_t C, proj_t W[3], const proj_t P, const proj_t A, const digit_t *S2, const digit_t e);
// 2^e-isogeny construction
void xisog_2e_2nd(proj_t C, const proj_t P, const proj_t A, const digit_t *S2, const digit_t e);

void xisog_3(proj_t C, proj_t UV, proj_t P, const proj_t A);                // 3-isogeny construction
void xeval_3(proj_t R, const proj_t Q, const proj_t P, const proj_t UV);    // 3-isogeny evaluation
// 3^e-isogeny construction + three evaluations
void xisog_3e_1st(proj_t C, proj_t W[3], const proj_t P, const proj_t A, const digit_t *S3, const digit_t e);
// 3^e-isogeny construction
void xisog_3e_2nd(proj_t C, const proj_t P, const proj_t A, const digit_t *S3, const digit_t e);

void xisog_f(proj_t C, proj_t P, const proj_t A);   // f-isogeny construction
#endif
