#ifndef _MONT_GPU_H_
#define _MONT_GPU_H_

#include <cuda.h>
#include <curand_kernel.h>

extern __device__ void set_initial_curve(proj_t A);        // projective Weierstrass curve (A : B) = (1,0)
extern __device__ int isinfinity( proj_t P);          // (X : 0) for any quadratic field element X
extern __device__ int proj_isequal( proj_t P, proj_t Q); // check if two projective points are equal
extern __device__ void proj_copy(proj_t Q, proj_t P);// stores P in Q

extern __device__ void random_affine_point(proj_t P, fp2_t A, curandStatePhilox4_32_10_t *state);                   // Random affine point of the Weierstrass curve E : y^2 = x^3 + Ax + B
extern __device__ void differience_point(proj_t PQ, proj_t P, proj_t Q, fp2_t A);// x(P - Q) given the affine points P and Q
extern __device__ int isfull_iorder(proj_t P2, proj_t P3,  proj_t P, proj_t A2); // full-order check
extern __device__ void init_basis(proj_t P, proj_t Q, proj_t PQ, proj_t A2, curandStatePhilox4_32_10_t *state); // random basis generators

extern __device__ void change_curvemodel(proj_t Q, proj_t P);
extern __device__ void coeff(fp2_t A, proj_t A2);
extern __device__ void j_invariant(fp2_t j, proj_t A);

extern __device__ void xadd(proj_t R, proj_t P, proj_t Q, proj_t PQ);
extern __device__ void xdbl(proj_t Q, proj_t P, proj_t A2);
extern __device__ void xdbladd(proj_t R, proj_t S, proj_t P, proj_t Q, proj_t PQ, proj_t A2);
extern __device__ void ladder3pt(proj_t R, uint64_t m, proj_t P, proj_t Q, proj_t PQ, proj_t A2, limb_t e);
extern __device__ void ladder3pt_long(proj_t R, fp_t m, proj_t P, proj_t Q, proj_t PQ, proj_t A2);
extern __device__ void xdble(proj_t Q, limb_t e, proj_t P, proj_t A2);
extern __device__ void xtpl(proj_t Q, proj_t P,  proj_t A3);
extern __device__ void xtple(proj_t Q, limb_t e, proj_t P, proj_t A3);
extern __device__ void xmul(proj_t Q, limb_t k, proj_t P, proj_t A2);

// x-only isogenies
extern __device__ void xisog_2(proj_t C, proj_t P);			    // 2-isogeny ruction
extern __device__ void xeval_2(proj_t R, proj_t Q, proj_t P);  // 2-isogeny evaluation
// 2^e-isogeny ruction + three evaluations
extern __device__ void xisog_2e_1st(proj_t C, proj_t W[3], proj_t P, proj_t A2, limb_t *S2, limb_t e);
// 2^e-isogeny ruction
extern __device__ void xisog_2e_2nd(proj_t C, proj_t P, proj_t A2, limb_t *S2, limb_t e);
// Just for testing
extern __device__ void full_xisog_2e_2nd(proj_t C, proj_t P, proj_t A2, limb_t *S2, limb_t e);

extern __device__ void xisog_3(proj_t C, proj_t K, proj_t P);              // 3-isogeny ruction
extern __device__ void xeval_3(proj_t R, proj_t Q, proj_t K);  // 3-isogeny evaluation
// 3^e-isogeny ruction + three evaluations
extern __device__ void xisog_3e_1st(proj_t C, proj_t W[3], proj_t P, proj_t A3, uint32_t *S3, uint32_t e);
// 3^e-isogeny ruction
extern __device__ void xisog_3e_2nd(proj_t C, proj_t P, proj_t A3, uint32_t *S3, uint32_t e);

//void xisog_f(proj_t C, proj_t P,  proj_t A);   // f-isogeny ruction
#endif
