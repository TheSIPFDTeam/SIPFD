#ifndef _MONT_H_
#define _MONT_H_

#include "api.h"

void proj_copy(proj_t Q, const proj_t P);           // stores P in Q

void xadd(proj_t R, const proj_t P, const proj_t Q, const proj_t PQ);
void xdbl(proj_t Q, const proj_t P, const proj_t A2);
void xdble(proj_t Q, digit_t e, const proj_t P, const proj_t A2);
void xmul(proj_t Q, const digit_t k, const proj_t P, proj_t const A2);

// x-only isogenies
void xisog_2(proj_t C, proj_t P);			                // 2-isogeny construction
void xeval_2(proj_t R, const proj_t Q, const proj_t P);		// 2-isogeny evaluation

void xisog_f(proj_t C, proj_t P, const proj_t A);           // f-isogeny construction
#endif
