#ifndef _PRECOMP_H_
#define _PRECOMP_H_

#include "api.h"

//extern "C" void precompute(proj_t *P, proj_t *Q, proj_t *PQ, proj_t *E, uint64_t path,
//        const proj_t b_P, const proj_t b_Q, const proj_t b_PQ, const proj_t b_E,
//        int e, const int level, const int depth);
extern "C" void precompute(fp2_t *P, fp2_t *Q, fp2_t *PQ, fp2_t *E, fp2_t *Z, uint64_t path,
    const proj_t basis[3], const proj_t curve, int e, const int level, const int depth);
#endif
