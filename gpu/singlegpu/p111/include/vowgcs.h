#ifndef _VOWGCS_API_H_
#define _VOWGCS_API_H_

#include "api.h"

typedef struct {
    point_t seed; // initial random point
    point_t tail; // distinguished point
    // integer length such that (_fn_)^length(seed) = tail point; that is, length function evaluations
    limb_t length; 
} vowgcs_t;
/*
typedef struct {
    limb_t omega_minus_one;   // 2^{omega_bits} - 1
    limb_t omegabits;         // omega = 2^omegabits
    limb_t omega;             // Limit: memory cells
    limb_t beta;
    double theta;               //2.25 x (omega / 2N) portion of distinguished point
    uint8_t n;                  // 1/theta = R*2^n with 0 <= R < 2
    uint8_t Rbits;
    limb_t distinguished;       // approximation for 2^Rbits/R
    limb_t betaXomega;          // (beta x omega) distinguished points per each PRF
    limb_t maxtrail;            // 10 / theta
    limb_t maxprf;              // Maximum number of PRF
    // Concerning number of cores (metrics: number of collisions)
    limb_t cores;
    linkedlist_t **address;     // Each core has its own list of pointers
    linkedlist_t **collisions;  // Sorted linked list of different collisions
    limb_t *index;              // Current number of different collisions
    limb_t *runtime_collision;  // Number of all collisions
    limb_t *runtime_different;  // Number of all different collisions
    limb_t *runtime;            // Number of function evaluations _fn_()
    uint8_t heuristic;          // For measuring vOW GCS heuristics not focused on the golden collision search
} ctx_vow_t;
*/

//extern __device__ void precompute(proj_t *P, proj_t *Q, proj_t *PQ, proj_t *E, uint64_t path,
//        proj_t b_P, proj_t b_Q, proj_t b_PQ, proj_t b_E, int e, int level, int depth);

extern __device__ void vowgcs(point_t *golden, uint8_t *finished, uint8_t *hashtable, 
        limb_t strategy_reduced[2][EXP0], fp2_t *P0, fp2_t *Q0, fp2_t *PQ0, fp2_t *E0, fp2_t *Z0, 
        fp2_t *P1, fp2_t *Q1, fp2_t *PQ1, fp2_t *E1, fp2_t *Z1, ctx_t *context, limb_t *expo, 
        limb_t *ebits, curandStatePhilox4_32_10_t *state);
#endif
