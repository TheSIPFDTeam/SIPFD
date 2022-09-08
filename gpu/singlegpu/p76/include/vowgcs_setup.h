#ifndef _VOW_SETUP_
#define _VOW_SETUP_ 1

#if defined(_vowgcs_)

#include "api.h"

#define OMEGABITS   10 // omega = 2^omegabits
#define OMEGA       1024 // Limit: memory cells
#define BETA        10
#define BETAXOMEGA  640 // (beta x omega) distinguished points per each PRF
#define THETA       0.140625 //2.25 x (omega / 2N) portion of distinguished point
#define MAXTRAIL    72 // 10 / theta
// We write 1/theta = R*2^n for 0 <= R < 2, and approximate 1/R to the nearest 1/(2^Rbtis)-th
#define N           2
#define RBITS       4
#define DISTINGUISHED 9
#define MAXPRF      1
#define TRIALBITS   7
#define TRIPLETBYTES 4

#define DEG 2
#define HEURISTIC 1

#define NUM_BLOCKS  1
#define NUM_THREADS 16

#define CORES 4

#endif

#endif

