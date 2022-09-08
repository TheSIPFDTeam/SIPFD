#ifndef _VOW_SETUP_
#define _VOW_SETUP_ 1

#if defined(_vowgcs_)

#include "api.h"

#define OMEGABITS   33 // omega = 2^omegabits
#define OMEGA       8589934592 // Limit: memory cells
#define BETA        10
#define BETAXOMEGA  3106892 // (beta x omega) distinguished points per each PRF
#define THETA       0.04971844555217913 //2.25 x (omega / 2N) portion of distinguished point
#define MAXTRAIL    202 // 10 / theta
// We write 1/theta = R*2^n for 0 <= R < 2, and approximate 1/R to the nearest 1/(2^Rbtis)-th
#define N           4
#define RBITS       4
#define DISTINGUISHED 12
#define MAXPRF      1
#define TRIALBITS   8
#define TRIPLETBYTES 8

#define DEG 2
#define HEURISTIC 1

#define NUM_BLOCKS  108
#define NUM_THREADS 256

#define CORES 16

#endif

#endif

