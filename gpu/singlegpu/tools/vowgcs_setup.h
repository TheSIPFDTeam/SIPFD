#ifndef _VOW_SETUP_
#define _VOW_SETUP_ 1

#if defined(_vowgcs_)

#include "api.h"

#define OMEGABITS   <omegabits> // omega = 2^omegabits
#define OMEGA       <omega> // Limit: memory cells
#define BETA        <beta>
#define BETAXOMEGA  <betaXomega> // (beta x omega) distinguished points per each PRF
#define THETA       <theta> //2.25 x (omega / 2N) portion of distinguished point
#define MAXTRAIL    <maxtrail> // 10 / theta
// We write 1/theta = R*2^n for 0 <= R < 2, and approximate 1/R to the nearest 1/(2^Rbtis)-th
#define N           <n>
#define RBITS       <Rbits>
#define DISTINGUISHED <distinguished>
#define MAXPRF      <maxprf>
#define TRIALBITS   <trailbits>
#define TRIPLETBYTES <tripletbytes>

#define DEG 2
#define HEURISTIC 1

#define NUM_BLOCKS  <numblocks>
#define NUM_THREADS <numthreads>

#define CORES <cores>

#endif

#endif
