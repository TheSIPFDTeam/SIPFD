#ifndef _ASM_ARITH_GPU_H_
#define _ASM_ARITH_GPU_H_

#include "api.h"

extern __device__ void fp_add(fp_t out, const fp_t arg1, const fp_t arg2); 
extern __device__ void fp_sub(fp_t out, const fp_t arg1, const fp_t arg2);
extern __device__ void fp_mul(fp_t out, const fp_t arg1, const fp_t arg2); 
extern __device__ void fp_sqr(fp_t out, const fp_t arg1);

#endif /* ASM_ARITH */
