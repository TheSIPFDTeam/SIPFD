#include "api.h"

#if RADIX == 64
/* Addition in Fp */
#define __fp_add(c0,c1,c2,a0,a1,a2,b0,b1,b2,p0,p1,p2)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;\n\t"                     \
        "add.cc.u64     %0, %3, %6;\n\t"        \
        "addc.cc.u64    %1, %4, %7;\n\t"        \
        "addc.cc.u64    %2, %5, %8;\n\t"        \
        "addc.u64       t0, 0, 0;\n\t"          \
        "sub.cc.u64     %0, %0, %9;\n\t"        \
        "subc.cc.u64    %1, %1, %10;\n\t"       \
        "subc.cc.u64    %2, %2, %11;\n\t"       \
        "subc.u64       t0, t0, 0;\n\t"         \
        "mov.u64        t1, t0;\n\t"            \
        "mov.u64        t2, t0;\n\t"            \
        "and.b64        t0, t0, %9;\n\t"        \
        "and.b64        t1, t1, %10;\n\t"       \
        "and.b64        t2, t2, %11;\n\t"       \
        "add.cc.u64     %0, %0, t0;\n\t"        \
        "addc.cc.u64    %1, %1, t1;\n\t"        \
        "addc.u64       %2, %2, t2;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2)          \
        : "l"(a0), "l"(a1), "l"(a2), "l"(b0), "l"(b1), "l"(b2),\
            "l"(p0), "l"(p1), "l"(p2) )

/* Subtraction in Fp */
#define __fp_sub(c0,c1,c2,a0,a1,a2,b0,b1,b2,p0,p1,p2)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;\n\t"                     \
        "sub.cc.u64     %0, %3, %6;\n\t"        \
        "subc.cc.u64    %1, %4, %7;\n\t"        \
        "subc.cc.u64    %2, %5, %8;\n\t"        \
        "subc.u64       t0, 0, 0;\n\t"          \
        "mov.u64        t1, t0;\n\t"            \
        "mov.u64        t2, t0;\n\t"            \
        "and.b64        t0, t0, %9;\n\t"        \
        "and.b64        t1, t1, %10;\n\t"       \
        "and.b64        t2, t2, %11;\n\t"       \
        "add.cc.u64     %0, %0, t0;\n\t"        \
        "addc.cc.u64    %1, %1, t1;\n\t"        \
        "addc.u64       %2, %2, t2;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2)          \
        : "l"(a0), "l"(a1), "l"(a2), "l"(b0), "l"(b1), "l"(b2),\
            "l"(p0), "l"(p1), "l"(p2) )

#define __fp_mul(c0,c1,c2,a0,a1,a2,b0,b1,b2,m0,m1,m2,p0,p1,p2)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"     \
        ".reg .u64 t4;" ".reg .u64 t5;\n\t"     \
        ".reg .u64 t6;" ".reg .u64 t7;\n\t"     \
        ".reg .u64 t8;" ".reg .u64 t9;\n\t"     \
        ".reg .u64 t10;" ".reg .u64 t11;\n\t"   \
        "mul.lo.u64     t0, %3, %6;\n\t"        \
        "mul.hi.u64     t1, %3, %6;\n\t"        \
        "mad.lo.cc.u64  t1, %3, %7, t1;\n\t"    \
        "mul.hi.u64     t2, %3, %7;\n\t"        \
        "madc.lo.cc.u64 t2, %3, %8, t2;\n\t"    \
        "madc.hi.u64    t3, %3, %8, 0;\n\t"     \
        "mul.lo.u64     t4, %4, %6;\n\t"        \
        "mul.hi.u64     t5, %4, %6;\n\t"        \
        "mad.lo.cc.u64  t5, %4, %7, t5;\n\t"    \
        "mul.hi.u64     t6, %4, %7;\n\t"        \
        "madc.lo.cc.u64 t6, %4, %8, t6;"        \
        "madc.hi.u64    t7, %4, %8, 0;\n\t"     \
        "add.cc.u64     t1, t1, t4;\n\t"        \
        "addc.cc.u64    t5, t5, t2;\n\t"        \
        "addc.cc.u64    t6, t6, t3;\n\t"        \
        "addc.u64       t7, t7, 0;\n\t"         \
        "mul.lo.u64     t2, %5, %6;\n\t"        \
        "mul.hi.u64     t3, %5, %6;\n\t"        \
        "mad.lo.cc.u64  t3, %5, %7, t3;\n\t"    \
        "mul.hi.u64     t4, %5, %7;\n\t"        \
        "madc.lo.cc.u64 t4, %5, %8, t4;\n\t"    \
        "madc.hi.u64    t8, %5, %8, 0;\n\t"     \
        "add.cc.u64     t2, t2, t5;\n\t"        \
        "addc.cc.u64    t3, t3, t6;\n\t"        \
        "addc.cc.u64    t4, t4, t7;\n\t"        \
        "addc.u64       t5, t8, 0;\n\t"         \
        "mov.u64        %0, t0;\n\t"/*Montgomery reduction, m = mu*c mod 2^(3*64)*/\
        "mad.lo.cc.u64  %1, %10, t0, t1;\n\t"   \
        "madc.hi.u64    %2, %10, t0, t2;\n\t"   \
        "mad.lo.u64     %2, %10, t1, %2;\n\t"   \
        "mad.lo.u64     %2, %11, t0, %2;\n\t"   \
        "mul.lo.u64     t6, %0, %12;\n\t"/* u = m*p */\
        "mul.hi.u64     t7, %0, %12;\n\t"       \
        "mad.lo.cc.u64  t7, %0, %13, t7;\n\t"   \
        "mul.hi.u64     t8, %0, %13;\n\t"       \
        "madc.lo.cc.u64 t8, %0, %14, t8;\n\t"   \
        "madc.hi.u64    t9, %0, %14, 0;\n\t"    \
        "mul.lo.u64     t10, %1, %12;\n\t"      \
        "mul.hi.u64     t11, %1, %12;\n\t"      \
        "mad.lo.cc.u64  t11, %1, %13, t11;\n\t" \
        "mul.hi.u64     %0, %1, %13;\n\t"       \
        "madc.lo.cc.u64 %0, %1, %14, %0;"       \
        "madc.hi.u64    %1, %1, %14, 0;\n\t"    \
        "add.cc.u64     t7, t7, t10;\n\t"       \
        "addc.cc.u64    t11, t11, t8;\n\t"      \
        "addc.cc.u64    %0, %0, t9;\n\t"        \
        "addc.u64       %1, %1, 0;\n\t"         \
        "mul.lo.u64     t8, %2, %12;\n\t"       \
        "mul.hi.u64     t9, %2, %12;\n\t"       \
        "mad.lo.cc.u64  t9, %2, %13, t9;\n\t"   \
        "mul.hi.u64     t10, %2, %13;\n\t"      \
        "madc.lo.cc.u64 t10, %2, %14, t10;\n\t" \
        "madc.hi.u64    %2, %2, %14, 0;\n\t"    \
        "add.cc.u64     t8, t8, t11;\n\t"       \
        "addc.cc.u64    t9, t9, %0;\n\t"        \
        "addc.cc.u64    t10, t10, %1;\n\t"      \
        "addc.u64       t11, %2, 0;\n\t"        \
        "mov.b64        %0, 0;\n\t"/*r = (c + u) div 2^(3*64)*/\
        "add.cc.u64     t0, t6, t0;\n\t"        \
        "addc.cc.u64    t1, t7, t1;\n\t"        \
        "addc.cc.u64    t2, t8, t2;\n\t"        \
        "addc.cc.u64    t3, t9, t3;\n\t"        \
        "addc.cc.u64    t4, t10, t4;\n\t"       \
        "addc.cc.u64    t5, t11, t5;\n\t"       \
        "addc.u64       %0, %0, 0;\n\t"         \
        "sub.cc.u64     t3, t3, %12;\n\t"       \
        "subc.cc.u64    t4, t4, %13;\n\t"       \
        "subc.cc.u64    t5, t5, %14;\n\t"       \
        "subc.u64       %0, %0, 0;\n\t"         \
        "mov.u64        %1, %0;\n\t"            \
        "mov.u64        %2, %0;\n\t"            \
        "and.b64        %0, %0, %12;\n\t"       \
        "and.b64        %1, %1, %13;\n\t"       \
        "and.b64        %2, %2, %14;\n\t"       \
        "add.cc.u64     %0, %0, t3;\n\t"        \
        "addc.cc.u64    %1, %1, t4;\n\t"        \
        "addc.u64       %2, %2, t5;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2)\
        : "l"(a0), "l"(a1), "l"(a2), "l"(b0), "l"(b1), "l"(b2),\
          "l"(m0), "l"(m1), "l"(m2), "l"(p0), "l"(p1), "l"(p2) )

#define __fp_sqr(c0,c1,c2,a0,a1,a2,m0,m1,m2,p0,p1,p2)\
    asm volatile ("{\n\t"                           \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"         \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"         \
        ".reg .u64 t4;" ".reg .u64 t5;\n\t"         \
        ".reg .u64 t6;" ".reg .u64 t7;\n\t"         \
        ".reg .u64 t8;" ".reg .u64 t9;\n\t"         \
        ".reg .u64 t10;" ".reg .u64 t11;\n\t"       \
        ".reg .u64 t12;" ".reg .u64 t13;\n\t"       \
        "mul.lo.u64     t0, %3, %3;\n\t"        \
        "mul.hi.u64     t1, %3, %3;\n\t"        \
        "mul.lo.u64     t8, %3, %4;\n\t"        \
        "mul.hi.u64     t9, %3, %4;\n\t"        \
        "mul.lo.u64     t10, %3, %5;\n\t"       \
        "mul.hi.u64     t11, %3, %5;\n\t"       \
        "add.cc.u64     t1, t1, t8;\n\t"        \
        "addc.cc.u64    t2, t9, t10;\n\t"       \
        "addc.u64       t3, t11, 0;\n\t"        \
        "mul.lo.u64     t12, %4, %5;\n\t"       \
        "mul.hi.u64     t13, %4, %5;\n\t"       \
        "mul.hi.u64     t6, %4, %4;\n\t"        \
        "mad.lo.cc.u64  t5, %4, %4, t9;\n\t"    \
        "addc.cc.u64    t6, t6, t12;\n\t"       \
        "addc.u64       t7, t13, 0;\n\t"        \
        "add.cc.u64     t1, t1, t8;\n\t"        \
        "addc.cc.u64    t5, t5, t2;\n\t"        \
        "addc.cc.u64    t6, t6, t3;\n\t"        \
        "addc.u64       t7, t7, 0;\n\t"         \
        "mul.lo.u64     t4, %5, %5;\n\t"        \
        "mul.hi.u64     t8, %5, %5;\n\t"        \
        "add.cc.u64     t3, t11, t12;\n\t"      \
        "addc.cc.u64    t4, t4, t13;\n\t"       \
        "addc.cc.u64    t8, t8, 0;\n\t"         \
        "add.cc.u64     t2, t10, t5;\n\t"       \
        "addc.cc.u64    t3, t3, t6;\n\t"        \
        "addc.cc.u64    t4, t4, t7;\n\t"        \
        "addc.u64       t5, t8, 0;\n\t"         \
        "mov.u64        %0, t0;\n\t"/*Montgomery reduction, m = mu*c mod 2^(64*3)*/\
        "mad.lo.cc.u64  %1, %7, t0, t1;\n\t"    \
        "madc.hi.u64    %2, %7, t0, t2;\n\t"    \
        "mad.lo.u64     %2, %7, t1, %2;\n\t"    \
        "mad.lo.u64     %2, %8, t0, %2;\n\t"    \
        "mul.lo.u64     t6, %0, %9;\n\t"/* u = m*p */\
        "mul.hi.u64     t7, %0, %9;\n\t"        \
        "mad.lo.cc.u64  t7, %0, %10, t7;\n\t"   \
        "mul.hi.u64     t8, %0, %10;\n\t"       \
        "madc.lo.cc.u64 t8, %0, %11, t8;\n\t"   \
        "madc.hi.u64    t9, %0, %11, 0;\n\t"    \
        "mul.lo.u64     t10, %1, %9;\n\t"       \
        "mul.hi.u64     t11, %1, %9;\n\t"       \
        "mad.lo.cc.u64  t11, %1, %10, t11;\n\t" \
        "mul.hi.u64     %0, %1, %10;\n\t"       \
        "madc.lo.cc.u64 %0, %1, %11, %0;"       \
        "madc.hi.u64    %1, %1, %11, 0;\n\t"    \
        "add.cc.u64     t7, t7, t10;\n\t"       \
        "addc.cc.u64    t11, t11, t8;\n\t"      \
        "addc.cc.u64    %0, %0, t9;\n\t"        \
        "addc.u64       %1, %1, 0;\n\t"         \
        "mul.lo.u64     t8, %2, %9;\n\t"        \
        "mul.hi.u64     t9, %2, %9;\n\t"        \
        "mad.lo.cc.u64  t9, %2, %10, t9;\n\t"   \
        "mul.hi.u64     t10, %2, %10;\n\t"      \
        "madc.lo.cc.u64 t10, %2, %11, t10;\n\t" \
        "madc.hi.u64    %2, %2, %11, 0;\n\t"    \
        "add.cc.u64     t8, t8, t11;\n\t"       \
        "addc.cc.u64    t9, t9, %0;\n\t"        \
        "addc.cc.u64    t10, t10, %1;\n\t"      \
        "addc.u64       t11, %2, 0;\n\t"        \
        "mov.b64        %0, 0;\n\t"/*r = (c + u) div 2^(64*3)*/\
        "add.cc.u64     t0, t6, t0;\n\t"        \
        "addc.cc.u64    t1, t7, t1;\n\t"        \
        "addc.cc.u64    t2, t8, t2;\n\t"        \
        "addc.cc.u64    t3, t9, t3;\n\t"        \
        "addc.cc.u64    t4, t10, t4;\n\t"       \
        "addc.cc.u64    t5, t11, t5;\n\t"       \
        "addc.u64       %0, %0, 0;\n\t"         \
        "sub.cc.u64     t3, t3, %9;\n\t"        \
        "subc.cc.u64    t4, t4, %10;\n\t"       \
        "subc.cc.u64    t5, t5, %11;\n\t"       \
        "subc.cc.u64    %0, %0, 0;\n\t"         \
        "mov.u64        %1, %0;\n\t"            \
        "mov.u64        %2, %0;\n\t"            \
        "and.b64        %0, %0, %9;\n\t"        \
        "and.b64        %1, %1, %10;\n\t"       \
        "and.b64        %2, %2, %11;\n\t"       \
        "add.cc.u64     %0, %0, t3;\n\t"        \
        "addc.cc.u64    %1, %1, t4;\n\t"        \
        "addc.u64       %2, %2, t5;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2)\
        : "l"(a0), "l"(a1), "l"(a2), "l"(m0), "l"(m1), "l"(m2),\
        "l"(p0), "l"(p1), "l"(p2) )

__device__ void fp_add(fp_t out, const fp_t arg1, const fp_t arg2) {
    __fp_add(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2],
            __p[0], __p[1], __p[2]);
}

__device__ void fp_sub(fp_t out, const fp_t arg1, const fp_t arg2) {
    __fp_sub(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2],
            __p[0], __p[1], __p[2]);
}

__device__ void fp_mul(fp_t out, const fp_t arg1, const fp_t arg2) {
    __fp_mul(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2],
            __mu[0], __mu[1], __mu[2], __p[0], __p[1], __p[2]);
}

__device__ void fp_sqr(fp_t out, const fp_t arg1) {
    __fp_sqr(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], __mu[0], __mu[1], __mu[2],
            __p[0], __p[1], __p[2]);
}

#elif RADIX == 32
// arith 32
#else
#error "Not implemented"
#endif
