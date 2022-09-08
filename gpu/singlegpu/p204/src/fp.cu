#include "api.h"

/* Addition in Fp */
#define __fp_add(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1,b2,b3,p0,p1,p2,p3)\
    asm volatile ("{\n\t"                                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"                     \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"                     \
        "add.cc.u64     %0, %4, %8;\n\t"        \
        "addc.cc.u64    %1, %5, %9;\n\t"        \
        "addc.cc.u64    %2, %6, %10;\n\t"       \
        "addc.cc.u64    %3, %7, %11;\n\t"       \
        "addc.u64       t0, 0, 0;\n\t"          \
        "sub.cc.u64     %0, %0, %12;\n\t"       \
        "subc.cc.u64    %1, %1, %13;\n\t"       \
        "subc.cc.u64    %2, %2, %14;\n\t"       \
        "subc.cc.u64    %3, %3, %15;\n\t"       \
        "subc.u64       t0, t0, 0;\n\t"         \
        "mov.u64        t1, t0;\n\t"            \
        "mov.u64        t2, t0;\n\t"            \
        "mov.u64        t3, t0;\n\t"            \
        "and.b64        t0, t0, %12;\n\t"       \
        "and.b64        t1, t1, %13;\n\t"       \
        "and.b64        t2, t2, %14;\n\t"       \
        "and.b64        t3, t3, %15;\n\t"       \
        "add.cc.u64     %0, %0, t0;\n\t"        \
        "addc.cc.u64    %1, %1, t1;\n\t"        \
        "addc.cc.u64    %2, %2, t2;\n\t"        \
        "addc.u64       %3, %3, t3;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2), "=l"(c3)\
        : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(b0), "l"(b1), "l"(b2), "l"(b3),\
            "l"(p0), "l"(p1), "l"(p2), "l"(p3) )

/* Subtraction in Fp */
#define __fp_sub(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1,b2,b3,p0,p1,p2,p3)\
    asm volatile ("{\n\t"                                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"                     \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"                     \
        "sub.cc.u64     %0, %4, %8;\n\t"        \
        "subc.cc.u64    %1, %5, %9;\n\t"        \
        "subc.cc.u64    %2, %6, %10;\n\t"       \
        "subc.cc.u64    %3, %7, %11;\n\t"       \
        "subc.u64       t0, 0, 0;\n\t"          \
        "mov.u64        t1, t0;\n\t"            \
        "mov.u64        t2, t0;\n\t"            \
        "mov.u64        t3, t0;\n\t"            \
        "and.b64        t0, t0, %12;\n\t"       \
        "and.b64        t1, t1, %13;\n\t"       \
        "and.b64        t2, t2, %14;\n\t"       \
        "and.b64        t3, t3, %15;\n\t"       \
        "add.cc.u64     %0, %0, t0;\n\t"        \
        "addc.cc.u64    %1, %1, t1;\n\t"        \
        "addc.cc.u64    %2, %2, t2;\n\t"        \
        "addc.u64       %3, %3, t3;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2), "=l"(c3)\
        : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(b0), "l"(b1), "l"(b2), "l"(b3),\
            "l"(p0), "l"(p1), "l"(p2), "l"(p3) )

/* Multiplication in Fp */
#define __fp_mul(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1,b2,b3,m0,m1,m2,m3,p0,p1,p2,p3)  \
    asm volatile ("{\n\t"                                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"                     \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"                     \
        ".reg .u64 t4;" ".reg .u64 t5;\n\t"                     \
        ".reg .u64 t6;" ".reg .u64 t7;\n\t"                     \
        ".reg .u64 t8;" ".reg .u64 t9;\n\t"                     \
        ".reg .u64 t10;" ".reg .u64 t11;\n\t"                   \
        ".reg .u64 t12;" ".reg .u64 t13;\n\t"                   \
        ".reg .u64 t14;" ".reg .u64 t15;\n\t"                   \
        ".reg .u64 t16;" ".reg .u64 t17;\n\t"                   \
        "mul.lo.u64     t0, %4, %8;\n\t"        \
        "mul.hi.u64     t1, %4, %8;\n\t"        \
        "mad.lo.cc.u64  t1, %4, %9, t1;\n\t"    \
        "mul.hi.u64     t2, %4, %9;\n\t"       \
        "madc.lo.cc.u64 t2, %4, %10, t2;\n\t"   \
        "mul.hi.u64     t3, %4, %10;\n\t"       \
        "madc.lo.cc.u64 t3, %4, %11, t3;\n\t"   \
        "madc.hi.u64    t4, %4, %11, 0;\n\t"    \
        "mul.lo.u64     t5, %5, %8;\n\t"       \
        "mul.hi.u64     t6, %5, %8;\n\t"       \
        "mad.lo.cc.u64  t6, %5, %9, t6;\n\t"   \
        "mul.hi.u64     t7, %5, %9;\n\t"       \
        "madc.lo.cc.u64 t7, %5, %10, t7;\n\t"   \
        "mul.hi.u64     t8, %5, %10;\n\t"       \
        "madc.lo.cc.u64 t8, %5, %11, t8;\n\t"   \
        "madc.hi.u64    t9, %5, %11, 0;\n\t"    \
        "add.cc.u64     t1, t1, t5;\n\t"        \
        "addc.cc.u64    t6, t6, t2;\n\t"        \
        "addc.cc.u64    t7, t7, t3;\n\t"        \
        "addc.cc.u64    t8, t8, t4;\n\t"        \
        "addc.u64       t9, t9, 0;\n\t"         \
        "mul.lo.u64     t2, %6, %8;\n\t"      \
        "mul.hi.u64     t3, %6, %8;\n\t"      \
        "mad.lo.cc.u64  t3, %6, %9, t3;\n\t"  \
        "mul.hi.u64     t4, %6, %9;\n\t"      \
        "madc.lo.cc.u64 t4, %6, %10, t4;\n\t"  \
        "mul.hi.u64     t5, %6, %10;\n\t"      \
        "madc.lo.cc.u64 t5, %6, %11, t5;\n\t"  \
        "madc.hi.u64    t10, %6, %11, 0;\n\t"  \
        "add.cc.u64     t2, t2, t6;\n\t"        \
        "addc.cc.u64    t3, t3, t7;\n\t"        \
        "addc.cc.u64    t4, t4, t8;\n\t"        \
        "addc.cc.u64    t5, t5, t9;\n\t"        \
        "addc.u64       t10, t10, 0;\n\t"       \
        "mul.lo.u64     t6, %7, %8;\n\t"      \
        "mul.hi.u64     t7, %7, %8;\n\t"      \
        "mad.lo.cc.u64  t7, %7, %9, t7;\n\t"  \
        "mul.hi.u64     t8, %7, %9;\n\t"      \
        "madc.lo.cc.u64 t8, %7, %10, t8;\n\t"  \
        "mul.hi.u64     t9, %7, %10;\n\t"      \
        "madc.lo.cc.u64 t9, %7, %11, t9;\n\t"  \
        "madc.hi.u64    t11, %7, %11, 0;\n\t"  \
        "add.cc.u64     t3, t3, t6;\n\t"        \
        "addc.cc.u64    t4, t4, t7;\n\t"        \
        "addc.cc.u64    t5, t5, t8;\n\t"        \
        "addc.cc.u64    t6, t9, t10;\n\t"       \
        "addc.u64       t7, t11, 0;\n\t"        \
        "mul.lo.u64     %0, t0, %12;\n\t"/*Montgomery reduction, m = mu*c mod 2^(64*4)*/\
        "mul.hi.u64     %1, t0, %12;\n\t"       \
        "mad.lo.cc.u64  %1, t0, %13, %1;\n\t"   \
        "mul.hi.u64     %2, t0, %13;\n\t"       \
        "madc.lo.cc.u64 %2, t0, %14, %2;\n\t"   \
        "mul.hi.u64     %3, t0, %14;\n\t"       \
        "madc.lo.u64    %3, t0, %15, %3;\n\t"   \
        "mul.lo.u64     t8, t1, %12;\n\t"       \
        "mul.hi.u64     t9, t1, %12;\n\t"       \
        "mad.lo.cc.u64  t9, t1, %13, t9;\n\t"   \
        "mul.hi.u64     t10, t1, %13;\n\t"      \
        "madc.lo.u64    t10, t1, %14, t10;\n\t" \
        "add.cc.u64     %1, %1, t8;\n\t"        \
        "addc.cc.u64    t9, t9, %2;\n\t"        \
        "addc.u64       t10, t10, %3;\n\t"      \
        "mul.lo.u64     %2, t2, %12;\n\t"       \
        "mul.hi.u64     %3, t2, %12;\n\t"       \
        "mad.lo.u64     %3, t2, %13, %3;\n\t"   \
        "add.cc.u64     %2, %2, t9;\n\t"        \
        "addc.u64       %3, %3, t10;\n\t"       \
        "mad.lo.u64     %3, t3, %12, %3;\n\t"   \
        "mul.lo.u64     t8, %0, %16;\n\t"/* u = m*p */\
        "mul.hi.u64     t9, %0, %16;\n\t"        \
        "mad.lo.cc.u64  t9, %0, %17, t9;\n\t"    \
        "mul.hi.u64     t10, %0, %17;\n\t"       \
        "madc.lo.cc.u64 t10, %0, %18, t10;\n\t"   \
        "mul.hi.u64     t11, %0, %18;\n\t"      \
        "madc.lo.cc.u64 t11, %0, %19, t11;"     \
        "madc.hi.u64    t12, %0, %19, 0;\n\t"   \
        "mul.lo.u64     t13, %1, %16;\n\t"      \
        "mul.hi.u64     t14, %1, %16;\n\t"      \
        "mad.lo.cc.u64  t14, %1, %17, t14;\n\t" \
        "mul.hi.u64     t15, %1, %17;\n\t"      \
        "madc.lo.cc.u64 t15, %1, %18, t15;\n\t" \
        "mul.hi.u64     t16, %1, %18;\n\t"      \
        "madc.lo.cc.u64 t16, %1, %19, t16;\n\t" \
        "madc.hi.u64    t17, %1, %19, 0;\n\t"   \
        "add.cc.u64     t9, t9, t13;\n\t"       \
        "addc.cc.u64    t14, t14, t10;\n\t"     \
        "addc.cc.u64    t15, t15, t11;\n\t"     \
        "addc.cc.u64    t16, t16, t12;\n\t"     \
        "addc.u64       t17, t17, 0;\n\t"       \
        "mul.lo.u64     t10, %2, %16;\n\t"      \
        "mul.hi.u64     t11, %2, %16;\n\t"      \
        "mad.lo.cc.u64  t11, %2, %17, t11;\n\t" \
        "mul.hi.u64     t12, %2, %17;\n\t"      \
        "madc.lo.cc.u64 t12, %2, %18, t12;\n\t" \
        "mul.hi.u64     t13, %2, %18;\n\t"      \
        "madc.lo.cc.u64 t13, %2, %19, t13;\n\t" \
        "madc.hi.u64    %0, %2, %19, 0;\n\t"    \
        "add.cc.u64     t10, t10, t14;\n\t"     \
        "addc.cc.u64    t11, t11, t15;\n\t"     \
        "addc.cc.u64    t12, t12, t16;\n\t"     \
        "addc.cc.u64    t13, t13, t17;\n\t"     \
        "addc.u64       %0, %0, 0;\n\t"         \
        "mul.lo.u64     t14, %3, %16;\n\t"      \
        "mul.hi.u64     t15, %3, %16;\n\t"      \
        "mad.lo.cc.u64  t15, %3, %17, t15;\n\t" \
        "mul.hi.u64     t16, %3, %17;\n\t"      \
        "madc.lo.cc.u64 t16, %3, %18, t16;\n\t" \
        "mul.hi.u64     t17, %3, %18;\n\t"      \
        "madc.lo.cc.u64 t17, %3, %19, t17;\n\t" \
        "madc.hi.u64    %1, %3, %19, 0;\n\t"    \
        "add.cc.u64     t11, t11, t14;\n\t"     \
        "addc.cc.u64    t12, t12, t15;\n\t"     \
        "addc.cc.u64    t13, t13, t16;\n\t"     \
        "addc.cc.u64    t14, %0, t17;\n\t"      \
        "addc.u64       t15, %1, 0;\n\t"        \
        "mov.b64        %0, 0;\n\t"/*r = (c + u) div 2^(4*64)*/\
        "add.cc.u64     t0, t8, t0;\n\t"        \
        "addc.cc.u64    t1, t9, t1;\n\t"        \
        "addc.cc.u64    t2, t10, t2;\n\t"        \
        "addc.cc.u64    t3, t11, t3;\n\t"        \
        "addc.cc.u64    t4, t12, t4;\n\t"        \
        "addc.cc.u64    t5, t13, t5;\n\t"        \
        "addc.cc.u64    t6, t14, t6;\n\t"        \
        "addc.cc.u64    t7, t15, t7;\n\t"        \
        "addc.u64       %0, %0, 0;\n\t"         \
        "sub.cc.u64     t4, t4, %16;\n\t"       \
        "subc.cc.u64    t5, t5, %17;\n\t"       \
        "subc.cc.u64    t6, t6, %18;\n\t"       \
        "subc.cc.u64    t7, t7, %19;\n\t"       \
        "subc.u64       %0, %0, 0;\n\t"         \
        "mov.u64        %1, %0;\n\t"             \
        "mov.u64        %2, %0;\n\t"             \
        "mov.u64        %3, %0;\n\t"             \
        "and.b64        %0, %0, %16;\n\t"        \
        "and.b64        %1, %1, %17;\n\t"        \
        "and.b64        %2, %2, %18;\n\t"        \
        "and.b64        %3, %3, %19;\n\t"        \
        "add.cc.u64     %0, %0, t4;\n\t"        \
        "addc.cc.u64    %1, %1, t5;\n\t"        \
        "addc.cc.u64    %2, %2, t6;\n\t"        \
        "addc.u64       %3, %3, t7;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2), "=l"(c3)\
        : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(b0), "l"(b1), "l"(b2), "l"(b3),\
          "l"(m0), "l"(m1), "l"(m2), "l"(m3), "l"(p0), "l"(p1), "l"(p2), "l"(p3) )

__device__ void fp_add(fp_t c, const fp_t a, const fp_t b) {
    __fp_add(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], 
            __p[0], __p[1], __p[2], __p[3]);
}

__device__ void fp_sub(fp_t c, const fp_t a, const fp_t b) {
    __fp_sub(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
            __p[0], __p[1], __p[2], __p[3]);
}

__device__ void fp_mul(fp_t c, const fp_t a, const fp_t b) {
    __fp_mul(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
            __mu[0], __mu[1], __mu[2], __mu[3], __p[0], __p[1], __p[2], __p[3]);
}

/* TODO: Implement sqr */
__device__ void fp_sqr(fp_t c, const fp_t a) {
    __fp_mul(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], a[0], a[1], a[2], a[3],
            __mu[0], __mu[1], __mu[2], __mu[3], __p[0], __p[1], __p[2], __p[3]);
}
/*
__device__ __forceinline__ void fp_sqr(limb_t *c, const limb_t a) {
    __fp_sqr(c->x, c->y, c->z, c->w, a.x, a.y, a.z, a.w, __mu.x, __mu.y, __mu.z, __mu.w,
            __p.x, __p.y, __p.z, __p.w);
}
*/
