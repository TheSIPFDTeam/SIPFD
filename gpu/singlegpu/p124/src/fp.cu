#include "api.h"

#if RADIX == 64
/* Addition in Fp */
#define __fp_add(c0,c1,a0,a1,b0,b1,p0,p1)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        "add.cc.u64     %0, %2, %4;\n\t"        \
        "addc.cc.u64    %1, %3, %5;\n\t"        \
        "addc.u64       t0, 0, 0;\n\t"          \
        "sub.cc.u64     %0, %0, %6;\n\t"        \
        "subc.cc.u64    %1, %1, %7;\n\t"        \
        "subc.u64       t0, t0, 0;\n\t"         \
        "mov.u64        t1, t0;\n\t"            \
        "and.b64        t0, t0, %6;\n\t"        \
        "and.b64        t1, t1, %7;\n\t"        \
        "add.cc.u64     %0, %0, t0;\n\t"        \
        "addc.u64       %1, %1, t1;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1)                    \
        : "l"(a0), "l"(a1), "l"(b0), "l"(b1),   \
            "l"(p0), "l"(p1) )

/* Subtraction in Fp */
#define __fp_sub(c0,c1,a0,a1,b0,b1,p0,p1)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        "sub.cc.u64     %0, %2, %4;\n\t"        \
        "subc.cc.u64    %1, %3, %5;\n\t"        \
        "subc.u64       t0, 0, 0;\n\t"          \
        "mov.u64        t1, t0;\n\t"            \
        "and.b64        t0, t0, %6;\n\t"        \
        "and.b64        t1, t1, %7;\n\t"        \
        "add.cc.u64     %0, %0, t0;\n\t"        \
        "addc.u64       %1, %1, t1;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1)                    \
        : "l"(a0), "l"(a1), "l"(b0), "l"(b1),   \
            "l"(p0), "l"(p1) )

#define __fp_mul(c0,c1,a0,a1,b0,b1,m0,m1,p0,p1)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"     \
        ".reg .u64 t4;" ".reg .u64 t5;\n\t"     \
        ".reg .u64 t6;" ".reg .u64 t7;\n\t"     \
        "mul.lo.u64     t0, %2, %4;\n\t"        \
        "mul.hi.u64     t1, %2, %4;\n\t"        \
        "mad.lo.cc.u64  t1, %2, %5, t1;\n\t"    \
        "madc.hi.cc.u64 t2, %2, %5, 0;\n\t"     \
        "mad.lo.cc.u64  t1, %3, %4, t1;\n\t"    \
        "madc.hi.cc.u64 t2, %3, %4, t2;\n\t"    \
        "madc.lo.cc.u64 t2, %3, %5, t2;\n\t"    \
        "madc.hi.u64    t3, %3, %5, 0;\n\t"     \
        "mul.lo.u64     %0, t0, %6;\n\t"/*Montgomery reduction, m = mu*c mod 2^(2*64)*/\
        "mul.hi.u64     %1, t0, %6;\n\t"        \
        "mad.lo.u64     %1, t0, %7, %1;\n\t"    \
        "mad.lo.u64     %1, t1, %6, %1;\n\t"    \
        "mul.lo.u64     t4, %0, %8;\n\t"/* u = m*p */\
        "mul.hi.u64     t5, %0, %8;\n\t"        \
        "mad.lo.cc.u64  t5, %0, %9, t5;\n\t"    \
        "madc.hi.cc.u64 t6, %0, %9, 0;\n\t"     \
        "mad.lo.cc.u64  t5, %1, %8, t5;\n\t"    \
        "madc.hi.cc.u64 t6, %1, %8, t6;\n\t"    \
        "madc.lo.cc.u64 t6, %1, %9, t6;\n\t"    \
        "madc.hi.u64    t7, %1, %9, 0;\n\t"     \
        "mov.b64        %0, 0;\n\t"/*r = (c + u) div 2^(2*64)*/\
        "add.cc.u64     t0, t4, t0;\n\t"        \
        "addc.cc.u64    t1, t5, t1;\n\t"        \
        "addc.cc.u64    t2, t6, t2;\n\t"        \
        "addc.cc.u64    t3, t7, t3;\n\t"        \
        "addc.u64       %0, %0, 0;\n\t"         \
        "sub.cc.u64     t2, t2, %8;\n\t"        \
        "subc.cc.u64    t3, t3, %9;\n\t"        \
        "subc.u64       %0, %0, 0;\n\t"         \
        "mov.u64        %1, %0;\n\t"            \
        "and.b64        %0, %0, %8;\n\t"        \
        "and.b64        %1, %1, %9;\n\t"        \
        "add.cc.u64     %0, %0, t2;\n\t"        \
        "addc.u64       %1, %1, t3;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1)\
        : "l"(a0), "l"(a1), "l"(b0), "l"(b1),   \
          "l"(m0), "l"(m1), "l"(p0), "l"(p1) )

#define __fp_sqr(c0,c1,a0,a1,m0,m1,p0,p1)\
    asm volatile ("{\n\t"                           \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"         \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"         \
        ".reg .u64 t4;" ".reg .u64 t5;\n\t"         \
        ".reg .u64 t6;" ".reg .u64 t7;\n\t"         \
        ".reg .u64 t8;" ".reg .u64 t9;\n\t"         \
        "mul.lo.u64     t0, %2, %2;\n\t"            \
        "mul.hi.u64     t1, %2, %2;\n\t"            \
        "mul.lo.u64     t4, %2, %3;\n\t"            \
        "mul.hi.u64     t5, %2, %3;\n\t"            \
        "add.cc.u64     t1, t1, t4;\n\t"            \
        "addc.cc.u64    t2, t5, 0;\n\t"             \
        "addc.cc.u64    t1, t1, t4;\n\t"            \
        "addc.cc.u64    t2, t2, t5;\n\t"            \
        "madc.lo.cc.u64 t2, %3, %3, t2;\n\t"        \
        "madc.hi.u64    t3, %3, %3, 0;\n\t"         \
        "mul.lo.u64     %0, t0, %4;\n\t"/*Montgomery reduction, m = mu*c mod 2^(2*64)*/\
        "mul.hi.u64     %1, t0, %4;\n\t"        \
        "mad.lo.u64     %1, t0, %5, %1;\n\t"    \
        "mad.lo.u64     %1, t1, %4, %1;\n\t"    \
        "mul.lo.u64     t4, %0, %6;\n\t"/* u = m*p */\
        "mul.hi.u64     t5, %0, %6;\n\t"        \
        "mad.lo.cc.u64  t5, %0, %7, t5;\n\t"    \
        "madc.hi.cc.u64 t6, %0, %7, 0;\n\t"     \
        "madc.lo.cc.u64 t5, %1, %6, t5;\n\t"    \
        "madc.hi.cc.u64 t6, %1, %6, t6;\n\t"    \
        "madc.lo.cc.u64 t6, %1, %7, t6;\n\t"    \
        "madc.hi.u64    t7, %1, %7, 0;\n\t"     \
        "mov.b64        %0, 0;\n\t"/*r = (c + u) div 2^(2*64)*/\
        "add.cc.u64     t0, t4, t0;\n\t"        \
        "addc.cc.u64    t1, t5, t1;\n\t"        \
        "addc.cc.u64    t2, t6, t2;\n\t"        \
        "addc.cc.u64    t3, t7, t3;\n\t"        \
        "addc.u64       %0, %0, 0;\n\t"         \
        "sub.cc.u64     t2, t2, %6;\n\t"        \
        "subc.cc.u64    t3, t3, %7;\n\t"        \
        "subc.u64       %0, %0, 0;\n\t"         \
        "mov.u64        %1, %0;\n\t"            \
        "and.b64        %0, %0, %6;\n\t"        \
        "and.b64        %1, %1, %7;\n\t"        \
        "add.cc.u64     %0, %0, t2;\n\t"        \
        "addc.u64       %1, %1, t3;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1)\
        : "l"(a0), "l"(a1), "l"(m0), "l"(m1),   \
        "l"(p0), "l"(p1) )

__device__ void fp_add(fp_t out, const fp_t arg1, const fp_t arg2) {
    __fp_add(out[0], out[1], arg1[0], arg1[1], arg2[0], arg2[1], __p[0], __p[1]);
}

__device__ void fp_sub(fp_t out, const fp_t arg1, const fp_t arg2) {
    __fp_sub(out[0], out[1], arg1[0], arg1[1], arg2[0], arg2[1], __p[0], __p[1]);
}

__device__ void fp_mul(fp_t out, const fp_t arg1, const fp_t arg2) {
    __fp_mul(out[0], out[1], arg1[0], arg1[1], arg2[0], arg2[1], __mu[0], __mu[1],
            __p[0], __p[1]);
}

__device__ void fp_sqr(fp_t out, const fp_t arg1) {
    __fp_sqr(out[0], out[1], arg1[0], arg1[1], __mu[0], __mu[1], __p[0], __p[1]);
}

#elif RADIX == 32

// Generic three limb addition base on 32bit operations.
#define __fp_add(c0,c1,c2,a0,a1,a2,b0,b1,b2,p0,p1,p2)          \
	asm volatile("{\n\t"                                            \
	     ".reg .u32     t0, t1, t2; \n\t"                           \
	     "add.cc.u32    %0, %3, %6; \n\t"                           \
	     "addc.cc.u32   %1, %4, %7; \n\t"                           \
	     "addc.cc.u32   %2, %5, %8; \n\t"                           \
	     "addc.u32      t0,  0,  0; \n\t"                           \
	     "sub.cc.u32    %0, %0, %9; \n\t"                           \
	     "subc.cc.u32   %1, %1, %10;\n\t"                           \
	     "subc.cc.u32   %2, %2, %11;\n\t"                           \
    	 "subc.u32      t0, t0,  0; \n\t"                           \
         "and.b32       t1, t0, %10;\n\t"                           \
         "and.b32       t2, t0, %11;\n\t"                           \
         "and.b32       t0, t0, %9; \n\t"                           \
         "add.cc.u32    %0, %0, t0; \n\t"                           \
         "addc.cc.u32   %1, %1, t1; \n\t"                           \
         "addc.u32      %2, %2, t2; \n\t"                           \
	     "}"                                                        \
	: "=r"(c0), "=r"(c1), "=r"(c2) /*0,2*/                          \
	: "r"(a0), "r"(a1), "r"(a2),   /*3,5*/                          \
	  "r"(b0), "r"(b1), "r"(b2),   /*6,8*/                          \
	  "r"(p0), "r"(p1), "r"(p2));  /*9,11*/ 

// Generic 96bit, 3 limb subtraction base on 32bit operations.
#define __fp_sub(c0,c1,c2,a0,a1,a2,b0,b1,b2,p0,p1,p2)      \
    asm volatile ("{\n\t"                                       \
    ".reg .u32      t0, t1, t2; \n\t"                           \
    "sub.cc.u32     %0, %3, %6; \n\t"                           \
    "subc.cc.u32    %1, %4, %7; \n\t"                           \
	"subc.cc.u32    %2, %5, %8; \n\t"                           \
	"subc.u32       t0, 0, 0;\n\t"                              \
    "and.b32        t1, t0, %10;\n\t"                           \
	"and.b32        t2, t0, %11;\n\t"                           \
	"and.b32        t0, t0, %9 ;\n\t"                           \
	"add.cc.u32     %0, %0, t0;\n\t"                            \
	"addc.cc.u32    %1, %1, t1;\n\t"                            \
	"addc.u32       %2, %2, t2;\n\t"                            \
    "}"                                                         \
    : "=r"(c0), "=r"(c1), "=r"(c2) /*0-2*/                      \
    : "r"(a0), "r"(a1), "r"(a2),   /*3-5*/                      \
      "r"(b0), "r"(b1), "r"(b2),   /*6-8*/                      \
      "r"(p0), "r"(p1), "r"(p2));  /*9-11*/
 

// 3x3 schoolbook multiplicaiton using `mad` instructions
// uses 4 tmp registers
// result is a 6 limb value
// c5c4c3c2c1c0 = a2a1a0 * b2b1b0
#define __school32_3x3(c0,c1,c2,c3,c4,c5,a0,a1,a2,b0,b1,b2) \
	asm volatile ("{\n\t"                       			\
			".reg .u32 t0,t1,t2,tc;\n\t"                   	\
			/*Multiplication involving a0*/        			\
			"mul.lo.u32     %0, %6, %9;\n\t"      			\
			"mul.hi.u32     t0, %6, %9;\n\t"      			\
			"mad.lo.cc.u32  t0, %6, %10, t0;\n\t"  			\
			"mul.hi.u32     t1, %6, %10;\n\t"      			\
			"madc.lo.cc.u32 t1, %6, %11, t1;\n\t"  			\
			"mul.hi.u32     t2, %6, %11;\n\t"      			\
			"addc.cc.u32    t2, t2, 0;\n\t"         		\
			"addc.u32       tc, 0, 0;\n\t"          		\
			/*Multiplication involving a1 first diagonal*/ 	\
			"mad.lo.cc.u32  %1, %7, %9, t0;\n\t"    		\
			"madc.hi.cc.u32 t1, %7, %9, t1;\n\t"    		\
			"madc.hi.cc.u32 t2, %7, %10,t2;\n\t"    		\
			"mul.hi.u32     t0, %7, %11;\n\t"        		\
			"addc.cc.u32    t0, t0 , tc;\n\t"    			\
			/*Multiplication involving a1 second diagonal*/	\
			"mad.lo.cc.u32  t1, %7, %10, t1;\n\t"   		\
			"madc.lo.cc.u32 t2, %7, %11, t2;\n\t"   	 	\
			"addc.cc.u32    t0, t0 , 0;\n\t"     			\
			"addc.u32       tc, tc , 0;\n\t"        		\
			/*Multiplication involving a2 first diagonal*/ 	\
			"mad.lo.cc.u32  %2, %8, %9, t1;\n\t"   	 		\
			"madc.hi.cc.u32 t2, %8, %9, t2;\n\t"   	  		\
			"madc.hi.cc.u32 t0, %8, %10,t0;\n\t"   	  		\
			"mul.hi.u32     t1, %8, %11;\n\t"       	  	\
			"addc.cc.u32    t1, t1 , tc;\n\t"    			\
			"addc.u32       tc, 0 , 0;\n\t"         		\
			/*Multiplication involving a2 second diagonal*/	\
			"mad.lo.cc.u32  %3, %8, %10, t2;\n\t"    		\
			"madc.lo.cc.u32 %4, %8, %11, t0;\n\t"    		\
			"addc.u32       %5, t1 , 0;\n\t"     			\
			"}\n"                                       	\
	: "=r"(c0), "=r"(c1), "=r"(c2), /*0,  2*/ 				\
	  "=r"(c3), "=r"(c4), "=r"(c5)  /*3,  5*/ 				\
	: "r"(a0), "r"(a1), "r"(a2),    /*6 , 8*/ 				\
	  "r"(b0), "r"(b1), "r"(b2));   /*9 ,11*/

// first part of montgomery reduction 
// essential its a multiplication a*mu mod 2**96
// This version does not use the 'mad' instructions
// needs 2 helper registers
// NonInplace Version: 
#define __school32_3x3_low(c0,c1,c2, a0,a1,a2, b0,b1,b2)			\
	asm volatile ("{\n\t"                       					\
			".reg .u32 t10,t20;\n\t"            					\
			/* t0-t2: m = a * mu mod 2^96 */    					\
			/* Multiplication involving a0 */   					\
			"mul.lo.u32     %0,  %6, %3;\n\t"						\
			"mul.hi.u32     %1,  %6, %3;\n\t"   					\
			"mul.lo.u32     t10, %6, %4;\n\t"   					\
			"mul.hi.u32     %2,  %6, %4;\n\t"   					\
			"mul.lo.u32     t20, %6, %5;\n\t"   					\
			/* now fixing the carry chains */   					\
  			"add.cc.u32     %1,  %1,  t10;\n\t" 					\
			"addc.u32       %2,  %2,  t20;\n\t" 					\
			/*Multiplication involving a1 second (low) diagonal*/   \
			"mul.lo.u32     t10, %7, %3;\n\t"   					\
			"mul.lo.u32     t20, %7, %4;\n\t"   					\
			"add.cc.u32     %1, %1, t10;\n\t"   					\
			"addc.u32       %2, %2, t20;\n\t"   					\
			/*Multiplication involving a1 first (high) diagonal*/ 	\
			"mul.hi.u32     t20, %7, %3;\n\t"   					\
			"add.u32        %2, %2, t20;\n\t"   					\
			/*Multiplication involving a2 second (low) diagonal*/  	\
			"mul.lo.u32     t20, %8,  %3;\n\t"  					\
			"add.u32         %2, %2, t20;\n\t"  					\
			"}\n"		                          					\
		: "=r"(c0),"=r"(c1),"=r"(c2)   /*0 , 2*/ 					\
		: "r"(b0),"r"(b1),"r"(b2)      /*3 , 5*/ 					\
		  "r"(a0),"r"(a1),"r"(a2)  	   /*6 , 8*/					\
	);

// second part of the montgomery reduction written by floyd
// NonInplace Version
// a2a1a0 = (i5i4i3i2i1i0) mod 2**(3*32)
#define __reduce32_sec3x3(c0,c1,c2, a0,a1,a2,a3,a4,a5, p0,p1,p2, i0,i1,i2,i3,i4,i5) \
	asm volatile ("{\n\t"                   							\
			/*%0-%3: r= (a + u) div 2^(3**32)*/      					\
			".reg.u32 		t0,t3,t4,t5;\n\t"							\
			/*a + i*/													\
			"add.cc.u32     %0, %12, %6;\n\t"        					\
			"addc.cc.u32    %1, %13, %7;\n\t"        					\
			"addc.cc.u32    %2, %14, %8;\n\t"        					\
			"addc.cc.u32    t3, %15, %9;\n\t"        					\
			"addc.cc.u32    t4, %16, %10;\n\t"       					\
			"addc.cc.u32    t5, %17, %11;\n\t"       					\
			"addc.u32       t0, 0, 0;\n\t"         						\
			/*-p*/														\
			"sub.cc.u32     t3, t3, %3;\n\t"       						\
			"subc.cc.u32    t4, t4, %4;\n\t"       						\
			"subc.cc.u32    t5, t5, %5;\n\t"       						\
			"subc.u32       t0, t0, 0;\n\t"         					\
			/*&p*/														\
			"and.b32        %0, t0, %3;\n\t"       						\
			"and.b32        %1, t0, %4;\n\t"       						\
			"and.b32        %2, t0, %5;\n\t"       						\
			"add.cc.u32     %0, %0, t3;\n\t"        					\
			"addc.cc.u32    %1, %1, t4;\n\t"        					\
			"addc.u32    	%2, %2, t5;\n\t"        					\
			"}\n"                                     					\
		: "=r"(c0),"=r"(c1),"=r"(c2)	/*0 , 2*/ 						\
		: "r"(p0),"r"(p1),"r"(p2),	 	/*3 , 5*/  						\
		  "r"(i0),"r"(i1),"r"(i2), 		/*6 , 8*/  						\
		  "r"(i3),"r"(i4),"r"(i5),      /*9, 11*/  						\
		  "r"(a0),"r"(a1),"r"(a2), 		/*12,14*/  						\
		  "r"(a3),"r"(a4),"r"(a5) 		/*15,17*/  						\
	);


// 3x3 schoolbook mutliplication over Fp.
#define __fp_mul(c0,c1,c2, a0,a1,a2, b0,b1,b2, mu0,mu1,mu2, p0,p1,p2)			\
	uint32_t f0,f1,f2,f3,f4,f5,u0,u1,u2,u3,u4,u5; 								\
	__school32_3x3(f0,f1,f2,f3,f4,f5, a0,a1,a2, b0,b1,b2)						\
	__school32_3x3_low(c0,c1,c2, f0,f1,f2, mu0,mu1,mu2)							\
	__school32_3x3(u0,u1,u2,u3,u4,u5, c0,c1,c2, p0,p1,p2)   					\
	__reduce32_sec3x3(c0,c1,c2, f0,f1,f2,f3,f4,f5, p0,p1,p2, u0,u1,u2,u3,u4,u5)

#define __fp_sqr(c0,c1,c2,a0,a1,a2,m0,m1,m2,p0,p1,p2)\
    asm volatile ("{\n\t"                           \
        ".reg .u32 t0;" ".reg .u32 t1;\n\t"         \
        ".reg .u32 t2;" ".reg .u32 t3;\n\t"         \
        ".reg .u32 t4;" ".reg .u32 t5;\n\t"         \
        ".reg .u32 t6;" ".reg .u32 t7;\n\t"         \
        ".reg .u32 t8;" ".reg .u32 t9;\n\t"         \
        ".reg .u32 t10;" ".reg .u32 t11;\n\t"       \
        ".reg .u32 t12;" ".reg .u32 t13;\n\t"       \
        "mul.lo.u32     t0, %3, %3;\n\t"        \
        "mul.hi.u32     t1, %3, %3;\n\t"        \
        "mul.lo.u32     t8, %3, %4;\n\t"        \
        "mul.hi.u32     t9, %3, %4;\n\t"        \
        "mul.lo.u32     t10, %3, %5;\n\t"       \
        "mul.hi.u32     t11, %3, %5;\n\t"       \
        "add.cc.u32     t1, t1, t8;\n\t"        \
        "addc.cc.u32    t2, t9, t10;\n\t"       \
        "addc.u32       t3, t11, 0;\n\t"        \
        "mul.lo.u32     t12, %4, %5;\n\t"       \
        "mul.hi.u32     t13, %4, %5;\n\t"       \
        "mul.hi.u32     t6, %4, %4;\n\t"        \
        "mad.lo.cc.u32  t5, %4, %4, t9;\n\t"    \
        "addc.cc.u32    t6, t6, t12;\n\t"       \
        "addc.u32       t7, t13, 0;\n\t"        \
        "add.cc.u32     t1, t1, t8;\n\t"        \
        "addc.cc.u32    t5, t5, t2;\n\t"        \
        "addc.cc.u32    t6, t6, t3;\n\t"        \
        "addc.u32       t7, t7, 0;\n\t"         \
        "mul.lo.u32     t4, %5, %5;\n\t"        \
        "mul.hi.u32     t8, %5, %5;\n\t"        \
        "add.cc.u32     t3, t11, t12;\n\t"      \
        "addc.cc.u32    t4, t4, t13;\n\t"       \
        "addc.cc.u32    t8, t8, 0;\n\t"         \
        "add.cc.u32     t2, t10, t5;\n\t"       \
        "addc.cc.u32    t3, t3, t6;\n\t"        \
        "addc.cc.u32    t4, t4, t7;\n\t"        \
        "addc.u32       t5, t8, 0;\n\t"         \
        "mov.u32        %0, t0;\n\t"/*Montgomery reduction, m = mu*c mod 2^(32*3)*/\
        "mad.lo.cc.u32  %1, %7, t0, t1;\n\t"    \
        "madc.hi.u32    %2, %7, t0, t2;\n\t"    \
        "mad.lo.u32     %2, %7, t1, %2;\n\t"    \
        "mad.lo.u32     %2, %8, t0, %2;\n\t"    \
        "mul.lo.u32     t6, %0, %9;\n\t"/* u = m*p */\
        "mul.hi.u32     t7, %0, %9;\n\t"        \
        "mad.lo.cc.u32  t7, %0, %10, t7;\n\t"   \
        "mul.hi.u32     t8, %0, %10;\n\t"       \
        "madc.lo.cc.u32 t8, %0, %11, t8;\n\t"   \
        "madc.hi.u32    t9, %0, %11, 0;\n\t"    \
        "mul.lo.u32     t10, %1, %9;\n\t"       \
        "mul.hi.u32     t11, %1, %9;\n\t"       \
        "mad.lo.cc.u32  t11, %1, %10, t11;\n\t" \
        "mul.hi.u32     %0, %1, %10;\n\t"       \
        "madc.lo.cc.u32 %0, %1, %11, %0;"       \
        "madc.hi.u32    %1, %1, %11, 0;\n\t"    \
        "add.cc.u32     t7, t7, t10;\n\t"       \
        "addc.cc.u32    t11, t11, t8;\n\t"      \
        "addc.cc.u32    %0, %0, t9;\n\t"        \
        "addc.u32       %1, %1, 0;\n\t"         \
        "mul.lo.u32     t8, %2, %9;\n\t"        \
        "mul.hi.u32     t9, %2, %9;\n\t"        \
        "mad.lo.cc.u32  t9, %2, %10, t9;\n\t"   \
        "mul.hi.u32     t10, %2, %10;\n\t"      \
        "madc.lo.cc.u32 t10, %2, %11, t10;\n\t" \
        "madc.hi.u32    %2, %2, %11, 0;\n\t"    \
        "add.cc.u32     t8, t8, t11;\n\t"       \
        "addc.cc.u32    t9, t9, %0;\n\t"        \
        "addc.cc.u32    t10, t10, %1;\n\t"      \
        "addc.u32       t11, %2, 0;\n\t"        \
        "mov.b32        %0, 0;\n\t"/*r = (c + u) div 2^(32*3)*/\
        "add.cc.u32     t0, t6, t0;\n\t"        \
        "addc.cc.u32    t1, t7, t1;\n\t"        \
        "addc.cc.u32    t2, t8, t2;\n\t"        \
        "addc.cc.u32    t3, t9, t3;\n\t"        \
        "addc.cc.u32    t4, t10, t4;\n\t"       \
        "addc.cc.u32    t5, t11, t5;\n\t"       \
        "addc.u32       %0, %0, 0;\n\t"         \
        "sub.cc.u32     t3, t3, %9;\n\t"        \
        "subc.cc.u32    t4, t4, %10;\n\t"       \
        "subc.cc.u32    t5, t5, %11;\n\t"       \
        "subc.cc.u32    %0, %0, 0;\n\t"         \
        "mov.u32        %1, %0;\n\t"            \
        "mov.u32        %2, %0;\n\t"            \
        "and.b32        %0, %0, %9;\n\t"        \
        "and.b32        %1, %1, %10;\n\t"       \
        "and.b32        %2, %2, %11;\n\t"       \
        "add.cc.u32     %0, %0, t3;\n\t"        \
        "addc.cc.u32    %1, %1, t4;\n\t"        \
        "addc.u32       %2, %2, t5;\n\t"        \
        "}"                                     \
        : "=r"(c0), "=r"(c1), "=r"(c2)			\
        : "r"(a0), "r"(a1), "r"(a2), 			\
 		  "r"(m0), "r"(m1), "r"(m2),			\
          "r"(p0), "r"(p1), "r"(p2));			\


__device__ void fp_add(fp_t out, const fp_t arg1, const fp_t arg2) {
    __fp_add(out[0], out[1], out[2],
			arg1[0], arg1[1], arg1[2],
			arg2[0], arg2[1], arg2[2],
			__p[0], __p[1], __p[2]);
}

__device__ void fp_sub(fp_t out, const fp_t arg1, const fp_t arg2) {
    __fp_sub(out[0], out[1], out[2],
			arg1[0], arg1[1], arg1[2],
			arg2[0], arg2[1], arg2[2],
			__p[0], __p[1], __p[2]);
}

__device__ void fp_mul(fp_t out, const fp_t arg1, const fp_t arg2) {
    __fp_mul(out[0], out[1], out[2],
			arg1[0], arg1[1], arg1[2],
			arg2[0], arg2[1], arg2[2],
			__mu[0], __mu[1], __mu[2],
			__p[0], __p[1], __p[2]);
}

__device__ void fp_sqr(fp_t out, const fp_t arg1) {
    __fp_sqr(out[0], out[1], out[2],
			arg1[0], arg1[1], arg1[2],
			__mu[0], __mu[1], __mu[2],
			__p[0], __p[1], __p[2]);
}

#else
#error "Not implemented"
#endif
