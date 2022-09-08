#include "p107_api.h"
#include "../../rng.h"

// Namespace concerning GF(p)
#if defined(_assembly_)
	#include "p107asm_api.h"
#else
	#define fp_mul fiat_fp107_mul
	#define fp_sqr fiat_fp107_square
	#define fp_add fiat_fp107_add
	#define fp_sub fiat_fp107_sub
	#define fp_neg fiat_fp107_opp
	#define from_montgomery fiat_fp107_from_montgomery
	#define to_montgomery fiat_fp107_to_montgomery
	#define fp_nonzero fiat_fp107_nonzero
	#define to_bytes fiat_fp107_to_bytes
	#define from_bytes fiat_fp107_from_bytes
	#define fp_set_one fiat_fp107_set_one
	#define fp_random fiat_fp107_random
	#define fp_copy fiat_fp107_copy
	#define fp_compare fiat_fp107_compare
	#define fp_iszero fiat_fp107_iszero
	#define fp_string fiat_fp107_string
	#define fp_printf fiat_fp107_printf
	#define fp_pow fiat_fp107_pow
	#define fp_inv fiat_fp107_inv
#endif

// GF(p²) implementation
#include "../../fpx.c"

// Short Weierstrass model
#if defined(_shortw_)
// Curve arithmetic
#include "../../shortw/curvemodel.c"
// Utility functions
#include "../../shortw/utils.c"
// MITM
#if defined(_mitm_)
#include "../../shortw/mitm-basic.c"
#include "../../shortw/mitm-dfs-2.c"
#include "../../shortw/mitm-dfs-3.c"
#endif
// vOW GCS
#if defined(_vowgcs_)
#include "../../shortw/vowgcs.c"
#endif

// Montgomery model
#elif defined(_mont_)
// Curve arithmetic
#include "../../mont/curvemodel.c"
// Utility functions
#include "../../mont/utils.c"
// MITM
#if defined(_mitm_)
#include "../../mont/mitm-basic.c"
#include "../../mont/mitm-dfs-2.c"
#include "../../mont/mitm-dfs-3.c"
#endif
// vOW GCS
#if defined(_vowgcs_)
#include<omp.h>
#include "../../mont/pcs_vect_bin.c"
#include "../../mont/pcs_struct_PRTL.c"
#include "../../mont/vowgcs.c"
#endif

#else
#error -- "Unsupported Curve Model"
#endif
