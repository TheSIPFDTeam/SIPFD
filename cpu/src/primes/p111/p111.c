#include "p111_api.h"
#include "../../rng.h"

// Namespace concerning GF(p)
#if defined(_assembly_)
	#include "p111asm_api.h"
#else
	#define fp_mul fiat_fp111_mul
	#define fp_sqr fiat_fp111_square
	#define fp_add fiat_fp111_add
	#define fp_sub fiat_fp111_sub
	#define fp_neg fiat_fp111_opp
	#define from_montgomery fiat_fp111_from_montgomery
	#define to_montgomery fiat_fp111_to_montgomery
	#define fp_nonzero fiat_fp111_nonzero
	#define to_bytes fiat_fp111_to_bytes
	#define from_bytes fiat_fp111_from_bytes
	#define fp_set_one fiat_fp111_set_one
	#define fp_random fiat_fp111_random
	#define fp_copy fiat_fp111_copy
	#define fp_compare fiat_fp111_compare
	#define fp_iszero fiat_fp111_iszero
	#define fp_string fiat_fp111_string
	#define fp_printf fiat_fp111_printf
	#define fp_pow fiat_fp111_pow
	#define fp_inv fiat_fp111_inv
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
