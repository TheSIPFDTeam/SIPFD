#include "api.h"

/* ----------------------------------------------------------------------------- *
 * Recursive function to precompute the 2-isogeny tree of a given depth. 
 * Nodes are saved to pc_table in the form of {P, Q, PQ, E} at the address
 * corresponding to the reverse of the least `depth` bits of k.
 * ----------------------------------------------------------------------------- */
void precompute(fp2_t *P, fp2_t *Q, fp2_t *PQ, fp2_t *E, fp2_t *Z, uint64_t path, 
    const proj_t basis[3], const proj_t curve, int e, const int level, const int depth)
{
    if (level == depth)
    {
        // Reverse of path
        uint64_t address = 0;
        fp2_t t0, t1, t2;

        for(int i = 0; i < depth; i++)
        {
            address <<= 1;
            address += (path >> i) & 1;
        }
        
        fp2_mul(t0, basis[0][1], basis[1][1]);
        fp2_mul(t1, t0, basis[2][1]);
        fp2_mul(E[address], curve[0], t1);
        fp2_mul(t1, t0, curve[1]);
        fp2_mul(PQ[address], basis[2][0], t1);
        fp2_mul(t1, basis[2][1], curve[1]);
        fp2_mul(t2, t1, basis[0][1]);
        fp2_mul(Q[address], basis[1][0], t2);
        fp2_mul(t2, t1, basis[1][1]);
        fp2_mul(P[address], basis[0][0], t2);
        fp2_mul(Z[address], t0, t1);
    }
    else
    {
        proj_t G0, B2, G2, DUAL, G0_diff, next_basis[3], next_curve;

        xdble(G0, e - level - 1, basis[0], curve);  // x([2^(e[c] - i - 1)]P)
        xdbl(DUAL, basis[1], curve);                // x([2]Q)
        xadd(B2, basis[0], basis[1], basis[2]);     // x(P + Q)
        xdble(G2, e - level - 1, B2, curve);        // x([2^(e[c] - i - 1)](P + Q)
        xadd(G0_diff, basis[2], basis[1], basis[0]);// x(P - [2]Q)

        // Branch corresponding to x(G0) = x([2^(e[c] - i - 1]P)

        xisog_2(next_curve, G0);                // 2-isogenous curve
        xeval_2(next_basis[0], basis[0], G0);        // 2-isogeny evaluation of x(P)
        xeval_2(next_basis[1], DUAL, G0);       // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_basis[2], G0_diff, G0);    // 2-isogeny evaluation of x([2](P - [2]Q))
        
        // Go to the next depth-level
        precompute(P, Q, PQ, E, Z, path << 1, next_basis, next_curve, e, level+1, depth); 

        // Branch corresponding to x(G2) = x(G0 + G1) = x([2^(e[c] - i - 1](P + Q))

        xisog_2(next_curve, G2);            // 2-isogenous curve
        xeval_2(next_basis[0], B2, G2);     // 2-isogeny evaluation of x(P+Q)
        xeval_2(next_basis[1], DUAL, G2);   // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_basis[2], basis[2], G2);   // 2-isogeny evaluation of x(P - Q)

        // Go to the next depth-level
        precompute(P, Q, PQ, E, Z, (path << 1) + 1, next_basis, next_curve, e, 
                level+1, depth); 
    }
}

