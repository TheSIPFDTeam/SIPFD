#include "api.h"
#include "vowgcs_setup.h"

/* ----------------------------------------------------------------------------- *
 *  point_compare()
 *  Inputs: two points a and b
 *  Output:
 *           0  if a = c,
 *          -1  if a < c, or
 *           1  if a > c.
 * ----------------------------------------------------------------------------- */
__device__ int point_compare(point_t *a, point_t *b) {
    if (a->c < b->c)
        return -1;
    else if (a->c > b->c)
        return 1;
    else if (a->k < b->k)
        return -1;
    else if (a->k > b->k)
        return 1;
    else
        return 0;
}

/* ------------------------------------------------------------------------------ *
   The next function reconstruct a collision between two trail that converge into
   a same distinguished point.
                     *      <~~~~~~ Distinguished point
                     |
                     *
                     .
                     .
                     .
                     *
                     |
                     *      <~~~~~~ Collision
                    / \
                   *   *
                  /     \
                 *       *
                .         .
               .           .
              .             .
             *               *
 * ------------------------------------------------------------------------------ */
__device__ limb_t reconstruction(point_t *collision, vowgcs_t *X, vowgcs_t *Y, 
        limb_t strategy[2][EXP0], fp2_t *P0, fp2_t *Q0, fp2_t *PQ0, fp2_t *E0, fp2_t *Z0, 
        fp2_t *P1, fp2_t *Q1, fp2_t *PQ1, fp2_t *E1, fp2_t *Z1, ctx_t *context, limb_t *expo, 
        limb_t *ebits) {
    limb_t i = 0, counter = 0;
    limb_t LENGTH;
    
    fp2_t Jx, Jy;
    point_t X_i, X_k = {0}, Y_i, Y_k = {0};

    X_i.k = X->seed.k;
    X_i.c = X->seed.c;
    Y_i.k = Y->seed.k;
    Y_i.c = Y->seed.c;

    if(X->length < Y->length) {
        LENGTH = X->length;
        for(i = X->length; i < Y->length; i++) {
            counter += 1;
            _fn_(&Y_i, Jy, &Y_i, context, strategy, P0, Q0, PQ0, E0, Z0, P1, Q1, PQ1, E1, Z1, 
                    expo, ebits);
        }
    }
    else {
        LENGTH = Y->length;
        for(i = Y->length; i < X->length; i++) {
            counter += 1;
            _fn_(&X_i, Jx, &X_i, context, strategy, P0, Q0, PQ0, E0, Z0, P1, Q1, PQ1, E1, Z1, 
                    expo, ebits);
        }
    }

    for (i = 0; i < LENGTH; i++) {
        counter += 2;
        X_k.k = X_i.k;
        X_k.c = X_i.c;
        Y_k.k = Y_i.k;
        Y_k.c = Y_i.c;
        _fn_(&X_i, Jx, &X_i, context, strategy, P0, Q0, PQ0, E0, Z0, P1, Q1, PQ1, E1, Z1, 
                expo, ebits);
        _fn_(&Y_i, Jy, &Y_i, context, strategy, P0, Q0, PQ0, E0, Z0, P1, Q1, PQ1, E1, Z1, 
                expo, ebits);

        if (fp_compare(Jx[0], Jy[0]) == 0)
            break;
    }

    // Special case when the collision is given at the tail (distinguished collision)
    // The golden collision must satisfy that the j-invariants are equal. If not, it
    // is a collision determined by the hash function MD5. Consequently, in order to
    // decide if the reached collision is the golden one, we must ensure that the
    // j-invariants J0 and J1 are equal.
    if ((counter == (X->length + Y->length)) && (fp2_compare(Jx, Jy) != 0)) {
        /* No golden collision reached at the tail of each trail

                                   *
                                  / \
                                 *   *
                                      \
                                       *

        *  In this case, the j-invariants never matched, so it is either a collision
        *  of the hash or a collision across different functions. We default to returning
        *  identical points so that the collision will be discarded by vowgcs.
        */
        X_k.k = X_i.k;
        X_k.c = X_i.c;
        Y_k.k = X_i.k;
        Y_k.c = X_i.c;
    }

    // Has the golden collision been reached? if X_k != Y_k, then we have found it!!!
    collision[0].k = X_k.k;
    collision[0].c = X_k.c;
    collision[1].k = Y_k.k;
    collision[1].c = Y_k.c;
    return counter;
}

// If point is distinguished, returns 1 and writes the address
// Note: this assumes both log(1/theta) and log(w) are less than sizeof(digit_t)
__device__ uint8_t is_distinguished(uint64_t *address, point_t *point, ctx_t *context) {
    uint64_t tmp;

    tmp = point->k ^ context->NONCE;

    // First n bits must be 0
    if ((tmp & ((1 << N) - 1)) != 0)
            return 0;
    else
        tmp >>= N;

    // Address is determined by the next omegabits bits
    *address = tmp & (((uint64_t)1 << OMEGABITS) - 1);
    tmp >>= OMEGABITS;

    // Next Rbits bits must be less than the threshold
    if ( (tmp & ( (1 << RBITS) - 1 )) >= DISTINGUISHED)
        return 0;
    else
        return 1;
}

// Compresses seed.k|seed.c|tail.k[useful bits only]|tail.c|length to a sequence of bytes
__device__ void point_compress(uint8_t *target, const vowgcs_t *point) {
    int byte = 0, bit = 0, i;

    for(byte = 0; byte < TRIPLETBYTES; byte++)
        target[byte] = 0;

    for(byte = 0; 8*byte < EBITS_MAX; byte++)
        target[byte] = (point->seed.k >> (8*byte)) & 0xFF;

    bit = EBITS_MAX & 7;
    byte -= (bit+7)/8;
    target[byte] += (point->seed.c & 1) << bit;
    bit = (bit + 1) & 7;
    byte += 1 - (bit+7)/8;
    target[byte] += ((point->tail.k >> (OMEGABITS + N)) & ( (1 << (8-bit)) - 1))<<bit;
    
    for(i = 0; 8*i + 8 - bit < EBITS_MAX - OMEGABITS - N; i++) {
        byte++;
        target[byte] = (point->tail.k >> (OMEGABITS + N + 8*i + 8 - bit)) & 0xFF;
    }

    bit = (bit + EBITS_MAX - OMEGABITS - N) & 7;
    byte += 1 - (bit+7)/8;
    target[byte] += (point->tail.c & 1) << bit;
    bit = (bit + 1) & 7;
    byte += 1 - (bit+7)/8;
    target[byte] += ((point->length) & ( (1 << (8-bit)) - 1)) << bit;

    for(i = 0; 8*i + 8 - bit < TRIALBITS; i++) {
        byte++;
        target[byte] = (point->length >> (8*i + 8 - bit)) & 0xFF;
    }

    bit = (bit + TRIALBITS) & 7;
    byte += 1 - (bit+7)/8;
    target[byte] += 1 << bit;        // Add a dirty bit
}

// Reads a compressed point, replacing length=0 if the dirty bit was off
__device__ void point_uncompress(vowgcs_t *point, uint8_t *source, uint64_t address, ctx_t *context) {
    int byte = 0, bit = 0, i;

    point->tail.k = 0;
    point->seed.k = 0;
    point->length = 0;

    for(byte = 0; 8*byte < EBITS_MAX; byte++)
        point->seed.k ^= (((uint64_t)(source[byte])) << 8*byte);

    bit = EBITS_MAX & 7;
    byte -= (bit+7)/8;
    point->seed.k &= ( ((uint64_t)1 << EXP1) - 1 );
    point->seed.c = (source[byte] >> bit) & 1;
    bit = (bit + 1) & 7;
    byte += 1 - (bit+7)/8;
    point->tail.k += (source[byte] >> bit);

    for(i = 0; 8*i + 8 - bit < EBITS_MAX - OMEGABITS - N; i++) {
        byte++;
        point->tail.k ^= (((uint64_t)(source[byte])) << (8*i + 8 - bit));
    }

    bit = (bit + EBITS_MAX - OMEGABITS - N) & 7;
    byte += 1 - (bit+7)/8;
    point->tail.k &= ( ((uint64_t)1 << (EBITS_MAX - OMEGABITS - N)) - 1 );
    point->tail.k <<= OMEGABITS;
    point->tail.k ^= address & (((uint64_t)1 << OMEGABITS) - 1 );
    point->tail.k <<= N;
    point->tail.k ^= context->NONCE & ( ((uint64_t)1 << (OMEGABITS + N)) - 1 );
    point->tail.c = (source[byte] >> bit) & 1;
    bit = (bit + 1) & 7;
    byte += 1 - (bit+7)/8;
    point->length = source[byte] >> bit;

    for(i = 0; 8*i + 8 - bit < TRIALBITS; i++) {
        byte++;
        point->length ^= (limb_t)(((limb_t)(source[byte])) << (8*i + 8 - bit));
    }

    bit = (bit + TRIALBITS) & 7;
    byte += 1 - (bit+7)/8;
    point->length &= ( 1 << TRIALBITS ) - 1;
    point->length *= (source[byte]>>bit) & 1; //If dirty bit was off, replace length=0
}

/* ----------------------------------------------------------------------------- *
   Finally, the next function is an implementation of the main block of the vOW
   GCS procedure, which corresponds with the golden collision search given a
   Pseudo Random Function (PRF). The maximum number of distinguished points per
   PRF is reached in a parallel model (i.e., the task is split into the threads).
 * ----------------------------------------------------------------------------- */
__device__ void vowgcs(point_t *golden, uint8_t *finished, uint8_t *hashtable, 
        limb_t strategy_reduced[2][EXP0], fp2_t *P0, fp2_t *Q0, fp2_t *PQ0, fp2_t *E0, fp2_t *Z0, 
        fp2_t *P1, fp2_t *Q1, fp2_t *PQ1, fp2_t *E1, fp2_t *Z1, ctx_t *context, limb_t *expo, 
        limb_t *ebits, curandStatePhilox4_32_10_t *state) 
{
    limb_t distinguished_points = 0;

    limb_t length;
    uint64_t address = 0;

    point_t seed = {0}, tail = {0};
    fp2_t J = {0};
    vowgcs_t node0, node1;
    point_t temporal_collision[2];

    while( (*finished == 0) && (distinguished_points < BETAXOMEGA) ) {
        // Random element in fp2
        fp2_random(J, state);

        // Random seed
        _gn_(&seed, J, context->NONCE);
        // tail of the trail
        tail.k = seed.k;
        tail.c = seed.c;
        
        length = 1;
        while(length < MAXTRAIL) {
            // function evaluation
            _fn_(&tail, J, &tail, context, strategy_reduced, P0, Q0, PQ0, E0, Z0, 
                    P1, Q1, PQ1, E1, Z1, expo, ebits);
        
            // t LSBs from MD5(2|X|NONCE)
            if(is_distinguished(&address, &tail, context))
                break;
            length += 1;
        }

        //runtime_distinguished += length;
        if(length < MAXTRAIL) {
            // New distinguished point reached
            distinguished_points += 1;
            node0.seed.k = seed.k;
            node0.seed.c = seed.c;
            node0.tail.k = tail.k;
            node0.tail.c = tail.c;
            node0.length = length;

            // -----------------------------------------------------------------------------------------
            // In radix-tree implementation this must be changed!
            // Accessing to the stored distinguished point
            point_uncompress(&node1, &hashtable[address*TRIPLETBYTES], address, context);
            //printf("Found 0x%lX, %d, 0x%lX, %d, %ld at 0x%lX\n",
            //        node0.seed.k, node0.seed.c, node0.tail.k, node0.tail.c, node0.length, address);
            //printf("Read 0x%lX, %d, 0x%lX, %d, %ld from 0x%lX\n",
            //        node1.seed.k, node1.seed.c, node1.tail.k, node1.tail.c, node1.length, address);
            // -----------------------------------------------------------------------------------------
            if((node1.length > 0) && (point_compare(&node0.tail, &node1.tail) == 0)) {
                // Reconstructing the collision
                //collisions += 1;
                reconstruction(temporal_collision, &node0, &node1, strategy_reduced, 
                        P0, Q0, PQ0, E0, Z0, P1, Q1, PQ1, E1, Z1, context, expo, ebits);

                // Is the golden collision? (collision with different points)
                if (temporal_collision[0].c != temporal_collision[1].c) {
                    // Storing the golden collision
                    //*finished = 1;
                    printf("//finished: %d thread: %d\n", *finished, threadIdx.x + blockIdx.x * blockDim.x);
                    printf("//point k0: 0x%lX\n", temporal_collision[0].k);
                    printf("//point k1: 0x%lX\n", temporal_collision[1].k);
                    printf("//c0: %d c1: %d\n", temporal_collision[0].c, temporal_collision[1].c);

                    uint8_t index = temporal_collision[0].c & 0x1;
                    golden[index].k = temporal_collision[0].k;
                    golden[index].c = temporal_collision[0].c;

                    index = temporal_collision[1].c & 0x1;
                    golden[index].k = temporal_collision[1].k;
                    golden[index].c = temporal_collision[1].c;
                    break;
                }
            }

            // Finally, old distinguished point is always replaced by the new one reached
            if(*finished == 0) {
                // We must ensure do not overwrite the golden collision!
                // --------------------------------------------------
                // In radix-tree implementation this must be modified
                point_compress(&hashtable[address*TRIPLETBYTES], &node0);
                //printf("Saved 0x%lX, %d, 0x%lX, %d, %ld to 0x%lX\n",
                //        node0.seed.k, node0.seed.c, node0.tail.k, node0.tail.c, node0.length, address);
                // --------------------------------------------------
            }
        }
    }
}

