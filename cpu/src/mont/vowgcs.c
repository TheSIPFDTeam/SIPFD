int point_compare(const point_t *a, const point_t *b)
{
    /* ----------------------------------------------------------------------------- *
     *  point_compare()
     *  Inputs: two points a and b
     *  Output:
     *           0  if a = c,
     *          -1  if a < c, or
     *           1  if a > c.
     * ----------------------------------------------------------------------------- */

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

void create_node(linkedlist_t *new_node, point_t new_point)
{
    /* It creates a node of a linked list: distinguished point */
    (&new_node->point)->k = (&new_point)->k;
    (&new_node->point)->c = (&new_point)->c;
    new_node->next = NULL;
}

linkedlist_t *middle_node(linkedlist_t *first_node, linkedlist_t *last_node)
{
    /* It gets the middle node of a linked list  first_node -> ... -> last_node -> NULL */
    if(first_node == NULL)
    {
        // Empty linked list
        return NULL;
    }
    linkedlist_t *slow_move = first_node;
    linkedlist_t *fast_move = first_node->next;
    while(fast_move != last_node)
    {
        fast_move = fast_move->next;        // Moving one position in the list
        if(fast_move != last_node)
        {
            slow_move = slow_move->next;    // Moving one position in the list
            fast_move = fast_move->next;    // Moving one position in the list
        }
    }
    return slow_move;   // At the end, slow_move is pointing to the middle node
}

linkedlist_t *linkedlist_binarysearch(linkedlist_t *head, point_t point)
{
    /* Binary search in a sorted linked list */
    linkedlist_t *current_node;
    linkedlist_t *first_node = head;
    linkedlist_t *last_node = NULL;

    int tmp;
    if(head == NULL)
    {
        // Empty liked list
        return NULL;
    }
    else
    {
        // Locating the node before the point of search;
        current_node = middle_node(first_node, last_node);	// Middle node
        while( (current_node != NULL) && (first_node != last_node) )
        {
            if(current_node == NULL)
                return NULL;

            tmp = point_compare(&current_node->point, &point);
            if(tmp == 0)
            {
                return current_node;
            }
            else if(tmp == -1)
                first_node = current_node->next;
            else
                last_node = current_node;

            if(first_node != last_node)
                current_node = middle_node(first_node, last_node);
        }
        // The input (distinguished) point is not in the input linked list
        return NULL;
    }
}

void sorted_insert(linkedlist_t **head, linkedlist_t *new_node)
{
    /* It inserts a node in a sorted linked list. */
    if(*head == NULL)
    {
        // Case when the input linked list is the empty list
        new_node->next = *head;
        *head = new_node;
    }
    else if(point_compare(&(*head)->point, &new_node->point) >= 1)
    {
        // Case when the input node has larger distinguished point than any in the input linked list
        new_node->next = *head;
        *head = new_node;
    }
    else
    {
        // First, to locate the node before the point of insertion
        linkedlist_t *current_node;
        linkedlist_t *first_node = *head;
        linkedlist_t *last_node = NULL;
        current_node = middle_node(first_node, last_node);	// Middle node

        while( (current_node->next != NULL) && (first_node != last_node) )
        {
            if( point_compare(&current_node->next->point, &new_node->point) == -1 )
                first_node = current_node->next;
            else
                last_node = current_node;

            if(first_node != last_node)
                current_node = middle_node(first_node, last_node);
        }
        new_node->next = first_node->next;
        first_node->next = new_node;
    }
}

digit_t reconstruction(point_t *collision, const vowgcs_t X, const vowgcs_t Y, const ctx_mitm_t context)
{
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
    digit_t i = 0, LENGTH, counter = 0;
    //printf(">>>\t%d\n", (int)(&context)->deg);
    fp2_t Jx, Jy;
    point_t X_i, X_k={0}, Y_i, Y_k={0};

    (&X_i)->k = (&(&X)->seed)->k;
    (&X_i)->c = (&(&X)->seed)->c;
    (&Y_i)->k = (&(&Y)->seed)->k;
    (&Y_i)->c = (&(&Y)->seed)->c;

    if(X.length < Y.length)
    {
        LENGTH = X.length;
        // X.length + (Y.length - X.length) = Y.length
        for(i = X.length; i < Y.length; i++)
        {
            counter += 1;
            _fn_(&Y_i, Jy, Y_i, context);
        }
    }
    else
    {
        LENGTH = Y.length;
        // Y.length + (X.length - Y.length) = X.length
        for(i = Y.length; i < X.length; i++)
        {
            counter += 1;
            _fn_(&X_i, Jx, X_i, context);
        }
    }

    for (i = 0; i < LENGTH; i++)
    {
        counter += 2;
        (&X_k)->k = (&X_i)->k;
        (&X_k)->c = (&X_i)->c;
        (&Y_k)->k = (&Y_i)->k;
        (&Y_k)->c = (&Y_i)->c;
        _fn_(&X_i, Jx, X_i, context);
        _fn_(&Y_i, Jy, Y_i, context);
        // printf("%lu,%d, %lu,%d:\n", (&X_k)->k,(&X_k)->c,(&Y_k)->k,(&Y_k)->c);
        // fp2_printf(Jx);
        // fp2_printf(Jy);

        #ifdef FROBENIUS
        if (fp2_compare_conj(Jx, Jy) == 0)
            break;
        #else
        if (fp2_compare(Jx, Jy) == 0)
            break;
        #endif

    }

    // Special case when the collision is given at the tail (distinguished collision)
    // The golden collision must satisfy that the j-invariants are equal. If not, it
    // is a collision determined by the hash function MD5. Consequently, in order to
    // decide if the reached collision is the golden one, we must ensure that the
    // j-invariants J0 and J1 are equal.
    #ifdef FROBENIUS
    if ((counter == (X.length + Y.length)) && (fp2_compare_conj(Jx, Jy) != 0))
    #else
    if ((counter == (X.length + Y.length)) && (fp2_compare(Jx, Jy) != 0))
    #endif
    {
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
        (&X_k)->k = (&X_i)->k;
        (&X_k)->c = (&X_i)->c;
        (&Y_k)->k = (&X_i)->k;
        (&Y_k)->c = (&X_i)->c;
        //_fn_(&X_i, Jx, X_i, context, (const digit_t **) ((&context)->S));
        //_fn_(&Y_i, Jy, Y_i, context, (const digit_t **) ((&context)->S));
        //printf("%lu,%d, %lu,%d\n", (&X_i)->k,(&X_i)->c,(&Y_i)->k,(&Y_i)->c);
        //fp2_printf(Jx);
        //fp2_printf(Jy);
    }

    // Has the golden collision been reached? if X_k != Y_k, then we have found it!!!
    (&collision[0])->k = (&X_k)->k;
    (&collision[0])->c = (&X_k)->c;
    (&collision[1])->k = (&Y_k)->k;
    (&collision[1])->c = (&Y_k)->c;
    //printf("%d\t", (int)(&collision[0])->b);fp_printf((&collision[0])->k);
    //printf("%d\t", (int)(&collision[1])->b);fp_printf((&collision[1])->k);
    return counter;
}

uint8_t is_distinguished(uint64_t *address, const point_t point, const ctx_mitm_t context, const ctx_vow_t ctx)
{
    // If point is distinguished, returns 1 and writes the address
    // Note: this assumes both log(1/theta) and log(w) are less than sizeof(digit_t)
    uint64_t tmp;

    tmp = (&point)->k ^ (&context)->NONCE;

    // First n bits must be 0
    if ( (tmp & ( (1 << (&ctx)->n) - 1) ) != 0)
            return 0;
    else
        tmp >>= (&ctx)->n;

    // Address is determined by the next omegabits bits
    *address = tmp & (((uint64_t)1 << (&ctx)->omegabits) - 1);
    tmp >>= (&ctx)->omegabits;

    // Next Rbits bits must be less than the threshold
    if ( (tmp & ( (1 << (&ctx)->Rbits) - 1 )) >= (&ctx)->distinguished)
        return 0;
    else
        return 1;
}

void point_compress(uint8_t *target, const vowgcs_t *point, const ctx_mitm_t context, const ctx_vow_t ctx)
{
    // Compresses seed.k|seed.c|tail.k[useful bits only]|tail.c|length to a sequence of bytes
    int byte = 0, bit = 0, i;
    for(byte = 0; byte < ctx.triplet_bytes; byte++)
        target[byte] = 0;
    for(byte = 0; 8*byte < context.ebits_max; byte++)
        target[byte] = (point->seed.k >> (8*byte)) & 255;
    bit = context.ebits_max & 7;
    byte -= (bit+7)/8;
    target[byte] += (point->seed.c & 1) << bit;
    bit = (bit + 1) & 7;
    byte += 1 - (bit+7)/8;
    target[byte] += ((point->tail.k >> (ctx.omegabits + ctx.n)) & ( (1 << (8-bit)) - 1))<<bit;
    for(i = 0; 8*i + 8 - bit < context.ebits_max - ctx.omegabits - ctx.n; i++)
    {
        byte++;
        target[byte] = (point->tail.k >> (ctx.omegabits + ctx.n + 8*i + 8 - bit)) & 255;
    }
    bit = (bit + context.ebits_max - ctx.omegabits - ctx.n) & 7;
    byte += 1 - (bit+7)/8;
    target[byte] += (point->tail.c & 1) << bit;
    bit = (bit + 1) & 7;
    byte += 1 - (bit+7)/8;
    target[byte] += ((point->length) & ( (1 << (8-bit)) - 1)) << bit;
    for(i = 0; 8*i + 8 - bit < ctx.trail_bits; i++)
    {
        byte++;
        target[byte] = (point->length >> (8*i + 8 - bit)) & 255;
    }
    bit = (bit + ctx.trail_bits) & 7;
    byte += 1 - (bit+7)/8;
    target[byte] += 1 << bit;        // Add a dirty bit
}

void point_uncompress(vowgcs_t *point, const uint8_t *source, const uint64_t address, const ctx_mitm_t context, const ctx_vow_t ctx)
{
    // Reads a compressed point, replacing length=0 if the dirty bit was off
    int byte = 0, bit = 0, i;

    point->tail.k = 0;
    point->seed.k = 0;
    point->length = 0;

    for(byte = 0; 8*byte < context.ebits_max; byte++)
        point->seed.k += ((uint64_t)(source[byte])) << 8*byte;
    bit = context.ebits_max & 7;
    byte -= (bit+7)/8;
    point->seed.k &= ( (1 << context.e[1]) - 1 );
    point->seed.c = (source[byte] >> bit) & 1;
    bit = (bit + 1) & 7;
    byte += 1 - (bit+7)/8;
    point->tail.k += source[byte] >> bit;
    for(i = 0; 8*i + 8 - bit < context.ebits_max - ctx.omegabits - ctx.n; i++)
    {
        byte++;
        point->tail.k += ((uint64_t)(source[byte])) << (8*i + 8 - bit);
    }
    bit = (bit + context.ebits_max - ctx.omegabits - ctx.n) & 7;
    byte += 1 - (bit+7)/8;
    point->tail.k &= ( (1 << (context.ebits_max - ctx.omegabits - ctx.n)) - 1 );
    point->tail.k <<= ctx.omegabits;
    point->tail.k += address & ( (1 << ctx.omegabits) - 1 );
    point->tail.k <<= ctx.n;
    point->tail.k ^= context.NONCE & ( (1 << (ctx.omegabits + ctx.n)) - 1 );
    point->tail.c = (source[byte] >> bit) & 1;
    bit = (bit + 1) & 7;
    byte += 1 - (bit+7)/8;
    point->length = source[byte] >> bit;
    for(i = 0; 8*i + 8 - bit < ctx.trail_bits; i++)
    {
        byte++;
        point->length += ((digit_t)(source[byte])) << (8*i + 8 - bit);
    }
    bit = (bit + ctx.trail_bits) & 7;
    byte += 1 - (bit+7)/8;
    point->length &= ( 1 << ctx.trail_bits ) - 1;
    point->length *= (source[byte]>>bit) & 1; //If dirty bit was off, replace length=0
}

void precompute(fp2_t **pc_table, uint64_t path, const proj_t basis[3], const proj_t curve, int e, const int level, const int depth)
{
    /* ----------------------------------------------------------------------------- *
     * Recursive function to precompute the 2-isogeny tree of a given depth. 
     * Nodes are saved to pc_table in the form of {P, Q, PQ, E} at the address
     * corresponding to the reverse of the least `depth` bits of k.
     * ----------------------------------------------------------------------------- */
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
        fp2_mul(pc_table[3][address], curve[0], t1);
        fp2_mul(t1, t0, curve[1]);
        fp2_mul(pc_table[2][address], basis[2][0], t1);
        fp2_mul(t1, basis[2][1], curve[1]);
        fp2_mul(t2, t1, basis[0][1]);
        fp2_mul(pc_table[1][address], basis[1][0], t2);
        fp2_mul(t2, t1, basis[1][1]);
        fp2_mul(pc_table[0][address], basis[0][0], t2);
        fp2_mul(pc_table[4][address], t0, t1);
    }
    else
    {
        proj_t G0, B2, G2, DUAL, G0_diff, next_basis[3], next_curve;

        xdble(G0, e - level - 1, basis[0], curve); // x([2^((&context)->e[(&context)->c] - i - 1)]P)
        xdbl(DUAL, basis[1], curve);                                                // x([2]Q)
        xadd(B2, basis[0], basis[1], basis[2]);                                         // x(P + Q)
        xdble(G2, e - level - 1, B2, curve);      // x([2^((&context)->e[(&context)->c] - i - 1)](P + Q)
        xadd(G0_diff, basis[2], basis[1], basis[0]);                                    // x(P - [2]Q)

        // Branch corresponding to x(G0) = x([2^((&context)->e[(&context)->c] - i - 1]P)

        xisog_2(next_curve, G0);          // 2-isogenous curve
        xeval_2(next_basis[0], basis[0], G0); // 2-isogeny evaluation of x(P)
        xeval_2(next_basis[1], DUAL, G0);    // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_basis[2], G0_diff, G0); // 2-isogeny evaluation of x([2](P - [2]Q))

        precompute(pc_table, path << 1, next_basis, next_curve, e, level+1, depth); // Go to the next depth-level

        // Branch corresponding to x(G2) = x(G0 + G1) = x([2^((&context)->e[(&context)->c] - i - 1](P + Q))

        xisog_2(next_curve, G2);          // 2-isogenous curve
        xeval_2(next_basis[0], B2, G2);      // 2-isogeny evaluation of x(P+Q)
        xeval_2(next_basis[1], DUAL, G2);    // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_basis[2], basis[2], G2); // 2-isogeny evaluation of x(P - Q)

        precompute(pc_table, (path << 1) + 1, next_basis, next_curve, e, level+1, depth); // Go to the next depth-level
    }
}

/* ----------------------------------------------------------------------------- *
   Finally, the next function is an implementation of the main block of the vOW
   GCS procedure, which corresponds with the golden collision search given a
   Pseudo Random Function (PRF). The maximum number of distinguished points per
   PRF is reached in a parallel model (i.e., the task is split into the threads).
 * ----------------------------------------------------------------------------- */
#ifdef STRUCT_HASH_TABLE
double vowgcs(point_t *golden, uint8_t *finished, uint8_t hashtable[], const ctx_mitm_t context, const ctx_vow_t ctx, const int id)
#else
double vowgcs(point_t *golden, uint8_t *finished, const ctx_mitm_t context, const ctx_vow_t ctx, const int id, const int prf_counter)
#endif
{
    ticks clockcycles_init, clockcycles_last;		// Clock Cycles counters
    digit_t collisions = 0, runtime_distinguished = 0, runtime_reconstruction = 0, distinguished_points = 0;

    digit_t length, tmp_k;
    uint64_t address = 0;

    clockcycles_init = getticks();		// Clock Cycles at the beginning

    point_t seed = {0}, tail = {0};
    fp2_t J = {0};
    vowgcs_t node0, node1;
    point_t temporal_collision[2];
    linkedlist_t *another_temporal;

    while( (*finished == 0) && (distinguished_points < (&ctx)->betaXomega) )
    {
        // Random element in fp2
        fp2_random(J);
        // Random seed
        _gn_(&seed, J, (&context)->NONCE, (&context)->deg, (&context)->bound[1], (&context)->ebits_max);
        // tail of the trail
        (&tail)->k = (&seed)->k;
        (&tail)->c = (&seed)->c;

        length = 1;
        while((int)length <  (int)(&ctx)->maxtrail)
        {
            // function evaluation
            _fn_(&tail, J, tail, context);
            // t LSBs from MD5(2|X|NONCE)
            if( is_distinguished(&address, tail, context, ctx) )
                break;
            length += 1;
        }

        runtime_distinguished += (digit_t)length;
        if( (int)length <  (int)(&ctx)->maxtrail )
        {

            // New distinguished point reached
            distinguished_points += 1;
            (&(&node0)->seed)->k = (&seed)->k;
            (&(&node0)->seed)->c = (&seed)->c;
            (&(&node0)->tail)->k = (&tail)->k;
            (&(&node0)->tail)->c = (&tail)->c;
            (&node0)->length = length;

            // Accessing to the stored distinguished point
#ifdef STRUCT_PRTL
            if(struct_add_PRTL(&node1, node0, prf_counter))
            {
                //printf("PRTL found collision on %lld,%lld\n", (&(&node0)->tail)->k, (&(&node0)->tail)->c);
            //} //ATTENTION: unncomment this line if BOTH structures are used for some comparison - otherwise won't compile
#endif
#ifdef STRUCT_HASH_TABLE
            point_uncompress(&node1, &(hashtable[address*(&ctx)->triplet_bytes]), address, context, ctx);
            //printf("Found %lu, %d, %lu, %d, %ld at %lu\n", node0.seed.k, node0.seed.c, node0.tail.k, node0.tail.c, node0.length, address);
            //printf("Read %lu, %d, %lu, %d, %ld from %lu\n", node1.seed.k, node1.seed.c, node1.tail.k, node1.tail.c, node1.length, address);
            // -----------------------------------------------------------------------------------------
            if( ((&node1)->length > 0) && (point_compare(&((&node0)->tail), &((&node1)->tail)) == 0) )
            {
                //printf("Hash table found collision on %lld,%lld\n", (&(&node1)->tail)->k, (&(&node1)->tail)->c);
#endif
                // Reconstructing the collision
                collisions += 1;
                runtime_reconstruction += reconstruction(temporal_collision, node0, node1, context);

                // Is the golden collision? (collision with different points)
                //if( point_compare(&temporal_collision[0], &temporal_collision[1]) != 0 )
                if ((&temporal_collision[0])->c != (&temporal_collision[1])->c)
                {
                    // Storing the golden collision
                    *finished = 1;

                    uint8_t index = ((&temporal_collision[0])->c) & 0x1;
                    (&golden[index])->k = (&temporal_collision[0])->k;
                    (&golden[index])->c = (&temporal_collision[0])->c;

                    index = ((&temporal_collision[1])->c) & 0x1;
                    (&golden[index])->k = (&temporal_collision[1])->k;
                    (&golden[index])->c = (&temporal_collision[1])->c;
                    break;
                }

                if((&ctx)->heuristic != 0)
                {
                    // At this step, No golden collision reached
                    // Let's count the number of different collisions
                    another_temporal = NULL;
                    for(tmp_k = 0; tmp_k < (&ctx)->cores; tmp_k++)
                    {
                        // Is the reached collision different from the stored ones?
                        another_temporal = linkedlist_binarysearch((&ctx)->collisions[tmp_k], temporal_collision[0]);
                        if(another_temporal != NULL)
                            break;
                    }

                    if(another_temporal == NULL)
                    {
                        // New different collision reached
                        assert((&ctx)->index[id] < 2*(&ctx)->omega - 1);   // Is there space to save a new collision?
                        another_temporal = &(&ctx)->address[id][(&ctx)->index[id]];
                        create_node(another_temporal, temporal_collision[0]);
                        sorted_insert(&(&ctx)->collisions[id], another_temporal);
                        (&ctx)->index[id] += 1;
                    }
                }
            }

            // Finally, old distinguished point is always replaced by the new one reached
#ifdef STRUCT_HASH_TABLE
            if(*finished == 0)
            {
                // We must ensure do not overwrite the golden collision!
                // --------------------------------------------------
                // In radix-tree implementation this is included in the search-insert function
                point_compress(&(hashtable[address*(&ctx)->triplet_bytes]), &(node0), context, ctx);
                //printf("Saved %lu, %d, %lu, %d, %ld to %lu\n", node0.seed.k, node0.seed.c, node0.tail.k, node0.tail.c, node0.length, address);
                // --------------------------------------------------
            }
#endif
        }
    }

    if((&ctx)->heuristic != 0)
    {
        // Number of collisions per core
        (&ctx)->runtime_collision[id] += collisions;
        // Number of different collisions per core
        (&ctx)->runtime_different[id] += (&ctx)->index[id];
    }

    // Running time per core
    (&ctx)->runtime[id] += runtime_reconstruction + runtime_distinguished;
    clockcycles_last = getticks();
    return elapsed(clockcycles_last, clockcycles_init);
}
