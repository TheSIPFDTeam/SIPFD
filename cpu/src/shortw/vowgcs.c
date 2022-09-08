int point_compare(const point_t *a, const point_t *c)
{
    /* ----------------------------------------------------------------------------- *
     *  point_compare()
     *  Inputs: two points a and b
     *  Output:
     *           0  if a = c,
     *          -1  if a < c, or
     *           1  if a > c.
     * ----------------------------------------------------------------------------- */
    int local_2nd = fp_compare(a->k, c->k);

    if (a->b < c->b)
        return -1;
    else if (a->b > c->b)
        return 1;
    else
        return local_2nd;
}

void create_node(linkedlist_t *new_node, point_t new_point)
{
    /* It creates a node of a linked list: distinguished point */
    fp_copy((&new_node->point)->k, (&new_point)->k);
    (&new_node->point)->b = (&new_point)->b;
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

    fp_copy((&X_i)->k, (&(&X)->seed)->k);
    (&X_i)->b = (&(&X)->seed)->b;
    fp_copy((&Y_i)->k, (&(&Y)->seed)->k);
    (&Y_i)->b = (&(&Y)->seed)->b;

    if(X.length < Y.length)
    {
        LENGTH = X.length;
        // X.length + (Y.length - X.length) = Y.length 
        for(i = X.length; i < Y.length; i++)
        {
            counter += 1;
            _fn_(&Y_i, Jy, Y_i, context, (const digit_t **)((&context)->S));
        }
    }
    else
    {
        LENGTH = Y.length;
        // Y.length + (X.length - Y.length) = X.length
        for(i = Y.length; i < X.length; i++)
        {
            counter += 1;
            _fn_(&X_i, Jx, X_i, context, (const digit_t **)((&context)->S));
        }
    }

    for (i = 0; i < LENGTH; i++)
    {
        counter += 2;
        fp_copy((&X_k)->k, (&X_i)->k);
        (&X_k)->b = (&X_i)->b;
        fp_copy((&Y_k)->k, (&Y_i)->k);
        (&Y_k)->b = (&Y_i)->b;
        _fn_(&X_i, Jx, X_i, context, (const digit_t **) ((&context)->S));
        _fn_(&Y_i, Jy, Y_i, context, (const digit_t **) ((&context)->S));

        if (fp2_compare(Jx, Jy) == 0)
            break;
    }

    // Special case when the collision is given at the tail (distinguished collision)
    // The golden collision must satisfy that the j-invariants are equal. If not, it
    // is a collision determined by the hash function MD5. Consequently, in order to
    // decide if the reached collision is the golden one, we must ensure that the
    // j-invariants J0 and J1 are equal.
    if ((counter == (X.length + Y.length)) && (fp2_compare(Jx, Jy) != 0))
    {
        /* No golden collision reached at the tail of each trail

                                   *
                                  / \
                                 *   *
                                      \
                                       *

         *  In this case, the j-invariants J0 and J1 are different.
         */
        fp_copy((&X_k)->k, (&X_i)->k);
        (&X_k)->b = (&X_i)->b;
        fp_copy((&Y_k)->k, (&Y_i)->k);
        (&Y_k)->b = (&Y_i)->b;
    }

    // Has the golden collision been reached? if X_k != Y_k, then we have found it!!!
    //fp2_printf(Jx);
    //fp2_printf(Jy);
    fp_copy((&collision[0])->k, (&X_k)->k);
    (&collision[0])->b = (&X_k)->b;
    fp_copy((&collision[1])->k, (&Y_k)->k);
    (&collision[1])->b = (&Y_k)->b;
    //printf("%d\t", (int)(&collision[0])->b);fp_printf((&collision[0])->k);
    //printf("%d\t", (int)(&collision[1])->b);fp_printf((&collision[1])->k);
    return counter;
}

digit_t tLSBs(uint8_t TYPE, const point_t point, const digit_t NONCE, const uint8_t t_bits)
{
    // We transform the point into an string
    digit_t mask = ((digit_t)1 << t_bits) - 1;
    char string_input[2*NBYTES_FIELD + 2*(int)sizeof(digit_t) + 3], v_string[2*(int)sizeof(digit_t) + 1];
    string_input[0] = '\0';

    v_string[0] = '\0';
    sprintf(v_string, "%x", TYPE);			// TYPE can be equal to 1,2, or 3
    v_string[strlen(v_string)] = '\0';
    strcat(string_input, v_string);
    string_input[strlen(string_input)] = '\0';

    v_string[0] = '\0';
    sprintf(v_string, "%x", (&point)->b & 0x3);			// b' : 2 bits determining the form of a kernel point generator
    v_string[strlen(v_string)] = '\0';
    strcat(string_input, v_string);
    string_input[strlen(string_input)] = '\0';

    char scalar[2*NBYTES_FIELD + 1];
    fp_string(scalar, (&point)->k);
    strcat(string_input, scalar);
    string_input[strlen(string_input)] = '\0';

    // We append the NONCE to string_input
    // each NONCE determines a different function to be used
    v_string[0] = '\0';
    SPRINTF(v_string, NONCE);
    v_string[strlen(v_string)] = '\0';
    strcat(string_input, v_string);
    string_input[strlen(string_input)] = '\0';
    //printf("%s, %jX\n", string_input, NONCE);

    unsigned char v_md5[MD5_DIGEST_LENGTH];

    // Now, we apply MD5
    MD5((unsigned char *)string_input, strlen(string_input), v_md5);

    // Finally, take the t_bits least significant bits of the output of MD5(TYPE | b' | k | NONCE )
    int i = 0, j = 0;
    digit_t point_tLSbits_MD5 = 0x0000000000000000;
    while(j < t_bits)
    {
        point_tLSbits_MD5 ^= ( (digit_t)(0xFF & v_md5[MD5_DIGEST_LENGTH - 1 - i]) ) << j;
        j += 8; i += 1;
    }
    return point_tLSbits_MD5 & mask;   // Ensure only t_bits are taken!
}

/* ----------------------------------------------------------------------------- *
   Finally, the next function is an implementation of the main block of the vOW
   GCS procedure, which corresponds with the golden collision search given a
   Pseudo Random Function (PRF). The maximum number of distinguished points per
   PRF is reached in a parallel model (i.e., the task is split into the threads).
 * ----------------------------------------------------------------------------- */
double vowgcs(point_t *golden, uint8_t *finished, vowgcs_t *hashtable, const ctx_mitm_t context, const ctx_vow_t ctx, const int id)
{
    ticks clockcycles_init, clockcycles_last;		// Clock Cycles counters
    digit_t collisions = 0, runtime_distinguished = 0, runtime_reconstruction = 0, distinguished_points = 0;

    digit_t tmp_j, tmp_i, tmp_k, TEMPORAL={0};

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
        _gn_(&seed, J, (&context)->NONCE, (&context)->deg, context);
        // tail of the trail
        fp_copy((&tail)->k, (&seed)->k);
        (&tail)->b = (&seed)->b;

        tmp_j = 0;
        while((int)tmp_j <  (int)(&ctx)->maxtrail)
        {
            tmp_j += 1;
            // function evaluation
            _fn_(&tail, J, tail, context, (const digit_t **) ((&context)->S));
            // t LSBs from MD5(2|X|NONCE)
            TEMPORAL = tLSBs(2, tail, (&context)->NONCE, 32);
            if(TEMPORAL <= (&ctx)->distinguished)
                break;
        }

        runtime_distinguished += (digit_t)tmp_j;
        if( TEMPORAL <= (&ctx)->distinguished)
        {
            // New distinguished point reached
            distinguished_points += 1;
            fp_copy((&(&node0)->seed)->k, (&seed)->k);
            (&(&node0)->seed)->b = (&seed)->b;
            fp_copy((&(&node0)->tail)->k, (&tail)->k);
            (&(&node0)->tail)->b = (&tail)->b;
            (&node0)->length = tmp_j;

            // -----------------------------------------------------------------------------------------
            // In radix-tree implementation this must be changed!
            // Position in the hashtable
            tmp_i = tLSBs(3, tail, (&context)->NONCE, (&ctx)->omegabits);
            // Accessing to the stored distinguished point
            fp_copy((&(&node1)->seed)->k, (&(&hashtable[tmp_i])->seed)->k);
            (&(&node1)->seed)->b = (&(&hashtable[tmp_i])->seed)->b;
            fp_copy((&(&node1)->tail)->k, (&(&hashtable[tmp_i])->tail)->k);
            (&(&node1)->tail)->b = (&(&hashtable[tmp_i])->tail)->b;
            (&node1)->length = (&hashtable[tmp_i])->length;
            // -----------------------------------------------------------------------------------------
            if( ((&node1)->length > 0) && (point_compare(&(&node0)->tail, &(&node1)->tail) == 0) )
            {
                // Reconstructing the collision
                collisions += 1;
                runtime_reconstruction += reconstruction(temporal_collision, node0, node1, context);

                // Is the golden collision? (collision with different points)
                //if( point_compare(&temporal_collision[0], &temporal_collision[1]) != 0 )
                if (((&temporal_collision[0])->b & 0x4) != ((&temporal_collision[1])->b & 0x4))
                {
                    // Storing the golden collision
                    *finished = 1;

                    uint8_t index = ((&temporal_collision[0])->b >> 2) & 0x1;
                    fp_copy((&golden[index])->k, (&temporal_collision[0])->k);
                    (&golden[index])->b = (&temporal_collision[0])->b;

                    index = ((&temporal_collision[1])->b >> 2) & 0x1;
                    fp_copy((&golden[index])->k, (&temporal_collision[1])->k);
                    (&golden[index])->b = (&temporal_collision[1])->b;
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
                        another_temporal = &(&ctx)->address[id][(&ctx)->index[id]];
                        create_node(another_temporal, temporal_collision[0]);
                        sorted_insert(&(&ctx)->collisions[id], another_temporal);
                        (&ctx)->index[id] += 1;
                    }
                }
            }

            // Finally, old distinguished point is always replaced by the new one reached
            if(*finished == 0)
            {
                // We must ensure do not overwrite the golden collision!
                // --------------------------------------------------
                // In radix-tree implementation this must be modified
                fp_copy((&(&hashtable[tmp_i])->seed)->k, (&seed)->k);
                (&(&hashtable[tmp_i])->seed)->b = (&seed)->b;
                fp_copy((&(&hashtable[tmp_i])->tail)->k, (&tail)->k);
                (&(&hashtable[tmp_i])->tail)->b = (&tail)->b;
                (&hashtable[tmp_i])->length = tmp_j;
                // --------------------------------------------------
            }
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

