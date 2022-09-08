void left_mitm_side_dfs_2(mitm_t leaves[], int *element, const fp_t path, const proj_t node[4], const ctx_dfs_t context, const int level)
{
    /* ----------------------------------------------------------------------------- *
     * left_mitm_side_dfs_2():
     * Processes a node at the "level"-th depth from the 2-isogeny tree. Only leaves
     * are stored at the "element" position with their corresponding path. A node is
     * {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[2^(EXPONENT2 / 2 - level)]
     * ----------------------------------------------------------------------------- */
    if( ((int)(&context)->e[(&context)->c] - level) == 1)
    {
        fp_t next_path = {0};
        fp2_t v = {0};
        proj_t C, next_node = {0};
        fp_copy(next_path, path);

        // Branch corresponding to x(P)
        lsh(next_path);                 // k := k << 1
        proj_copy(next_node, node[0]);
        xisog_2(C, v, next_node, node[3]);
        j_invariant((&leaves[*element])->jinvariant, C);
        fp_copy((&(&leaves[*element])->point)->k, next_path);
        *element += 1;

        // Branch corresponding to x(P + Q) = x(P - Q)
        next_path[0] ^= 1;              // k := k + 1
        proj_copy(next_node, node[2]);
        xisog_2(C, v, next_node, node[3]);
        j_invariant((&leaves[*element])->jinvariant, C);
        fp_copy((&(&leaves[*element])->point)->k, next_path);
        *element += 1;
    }
    else
    {
        fp_t next_path = {0};
        fp2_t v = {0};
        proj_t G0, B2, G2, DUAL, G0_diff, next_node[4];

        xdble(G0, (int)(&context)->e[(&context)->c] - level - 1, node[0], node[3]);  // x([2^((int)(&context)->e[(&context)->c] - i - 1)]P)
        xdbl(DUAL, node[1], node[3]);                                   // x([2]Q)
        xadd(B2, node[0], node[1], node[2], node[3]);                   // x(P + Q)
        xdble(G2, (int)(&context)->e[(&context)->c] - level - 1, B2, node[3]);       // x([2^((int)(&context)->e[(&context)->c] - i - 1)](P + Q)
        xadd(G0_diff, node[2], node[1], node[0], node[3]);              // x(P - [2]Q)

        // Branch corresponding to x(G0) = x([2^((int)(&context)->e[(&context)->c] - i - 1]P)
        fp_copy(next_path, path);                                   // k := current path
        lsh(next_path);                                             // k := k << 1

        xisog_2(next_node[3], v, G0, node[3]);                      // 2-isogenous curve
        xeval_2(next_node[0], node[0], G0, v);                      // 2-isogeny evaluation of x(P)
        xeval_2(next_node[1], DUAL, G0, v);                         // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_node[2], G0_diff, G0, v);                      // 2-isogeny evaluation of x([2](P - [2]Q))

        left_mitm_side_dfs_2(leaves, element, next_path, next_node, context, level + 1);  // Go to the next depth-level

        // Branch corresponding to x(G2) = x(G0 + G1) = x([2^((int)(&context)->e[(&context)->c] - i - 1](P + Q))
        fp_copy(next_path, path);                                   // k := current path
        lsh(next_path);                                             // k := k << 1
        next_path[0] ^= 1;                                          // k := k + 1;

        xisog_2(next_node[3], v, G2, node[3]);                      // 2-isogenous curve
        xeval_2(next_node[0], B2, G2, v);                           // 2-isogeny evaluation of x(P+Q)
        xeval_2(next_node[1], DUAL, G2, v);                         // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_node[2], node[2], G2, v);                      // 2-isogeny evaluation of x(P - Q)

        left_mitm_side_dfs_2(leaves, element, next_path, next_node, context, level + 1);  // Go to the next depth-level
    }
}

void right_mitm_side_dfs_2(point_t *collision, uint8_t *finished, mitm_t *leaves[], const fp_t path, const proj_t node[4], const ctx_dfs_t context, const int level)
{
    /* ----------------------------------------------------------------------------- *
     * right_mitm_side_dfs_2():
     * Processes a node at the "level"-th depth from the 2-isogeny tree. Collision is
     * checked by pairs of leaves from the left and right 2-isogeny trees. Only the
     * collision is stored with their corresponding paths. A node is
     * {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[2^(EXPONENT2 / 2 - level)]
     * ----------------------------------------------------------------------------- */
    if(*finished == 0)
    {
        if( ((int)(&context)->e[(&context)->c ^ 1] - level) == 1)
        {
            int l, element;
            fp_t next_path = {0};
            fp2_t j_inv = {0};
            proj_t C, next_node = {0};
            fp_copy(next_path, path);

            // Branch corresponding to x(P)
            lsh(next_path);                            // k := k << 1
            proj_copy(next_node, node[0]);
            xisog_2(C, j_inv, next_node, node[3]);

            j_invariant(j_inv, C);
            element = -1;
            for(l = 0; (l < (&context)->cores) && (element == -1); l++)
                element = hashtable_binarysearch(leaves[l], j_inv, 0, (int)(&context)->runtime[(&context)->c][0] - 1);
            l--;
            if(element != -1)
            {
                // Side corresponding to E[c ^ 1]
                fp_copy((&collision[(&context)->c])->k, (&(&leaves[l][element])->point)->k);
                // Side corresponding to E[c]
                fp_copy((&collision[(&context)->c ^ 1])->k, next_path);

                *finished = 1;
                return ;
            }

            // Branch corresponding to x(P + Q) = x(P - Q)
            next_path[0] ^= 1;
            proj_copy(next_node, node[2]);
            xisog_2(C, j_inv, next_node, node[3]);
            j_invariant(j_inv, C);
            element = -1;
            for(l = 0; (l < (&context)->cores) && (element == -1); l++)
                element = hashtable_binarysearch(leaves[l], j_inv, 0, (int)(&context)->runtime[(&context)->c][0] - 1);
            l--;

            if(element != -1)
            {
                // Side corresponding to E[c ^ 1]
                fp_copy((&collision[(&context)->c])->k, (&(&leaves[l][element])->point)->k);
                // Side corresponding to E[c]
                fp_copy((&collision[(&context)->c ^ 1])->k, next_path);

                *finished = 1;
                return ;
            }
        }
        else
        {
            fp_t next_path = {0};
            fp2_t v = {0};
            proj_t G0, B2, G2, DUAL, G0_diff, next_node[4];

            xdble(G0, (int)(&context)->e[(&context)->c ^ 1] - level - 1, node[0], node[3]);  // x([2^((&context)->e[(&context)->c] - i - 1)]P)
            xdbl(DUAL, node[1], node[3]);                               // x([2]Q)

            xadd(B2, node[0], node[1], node[2], node[3]);               // x(P + Q)
            xdble(G2, (int)(&context)->e[(&context)->c ^ 1] - level - 1, B2, node[3]);       // x([2^((&context)->e[(&context)->c] - i - 1)](P + Q)

            xadd(G0_diff, node[2], node[1], node[0], node[3]);          // x(P - [2]Q)

            // Branch corresponding to x(G0) = x([2^((&context)->e[(&context)->c] - i - 1]P)
            fp_copy(next_path, path);                                   // k := current path
            lsh(next_path);                                             // k := k << 1

            xisog_2(next_node[3], v, G0, node[3]);                      // 2-isogenous curve
            xeval_2(next_node[0], node[0], G0, v);                      // 2-isogeny evaluation of x(P)
            xeval_2(next_node[1], DUAL, G0, v);                         // 2-isogeny evaluation of x([2]Q)
            xeval_2(next_node[2], G0_diff, G0, v);                      // 2-isogeny evaluation of x([2](P - [2]Q))

            right_mitm_side_dfs_2(collision, finished, leaves, next_path, next_node, context, level + 1); // Go to the next depth-level

            // Branch corresponding to x(G2) = x(G0 + G1) = x([2^((&context)->e[(&context)->c] - i - 1](P + Q))
            fp_copy(next_path, path);                                   // k := current path
            lsh(next_path);                                             // k := k << 1
            next_path[0] ^= 1;                                          // k := k + 1;

            xisog_2(next_node[3], v, G2, node[3]);                      // 2-isogenous curve
            xeval_2(next_node[0], B2, G2, v);                           // 2-isogeny evaluation of x(P+Q)
            xeval_2(next_node[1], DUAL, G2, v);                         // 2-isogeny evaluation of x([2]Q)
            xeval_2(next_node[2], node[2], G2, v);                      // 2-isogeny evaluation of x(P - Q)

            right_mitm_side_dfs_2(collision, finished, leaves, next_path, next_node, context, level + 1); // Go to the next depth-level
        }
    }
}

void get_nodes_2(proj_t *starting_nodes[], fp_t *starting_path, int *element, fp_t path, proj_t node[4], const ctx_dfs_t context, const digit_t i)
{
    /* ----------------------------------------------------------------------------- *
     * get_nodes_2():
     * Processes a node at the "level"-th depth from the 2-isogeny tree. Nodes for each
     * core are stored at the "element" position with their corresponding path. A node
     * is {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[2^(EXPONENT2 / 2 - level)]
     * There as nodes as cores: pre-computation step
     * ----------------------------------------------------------------------------- */
    if( ((int)(&context)->depth - (int)i) == -1)
    {
        fp_copy(starting_path[*element], path);
        proj_copy(starting_nodes[*element][0], node[0]);
        proj_copy(starting_nodes[*element][1], node[1]);
        proj_copy(starting_nodes[*element][2], node[2]);
        proj_copy(starting_nodes[*element][3], node[3]);
        *element += 1;
    }
    else
    {
        fp_t next_path = {0};
        fp2_t v = {0};
        proj_t G0, B2, G2, DUAL, G0_diff, next_node[4];

        xdble(G0, (int)(&context)->e[(&context)->c] - (int)i - 1, node[0], node[3]);  // x([2^((&context)->e[(&context)->c] - i - 1)]P)
        xdbl(DUAL, node[1], node[3]);                               // x([2]Q)
        xadd(B2, node[0], node[1], node[2], node[3]);               // x(P + Q)
        xdble(G2, (int)(&context)->e[(&context)->c] - (int)i - 1, B2, node[3]);       // x([2^((&context)->e[(&context)->c] - i - 1)](P + Q)
        xadd(G0_diff, node[2], node[1], node[0], node[3]);          // x(P - [2]Q)

        // Branch corresponding to x(G0) = x([2^((&context)->e[(&context)->c] - i - 1]P)
        fp_copy(next_path, path);                                   // k := current path
        lsh(next_path);                                             // k := k << 1

        xisog_2(next_node[3], v, G0, node[3]);                      // 2-isogenous curve
        xeval_2(next_node[0], node[0], G0, v);                      // 2-isogeny evaluation of x(P)
        xeval_2(next_node[1], DUAL, G0, v);                         // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_node[2], G0_diff, G0, v);                      // 2-isogeny evaluation of x([2](P - [2]Q))

        get_nodes_2(starting_nodes, starting_path, element, next_path, next_node, context, i + 1);                   // Go to the next depth-level

        // Branch corresponding to x(G2) = x(G0 + G1) = x([2^((&context)->e[(&context)->c] - i - 1](P + Q))
        fp_copy(next_path, path);                                   // k := current path
        lsh(next_path);                                             // k := k << 1
        next_path[0] ^= 1;                                          // k := k + 1;

        xisog_2(next_node[3], v, G2, node[3]);                      // 2-isogenous curve
        xeval_2(next_node[0], B2, G2, v);                           // 2-isogeny evaluation of x(P+Q)
        xeval_2(next_node[1], DUAL, G2, v);                         // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_node[2], node[2], G2, v);                      // 2-isogeny evaluation of x(P - Q)

        get_nodes_2(starting_nodes, starting_path, element, next_path, next_node, context, i + 1);                   // Go to the next depth-level
    }
}

void get_roots_2(proj_t *starting_nodes[], fp_t *starting_path, const proj_t BASIS[3], const proj_t E, const ctx_dfs_t context)
{
    /* ----------------------------------------------------------------------------- *
     * get_roots_2():
     * Stores the starting nodes (roots) and their corresponding path for each core.
     * A node is {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[3^(EXPONENT3 / 2 - level)]
     * There as nodes as cores: pre-computation step of 3 x 2^cores
     * ----------------------------------------------------------------------------- */
    int element = 0;
    proj_t  node[4];
    proj_copy(node[0], BASIS[0]);
    proj_copy(node[1], BASIS[1]);
    proj_copy(node[2], BASIS[2]);
    proj_copy(node[3], E);

    fp_t path = {0};
    fp2_t v = {0}, z = {0};
    proj_t G0, G1, G2, G3, B2, DUAL, G1_dual, G0_diff, G1_diff, next_node[4];

    xdbl(DUAL, node[1], node[3]);                       // x([2]Q)
    xdbl(G1_dual, node[0], node[3]);                    // x([2]P)
    xdble(G0, (int)(&context)->e[(&context)->c] - 2, G1_dual, node[3]);   // x([2^((&context)->e[(&context)->c] - 1)]P)
    xadd(G0_diff, node[2], node[1], node[0], node[3]);  // x(P - [2]Q)
    xdble(G1, (int)(&context)->e[(&context)->c] - 2, DUAL, node[3]);      // x([2^((&context)->e[(&context)->c] - 1)]Q)
    xadd(G1_diff, node[2], node[0], node[1], node[3]);  // x([2]P - Q)
    xadd(B2, node[0], node[1], node[2], node[3]);       // x(P + Q)

    // Branch corresponding to x(G0)
    proj_copy(G3, G0);
    xisog_2(next_node[3], v, G3, node[3]);              // 2-isogenous curve
    xeval_2(next_node[0], node[0], G3, v);              // 2-isogeny evaluation of x(P)
    xeval_2(next_node[1], DUAL, G3, v);                 // 2-isogeny evaluation of x([2]Q)
    xeval_2(next_node[2], G0_diff, G3, v);              // 2-isogeny evaluation of x(P - [2]Q)

    get_nodes_2(starting_nodes, starting_path, &element, path, next_node, context, 1);

    // Branch corresponding to x(G0 + G1)
    // Recall that 
    //             [x - x(G0)][x - x(G1)][x - x(G0 + G1)] = x^3 + Ax + B then,
    //              x(G0 + G1) = - [x(G0) + x(G1)] in affine coordinates. Thus
    //
    //             [1] X(G0 + G1) = -[(X0 * Z1) + (X1 * Z0)], and 
    //             [2] Z(G0 + G1) = Z0 * Z1. 
    //                                                     COST : 3M + 2a
    fp2_copy(v, z);
    fp2_mul(G2[0], G0[0], G1[1]);  //   (X0 * Z1)
    fp2_mul(G2[1], G0[1], G1[0]);  //               (Z0 * X1)
    fp2_add(G2[0], G2[0], G2[1]);  //   (X0 * Z1) + (Z0 * X1)
    fp2_sub(G2[0], v, G2[0]);      // -[(X0 * Z1) + (Z0 * X1)]
    fp2_mul(G2[1], G0[1], G1[1]);  //   (Z0 * Z1)

    xisog_2(next_node[3], v, G2, node[3]);              // 2-isogenous curve
    xeval_2(next_node[0], B2, G2, v);                   // 2-isogeny evaluation of x(P+Q)
    xeval_2(next_node[1], DUAL, G2, v);                 // 2-isogeny evaluation of x([2]Q)
    xeval_2(next_node[2], node[2], G2, v);              // 2-isogeny evaluation of x(P - Q)

    path[0] = 1;                                        // k := k + 1;
    get_nodes_2(starting_nodes, starting_path, &element, path, next_node, context, 1);

    // Branch corresponding to x(G1)
    xisog_2(next_node[3], v, G1, node[3]);              // 2-isogenous curve
    xeval_2(next_node[0], node[1], G1, v);              // 2-isogeny evaluation of x(Q)
    xeval_2(next_node[1], G1_dual, G1, v);              // 2-isogeny evaluation of x([2]P)
    xeval_2(next_node[2], G1_diff, G1, v);              // 2-isogeny evaluation of x([2](Q - [2]P))

    path[0] = 2;                                        // k := 2;
    get_nodes_2(starting_nodes, starting_path, &element, path, next_node, context, 1);
}

void from_dfs_to_collision_2(point_t *collision, const ctx_mitm_t context)
{
    /* ----------------------------------------------------------------------------- *
     * This functions reconstruct a a pair of paths (the collision in DFS form) as
     * a pair of elements in {0,1,2} x |[0, 2^{EXPONENT/2 - 1} - 1]|.
     * ----------------------------------------------------------------------------- */
    fp_t t, s, z = {0};
    int j, l;
    for(l = 0; l < 2; l++)
    {
        fp_copy(t, (&collision[l])->k);
        fp_copy(s, z);

        for(j = 1; j < (int)(&context)->e[l]; j++)
        {
            lsh(s);
            s[0] ^= (t[0] & 0x1);
            rsh(t);
        }

        if( (t[0] & 0x3) < 2 )
        {
            lsh(s);
            s[0] ^= (t[0] & 0x1);
            fp_copy((&collision[l])->k, s);
            (&collision[l])->b = 0 ^ (l << 2);
        }
        else
        {
            fp_copy((&collision[l])->k, s);
            (&collision[l])->b = 2 ^ (l << 2);
        }
    }
}