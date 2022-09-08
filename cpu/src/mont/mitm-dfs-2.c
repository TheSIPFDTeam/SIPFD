void left_mitm_side_dfs_2(mitm_t leaves[], uint64_t *element, const uint64_t path, const proj_t node[4], const ctx_dfs_t context, const int level)
{
    /* ----------------------------------------------------------------------------- *
     * left_mitm_side_dfs_2():
     * Processes a node at the "level"-th depth from the 2-isogeny tree. Only leaves
     * are stored at the "element" position with their corresponding path. A node is
     * {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[2^(EXPONENT2 / 2 - level)] and E
     * is given in terms of the A2 coefficient
     * ----------------------------------------------------------------------------- */
    if (level == (int)(&context)->e[(&context)->c])
    {
        uint64_t next_path = path;
        proj_t C, next_node = {0};

        // Branch corresponding to x(P)
        next_path <<= 1; // k := k << 1
        proj_copy(next_node, node[0]);
        xisog_2(C, next_node);
        j_invariant((&leaves[*element])->jinvariant, C);
        (&(&leaves[*element])->point)->k = next_path;
        *element += 1;

        // Branch corresponding to x(P + Q) = x(P - Q)
        next_path ^= 1; // k := k + 1
        proj_copy(next_node, node[2]);
        xisog_2(C, next_node);
        j_invariant((&leaves[*element])->jinvariant, C);
        (&(&leaves[*element])->point)->k = next_path;
        *element += 1;
    }
    else
    {
        uint64_t next_path = 0;
        proj_t G0, B2, G2, DUAL, G0_diff, next_node[4];

        xdble(G0, (int)(&context)->e[(&context)->c] - level, node[0], node[3]); // x([2^((int)(&context)->e[(&context)->c] - i - 1)]P)
        xdbl(DUAL, node[1], node[3]);                                           // x([2]Q)
        xadd(B2, node[0], node[1], node[2]);                                    // x(P + Q)
        xdble(G2, (int)(&context)->e[(&context)->c] - level, B2, node[3]);      // x([2^((int)(&context)->e[(&context)->c] - i - 1)](P + Q)
        xadd(G0_diff, node[2], node[1], node[0]);                               // x(P - [2]Q)

        // Branch corresponding to x(G0) = x([2^((int)(&context)->e[(&context)->c] - i - 1]P)
        next_path = path; // k := current path
        next_path <<= 1;           // k := k << 1

        xisog_2(next_node[3], G0);          // 2-isogenous curve
        xeval_2(next_node[0], node[0], G0); // 2-isogeny evaluation of x(P)
        xeval_2(next_node[1], DUAL, G0);    // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_node[2], G0_diff, G0); // 2-isogeny evaluation of x((P - [2]Q))

        left_mitm_side_dfs_2(leaves, element, next_path, next_node, context, level + 1); // Go to the next depth-level

        // Branch corresponding to x(G2) = x(G0 + G1) = x([2^((int)(&context)->e[(&context)->c] - i - 1](P + Q))
        next_path = path; // k := current path
        next_path <<= 1;           // k := k << 1
        next_path ^= 1;        // k := k + 1;

        xisog_2(next_node[3], G2);          // 2-isogenous curve
        xeval_2(next_node[0], B2, G2);      // 2-isogeny evaluation of x(P+Q)
        xeval_2(next_node[1], DUAL, G2);    // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_node[2], node[2], G2); // 2-isogeny evaluation of x(P - Q)

        left_mitm_side_dfs_2(leaves, element, next_path, next_node, context, level + 1); // Go to the next depth-level
    }
}

void right_mitm_side_dfs_2(point_t *collision, uint8_t *finished, mitm_t *leaves[], const uint64_t path, const proj_t node[4], const ctx_dfs_t context, const int level)
{
    /* ----------------------------------------------------------------------------- *
     * right_mitm_side_dfs_2():
     * Processes a node at the "level"-th depth from the 2-isogeny tree. Collision is
     * checked by pairs of leaves from the left and right 2-isogeny trees. Only the
     * collision is stored with their corresponding paths. A node is
     * {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[2^(EXPONENT2 / 2 - level)] and E
     * is given in terms of the A2 coefficient
     * ----------------------------------------------------------------------------- */
    if (*finished == 0)
    {
        if (level == (int)(&context)->e[(&context)->c ^ 1])
        {
            int l;
            uint64_t element;
            uint64_t next_path = path;
            fp2_t j_inv = {0};
            proj_t C, next_node = {0};

            // Branch corresponding to x(P)
            next_path <<= 1; // k := k << 1
            proj_copy(next_node, node[0]);
            xisog_2(C, next_node);

            j_invariant(j_inv, C);
            element = -1;
            for (l = 0; (l < (&context)->cores) && (element == -1); l++)
                element = hashtable_binarysearch(leaves[l], j_inv, 0, (&context)->runtime[(&context)->c] - 1);
            l--;
            if (element != -1)
            {
                // Side corresponding to E[c ^ 1]
                (&collision[(&context)->c])->k = (&(&leaves[l][element])->point)->k;
                // Side corresponding to E[c]
                (&collision[(&context)->c ^ 1])->k = next_path;

                *finished = 1;
                return;
            }

            // Branch corresponding to x(P + Q) = x(P - Q)
            next_path ^= 1;
            proj_copy(next_node, node[2]);
            xisog_2(C, next_node);
            j_invariant(j_inv, C);
            element = -1;
            for (l = 0; (l < (&context)->cores) && (element == -1); l++)
                element = hashtable_binarysearch(leaves[l], j_inv, 0, (int)(&context)->runtime[(&context)->c] - 1);
            l--;

            if (element != -1)
            {
                // Side corresponding to E[c ^ 1]
                (&collision[(&context)->c])->k = (&(&leaves[l][element])->point)->k;
                // Side corresponding to E[c]
                (&collision[(&context)->c ^ 1])->k = next_path;

                *finished = 1;
                return;
            }
        }
        else
        {
            uint64_t next_path = 0;
            proj_t G0, B2, G2, DUAL, G0_diff, next_node[4];

            xdble(G0, (int)(&context)->e[(&context)->c ^ 1] - level, node[0], node[3]); // x([2^((&context)->e[(&context)->c] - i - 1)]P)
            xdbl(DUAL, node[1], node[3]);                                               // x([2]Q)

            xadd(B2, node[0], node[1], node[2]);                                   // x(P + Q)
            xdble(G2, (int)(&context)->e[(&context)->c ^ 1] - level, B2, node[3]); // x([2^((&context)->e[(&context)->c] - i - 1)](P + Q)

            xadd(G0_diff, node[2], node[1], node[0]); // x(P - [2]Q)

            // Branch corresponding to x(G0) = x([2^((&context)->e[(&context)->c] - i - 1]P)
            next_path = path; // k := current path
            next_path <<= 1;           // k := k << 1

            xisog_2(next_node[3], G0);          // 2-isogenous curve
            xeval_2(next_node[0], node[0], G0); // 2-isogeny evaluation of x(P)
            xeval_2(next_node[1], DUAL, G0);    // 2-isogeny evaluation of x([2]Q)
            xeval_2(next_node[2], G0_diff, G0); // 2-isogeny evaluation of x([2](P - [2]Q))

            right_mitm_side_dfs_2(collision, finished, leaves, next_path, next_node, context, level + 1); // Go to the next depth-level

            // Branch corresponding to x(G2) = x(G0 + G1) = x([2^((&context)->e[(&context)->c] - i - 1](P + Q))
            next_path = path; // k := current path
            next_path <<= 1;           // k := k << 1
            next_path ^= 1;        // k := k + 1;

            xisog_2(next_node[3], G2);          // 2-isogenous curve
            xeval_2(next_node[0], B2, G2);      // 2-isogeny evaluation of x(P+Q)
            xeval_2(next_node[1], DUAL, G2);    // 2-isogeny evaluation of x([2]Q)
            xeval_2(next_node[2], node[2], G2); // 2-isogeny evaluation of x(P - Q)

            right_mitm_side_dfs_2(collision, finished, leaves, next_path, next_node, context, level + 1); // Go to the next depth-level
        }
    }
}

void get_nodes_2(proj_t *starting_nodes[], uint64_t *starting_path, uint64_t *element, uint64_t path, proj_t node[4], const ctx_dfs_t context, const digit_t i)
{
    /* ----------------------------------------------------------------------------- *
     * get_nodes_2():
     * Processes a node at the "level"-th depth from the 2-isogeny tree. Nodes for each
     * core are stored at the "element" position with their corresponding path. A node
     * is {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[2^(EXPONENT2 / 2 - level)]
     * There as nodes as cores: pre-computation step
     * ----------------------------------------------------------------------------- */
    if ((int)(&context)->depth == (int)i)
    {
        starting_path[*element] = path;
        proj_copy(starting_nodes[*element][0], node[0]);
        proj_copy(starting_nodes[*element][1], node[1]);
        proj_copy(starting_nodes[*element][2], node[2]);
        proj_copy(starting_nodes[*element][3], node[3]);
        *element += 1;
    }
    else
    {
        uint64_t next_path = 0;
        proj_t G0, B2, G2, DUAL, G0_diff, next_node[4];

        xdble(G0, (int)(&context)->e[(&context)->c] - (int)i - 1, node[0], node[3]); // x([2^((&context)->e[(&context)->c] - i - 1)]P)
        xdbl(DUAL, node[1], node[3]);                                                // x([2]Q)
        xadd(B2, node[0], node[1], node[2]);                                         // x(P + Q)
        xdble(G2, (int)(&context)->e[(&context)->c] - (int)i - 1, B2, node[3]);      // x([2^((&context)->e[(&context)->c] - i - 1)](P + Q)
        xadd(G0_diff, node[2], node[1], node[0]);                                    // x(P - [2]Q)

        // Branch corresponding to x(G0) = x([2^((&context)->e[(&context)->c] - i - 1]P)
        next_path = path; // k := current path
        next_path <<= 1;           // k := k << 1

        xisog_2(next_node[3], G0);          // 2-isogenous curve
        xeval_2(next_node[0], node[0], G0); // 2-isogeny evaluation of x(P)
        xeval_2(next_node[1], DUAL, G0);    // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_node[2], G0_diff, G0); // 2-isogeny evaluation of x([2](P - [2]Q))

        get_nodes_2(starting_nodes, starting_path, element, next_path, next_node, context, i + 1); // Go to the next depth-level

        // Branch corresponding to x(G2) = x(G0 + G1) = x([2^((&context)->e[(&context)->c] - i - 1](P + Q))
        next_path = path; // k := current path
        next_path <<= 1;           // k := k << 1
        next_path ^= 1;        // k := k + 1;

        xisog_2(next_node[3], G2);          // 2-isogenous curve
        xeval_2(next_node[0], B2, G2);      // 2-isogeny evaluation of x(P+Q)
        xeval_2(next_node[1], DUAL, G2);    // 2-isogeny evaluation of x([2]Q)
        xeval_2(next_node[2], node[2], G2); // 2-isogeny evaluation of x(P - Q)

        get_nodes_2(starting_nodes, starting_path, element, next_path, next_node, context, i + 1); // Go to the next depth-level
    }
}

void get_roots_2(proj_t *starting_nodes[], uint64_t *starting_path, const proj_t BASIS[3], const proj_t E, const ctx_dfs_t context)
{
    /* ----------------------------------------------------------------------------- *
     * get_roots_2():
     * Stores the starting nodes (roots) and their corresponding path for each core.
     * A node is {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[2^(EXPONENT2 / 2 - level)]
     * There as nodes as cores: pre-computation step of 3 x 2^cores
     * ----------------------------------------------------------------------------- */
    uint64_t element = 0;
    proj_t node[4];
    proj_copy(node[0], BASIS[0]);
    proj_copy(node[1], BASIS[1]);
    proj_copy(node[2], BASIS[2]);
    proj_copy(node[3], E);

    uint64_t path = 0;
    proj_t G0, G2, B2, DUAL, G0_diff, next_node[4];

    xdble(G0, (int)(&context)->e[(&context)->c] - 1, node[0], node[3]); // x([2^((&context)->e[(&context)->c] - 1)]P)
    xdbl(DUAL, node[1], node[3]);                                       // x([2]Q)
    xadd(B2, node[0], node[1], node[2]);                                // x(P + Q)
    xdble(G2, (int)(&context)->e[(&context)->c] - 1, B2, node[3]);      // x([2^((&context)->e[(&context)->c] - 1)](P + Q)
    xadd(G0_diff, node[2], node[1], node[0]);                           // x(P - [2]Q)

    // Branch corresponding to x(G0)
    xisog_2(next_node[3], G0);          // 2-isogenous curve
    xeval_2(next_node[0], node[0], G0); // 2-isogeny evaluation of x(P)
    xeval_2(next_node[1], DUAL, G0);    // 2-isogeny evaluation of x([2]Q)
    xeval_2(next_node[2], G0_diff, G0); // 2-isogeny evaluation of x(P - [2]Q)

    get_nodes_2(starting_nodes, starting_path, &element, path, next_node, context, 1);

    // Branch corresponding to x(G0 + G1)
    xisog_2(next_node[3], G2);          // 2-isogenous curve
    xeval_2(next_node[0], B2, G2);      // 2-isogeny evaluation of x(P+Q)
    xeval_2(next_node[1], DUAL, G2);    // 2-isogeny evaluation of x([2]Q)
    xeval_2(next_node[2], node[2], G2); // 2-isogeny evaluation of x(P - Q)

    path = 1; // k := k + 1;
    get_nodes_2(starting_nodes, starting_path, &element, path, next_node, context, 1);
}

void from_dfs_to_collision_2(point_t *collision, const ctx_mitm_t context)
{
    /* ----------------------------------------------------------------------------- *
     * This functions reconstruct a a pair of paths (the collision in DFS form) as
     * a pair of elements in {0,1,2} x |[0, 2^{EXPONENT/2 - 1} - 1]|.
     * ----------------------------------------------------------------------------- */
    uint64_t t, s;
    int j, l;

    for (l = 0; l < 2; l++)
    {
        t = (&collision[l])->k;
        s = 0;

        for (j = 0; j < (int)(&context)->e[l]; j++)
        {
            s <<= 1;
            s ^= (t & 0x1);
            t >>= 1;
        }
        (&collision[l])->k = s;
        (&collision[l])->c = l;
    }
}
