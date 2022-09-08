void left_mitm_side_dfs_3(mitm_t leaves[], int *element, const fp_t path, const proj_t node[4], const ctx_dfs_t context, const int level)
{
    /* ----------------------------------------------------------------------------- *
     * left_mitm_side_dfs_3():
     * Processes a node at the "level"-th depth from the 3-isogeny tree. Only leaves
     * are stored at the "element" position with their corresponding path. A node is
     * {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[3^(EXPONENT3 / 2 - level)]
     * ----------------------------------------------------------------------------- */
    if( ((int)(&context)->e[(&context)->c] - level) == 1)
    {
        proj_t C, G, uv = {0}, next_node = {0};

        fp_t next_path = {0};
        fp_copy(next_path, path);          // k := current path
        lsh(next_path);                    // k := k << 1
        lsh(next_path);                    // k := k << 1

        // Branch corresponding to x(P)
        proj_copy(next_node, node[0]);
        xisog_3(C, uv, next_node, node[3]);
        j_invariant((&leaves[*element])->jinvariant, C);
        fp_copy((&(&leaves[*element])->point)->k, next_path);
        *element += 1;

        // Branch corresponding to x(P + Q)
        // Recall that
        //             3[x - x(P)][x - x(Q)][x - x(P + Q)][x - x(P + [2]Qt)] must be
        //             equal to the division polynomial 3x^4 + 6Ax^2 + 12Bx - A^2 then,
        //             x(P + [2]Q) = - [x(P) + x(Q) + x(P + Q)] in affine coordinates.
        // In particular,
        //             [1] X(P + Q) = -{[ZQ * Z(P+[2]Q)]*XP + [ZP * Z(P+[2]Q)]*XQ + (ZP * ZQ)*X(P + [2]Q)}
        //             [2] Z(P + Q) = ZP * ZQ * Z(P + [2]Q)
        // Notice, x(P + [2]Q) = x(P - Q)
        //                                                                COST: 6M + 5a
        fp2_mul(uv[0], node[0][0], node[1][0]); // (XP * XQ)
        fp2_mul(uv[1], node[0][1], node[1][1]); // (ZP * ZQ)
        fp2_add(G[0], node[0][0], node[0][1]);  // (XP + ZP)
        fp2_add(G[1], node[1][0], node[1][1]);  // (XQ + ZQ)
        fp2_mul(G[0], G[0], G[1]);              //             (XP + ZP) * (XQ + ZQ)
        fp2_sub(G[0], uv[0], G[0]);             // (XP * XQ)             - (XP + ZP) * (XQ + ZQ)
        fp2_add(G[0], uv[1], G[0]);             // (XQ * XQ) + (XZ * ZQ) - (XP + ZP) * (XQ + ZQ) := - [(ZQ * XP) + (ZP * XQ)]
        fp2_mul(G[0], G[0], node[2][1]);        // - ZPQ * [(ZQ * XP) + (ZP * XQ)]
        fp2_mul(uv[0], uv[1], node[2][0]);      // (ZP * ZQ * XPQ)
        fp2_sub(G[0], G[0], uv[0]);             // -[(ZQ * ZPQ * XP) + (ZP * ZPQ * XQ) + (ZP * ZQ * XPQ)]
        fp2_mul(G[1], uv[1], node[2][1]);       // (ZP * ZQ * ZPQ)

        fp_copy(next_path, path);   // k := current path
        lsh(next_path);             // k := k << 1
        lsh(next_path);             // k := k << 1
        next_path[0] ^= 1;

        xisog_3(C, uv, G, node[3]);
        j_invariant((&leaves[*element])->jinvariant, C);
        fp_copy((&(&leaves[*element])->point)->k, next_path);
        *element += 1;

        // Branch corresponding to x(P + [2]Q) = x(P - Q)
        fp_copy(next_path, path);   // k := current path
        lsh(next_path);             // k := k << 1
        lsh(next_path);             // k := k << 1
        next_path[0] ^= 2;

        proj_copy(next_node, node[2]);
        xisog_3(C, uv, next_node, node[3]);
        j_invariant((&leaves[*element])->jinvariant, C);
        fp_copy((&(&leaves[*element])->point)->k, next_path);
        *element += 1;
    }
    else
    {
        fp_t next_path;
        proj_t uv = {0}, G0, G2, G3, B2, B3, DUAL, G0_diff, G2_diff, next_node[4];

        xadd(B2, node[0], node[1], node[2], node[3]);           // x(P + Q)
        xadd(B3, B2, node[1], node[0], node[3]);                // x(P + [2]Q)
        xtpl(DUAL, node[1], node[3]);                           // x([3]Q)
        xtple(G0, (int)(&context)->e[(&context)->c] - level - 1, node[0], node[3]);   // x([3^(EXPONENT3/2 - 1)]P)
        xtple(G2, (int)(&context)->e[(&context)->c] - level - 1, B2, node[3]);        // x([3^(EXPONENT3/2 - 1)](P + Q))
        xadd(G2_diff, node[2], node[1], node[0], node[3]);      // x(P - [2]Q)
        xadd(G0_diff, G2_diff, node[1], node[2], node[3]);      // x(P - [3]Q)
        xtple(G3, (int)(&context)->e[(&context)->c] - level - 1, B3, node[3]);        // x([3^(EXPONENT3/2 - 1)](P + [2]Q))

        // Branch corresponding to x(G0)
        fp_copy(next_path, path);   // k := current path
        lsh(next_path);             // k := k << 1
        lsh(next_path);             // k := k << 1

        xisog_3(next_node[3], uv, G0, node[3]);             // 3-isogenous curve
        xeval_3(next_node[0], node[0], G0, uv);             // 3-isogeny evaluation of x(P)
        xeval_3(next_node[1], DUAL, G0, uv);                // 3-isogeny evaluation of x([3]Q)
        xeval_3(next_node[2], G0_diff, G0, uv);             // 3-isogeny evaluation of x((P - [3]Q))

        left_mitm_side_dfs_3(leaves, element, next_path, next_node, context, level + 1);  // Go to the next depth-level

        // Branch corresponding to x(G0 + G1)
        fp_copy(next_path, path);   // k := current path
        lsh(next_path);             // k := k << 1
        lsh(next_path);             // k := k << 1
        next_path[0] ^= 1;                                  // k := k + 1;

        xisog_3(next_node[3], uv, G2, node[3]);             // 3-isogenous curve
        xeval_3(next_node[0], B2, G2, uv);                  // 3-isogeny evaluation of x(P + Q)
        xeval_3(next_node[1], DUAL, G2, uv);                // 3-isogeny evaluation of x([3]Q)
        xeval_3(next_node[2], G2_diff, G2, uv);             // 3-isogeny evaluation of x(P - [2]Q)

        // Now we proceed the branch corresponding to x([3^(EXPONENT3/2 - i - 1)](P + Q))
        left_mitm_side_dfs_3(leaves, element, next_path, next_node, context, level + 1);  // Go to the next depth-level

        // Branch corresponding to x(G0 + [2]G1)
        fp_copy(next_path, path);   // k := current path
        lsh(next_path);             // k := k << 1
        lsh(next_path);             // k := k << 1
        next_path[0] ^= 2;                                  // k := k + 1;

        xisog_3(next_node[3], uv, G3, node[3]);             // 3-isogenous curve
        xeval_3(next_node[0], B3, G3, uv);                  // 3-isogeny evaluation of x(P + [2]Q)
        xeval_3(next_node[1], DUAL, G3, uv);                // 3-isogeny evaluation of x([3]Q)
        xeval_3(next_node[2], node[2], G3, uv);             // 3-isogeny evaluation of x(P - Q)

        // Now we proceed the branch corresponding to x([3^(EXPONENT3/2 - i - 1)](P + Q))
        left_mitm_side_dfs_3(leaves, element, next_path, next_node, context, level + 1);  // Go to the next depth-level
    }
}

void right_mitm_side_dfs_3(point_t *collision, uint8_t *finished, mitm_t *leaves[], const fp_t path, const proj_t node[4], const ctx_dfs_t context, const int level)
{
    /* ----------------------------------------------------------------------------- *
     * right_mitm_side_dfs_3():
     * Processes a node at the "level"-th depth from the 3-isogeny tree. Collision is
     * checked by pairs of leaves from the left and right 3-isogeny trees. Only the
     * collision is stored with their corresponding paths. A node is
     * {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[3^(EXPONENT3 / 2 - level)]
     * ----------------------------------------------------------------------------- */
    if(*finished == 0)
    {
        if( ((int)(&context)->e[(&context)->c ^ 1] - level) == 1)
        {
            int l, element;
            fp2_t j_inv = {0};
            proj_t C, G, uv = {0}, next_node = {0};

            fp_t next_path;
            // Branch corresponding to x(P)
            proj_copy(next_node, node[0]);
            xisog_3(C, uv, next_node, node[3]);

            j_invariant(j_inv, C);
            element = -1;
            for(l = 0; (l < (&context)->cores) && (element == -1); l++)
                element = hashtable_binarysearch(leaves[l], j_inv, 0, (int)(&context)->runtime[(&context)->c][0] - 1);
            l--;
            if(element != -1)
            {
                fp_copy(next_path, path);   // k := current path
                lsh(next_path);             // k := k << 1
                lsh(next_path);             // k := k << 1

                // Side corresponding to E[c ^ 1]
                fp_copy((&collision[(&context)->c])->k, (&(&leaves[l][element])->point)->k);
                // Side corresponding to E[c]
                fp_copy((&collision[(&context)->c ^ 1])->k, next_path);

                *finished = 1;
                return ;
            }

            // Branch corresponding to x(P + Q)
            // Recall that
            //             3[x - x(P)][x - x(Q)][x - x(P + Q)][x - x(P + [2]Qt)] must be
            //             equal to the division polynomial 3x^4 + 6Ax^2 + 12Bx - A^2 then,
            //             x(P + [2]Q) = - [x(P) + x(Q) + x(P + Q)] in affine coordinates.
            // In particular,
            //             [1] X(P + Q) = -{[ZQ * Z(P+[2]Q)]*XP + [ZP * Z(P+[2]Q)]*XQ + (ZP * ZQ)*X(P + [2]Q)}
            //             [2] Z(P + Q) = ZP * ZQ * Z(P + [2]Q)
            // Notice, x(P + [2]Q) = x(P - Q)
            //                                                                COST: 6M + 5a
            fp2_mul(uv[0], node[0][0], node[1][0]); // (XP * XQ)
            fp2_mul(uv[1], node[0][1], node[1][1]); // (ZP * ZQ)
            fp2_add(G[0], node[0][0], node[0][1]);  // (XP + ZP)
            fp2_add(G[1], node[1][0], node[1][1]);  // (XQ + ZQ)
            fp2_mul(G[0], G[0], G[1]);              //             (XP + ZP) * (XQ + ZQ)
            fp2_sub(G[0], uv[0], G[0]);             // (XP * XQ)             - (XP + ZP) * (XQ + ZQ)
            fp2_add(G[0], uv[1], G[0]);             // (XQ * XQ) + (XZ * ZQ) - (XP + ZP) * (XQ + ZQ) := - [(ZQ * XP) + (ZP * XQ)]
            fp2_mul(G[0], G[0], node[2][1]);        // - ZPQ * [(ZQ * XP) + (ZP * XQ)]
            fp2_mul(uv[0], uv[1], node[2][0]);      // (ZP * ZQ * XPQ)
            fp2_sub(G[0], G[0], uv[0]);             // -[(ZQ * ZPQ * XP) + (ZP * ZPQ * XQ) + (ZP * ZQ * XPQ)]
            fp2_mul(G[1], uv[1], node[2][1]);       // (ZP * ZQ * ZPQ)

            xisog_3(C, uv, G, node[3]);

            j_invariant(j_inv, C);
            element = -1;
            for(l = 0; (l < (&context)->cores) && (element == -1); l++)
                element = hashtable_binarysearch(leaves[l], j_inv, 0, (int)(&context)->runtime[(&context)->c][0] - 1);
            l--;
            if(element != -1)
            {
                fp_copy(next_path, path);   // k := current path
                lsh(next_path);             // k := k << 1
                lsh(next_path);             // k := k << 1
                next_path[0] ^= 1;

                // Side corresponding to E[c ^ 1]
                fp_copy((&collision[(&context)->c])->k, (&(&leaves[l][element])->point)->k);
                // Side corresponding to E[c]
                fp_copy((&collision[(&context)->c ^ 1])->k, next_path);

                *finished = 1;
                return ;
            }

            // Branch corresponding to x(P + [2]Q) = x(P - Q)
            proj_copy(next_node, node[2]);
            xisog_3(C, uv, next_node, node[3]);

            j_invariant(j_inv, C);
            element = -1;
            for(l = 0; (l < (&context)->cores) && (element == -1); l++)
                element = hashtable_binarysearch(leaves[l], j_inv, 0, (int)(&context)->runtime[(&context)->c][0] - 1);
            l--;
            if(element != -1)
            {
                fp_copy(next_path, path);   // k := current path
                lsh(next_path);             // k := k << 1
                lsh(next_path);             // k := k << 1
                next_path[0] ^= 2;

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
            proj_t uv = {0}, G0, G2, G3, B2, B3, DUAL, G0_diff, G2_diff, next_node[4];

            xadd(B2, node[0], node[1], node[2], node[3]);       // x(P + Q)
            xadd(B3, B2, node[1], node[0], node[3]);            // x(P + [2]Q)
            xtpl(DUAL, node[1], node[3]);                       // x([3]Q)
            xtple(G0, (int)(&context)->e[(&context)->c ^ 1] - level - 1, node[0], node[3]);   // x([3^(e_3_halves - 1)]P)
            xtple(G2, (int)(&context)->e[(&context)->c ^ 1] - level - 1, B2, node[3]);        // x([3^(e_3_halves - 1)](P + Q))
            xadd(G2_diff, node[2], node[1], node[0], node[3]);  // x(P - [2]Q)
            xadd(G0_diff, G2_diff, node[1], node[2], node[3]);  // x(P - [3]Q)
            xtple(G3, (int)(&context)->e[(&context)->c ^ 1] - level - 1, B3, node[3]);        // x([3^(e_3_halves - 1)](P + [2]Q))

            // Branch corresponding to x(G0)
            fp_copy(next_path, path);   // k := current path
            lsh(next_path);             // k := k << 1
            lsh(next_path);             // k := k << 1

            xisog_3(next_node[3], uv, G0, node[3]);             // 3-isogenous curve
            xeval_3(next_node[0], node[0], G0, uv);             // 3-isogeny evaluation of x(P)
            xeval_3(next_node[1], DUAL, G0, uv);                // 3-isogeny evaluation of x([3]Q)
            xeval_3(next_node[2], G0_diff, G0, uv);             // 3-isogeny evaluation of x((P - [3]Q))

            right_mitm_side_dfs_3(collision, finished, leaves, next_path, next_node, context, level + 1); // Go to the next depth-level

            // Branch corresponding to x(G0 + G1)
            fp_copy(next_path, path);   // k := current path
            lsh(next_path);             // k := k << 1
            lsh(next_path);             // k := k << 1
            next_path[0] ^= 1;                                  // k := k + 1;

            xisog_3(next_node[3], uv, G2, node[3]);             // 3-isogenous curve
            xeval_3(next_node[0], B2, G2, uv);                  // 3-isogeny evaluation of x(P + Q)
            xeval_3(next_node[1], DUAL, G2, uv);                // 3-isogeny evaluation of x([3]Q)
            xeval_3(next_node[2], G2_diff, G2, uv);             // 3-isogeny evaluation of x(P - [2]Q)

            right_mitm_side_dfs_3(collision, finished, leaves, next_path, next_node, context, level + 1); // Go to the next depth-level

            // Branch corresponding to x(G0 + [2]G1)
            fp_copy(next_path, path);   // k := current path
            lsh(next_path);             // k := k << 1
            lsh(next_path);             // k := k << 1
            next_path[0] ^= 2;                                  // k := k + 1;

            xisog_3(next_node[3], uv, G3, node[3]);             // 3-isogenous curve
            xeval_3(next_node[0], B3, G3, uv);                  // 3-isogeny evaluation of x(P + [2]Q)
            xeval_3(next_node[1], DUAL, G3, uv);                // 3-isogeny evaluation of x([3]Q)
            xeval_3(next_node[2], node[2], G3, uv);             // 3-isogeny evaluation of x(P - Q)

            right_mitm_side_dfs_3(collision, finished, leaves, next_path, next_node, context, level + 1); // Go to the next depth-level
        }
    }
}

void get_nodes_3(proj_t *starting_nodes[], fp_t *starting_path, int *element, fp_t path, proj_t node[4], const ctx_dfs_t context, const digit_t i)
{
    /* ----------------------------------------------------------------------------- *
     * get_nodes_3():
     * Processes a node at the "level"-th depth from the 3-isogeny tree. Nodes for each
     * core are stored at the "element" position with their corresponding path. A node
     * is {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[3^(EXPONENT3 / 2 - level)]
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
        proj_t uv = {0}, G0, G2, G3, B2, B3, DUAL, G0_diff, G2_diff, next_node[4];

        xadd(B2, node[0], node[1], node[2], node[3]);       // x(P + Q)
        xadd(B3, B2, node[1], node[0], node[3]);            // x(P + [2]Q)
        xtpl(DUAL, node[1], node[3]);                       // x([3]Q)
        xtple(G0, (int)(&context)->e[(&context)->c] - i - 1, node[0], node[3]);   // x([3^(e_3_halves - 1)]P)
        xtple(G2, (int)(&context)->e[(&context)->c] - i - 1, B2, node[3]);        // x([3^(e_3_halves - 1)](P + Q))
        xadd(G2_diff, node[2], node[1], node[0], node[3]);  // x(P - [2]Q)
        xadd(G0_diff, G2_diff, node[1], node[2], node[3]);  // x(P - [3]Q)
        xtple(G3, (int)(&context)->e[(&context)->c] - i - 1, B3, node[3]);        // x([3^(e_3_halves - 1)](P + [2]Q))

        // Branch corresponding to x(G0)
        fp_copy(next_path, path);   // k := current path
        lsh(next_path);             // k := k << 1
        lsh(next_path);             // k := k << 1

        xisog_3(next_node[3], uv, G0, node[3]);             // degree-3 isogenous curve
        xeval_3(next_node[0], node[0], G0, uv);             // degree-3 isogeny evaluation of x(P)
        xeval_3(next_node[1], DUAL, G0, uv);                // degree-3 isogeny evaluation of x([3]Q)
        xeval_3(next_node[2], G0_diff, G0, uv);             // degree-3 isogeny evaluation of x((P - [3]Q))

        get_nodes_3(starting_nodes, starting_path, element, next_path, next_node, context, i + 1);                   // Go to the next depth-level

        // Branch corresponding to x(G0 + G1)
        fp_copy(next_path, path);   // k := current path
        lsh(next_path);             // k := k << 1
        lsh(next_path);             // k := k << 1
        next_path[0] ^= 1;                                  // k := k + 1;

        xisog_3(next_node[3], uv, G2, node[3]);             // degree-3 isogenous curve
        xeval_3(next_node[0], B2, G2, uv);                  // degree-3 isogeny evaluation of x(P + Q)
        xeval_3(next_node[1], DUAL, G2, uv);                // degree-3 isogeny evaluation of x([3]Q)
        xeval_3(next_node[2], G2_diff, G2, uv);             // degree-3 isogeny evaluation of x(P - [2]Q)

        get_nodes_3(starting_nodes, starting_path, element, next_path, next_node, context, i + 1);                   // Go to the next depth-level

        // Branch corresponding to x(G0 + [2]G1)
        fp_copy(next_path, path);   // k := current path
        lsh(next_path);             // k := k << 1
        lsh(next_path);             // k := k << 1
        next_path[0] ^= 2;                                  // k := k + 1;

        xisog_3(next_node[3], uv, G3, node[3]);             // degree-3 isogenous curve
        xeval_3(next_node[0], B3, G3, uv);                  // degree-3 isogeny evaluation of x(P + [2]Q)
        xeval_3(next_node[1], DUAL, G3, uv);                // degree-3 isogeny evaluation of x([3]Q)
        xeval_3(next_node[2], node[2], G3, uv);             // degree-3 isogeny evaluation of x(P - Q)

        get_nodes_3(starting_nodes, starting_path, element, next_path, next_node, context, i + 1);                   // Go to the next depth-level
    }
}

void get_roots_3(proj_t *starting_nodes[], fp_t *starting_path, const proj_t BASIS[3], const proj_t E, const ctx_dfs_t context)
{
    /* ----------------------------------------------------------------------------- *
     * get_roots_3():
     * Stores the starting nodes (roots) and their corresponding path for each core.
     * A node is {x(P), x(Q), x(P-Q), E} where <P, Q> = E(fp2)[3^(EXPONENT3 / 2 - level)]
     * There as nodes as cores: pre-computation step of 4 x 3^cores
     * ----------------------------------------------------------------------------- */
    int element = 0;
    proj_t  node[4];
    proj_copy(node[0], BASIS[0]);
    proj_copy(node[1], BASIS[1]);
    proj_copy(node[2], BASIS[2]);
    proj_copy(node[3], E);

    fp_t path = {0};
    proj_t uv = {0}, G0, G1, G2, G3, B2, B3, DUAL, G1_dual, G0_diff, G1_diff, G2_diff, next_node[4];

    xadd(B2, node[0], node[1], node[2], node[3]);       // x(P + Q)
    xadd(B3, B2, node[1], node[0], node[3]);            // x(P + [2]Q)
    xtpl(DUAL, node[1], node[3]);                       // x([3]Q)
    xtpl(G1_dual, node[0], node[3]);                    // x([3]P)
    xtple(G1, (int)(&context)->e[(&context)->c] - 2, DUAL, node[3]);      // x([3^(e_3_halves - 1)]Q)
    xadd(G1_diff, node[2], node[0], node[1], node[3]);  // x([2]P - Q) = x(Q - [2]P)
    xadd(G1_diff, G1_diff, node[0], node[2], node[3]);  // x([3]P - Q) = x(Q - [3]P)
    xtple(G0, (int)(&context)->e[(&context)->c] - 2, G1_dual, node[3]);   // x([3^(e_3_halves - 1)]P)
    xtple(G2, (int)(&context)->e[(&context)->c] - 1, B2, node[3]);        // x([3^(e_3_halves - 1)](P + Q))
    xadd(G2_diff, node[2], node[1], node[0], node[3]);  // x(P - [2]Q)
    xadd(G0_diff, G2_diff, node[1], node[2], node[3]);  // x(P - [3]Q)

    // Branch corresponding to x(G0)
    xisog_3(next_node[3], uv, G0, node[3]);             // degree-3 isogenous curve
    xeval_3(next_node[0], node[0], G0, uv);             // degree-3 isogeny evaluation of x(P)
    xeval_3(next_node[1], DUAL, G0, uv);                // degree-3 isogeny evaluation of x([3]Q)
    xeval_3(next_node[2], G0_diff, G0, uv);             // degree-3 isogeny evaluation of x((P - [3]Q))

    get_nodes_3(starting_nodes, starting_path, &element, path, next_node, context, 1);

    // Branch corresponding to x(G0 + G1)
    xisog_3(next_node[3], uv, G2, node[3]);             // degree-3 isogenous curve
    xeval_3(next_node[0], B2, G2, uv);                  // degree-3 isogeny evaluation of x(P + Q)
    xeval_3(next_node[1], DUAL, G2, uv);                // degree-3 isogeny evaluation of x([3]Q)
    xeval_3(next_node[2], G2_diff, G2, uv);             // degree-3 isogeny evaluation of x(P - [2]Q)

    path[0] = 1;                                        // k := 1;
    get_nodes_3(starting_nodes, starting_path, &element, path, next_node, context, 1);

    // Branch corresponding to x(G0 + [2]G1)
    // Recall that
    //             3[x - x(P)][x - x(Q)][x - x(P + Q)][x - x(P + [2]Qt)] must be
    //             equal to the division polynomial 3x^4 + 6Ax^2 + 12Bx + A^2 then,
    //             x(P + [2]Q) = - [x(P) + x(Q) + x(P + Q)] in affine coordinates.
    // In particular,
    //             [1] X(P + [2]Q) = -{[ZQ * Z(P + Q)]*XP + [ZP * Z(P + Q)]*XQ + (ZP * ZQ)*X(P + Q)}
    //             [2] Z(P + [2]Q) = ZP * ZQ * Z(P + Q)
    // Notice, x(P + [2]Q) = x(P - Q)
    //                                                                COST: 6M + 5a
    fp2_mul(uv[0], G0[0], G1[0]);   // (XP * XQ)
    fp2_mul(uv[1], G0[1], G1[1]);   // (ZP * ZQ)
    fp2_add(G3[0], G0[0], G0[1]);   // (XP + ZP)
    fp2_add(G3[1], G1[0], G1[1]);   // (XQ + ZQ)
    fp2_mul(G3[0], G3[0], G3[1]);   //                         (XP + ZP) * (XQ + ZQ)
    fp2_sub(G3[0], uv[0], G3[0]);   // (XP * XQ)             - (XP + ZP) * (XQ + ZQ)
    fp2_add(G3[0], uv[1], G3[0]);   // (XQ * XQ) + (XZ * ZQ) - (XP + ZP) * (XQ + ZQ) := - [(ZQ * XP) + (ZP * XQ)]
    fp2_mul(G3[0], G3[0], G2[1]);   // - ZPQ * [(ZQ * XP) + (ZP * XQ)]
    fp2_mul(uv[0], uv[1], G2[0]);   //                                       (ZP * ZQ * XPQ)
    fp2_sub(G3[0], G3[0], uv[0]);   // -[(ZQ * ZPQ * XP) + (ZP * ZPQ * XQ) + (ZP * ZQ * XPQ)]
    fp2_mul(G3[1], uv[1], G2[1]);   // (ZP * ZQ * ZPQ)

    xisog_3(next_node[3], uv, G3, node[3]);             // degree-3 isogenous curve
    xeval_3(next_node[0], B3, G3, uv);                  // degree-3 isogeny evaluation of x(P + [2]Q)
    xeval_3(next_node[1], DUAL, G3, uv);                // degree-3 isogeny evaluation of x([3]Q)
    xeval_3(next_node[2], node[2], G3, uv);             // degree-3 isogeny evaluation of x(P - Q)

    path[0] = 2;                                        // k := 2;
    get_nodes_3(starting_nodes, starting_path, &element, path, next_node, context, 1);

    // Branch corresponding to x(G1)
    xisog_3(next_node[3], uv, G1, node[3]);             // degree-2 isogenous curve
    xeval_3(next_node[0], node[1], G1, uv);             // degree-2 isogeny evaluation of x(Q)
    xeval_3(next_node[1], G1_dual, G1, uv);             // degree-2 isogeny evaluation of x([3]P)
    xeval_3(next_node[2], G1_diff, G1, uv);             // degree-2 isogeny evaluation of x(Q - [3]P)

    path[0] = 3;                                        // k := 3;
    get_nodes_3(starting_nodes, starting_path, &element, path, next_node, context, 1);
}

void from_dfs_to_collision_3(point_t *collision, const ctx_mitm_t context)
{
    /* ----------------------------------------------------------------------------- *
     * This functions reconstruct a a pair of paths (the collision in DFS form) as
     * a pair of elements in {0,1,2,3} x |[0, 3^{EXPONENT3/2 - 1} - 1]|.
     * ----------------------------------------------------------------------------- */

    fp_t t, s, z = {0}, step[3], three = {0};
    fp_copy(step[0], z);                // 0
    fp_set_one(step[1]);                // 1
    fp_add(step[2], step[1], step[1]);  // 2
    fp_add(three, step[1], step[2]);    // 3

    int j, l;
    for(l = 0; l < 2; l++)
    {
        fp_copy(t, (&collision[l])->k);
        fp_copy(s, z);

        for(j = 1; j < (int)(&context)->e[l]; j++)
        {
            fp_mul(s, s, three);            // multiplication by 3
            fp_add(s, s, step[t[0] & 0x3]); // adding either 0, 1, or 2
            rsh(t);
            rsh(t);
        }

        if( (t[0] & 0x3) < 3 )
        {
            fp_mul(s, s, three);            // multiplication by 3
            fp_add(s, s, step[t[0] & 0x3]); // adding either 0, 1, or 2
            from_montgomery((&collision[l])->k, s);
            (&collision[l])->b = 0 ^ (l << 2);
        }
        else
        {
            from_montgomery((&collision[l])->k, s);
            (&collision[l])->b = 3 ^ (l << 2);
        }
    }
}