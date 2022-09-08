void init_context_dfs(ctx_dfs_t *context, digit_t deg)
{
    if (deg == 2)
    {
        context->deg = 2;
        context->e[0] = EXPONENT2 - (EXPONENT2 >> 1);
        context->e[1] = EXPONENT2 >> 1;
        context->left_mitm_side_dfs = &left_mitm_side_dfs_2;
        context->right_mitm_side_dfs = &right_mitm_side_dfs_2;
        context->get_nodes = &get_nodes_2;
        context->get_roots = &get_roots_2;
        context->from_dfs_to_collision = &from_dfs_to_collision_2;
    }
    else if (deg == 3)
    {
        context->deg = 3;
        context->e[0] = EXPONENT3 - (EXPONENT3 >> 1);
        context->e[1] = EXPONENT3 >> 1;
        context->left_mitm_side_dfs = &left_mitm_side_dfs_3;
        context->right_mitm_side_dfs = &right_mitm_side_dfs_3;
        context->get_nodes = &get_nodes_3;
        context->get_roots = &get_roots_3;
        context->from_dfs_to_collision = &from_dfs_to_collision_3;
    }
    else
        assert((deg == 2) || (deg == 3));
}

void hashtable_swap(mitm_t *a, mitm_t *b)
{
    /* swap between hashtable elements */
    mitm_t t;
    memcpy(&t, a, sizeof(mitm_t));
    memcpy(a,  b, sizeof(mitm_t));
    memcpy(b, &t, sizeof(mitm_t));
}

int hashtable_partition (mitm_t arr[], int low, int high)
{
    /* ----------------------------------------------------------------------------- *
     * This function takes last element as pivot, places the pivot element at its
     * correct position in sorted array, and places all smaller (smaller than pivot)
     * to left of pivot and all greater elements to right of pivot
     * ----------------------------------------------------------------------------- */
    fp2_t pivot;
    fp2_copy(pivot, arr[high].jinvariant);
    int j, i = (low - 1);  // Index of smaller element

    for(j = low; j <= (high - 1); j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if(fp2_compare(arr[j].jinvariant, pivot) <= 0)
        {
            i++;    // increment index of smaller element
            hashtable_swap(&arr[i], &arr[j]);
        }
    }
    hashtable_swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void hashtable_quicksort(mitm_t arr[], int low, int high)
{
    /* Quick sort algorithm implementation (hashtable elements) */
    if (low < high)
    {
        //printf("%d, %d;\n", low, high);
        /* pi is hashtable_partitioning index, arr[p] is now
         * at right place */
        int pi = hashtable_partition(arr, low, high);

        // Separately sort elements before
        // hashtable_partition and after hashtable_partition
        hashtable_quicksort(arr, low, pi - 1);
        hashtable_quicksort(arr, pi + 1, high);
    }
}

int hashtable_binarysearch(mitm_t local_seq[], fp2_t jinvariant, int low, int high)
{
    /* Binary search on hashtables */
    int mid;
    while((int)high >= (int)low)
    {
        mid = (low + high)/2;
        if( fp2_compare(local_seq[mid].jinvariant, jinvariant) == 0)
            return mid;
        else if( fp2_compare(local_seq[mid].jinvariant, jinvariant) == -1)
            low = mid + 1;
        else
            high = mid - 1;
    }

    return -1;
}

void runtime_per_thread(ctx_mitm_t *context, int cores)
{
    // Assuming runtime is deg^{e/2-1}... Recall, if e is odd, then the output must be multiplied by deg
    fp_t t0 = {0}, t1 = {0}, t2 = {0};
    t0[0] = context->deg;
    to_montgomery(t1, t0);
    fp_inv(t1);                     // 1/deg
    fp_set_one(t0);                 // 1
    int i;
    for(i = 0; i < cores; i++)
        fp_mul(t0, t0, t1);         // 1/deg^{i+1}

    // At this point t0 = 1 / (deg ^ cores)
    to_montgomery(t2, context->bound[0]);
    fp_mul(t1, t2, t0);             // bound[0] / deg^{cores}
    from_montgomery((context->runtime)[0], t1);
    to_montgomery(t2, context->bound[1]);
    fp_mul(t1, t2, t0);             // bound[1] / deg^{cores}
    from_montgomery((context->runtime)[1], t1);

    context->cores = (int)pow((int)context->deg, cores);
}

double left_mitm_side_basic(mitm_t leaves[], const ctx_mitm_t context, const int id)
{
    /* Left MITM side computation (basic: deg^e-isogeny constructions at E_c [either E0, EA, or EB]) */
    ticks clockcycles_init, clockcycles_last;
    clockcycles_init = getticks();
    // +++
    int i, j, degree = (int)(&context)->deg;
    proj_t E = {0}, degP = {0}, T[degree + 1], G_NEXT[degree + 1], G_PREV[degree + 1];
    (&context)->xmul_deg(degP, (&context)->BASIS[(&context)->c][0], (&context)->E[(&context)->c]);
    // +++
    point_t x = {0}, y = {0};
    fp_t runtime = {0};
    fp_copy((&x)->k, runtime);
    fp_copy((&y)->k, runtime);
    runtime[0] = 1;
    for(i = 0; i < id; i++)
        fp_add((&x)->k, (&x)->k, (&context)->runtime[(&context)->c]);

    fp_add((&y)->k, (&x)->k, runtime);
    for(i = 0; i < (degree + 1); i++)
    {
        (&x)->b = (uint8_t)i & 0x3;
        (&y)->b = (uint8_t)i & 0x3;
        _h_(G_PREV[i], (&context)->BASIS[(&context)->c], (&context)->E[(&context)->c], x, (&context)->deg, (&context)->bound[(&context)->c]);
        _h_(G_NEXT[i], (&context)->BASIS[(&context)->c], (&context)->E[(&context)->c], y, (&context)->deg, (&context)->bound[(&context)->c]);
    }

    // +++ Main loop +++
    // R = P + [k]Q, S = P + [k+1]Q => S - R = Q
    // R = [DEGREE x k]P + Q, S = [DEGREE x (k+1)]P + Q => S - R = [DEGREE]P
    for(i = 0; i < (degree + 1) * (&context)->runtime[(&context)->c][0]; i+=(degree + 1))
    {
        for(j = 0; j < (degree + 1); j++)
        {
            (&context)->xisoge_2nd(E, G_PREV[j], (&context)->E[(&context)->c], (&context)->S[(&context)->c], (&context)->e[(&context)->c]);
            j_invariant((&leaves[i + j])->jinvariant, E);
            fp_copy((&(&leaves[i + j])->point)->k, (&x)->k);
            (&(&leaves[i + j])->point)->b = ((uint8_t)j & 0x3) ^ ((&context)->c << 2);
        }
        // Updating next kernel
        for(j = 0; j < degree; j++)
            xadd(T[j], G_NEXT[j], (&context)->BASIS[(&context)->c][1], G_PREV[j], (&context)->E[(&context)->c]);
        xadd(T[degree], G_NEXT[degree], degP, G_PREV[degree], (&context)->E[(&context)->c]);
        // The next kernel is in G_NEXT
        for(j = 0; j < (degree + 1); j++)
        {
            proj_copy(G_PREV[j], G_NEXT[j]);
            proj_copy(G_NEXT[j], T[j]);
        }
        // Updating next scalar
        fp_add((&x)->k, (&x)->k, runtime);
    }
    // +++
    clockcycles_last = getticks();
    return elapsed(clockcycles_last, clockcycles_init);
}

double right_mitm_side_basic(point_t *collision, uint8_t *finished, mitm_t *leaves[], const ctx_mitm_t context, const int id)
{
    /* Right MITM side computation (basic: deg^e-isogeny constructions at E_c [either E0, EA, or EB]) */
    ticks clockcycles_init, clockcycles_last;
    clockcycles_init = getticks();
    // +++
    int element, i, j, l, degree = (int)(&context)->deg;
    proj_t E = {0}, degP = {0}, T[degree + 1], G_NEXT[degree + 1], G_PREV[degree + 1];
    (&context)->xmul_deg(degP, (&context)->BASIS[(&context)->c ^ 1][0], (&context)->E[(&context)->c ^ 1]);
    // +++
    point_t x = {0}, y = {0};
    fp_t runtime = {0};
    fp_copy((&x)->k, runtime);
    fp_copy((&y)->k, runtime);
    runtime[0] = 1;
    for(i = 0; i < id; i++)
        fp_add((&x)->k, (&x)->k, (&context)->runtime[(&context)->c ^ 1]);

    fp_add((&y)->k, (&x)->k, runtime);
    for(i = 0; i < (degree + 1); i++)
    {
        (&x)->b = (uint8_t)i & 0x3;
        (&y)->b = (uint8_t)i & 0x3;
        _h_(G_PREV[i], (&context)->BASIS[(&context)->c ^ 1], (&context)->E[(&context)->c ^ 1], x, (&context)->deg, (&context)->bound[(&context)->c ^ 1]);
        _h_(G_NEXT[i], (&context)->BASIS[(&context)->c ^ 1], (&context)->E[(&context)->c ^ 1], y, (&context)->deg, (&context)->bound[(&context)->c ^ 1]);
    }

    // +++ Main loop +++
    fp2_t jinv = {0};
    // R = P + [k]Q, S = P + [k+1]Q => S - R = Q
    // R = [DEGREE x k]P + Q, S = [DEGREE x (k+1)]P + Q => S - R = [DEGREE]P
    for(i = 0; i < (degree + 1) * (&context)->runtime[(&context)->c ^ 1][0] && (*finished == 0); i+=(degree + 1))
    {
        for(j = 0; j < (degree + 1); j++)
        {
            (&context)->xisoge_2nd(E, G_PREV[j], (&context)->E[(&context)->c ^ 1], (&context)->S[(&context)->c ^ 1], (&context)->e[(&context)->c ^ 1]);
            j_invariant(jinv, E);
            // Looking for collision
            element = -1;
            for(l = 0; (l < (&context)->cores) && (element == -1); l++)
                element = hashtable_binarysearch(leaves[l], jinv, 0, (degree + 1) * (int)(&context)->runtime[(&context)->c][0] - 1);
            l--;
            if(element != -1)
            {
                // Side corresponding to E[(&context)->c ^ 1]
                fp_copy((&collision[(&context)->c])->k, (&(&leaves[l][element])->point)->k);
                (&collision[(&context)->c])->b = (&(&leaves[l][element])->point)->b ^ ((&context)->c << 2);
                // Side corresponding to E[(&context)->c]
                fp_copy((&collision[(&context)->c ^ 1])->k, (&x)->k);
                (&collision[(&context)->c ^ 1])->b = ((uint8_t)j & 0x3) ^ (((&context)->c ^ 1) << 2);
                *finished = 1;
            }
        }
        // Updating next kernel
        for(j = 0; j < degree; j++)
            xadd(T[j], G_NEXT[j], (&context)->BASIS[(&context)->c ^ 1][1], G_PREV[j], (&context)->E[(&context)->c ^ 1]);
        xadd(T[degree], G_NEXT[degree], degP, G_PREV[degree], (&context)->E[(&context)->c ^ 1]);
        // The next kernel is in G_NEXT
        for(j = 0; j < (degree + 1); j++)
        {
            proj_copy(G_PREV[j], G_NEXT[j]);
            proj_copy(G_NEXT[j], T[j]);
        }
        // Updating next scalar
        fp_add((&x)->k, (&x)->k, runtime);
    }
    // +++
    clockcycles_last = getticks();
    return elapsed(clockcycles_last, clockcycles_init);
}
