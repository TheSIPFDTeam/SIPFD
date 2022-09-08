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
    memcpy(a, b, sizeof(mitm_t));
    memcpy(b, &t, sizeof(mitm_t));
}

uint64_t hashtable_partition(mitm_t arr[], uint64_t low, uint64_t high)
{
    /* ----------------------------------------------------------------------------- *
     * This function takes last element as pivot, places the pivot element at its
     * correct position in sorted array, and places all smaller (smaller than pivot)
     * to left of pivot and all greater elements to right of pivot
     * ----------------------------------------------------------------------------- */
    fp2_t pivot;
    fp2_copy(pivot, arr[high].jinvariant);
    int j, i = (low - 1); // Index of smaller element

    for (j = low; j <= (high - 1); j++)
    {
        // If current element is smaller than or
        // equal to pivot
        #ifdef FROBENIUS
        if (fp2_compare_conj(arr[j].jinvariant, pivot) <= 0)
        #else
        if (fp2_compare(arr[j].jinvariant, pivot) <= 0)
        #endif
        {
            i++; // increment index of smaller element
            hashtable_swap(&arr[i], &arr[j]);
        }
    }
    hashtable_swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void hashtable_quicksort(mitm_t arr[], uint64_t low, uint64_t high)
{
    /* Quick sort algorithm implementation (hashtable elements) */
    if ( low < high && high != -1)
    {
        //printf("%d, %d;\n", low, high);
        /* pi is hashtable_partitioning index, arr[p] is now
         * at right place */
        uint64_t pi = hashtable_partition(arr, low, high);

        // Separately sort elements before
        // hashtable_partition and after hashtable_partition
        hashtable_quicksort(arr, low, pi - 1);
        hashtable_quicksort(arr, pi + 1, high);
    }
}

uint64_t hashtable_binarysearch(mitm_t local_seq[], fp2_t jinvariant, uint64_t low, uint64_t high)
{
    /* Binary search on hashtables */
    uint64_t mid;
    while ((int)high >= (int)low)
    {
        mid = (low + high) / 2;

    #ifdef FROBENIUS
        if (fp2_compare_conj(local_seq[mid].jinvariant, jinvariant) == 0)
            return mid;
        else if (fp2_compare_conj(local_seq[mid].jinvariant, jinvariant) == -1)
            low = mid + 1;
        else
            high = mid - 1;
    #else
        if (fp2_compare(local_seq[mid].jinvariant, jinvariant) == 0)
            return mid;
        else if (fp2_compare(local_seq[mid].jinvariant, jinvariant) == -1)
            low = mid + 1;
        else
            high = mid - 1;
    #endif
    }

    return -1;
}

void runtime_per_thread(ctx_mitm_t *context, int cores)
{
    // Assuming runtime is deg^{e/2-1}... Recall, if e is odd, then the output must be multiplied by deg


    context->cores = (int)pow((int)context->deg, cores);
    context->runtime[0] = (uint64_t)pow(context->deg, context->e[0] - cores);
    context->runtime[1] = (uint64_t)pow(context->deg, context->e[1] - cores);
}

double left_mitm_side_basic(mitm_t leaves[], const ctx_mitm_t context, const int id)
{
    /* Left MITM side computation (basic: deg^e-isogeny constructions at E_c [either E0, EA, or EB]) */
    ticks clockcycles_init, clockcycles_last;
    clockcycles_init = getticks();
    // +++
    int i, degree = (int)(&context)->deg;
    proj_t E = {0}, T, G_NEXT, G_PREV;
    // +++
    point_t x = {0}, y = {0};
    (&x)->k = 0;
    (&y)->k = 0;
    for (i = 0; i < id; i++)
        (&x)->k += (&context)->runtime[(&context)->c];

    (&y)->k = (&x)->k + 1;
    _h_(G_PREV, (&context)->BASIS[(&context)->c], (&context)->A2[(&context)->c], x, (&context)->deg, (&context)->ebits[(&context)->c]);
    _h_(G_NEXT, (&context)->BASIS[(&context)->c], (&context)->A2[(&context)->c], y, (&context)->deg, (&context)->ebits[(&context)->c]);

    // +++ Main loop +++
    // R = P + [k]Q, S = P + [k+1]Q => S - R = Q
    for (i = 0; i < (&context)->runtime[(&context)->c]; i += 1)
    {
        (&context)->xisoge_2nd(E, G_PREV, (&context)->E[(&context)->c], (&context)->S[(&context)->c], (&context)->e[(&context)->c]);
        if (degree == 3)
            change_curvemodel(E, E);
        j_invariant((&leaves[i])->jinvariant, E);
        (&(&leaves[i])->point)->k = (&x)->k;
        (&(&leaves[i])->point)->c = (&context)->c;
        // Updating next kernel
        xadd(T, G_NEXT, (&context)->BASIS[(&context)->c][1], G_PREV);
        // The next kernel is in G_NEXT
        proj_copy(G_PREV, G_NEXT);
        proj_copy(G_NEXT, T);
        // Updating next scalar
        (&x)->k += 1;
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
    int i, l, degree = (int)(&context)->deg;
    uint64_t element;
    proj_t E = {0}, T, G_NEXT, G_PREV;
    // +++
    point_t x = {0}, y = {0};
    uint64_t runtime;
    (&x)->k = 0;
    (&y)->k = 0;
    runtime = 1;
    for (i = 0; i < id; i++)
        (&x)->k += (&context)->runtime[(&context)->c ^ 1];

    (&y)->k = (&x)->k + runtime;
    _h_(G_PREV, (&context)->BASIS[(&context)->c ^ 1], (&context)->A2[(&context)->c ^ 1], x, (&context)->deg, (&context)->ebits[(&context)->c ^ 1]);
    _h_(G_NEXT, (&context)->BASIS[(&context)->c ^ 1], (&context)->A2[(&context)->c ^ 1], y, (&context)->deg, (&context)->ebits[(&context)->c ^ 1]);

    // +++ Main loop +++
    fp2_t jinv = {0};
    // R = P + [k]Q, S = P + [k+1]Q => S - R = Q
    for (i = 0; i < (&context)->runtime[(&context)->c ^ 1] && (*finished == 0); i += 1)
    {
        (&context)->xisoge_2nd(E, G_PREV, (&context)->E[(&context)->c ^ 1], (&context)->S[(&context)->c ^ 1], (&context)->e[(&context)->c ^ 1]);
        if (degree == 3)
            change_curvemodel(E, E);
        j_invariant(jinv, E);
        // Looking for collision
        element = -1;
        for (l = 0; (l < (&context)->cores) && (element == -1); l++)
            element = hashtable_binarysearch(leaves[l], jinv, 0, (&context)->runtime[(&context)->c] - 1);
        l--;
        if (element != -1)
        {
            // Side corresponding to E[(&context)->c ^ 1]
            (&collision[(&context)->c])->k =  (&(&leaves[l][element])->point)->k;
            (&collision[(&context)->c])->c = (&context)->c;
            // Side corresponding to E[(&context)->c]
            (&collision[(&context)->c ^ 1])->k = (&x)->k;
            (&collision[(&context)->c ^ 1])->c = ((&context)->c ^ 1);
            *finished = 1;
        }
        // Updating next kernel
        xadd(T, G_NEXT, (&context)->BASIS[(&context)->c ^ 1][1], G_PREV);
        // The next kernel is in G_NEXT
        proj_copy(G_PREV, G_NEXT);
        proj_copy(G_NEXT, T);
        // Updating next scalar
        (&x)->k += runtime;
    }
    // +++
    clockcycles_last = getticks();
    return elapsed(clockcycles_last, clockcycles_init);
}
