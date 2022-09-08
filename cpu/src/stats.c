#include "stats.h"

void swapping(ticks* a, ticks* b)
{
    /* A utility function to swap two elements */
    ticks t = *a;
    *a = *b;
    *b = t;
}

int partition(ticks arr[], int low, int high)
{
    /* ---------------------------------------------------------------------------------- *
    * This function takes last element as pivot, places the pivot element at its correct
    * position in sorted array, and places all smaller (smaller than pivot) to left of
    * pivot and all greater elements to right of pivot
    * --------------------------------------------------------------------------------- */
    ticks pivot = arr[high];    // pivot
    int i = (low - 1);          // Index of smaller element
    int j;
    for (j = low; j <= high- 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++;    // increment index of smaller element
            swapping(&arr[i], &arr[j]);
        }
    }
    swapping(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quicksort(ticks arr[], int low, int high)
{
    /* Quick sort algorithm implementation */
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
        at right place */
        int pi = partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}