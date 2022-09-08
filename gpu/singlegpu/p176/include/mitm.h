#ifndef _MITM_API_H_
#define _MITM_API_H_

#include "api.h"

typedef struct {
    uint64_t NONCE;
    int cores;
    int depth;
    int deg;
} ctx_t;

void init_context_dfs(ctx_t *context, uint32_t deg);

// Sorting and searching functions concerning the hashtable element
typedef struct {
    fp2_t jinvariant;   // j-invariant : key of the hashtable element
    point_t point;      // point: value of the hashtable element
} mitm_t;

void hashtable_swap(mitm_t *a, mitm_t *b);
int hashtable_partition(mitm_t arr[], int low, int high);
void hashtable_quicksort(mitm_t arr[], int low, int high);
int hashtable_binarysearch(mitm_t local_seq[], fp2_t jinvariant, int low, int high);

void runtime_per_thread(ctx_t *context, int cores);
// mitm-basic
//double left_mitm_side_basic(mitm_t leaves[], const ctx_mitm_t context, const int id);
//double right_mitm_side_basic(point_t *collision, uint8_t *finished, mitm_t *leaves[], const ctx_mitm_t context, int id);

// mitm-dfs
extern __device__ void left_mitm_side_dfs_2(mitm_t leaves[], int *element, const fp_t path, const proj_t node[4], const ctx_t *context, const int level);
extern __device__ void right_mitm_side_dfs_2(point_t *collision, uint8_t *finished, mitm_t *leaves[], const fp_t path, const proj_t node[4], const ctx_t context, const int level);
void get_nodes_2(proj_t *starting_nodes[], fp_t *starting_path, int *element, fp_t path, proj_t node[4], const ctx_t context, const uint32_t i);
void get_roots_2(proj_t *starting_nodes[], fp_t *starting_path, const proj_t BASIS[3], const proj_t E, const ctx_t context);
void from_dfs_to_collision_2(point_t *collision, const ctx_t context);
/*
void left_mitm_side_dfs_3(mitm_t leaves[], int *element, const fp_mont_fe path, const proj_t node[4], const ctx_dfs_t context, const int level);
void right_mitm_side_dfs_3(point_t *collision, uint8_t *finished, mitm_t *leaves[], const fp_mont_fe path, const proj_t node[4], const ctx_dfs_t context, const int level);
void get_nodes_3(proj_t *starting_nodes[], fp_mont_fe *starting_path, int *element, fp_mont_fe path, proj_t node[4], const ctx_dfs_t context, const uint32_t i);
void get_roots_3(proj_t *starting_nodes[], fp_mont_fe *starting_path, const proj_t BASIS[3], const proj_t E, const ctx_dfs_t context);
void from_dfs_to_collision_3(point_t *collision, const ctx_dfs_t context);
*/
#endif
