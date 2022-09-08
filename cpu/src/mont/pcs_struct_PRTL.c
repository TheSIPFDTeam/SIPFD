
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include <omp.h>
#include "pcs_struct_PRTL.h"

static uint8_t k_bits;
static uint8_t c_bits;
static uint8_t lenght_bits;
static uint8_t level;
static _vect_bin_chain_t memory_buffer[__NB_STORED_POINTS__]; //memory is static so no need to use _vect_bin_alloc_size
static _cursor_type memory_buffer_cursor;
static int chain_array_size;
static omp_lock_t *locks;
static omp_lock_t cursor_lock;
static int *slot_prf_counter;
static int seed_k_start;
static int seed_c_start;
static int tail_k_start;
static int tail_c_start;
static int length_start;
static int suffix_len;
static int distinguished_bits;
static int nb_stored_points;

void test_func()
{
	printf("test\n");
}
/** Serialize a node in vowgcs_t form.
 *
 */
int compute_optimal_level(digit_t M){

	digit_t cmpt=1;

	while (M>pow(2,cmpt)*(cmpt*log(2)+0.577)){

	   cmpt+=1;
	 }
	return cmpt-1;
}

#ifdef NODE_STRUCT_vowgcs_t
void node_to_vect(_vect_bin_t *v, vowgcs_t node_in)
{
	vect_bin_set_int64(v, tail_k_start, (&(&node_in)->tail)->k, level+distinguished_bits, k_bits);
	vect_bin_set_int8(v, tail_c_start, (&(&node_in)->tail)->c, 1);
	vect_bin_set_int64(v, seed_k_start, (&(&node_in)->seed)->k, 0, k_bits);
	vect_bin_set_int8(v, seed_c_start, (&(&node_in)->seed)->c, 1);
	vect_bin_set_int64(v, length_start, (&node_in)->length, 0, lenght_bits);
}
#endif

/** Deserialize a node in vowgcs_t form.
 *
 */

#ifdef NODE_STRUCT_vowgcs_t
void vect_to_node(vowgcs_t *node_out, _vect_bin_t *v, int key)
{
	(&(node_out->tail))->k = vect_bin_get_int64(v, tail_k_start, level+distinguished_bits, k_bits);
	(&(node_out->tail))->k = ((&(node_out)->tail)->k | key);
	(&(node_out->tail))->c = vect_bin_get_int8(v, tail_c_start, 1);
	(&(node_out->seed))->k = vect_bin_get_int64(v, seed_k_start, 0, k_bits);
	(&(node_out->seed))->c = vect_bin_get_int8(v, seed_c_start, 1);
	node_out->length = vect_bin_get_int64(v, length_start, 0, lenght_bits);
}
#endif

/** Get prefix from node in vowgcs_t form.
 *
 */

#ifdef NODE_STRUCT_vowgcs_t
uint64_t node_get_prefix(vowgcs_t node_in)
{
	uint64_t key = (&(&node_in)->tail)->k;
	key = key >> distinguished_bits;
	int mask = pow(2, level);
	mask--;
	key = key & mask;
	return key;
}
#endif

/** Compare a data vetor with a node in vowgcs_t form.
 *
 */

#ifdef NODE_STRUCT_vowgcs_t
int cmp_point(_vect_bin_t *t, vowgcs_t node_in) {
	return vect_bin_cmp_point(t, tail_k_start, (&(&node_in)->tail), level+distinguished_bits, k_bits, c_bits);
}
#endif

/** Testing the serialization for a node in vowgcs_t form.
 *
 */

#ifdef NODE_STRUCT_vowgcs_t
void check_serialization(vowgcs_t *some_node_out, vowgcs_t some_node_in)
{
	int key;
	_vect_bin_chain_t *new;
	_vect_bin_chain_t_new(new);
	vect_bin_t_reset(new->v);

	printf("In: %lu, %d, %lu, %d, %ld\n", some_node_in.tail.k, some_node_in.tail.c, some_node_in.seed.k, some_node_in.seed.c, some_node_in.length);
	key = node_get_prefix(some_node_in);
	node_to_vect(new->v, some_node_in);
	printf("storing: %s\n", vect_bin_to_binary_string(new->v, NULL));
	printf("compare works: %d\n", cmp_point(new->v, some_node_in));
	vect_to_node(some_node_out, new->v, key);
	printf("Out: %lu, %d, %lu, %d, %ld\n", (*some_node_out).tail.k, (*some_node_out).tail.c, (*some_node_out).seed.k, (*some_node_out).seed.c, (*some_node_out).length);
	printf("compare works: %d\n", cmp_point(new->v, *some_node_out));

	_vect_bin_chain_t_free(new);
}
#endif

#ifdef NODE_STRUCT_vowgcs_t
void print_memory(vowgcs_t *some_node_out)
{
		printf("-------------------\n");
		for(_cursor_type i = 0; i < memory_buffer_cursor; i++)
		{
				if(!vect_bin_is_empty(memory_buffer[i].v))
				{
						printf("%hd : \n", i);
						printf("bit-vector: %s\n", vect_bin_to_binary_string(memory_buffer[i].v, NULL));
						vect_to_node(some_node_out, memory_buffer[i].v, 0);
						printf("Corresponds to node (modulo the key): %lu, %d, %lu, %d, %ld\n", (*some_node_out).tail.k, (*some_node_out).tail.c, (*some_node_out).seed.k, (*some_node_out).seed.c, (*some_node_out).length);
						printf("Next: %hd\n", memory_buffer[i].nxt);
				}
		}
		printf("-------------------\n");
}

int check_memory_consistent(vowgcs_t *some_node_out, const int prf_counter)
{
	_vect_bin_chain_t *last;
	_vect_bin_chain_t *next;
	int cmp;
	for(int i = 0; i < chain_array_size; i++)
	{
		if(slot_prf_counter[i] == prf_counter) // should  not check otherwise
                {
		next = &memory_buffer[i];
		while(next->nxt != 0)
		{
			last = next;
			next = &memory_buffer[next->nxt];
			vect_to_node(some_node_out, next->v, i);
			cmp = cmp_point(last->v, *some_node_out);
			if(cmp > 0)
			{
				printf("Error in %d\n", i);
				return 1;
			}
		}
		}
	}
	return 0;
}
#endif


/***Memory limiting feature is turned off
static unsigned long long int memory_limit;
 ***/
static unsigned long long int memory_alloc;

/** Initialize the Packed Radix-Tree-List.
 *
 *	@brief Initialize the PRTL, allocate memory and
 *	create mask which will be used to map a stored point
 *	to an index of the chain array.
 *
 */
void struct_init_PRTL(uint8_t _level, int d, int omega) {
	int i;
	k_bits = __FP_BITS__ ;
	nb_stored_points = omega;
	assert(nb_stored_points <= __NB_STORED_POINTS__);
	distinguished_bits = d;
	c_bits = 1;
	lenght_bits = 16; //see if it can be less / more precise
	level = _level;
  suffix_len = k_bits - level - distinguished_bits + c_bits;
  tail_k_start = 0;
	tail_c_start = k_bits - level - distinguished_bits;
	seed_k_start = suffix_len;
	seed_c_start = seed_k_start + k_bits;
	length_start = seed_c_start + c_bits;
	chain_array_size = pow(2, level);

	if(__NB_STORED_POINTS__ > nb_stored_points)
	{
		printf("Warning! Memory allocation for PRTL is not set correctly. Program allocates unused memory. \n__NB_STORED_POINTS__ in pcs_struct_PRTL.h should be set to 2^w.\n");
	}

	//printf("%d,%d,%d,%d,%d\n",tail_k_start,tail_c_start,seed_k_start,seed_c_start,length_start);


    /***Memory limiting feature is turned off
	memory_limit = 100000000;
	 ***/
#ifndef LOCKFREE
    locks = malloc(sizeof(omp_lock_t) * chain_array_size);
	omp_init_lock(&cursor_lock);
	for(i = 0; i < chain_array_size; i++) {
		omp_init_lock(&locks[i]);
	}
#endif

    memory_alloc = sizeof(omp_lock_t) * chain_array_size;
	slot_prf_counter = malloc(sizeof(int) * chain_array_size);
	memory_alloc += sizeof(int) * chain_array_size;
	memory_alloc += sizeof(_vect_bin_chain_t) * __NB_STORED_POINTS__;

	for(i = 0; i < chain_array_size; i++) {
		slot_prf_counter[i] = 0;
	}

	for(i = 0; i < __NB_STORED_POINTS__; i++){
		vect_bin_t_reset(memory_buffer[i].v);
		memory_buffer[i].nxt = 0;
	}

	memory_buffer_cursor = chain_array_size;
}

/** Indicate prf change.
 *
 *	@brief When prf changes the structure is
 *	emptied which is done by replacing the
 * 	memory_buffer_cursor at the beginning.
 *
 */
void struct_change_prf()
{
		memory_buffer_cursor = chain_array_size;
		//printf("Restarted memory\n");
}

// lockfree https://gcc.gnu.org/onlinedocs/gcc-12.1.0/gcc/_005f_005fatomic-Builtins.html#g_t_005f_005fatomic-Builtins
#ifdef NODE_STRUCT_vowgcs_t
int struct_add_PRTL(vowgcs_t *node_out, vowgcs_t node_in, const int prf_counter)
{
	//print_memory(node_out);
	//check_serialization(node_out, node_in);
	uint8_t retval = 0;
	int key;
	int cmp = -1;
	_cursor_type new_point_cursor;
	_vect_bin_chain_t *new;
	_vect_bin_chain_t *last;
	_vect_bin_chain_t *next;

	key = node_get_prefix(node_in);
	//printf("Key : %d\n", key);
    omp_set_lock(&locks[key]);
    next = &memory_buffer[key];
	last = next; // no need but to remove warning

    if(vect_bin_is_empty(next->v)) {
		node_to_vect(next->v, node_in);
        next->nxt=0;
		slot_prf_counter[key] = prf_counter;
    }
    else {
		if(slot_prf_counter[key] != prf_counter) // should start at the beginning of chain
		{
					vect_bin_t_reset(next->v);
					node_to_vect(next->v, node_in);
					next->nxt=0;
					slot_prf_counter[key] = prf_counter;
		}
		else // add in current List
				{
		        cmp = cmp_point(next->v, node_in);
		        while(cmp < 0 && next->nxt != 0)
		        {
			          last = next;
			      		next = &memory_buffer[next->nxt];
								cmp = cmp_point(next->v, node_in);
		        }
		        if(cmp == 0) //collision
		        {
								//printf("***collision\n");
								vect_to_node(node_out, next->v, key);

								//replace previous point
								vect_bin_t_reset(next->v);
								node_to_vect(next->v, node_in);

		            retval = 1;
		        }
		        else
		        {
		            omp_set_lock(&cursor_lock);
								new_point_cursor = memory_buffer_cursor++;
								omp_unset_lock(&cursor_lock);
								if(new_point_cursor < nb_stored_points)
								{
										new = &memory_buffer[new_point_cursor];

										if(next == &memory_buffer[key])
						        {
												if(cmp > 0) //add at the beginning
												{
														//printf("***add beginning\n");
								        		vect_bin_cpy(new->v, next->v);
								            new->nxt = next->nxt;

								            vect_bin_t_reset(next->v);
														node_to_vect(next->v, node_in);

												}
												else // add at the end - while stopped because next was 0
												{
														//printf("***add end --\n");
														vect_bin_t_reset(new->v);
														node_to_vect(new->v, node_in);
														new->nxt = 0;
												}
												next->nxt = new_point_cursor;
						      	}
						        else // add in the middle or at the end
						        {
												vect_bin_t_reset(new->v);
												node_to_vect(new->v, node_in);
												if(cmp > 0) // add in the middle
												{
														//printf("***add middle\n");
														new->nxt = last->nxt;
														last->nxt = new_point_cursor;
												}
												else
												{
														//printf("***add end\n");
														new->nxt = 0;
														next->nxt = new_point_cursor;
												}
										}
								}
								else
								{
									//replace previous point
									vect_bin_t_reset(next->v);
									node_to_vect(next->v, node_in);
									//printf("Reached memory limit. The point will not be stored\n");
								}
		        }
				}
    }
	omp_unset_lock(&locks[key]);
	return retval;
}
#endif

/** Free the allocated memory for the Packed Radix-Tree-List.
 *
 */
void struct_free_PRTL(void)
{
  	int i;

#ifndef LOCKFREE
  	omp_destroy_lock(&_vect_alloc_size_lock);
	for(i = 0; i < chain_array_size; i++) {
        omp_destroy_lock(&locks[i]);
	}
	omp_destroy_lock(&cursor_lock);
	free(locks);
#endif

	free(slot_prf_counter);
}
