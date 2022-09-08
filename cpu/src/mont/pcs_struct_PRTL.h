#define NODE_STRUCT_vowgcs_t
#define __NB_STORED_POINTS__ 1024
// #define LOCKFREE

void test_func(void);
void struct_init_PRTL(uint8_t _level, int d, int nb_stored_points);
void struct_change_prf(void);
#ifdef NODE_STRUCT_vowgcs_t
int struct_add_PRTL(vowgcs_t *node_out, vowgcs_t node_in, const int prf_counter);
#endif
void struct_free_PRTL(void);
int compute_optimal_level(digit_t M);
