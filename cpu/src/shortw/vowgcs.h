#ifndef _VOWGCS_H_
#define _VOWGCS_H_

// Sorting and searching functions concerning the linked list element (for metrics)
typedef struct linkedlist_struct{
    struct linkedlist_struct *next; // next element
    point_t point;                  // point: value of the linked list element
} linkedlist_t;

typedef struct {
    digit_t omega_minus_one;    // 2^{omega_bits} - 1
    digit_t omegabits;          // omega = 2^omegabits
    digit_t omega;              // Limit: memory cells
    digit_t beta;
    double theta;               // 2.25 x (omega / 2N) portion of distinguished point
    uint8_t n;                  // 1/theta = R*2^n with 0 <= R < 2
    uint8_t Rbits;              
    digit_t distinguished;      // approximation for 2^Rbits/R
    digit_t betaXomega;         // (beta x omega) distinguished points per each PRF
    digit_t maxtrail;           // 10 / theta
    digit_t maxprf;             // Maximum number of PRF
    // Concerning number of cores (metrics: number of collisions)
    digit_t cores;
    linkedlist_t **address;     // Each core has its own list of pointers
    linkedlist_t **collisions;  // Sorted linked list of different collisions
    digit_t *index;             // Current number of different collisions
    digit_t *runtime_collision; // Number of all collisions
    digit_t *runtime_different; // Number of all different collisions
    digit_t *runtime;           // Number of function evaluations _fn_()
    uint8_t heuristic;          // For measuring vOW GCS heuristics not focused on the golden collision search
} ctx_vow_t;

void create_node(linkedlist_t *new_node, point_t new_point);                    // It creates a new node (distinguished point)
linkedlist_t *middle_node(linkedlist_t *first_node, linkedlist_t *last_node);   // It gets the middle element of a linked list
linkedlist_t *linkedlist_binarysearch(linkedlist_t *head, point_t point);       // Binary search in a linked list
void sorted_insert(linkedlist_t **head, linkedlist_t *new_node);                // It inserts an element in a sorted linked list

typedef struct {
    point_t seed;   // initial random point
    point_t tail;   // distinguished point
    digit_t length; // integer length such that (_fn_)^length(seed) = tail point; that is, length function evaluations
} vowgcs_t;

digit_t reconstruction(point_t *collision, const vowgcs_t X, const vowgcs_t Y, const ctx_mitm_t context);
double vowgcs(point_t *golden, uint8_t *finished, vowgcs_t *hashtable,const ctx_mitm_t context, const ctx_vow_t ctx, const int id);
#endif
