
/// type of boolean
typedef char _bool_t;
#define _true 1
#define _false 0

typedef int _cursor_type;

#define __FP_WORDS__ 2

extern unsigned long long _vect_bin_alloc_size;
extern omp_lock_t _vect_alloc_size_lock;

/// type of one cell of the binary vector
typedef char _vect_bin_t;

/// Size - in bits - of the binary vector
static const unsigned int _vect_bin_size = (__DATA_SIZE_IN_BYTES__ * (sizeof(_vect_bin_t) << 3));

/// Size - computed - of needed _vect_bin_t cells to store the binary
/// vector and a pointer to the next cell.
static const unsigned int _vect_bin_array_size = __DATA_SIZE_IN_BYTES__;

/// Main structure dedidacted to store binary vectors of at most
/// _STATIC_VECT_BIN_SIZE_ bits
/// __attribute__((packed)) is needed to avoid struct memory alignment
/// and then unintended padding
typedef struct __vect_bin_list_t {
  _vect_bin_t v[__DATA_SIZE_IN_BYTES__];
  _cursor_type nxt;
} __attribute__((packed)) _vect_bin_chain_t;

/// call it when you allocate several (_n) cells that will start
/// each one list.
#define _vect_bin_t_count_memory(_n) \
  _vect_bin_alloc_size += (sizeof(_vect_bin_chain_t) * _n)

/// Initialization of one cell that will be chained
#define _vect_bin_chain_t_new(_v) \
    omp_set_lock(&_vect_alloc_size_lock); \
  _v = (_vect_bin_chain_t *) malloc(sizeof(_vect_bin_chain_t)); \
  _v->nxt = 0; \
  _vect_bin_alloc_size += sizeof(_vect_bin_chain_t); \
  omp_unset_lock(&_vect_alloc_size_lock)


  /// Free memory of one cell that is chained
  #define _vect_bin_chain_t_free(_v) \
    free(_v); \
    _vect_bin_alloc_size -= sizeof(_vect_bin_chain_t)

/// ----------------------------------- prototypes

void _vect_bin_t_initiate(void);
void print_vect_bin(_vect_bin_t *);
_vect_bin_t *vect_bin_t_reset(_vect_bin_t *);
_bool_t vect_bin_get_bit(_vect_bin_t *, int);
void vect_bin_set_1(_vect_bin_t *, int);
void vect_bin_set_0(_vect_bin_t *, int);
char *vect_bin_to_binary_string(_vect_bin_t *, char *);
_vect_bin_t *vect_bin_set_int(_vect_bin_t *, int, int);
int vect_bin_get_int(_vect_bin_t *, int);
void vect_bin_get_vect_bin_from(_vect_bin_t *, int, int, _vect_bin_t *);
_vect_bin_t *vect_bin_set_int64(_vect_bin_t *t, int from_bit, uint64_t value, int from_bit_value, int real_len);
uint64_t vect_bin_get_int64(_vect_bin_t *t, int from_bit, int from_bit_value, int real_len);
_vect_bin_t *vect_bin_set_int8(_vect_bin_t *t, int from_bit, uint8_t value, int real_len);
uint8_t vect_bin_get_int8(_vect_bin_t *t, int from_bit, int real_len);
 _vect_bin_t *vect_bin_set_fp(_vect_bin_t *t, int from_bit, uint64_t value[__FP_WORDS__], int from_bit_value);
void vect_bin_get_fp(uint64_t value[__FP_WORDS__], _vect_bin_t *t, int from_bit_vect, int from_bit_value);
void vect_bin_cpy(_vect_bin_t *out, _vect_bin_t *in);
int vect_bin_is_empty(_vect_bin_t *v);
int vect_bin_cmp_point(_vect_bin_t *t, int from_bit_vect, point_t *dist, int from_bit_dist, int len1, int len2);
