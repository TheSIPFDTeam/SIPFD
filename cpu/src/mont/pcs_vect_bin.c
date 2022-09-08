
#include<assert.h>
#include<stdlib.h>
#include <stdio.h>
#include<string.h>
#include "pcs_vect_bin.h"

/// Adressing bytes (and bits) is like this in _vect_bin_t type
///      0         1        2         3         4
/// [N-------][--------][--------][--------][-------0]
/// [--------][--------][--------][--------][--------]

unsigned long long _vect_bin_alloc_size;
omp_lock_t _vect_alloc_size_lock;

/// call once at the begin of each program
void _vect_bin_t_initiate()
{
	_vect_bin_alloc_size = 0ULL;
	omp_init_lock(&_vect_alloc_size_lock);
}

/// get bit at rank return _true or _false respectively to 1 and 0
inline _bool_t vect_bin_get_bit(_vect_bin_t *t, int rank) {
//  printf(" Rank(%d) : t[%lu] @ Rank(%lu)\n", rank, _vect_bin_array_size - 1 - (rank / (sizeof(_vect_bin_t) << 3)), (rank % (sizeof(_vect_bin_t) << 3)));
  return((t[_vect_bin_array_size - 1 - (rank / (sizeof(_vect_bin_t) << 3))] & ((_vect_bin_t) 1 << (rank % (sizeof(_vect_bin_t) << 3)))) ? 1 : 0);
}

/// set 'rank' bit to 1
inline void vect_bin_set_1(_vect_bin_t *t, int rank) {
	assert(t != NULL);
  t[_vect_bin_array_size - 1 - (rank / (sizeof(_vect_bin_t) << 3))] |= ((_vect_bin_t) 1 << (rank % (sizeof(_vect_bin_t) << 3)));
}

/// set 'rank' bit to 0
inline void vect_bin_set_0(_vect_bin_t *t, int rank) {
  t[_vect_bin_array_size - 1 - (rank / (sizeof(_vect_bin_t) << 3))] &= (~((_vect_bin_t) 1 << (rank % (sizeof(_vect_bin_t) << 3))));
}

/// set an int value to a _vect_bin_t type starting at a given bit number
_vect_bin_t *vect_bin_set_int(_vect_bin_t *t, int from_bit, int value) {
  const int int_bitsize = (sizeof(int) << 3);
  for(int i = 0; i < int_bitsize; ++i)
    if(value & (1 << i)) vect_bin_set_1(t, i + from_bit);
  return(t);
}

/// get an int value from a _vect_bin_t type starting at a given bit number
int vect_bin_get_int(_vect_bin_t *t, int from_bit) {
  int out = 0;
  const int int_bitsize = (sizeof(int) << 3);
  for(int i = 0; i < int_bitsize; ++i)
    if(vect_bin_get_bit(t, from_bit + i)) out |= (1 << i);
  return(out);
}

/// set an uint64_t value to a _vect_bin_t type starting at a given bit number
_vect_bin_t *vect_bin_set_int64(_vect_bin_t *t, int from_bit, uint64_t value, int from_bit_value, int real_len) {
	uint64_t const1_64 = 1;
	for(int i = 0; i < real_len - from_bit_value; ++i)
	{
		//printf("64: Setting vect_bit %d to bit %d : %d\n", i + from_bit, i + from_bit_value, value & (const1_64 << i + from_bit_value));
		if(value & (const1_64 << (i + from_bit_value))) vect_bin_set_1(t, i + from_bit);
	}

	return(t);
}

/// get an uint64_t value from a _vect_bin_t type starting at a given bit number
uint64_t vect_bin_get_int64(_vect_bin_t *t, int from_bit, int from_bit_value, int real_len) {
	uint64_t out = 0;
	uint64_t const1_64 = 1;
	for(int i = 0; i < real_len - from_bit_value; ++i)
	{
		if(vect_bin_get_bit(t, i + from_bit)) out |= (const1_64 << (i + from_bit_value));
		//printf("64:Getting vect_bit %d for bit %d : %d - res: %d\n", i + from_bit, i + from_bit_value, vect_bin_get_bit(t, i + from_bit), out);
	}
	return out;
}

/// set an uint8_t value to a _vect_bin_t type starting at a given bit number
_vect_bin_t *vect_bin_set_int8(_vect_bin_t *t, int from_bit, uint8_t value, int real_len) {
	for(int i = 0; i < real_len; ++i)
	{
		//printf("setting vect_bit %d to bit %d : res %d\n",i + from_bit, i, value & (1 << i));
		if(value & (1 << i)) vect_bin_set_1(t, i + from_bit);
	}
	return(t);
}

/// get an uint8_t value from a _vect_bin_t type starting at a given bit number
uint8_t vect_bin_get_int8(_vect_bin_t *t, int from_bit, int real_len) {
	uint8_t out = 0;
	for(int i = 0; i < real_len; ++i)
	{
		if(vect_bin_get_bit(t, i + from_bit)) out |= (1 << i);
		//printf("getting vect_bit %d for bit %d - res %d\n",i + from_bit,i, out);
	}
	return out;
}

/// set an fp value to a _vect_bin_t type starting at a given bit number
_vect_bin_t *vect_bin_set_fp(_vect_bin_t *t, int from_bit, uint64_t value[__FP_WORDS__], int from_bit_value) {
	int i_vect = from_bit;
	int j = from_bit_value / 64;
	int i = from_bit_value % 64;
	//printf("start from i:%d, j:%d\n",i,j);
	while(i < 64)
	{
		if(j*64 + i >= __FP_BITS__) break;
		//printf("putting %d,%d to %d--bit is %d\n",j,i,i_vect,value[j] & (1 << i));
		if(value[j] & (1 << i)) vect_bin_set_1(t, i_vect);
		i++;
		i_vect++;
	}
	j++;
	while(j < __FP_WORDS__)
	{
		for(int i = 0; i < 64; ++i)
		{
			if(j*64 + i >= __FP_BITS__) break;
			//printf("putting %d,%d to %d--bit is %d\n",j,i,i_vect,value[j] & (1 << i));
			if(value[j] & (1 << i)) vect_bin_set_1(t, i_vect);
			i_vect++;
		}
		j++;
	}
	return(t);
}

/// get an fp value from a _vect_bin_t type starting at a given bit number
void vect_bin_get_fp(uint64_t value[__FP_WORDS__], _vect_bin_t *t, int from_bit_vect, int from_bit_value) {
	int i_vect = from_bit_vect;
	int j = from_bit_value / 64;
	int i = from_bit_value % 64;
	//printf("start from i:%d, j:%d\n",i,j);
	value[j] = 0;
	while(i < 64)
	{
		if(j*64 + i >= __FP_BITS__) break;
		//printf("taking from %d to %d,%d--bit is %d\n",i_vect, j, i,vect_bin_get_bit(t, j*64 + i_vect));
		if(vect_bin_get_bit(t, i_vect)) value[j] |= (1 << i);
		i++;
		i_vect++;
	}
	j++;
	while(j < __FP_WORDS__)
	{
		value[j] = 0;
		for(int i = 0; i < 64; ++i)
		{
			if(j*64 + i >= __FP_BITS__) break;
			//printf("taking from %d to %d,%d -- bit %d\n",i_vect, j, i, vect_bin_get_bit(t, i_vect));
			if(vect_bin_get_bit(t, i_vect)) value[j] |= (1 << i);
			i_vect++;
		}
		j++;
	}
}

///compare a point with a _vect_bin_t type starting at a given bit number
/// if dist < vect return -1, if dist > vect return 1, if dist == vect return 0
///from_bit_dist has to be smaller than len. the second param is checked in full - can not be part od the PRTL prefix
int vect_bin_cmp_point(_vect_bin_t *t, int from_bit_vect, point_t *dist, int from_bit_dist, int len1, int len2) {
	uint64_t const1_64 = 1;
	for(int i = len1 - 1; i >= from_bit_dist; i--)
	{
		//printf("Comparing vect_bit %d with bit %d : %d, %d\n", from_bit_vect + i - from_bit_dist, i, vect_bin_get_bit(t, from_bit_vect + i - from_bit_dist), (((dist->k & (const1_64 << i)) !=0) ? 1 : 0));
		if(vect_bin_get_bit(t, from_bit_vect + i - from_bit_dist) > (((dist->k & (const1_64 << i)) !=0) ? 1 : 0)) return 1;
		else if(vect_bin_get_bit(t, from_bit_vect + i - from_bit_dist) < (((dist->k & (const1_64 << i)) !=0) ? 1 : 0)) return -1;
	}
	for(int i = len2 - 1; i >= 0; i--)
	{
		//printf("Comparing vect_bit %d with bit %d : %d, %d\n", from_bit_vect + i + len1 - from_bit_dist, i, vect_bin_get_bit(t, from_bit_vect + i + len1 - from_bit_dist), (((dist->c & (const1_64 << i)) !=0) ? 1 : 0));
		if(vect_bin_get_bit(t, from_bit_vect + i + len1 - from_bit_dist) > (((dist->c & (const1_64 << i)) !=0) ? 1 : 0)) return 1;
		else if(vect_bin_get_bit(t, from_bit_vect + i + len1 - from_bit_dist) < (((dist->c & (const1_64 << i)) !=0) ? 1 : 0)) return -1;
	}
	return 0;
}

void print_vect_bin(_vect_bin_t *v) {
  for(int i = 0; i < _vect_bin_array_size; ++i) printf(" %d", v[i]);
  printf("\n");
}

//check if _vect_bin_t is zero
int vect_bin_is_empty(_vect_bin_t *v)
{
    for(int i = 0; i < _vect_bin_array_size; ++i) if(v[i] != 0) return 0;
    return 1;
}

//deep clone _vect_bin_t
void vect_bin_cpy(_vect_bin_t *out, _vect_bin_t *in)
{
    memcpy(out, in, _vect_bin_array_size);
}

/// return a string that represents the vect_bin_t number in decimal format
char *vect_bin_to_binary_string(_vect_bin_t *t, char *s) {
  char *out = ((s == NULL) ? (char *)malloc(sizeof(char) * (_vect_bin_size + 1)) : s);
  for(int i = _vect_bin_size - 1; i >= 0; --i)
    out[_vect_bin_size - 1 - i] = (vect_bin_get_bit(t, i) ? '1' : '0');
  out[_vect_bin_size] = '\0';
  return(out);
}

/// _vect_bin_t assigned to 0
_vect_bin_t *vect_bin_t_reset(_vect_bin_t *_v) {
  if(_v == NULL) return(NULL);
  for(int i = 0; i < _vect_bin_array_size; ++i) _v[i] = 0;
  return _v;
}

/// return a substring from t from from_bit to to_bit. o could be NULL. It's then allocated
void vect_bin_get_vect_bin_from(_vect_bin_t *t, int from_bit, int to_bit, _vect_bin_t *o) {
  int i, j;
  for(i = from_bit, j = 0; i < to_bit; ++i, ++j)
    if(vect_bin_get_bit(t, i)) vect_bin_set_1(o, j);
    else vect_bin_set_0(o, j);
}
