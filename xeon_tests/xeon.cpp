// TODO: make free in tests, make allocation of pseudo-random sizes between itearations. use -O0
#define __STDC_LIMIT_MACROS // enable intN_t limits from stdint.h
#include <stdint.h>

#include "templ.h"

#ifdef KNC_H
//    #include "knc.h"
#else
//    #include "knc-i1x16.h"
    #include <stdio.h>
#endif

void gather32_double(double  *d  , int *mask);
void gather32_float (float   *f  , int *mask);
void gather32_i8    (int8_t  *i8 , int *mask);
void gather32_i16   (int16_t *i16, int *mask);
void gather32_i32   (int32_t *i32, int *mask);
void gather32_i64   (int64_t *i64, int *mask);

void gather64_double(double  *d  , int *mask);
void gather64_float (float   *f  , int *mask);
void gather64_i8    (int8_t  *i8 , int *mask);
void gather64_i16   (int16_t *i16, int *mask);
void gather64_i32   (int32_t *i32, int *mask);
void gather64_i64   (int64_t *i64, int *mask);


void gather_base_offsets32_double(double  *d  , int *mask);
void gather_base_offsets32_float (float   *f  , int *mask);
void gather_base_offsets32_i8    (int8_t  *i8 , int *mask);
void gather_base_offsets32_i16   (int16_t *i16, int *mask);
void gather_base_offsets32_i32   (int32_t *i32, int *mask);
void gather_base_offsets32_i64   (int64_t *i64, int *mask);

void gather_base_offsets64_double(double  *d  , int *mask);
void gather_base_offsets64_float (float   *f  , int *mask);
void gather_base_offsets64_i8    (int8_t  *i8 , int *mask);
void gather_base_offsets64_i16   (int16_t *i16, int *mask);
void gather_base_offsets64_i32   (int32_t *i32, int *mask);
void gather_base_offsets64_i64   (int64_t *i64, int *mask);


void scatter32_double(double  *d  , int *mask);
void scatter32_float (float   *f  , int *mask);
void scatter32_i8    (int8_t  *i8 , int *mask);
void scatter32_i16   (int16_t *i16, int *mask);
void scatter32_i32   (int32_t *i32, int *mask);
void scatter32_i64   (int64_t *i64, int *mask);

void scatter64_double(double  *d  , int *mask);
void scatter64_float (float   *f  , int *mask);
void scatter64_i8    (int8_t  *i8 , int *mask);
void scatter64_i16   (int16_t *i16, int *mask);
void scatter64_i32   (int32_t *i32, int *mask);
void scatter64_i64   (int64_t *i64, int *mask);


void scatter_base_offsets32_double(double  *d  , int *mask);
void scatter_base_offsets32_float (float   *f  , int *mask);
void scatter_base_offsets32_i8    (int8_t  *i8 , int *mask);
void scatter_base_offsets32_i16   (int16_t *i16, int *mask);
void scatter_base_offsets32_i32   (int32_t *i32, int *mask);
void scatter_base_offsets32_i64   (int64_t *i64, int *mask);

void scatter_base_offsets64_double(double  *d  , int *mask);
void scatter_base_offsets64_float (float   *f  , int *mask);
void scatter_base_offsets64_i8    (int8_t  *i8 , int *mask);
void scatter_base_offsets64_i16   (int16_t *i16, int *mask);
void scatter_base_offsets64_i32   (int32_t *i32, int *mask);
void scatter_base_offsets64_i64   (int64_t *i64, int *mask);


void masked_load_double(double  *d  , int *mask);
void masked_load_float (float   *f  , int *mask);
void masked_load_i8    (int8_t  *i8 , int *mask);
void masked_load_i16   (int16_t *i16, int *mask);
void masked_load_i32   (int32_t *i32, int *mask);
void masked_load_i64   (int64_t *i64, int *mask);


void masked_store_double(double  *d  , int *mask);
void masked_store_float (float   *f  , int *mask);
void masked_store_i8    (int8_t  *i8 , int *mask);
void masked_store_i16   (int16_t *i16, int *mask);
void masked_store_i32   (int32_t *i32, int *mask);
void masked_store_i64   (int64_t *i64, int *mask);

void masked_store_blend_double(double  *d  , int *mask);
void masked_store_blend_float (float   *f  , int *mask);
void masked_store_blend_i8    (int8_t  *i8 , int *mask);
void masked_store_blend_i16   (int16_t *i16, int *mask);
void masked_store_blend_i32   (int32_t *i32, int *mask);
void masked_store_blend_i64   (int64_t *i64, int *mask);


void movmsk(int *mask);


void insert_extract_element_double(double  *d  );
void insert_extract_element_float (float   *f  );
void insert_extract_element_i8    (int8_t  *i8 );
void insert_extract_element_i16   (int16_t *i16);
void insert_extract_element_i32   (int32_t *i32);
void insert_extract_element_i64   (int64_t *i64);


void load_double(double  *d  );
void load_float (float   *f  );
void load_i8    (int8_t  *i8 );
void load_i16   (int16_t *i16);
void load_i32   (int32_t *i32);
void load_i64   (int64_t *i64);


void store_double(double  *d  );
void store_float (float   *f  );
void store_i8    (int8_t  *i8 );
void store_i16   (int16_t *i16);
void store_i32   (int32_t *i32);
void store_i64   (int64_t *i64);


void smear_double(double  *d  );
void smear_float (float   *f  );
void smear_i8    (int8_t  *i8 );
void smear_i16   (int16_t *i16);
void smear_i32   (int32_t *i32);
void smear_i64   (int64_t *i64);


void setzero_double();
void setzero_float ();
void setzero_i8    ();
void setzero_i16   ();
void setzero_i32   ();
void setzero_i64   ();


void select_double(double  *d  , int *mask);
void select_float (float   *f  , int *mask);
void select_i8    (int8_t  *i8 , int *mask);
void select_i16   (int16_t *i16, int *mask);
void select_i32   (int32_t *i32, int *mask);
void select_i64   (int64_t *i64, int *mask);

void select_double_cond(double  *d  , int *mask);
void select_float_cond (float   *f  , int *mask);
void select_i8_cond    (int8_t  *i8 , int *mask);
void select_i16_cond   (int16_t *i16, int *mask);
void select_i32_cond   (int32_t *i32, int *mask);
void select_i64_cond   (int64_t *i64, int *mask);


void broadcast_double(double  *d  );
void broadcast_float (float   *f  );
void broadcast_i8    (int8_t  *i8 );
void broadcast_i16   (int16_t *i16);
void broadcast_i32   (int32_t *i32);
void broadcast_i64   (int64_t *i64);


void rotate_double(double  *d  );
void rotate_float (float   *f  );
void rotate_i8    (int8_t  *i8 );
void rotate_i16   (int16_t *i16);
void rotate_i32   (int32_t *i32);
void rotate_i64   (int64_t *i64);


void shift_double(double  *d  );
void shift_float (float   *f  );
void shift_i8    (int8_t  *i8 );
void shift_i16   (int16_t *i16);
void shift_i32   (int32_t *i32);
void shift_i64   (int64_t *i64);


void shuffle_double(double  *d  );
void shuffle_float (float   *f  );
void shuffle_i8    (int8_t  *i8 );
void shuffle_i16   (int16_t *i16);
void shuffle_i32   (int32_t *i32);


void cast_i64_i32(int32_t *i32);
void cast_i64_i16(int16_t *i16);
void cast_i64_i8 (int8_t  *i8);
void cast_i64_i1 (bool    *i1);
void cast_i32_i16(int16_t *i16);
void cast_i32_i8 (int8_t  *i8);
void cast_i32_i1 (bool    *i1);
void cast_i16_i8 (int8_t  *i8);
void cast_i16_i1 (bool    *i1);
void cast_i8_i1 (bool    *i1);

void cast_ui64_ui32(uint32_t *ui32);
void cast_ui64_ui16(uint16_t *ui16);
void cast_ui64_ui8 (uint8_t  *ui8);
void cast_ui64_ui1 (bool      *ui1);
void cast_ui32_ui16(uint16_t *ui16);
void cast_ui32_ui8 (uint8_t  *ui8);
void cast_ui32_ui1 (bool      *ui1);
void cast_ui16_ui8 (uint8_t  *ui8);
void cast_ui16_ui1 (bool      *ui1);
void cast_ui8_ui1 (bool      *ui1);

void trunk_i32_i64(int64_t *i64);
void trunk_i16_i64(int64_t *i64);
void trunk_i8_i64 (int64_t *i64);
void trunk_i16_i32(int32_t *i32);
void trunk_i8_i32 (int32_t *i32);
void trunk_i8_i16 (int16_t *i16);

void cast_f_i8 (int8_t *i8);
void cast_f_i16(int16_t *i16);
void cast_f_i32(int32_t *i32);
void cast_f_i64(int64_t *i64);
void cast_d_i8 (int8_t *i8);
void cast_d_i16(int16_t *i16);
void cast_d_i32(int32_t *i32);
void cast_d_i64(int64_t *i64);

void cast_f_ui8 (uint8_t *ui8);
void cast_f_ui16(uint16_t *ui16);
void cast_f_ui32(uint32_t *ui32);
void cast_f_ui64(uint64_t *ui64);
void cast_d_ui8 (uint8_t *ui8);
void cast_d_ui16(uint16_t *ui16);
void cast_d_ui32(uint32_t *ui32);
void cast_d_ui64(uint64_t *ui64);

void cast_i8_f (float  *f_8);
void cast_i16_f(float  *f_16);
void cast_i32_f(float  *f_32);
void cast_i64_f(float  *f_64);
void cast_i8_d (double *d_8);
void cast_i16_d(double *d_16);
void cast_i32_d(double *d_32);
void cast_i64_d(double *d_64);

void cast_ui8_f (float  *f_u8);
void cast_ui16_f(float  *f_u16);
void cast_ui32_f(float  *f_u32);
void cast_ui64_f(float  *f_u64);
void cast_ui8_d (double *d_u8);
void cast_ui16_d(double *d_u16);
void cast_ui32_d(double *d_u32);
void cast_ui64_d(double *d_u64);

void cast_f_d(double *d);
void cast_d_f(float  *f);


void add_double(double  *d_1  , double  *d_2  );
void add_float (float   *f_1  , float   *f_2  );
void add_i8    (int8_t  *i8_1 , int8_t  *i8_2 );
void add_i16   (int16_t *i16_1, int16_t *i16_2);
void add_i32   (int32_t *i32_1, int32_t *i32_2);
void add_i64   (int64_t *i64_1, int64_t *i64_2);

void sub_double(double  *d_1  , double  *d_2  );
void sub_float (float   *f_1  , float   *f_2  );
void sub_i8    (int8_t  *i8_1 , int8_t  *i8_2 );
void sub_i16   (int16_t *i16_1, int16_t *i16_2);
void sub_i32   (int32_t *i32_1, int32_t *i32_2);
void sub_i64   (int64_t *i64_1, int64_t *i64_2);

void mul_double(double  *d_1  , double  *d_2  );
void mul_float (float   *f_1  , float   *f_2  );
void mul_i8    (int8_t  *i8_1 , int8_t  *i8_2 );
void mul_i16   (int16_t *i16_1, int16_t *i16_2);
void mul_i32   (int32_t *i32_1, int32_t *i32_2);
void mul_i64   (int64_t *i64_1, int64_t *i64_2);

void or_i8 (int8_t  *i8_1 , int8_t  *i8_2 );
void or_i16(int16_t *i16_1, int16_t *i16_2);
void or_i32(int32_t *i32_1, int32_t *i32_2);
void or_i64(int64_t *i64_1, int64_t *i64_2);

void and_i8 (int8_t  *i8_1 , int8_t  *i8_2 );
void and_i16(int16_t *i16_1, int16_t *i16_2);
void and_i32(int32_t *i32_1, int32_t *i32_2);
void and_i64(int64_t *i64_1, int64_t *i64_2);

void xor_i8 (int8_t  *i8_1 , int8_t  *i8_2 );
void xor_i16(int16_t *i16_1, int16_t *i16_2);
void xor_i32(int32_t *i32_1, int32_t *i32_2);
void xor_i64(int64_t *i64_1, int64_t *i64_2);

void shl_i8 (int8_t  *i8_1 , int8_t  *i8_2 );
void shl_i16(int16_t *i16_1, int16_t *i16_2);
void shl_i32(int32_t *i32_1, int32_t *i32_2);
void shl_i64(int64_t *i64_1, int64_t *i64_2);

void udiv_ui8 (uint8_t  *ui8_1 , uint8_t  *ui8_2 );
void udiv_ui16(uint16_t *ui16_1, uint16_t *ui16_2);
void udiv_ui32(uint32_t *ui32_1, uint32_t *ui32_2);
void udiv_ui64(uint64_t *ui64_1, uint64_t *ui64_2);

void sdiv_i8 (int8_t  *i8_1 , int8_t  *i8_2 );
void sdiv_i16(int16_t *i16_1, int16_t *i16_2);
void sdiv_i32(int32_t *i32_1, int32_t *i32_2);
void sdiv_i64(int64_t *i64_1, int64_t *i64_2);

void div_double(double *d_1, double *d_2);
void div_float (float  *f_1, float  *f_2);

void urem_ui8 (uint8_t  *ui8_1 , uint8_t  *ui8_2 );
void urem_ui16(uint16_t *ui16_1, uint16_t *ui16_2);
void urem_ui32(uint32_t *ui32_1, uint32_t *ui32_2);
void urem_ui64(uint64_t *ui64_1, uint64_t *ui64_2);

void srem_i8 (int8_t  *i8_1 , int8_t  *i8_2 );
void srem_i16(int16_t *i16_1, int16_t *i16_2);
void srem_i32(int32_t *i32_1, int32_t *i32_2);
void srem_i64(int64_t *i64_1, int64_t *i64_2);

void lshr_ui8 (uint8_t  *ui8_1 , uint8_t  *ui8_2 );
void lshr_ui16(uint16_t *ui16_1, uint16_t *ui16_2);
void lshr_ui32(uint32_t *ui32_1, uint32_t *ui32_2);
void lshr_ui64(uint64_t *ui64_1, uint64_t *ui64_2);

void ashr_i8 (int8_t  *i8_1 , int8_t  *i8_2 );
void ashr_i16(int16_t *i16_1, int16_t *i16_2);
void ashr_i32(int32_t *i32_1, int32_t *i32_2);
void ashr_i64(int64_t *i64_1, int64_t *i64_2);

void lshr_ui8_uniform (uint8_t  *ui8 , int32_t *i32);
void lshr_ui16_uniform(uint16_t *ui16, int32_t *i32);
void lshr_ui32_uniform(uint32_t *ui32, int32_t *i32);
void lshr_ui64_uniform(uint64_t *ui64, int32_t *i32);

void ashr_i8_uniform (int8_t  *i8 , int32_t *i32);
void ashr_i16_uniform(int16_t *i16, int32_t *i32);
void ashr_i32_uniform(int32_t *i32_1, int32_t *i32);
void ashr_i64_uniform(int64_t *i64, int32_t *i32);


void equal_double(double  *d  );
void equal_float (float   *f  );
void equal_i8    (int8_t  *i8 );
void equal_i16   (int16_t *i16);
void equal_i32   (int32_t *i32);
void equal_i64   (int64_t *i64);

void not_equal_double(double  *d  );
void not_equal_float (float   *f  );
void not_equal_i8    (int8_t  *i8 );
void not_equal_i16   (int16_t *i16);
void not_equal_i32   (int32_t *i32);
void not_equal_i64   (int64_t *i64);

void unsigned_less_equal_i8  (uint8_t  *ui8 );
void unsigned_less_equal_i16 (uint16_t *ui16);
void unsigned_less_equal_i32 (uint32_t *ui32);
void unsigned_less_equal_i64 (uint64_t *ui64);

void signed_less_equal_i8  (int8_t  *i8 );
void signed_less_equal_i16 (int16_t *i16);
void signed_less_equal_i32 (int32_t *i32);
void signed_less_equal_i64 (int64_t *i64);

void less_equal_double(double *d);
void less_equal_float (float  *f);

void unsigned_greater_equal_i8  (uint8_t  *ui8 );
void unsigned_greater_equal_i16 (uint16_t *ui16);
void unsigned_greater_equal_i32 (uint32_t *ui32);
void unsigned_greater_equal_i64 (uint64_t *ui64);

void signed_greater_equal_i8  (int8_t  *i8 );
void signed_greater_equal_i16 (int16_t *i16);
void signed_greater_equal_i32 (int32_t *i32);
void signed_greater_equal_i64 (int64_t *i64);

void greater_equal_double(double *d);
void greater_equal_float (float  *f);

void unsigned_less_than_i8  (uint8_t  *ui8 );
void unsigned_less_than_i16 (uint16_t *ui16);
void unsigned_less_than_i32 (uint32_t *ui32);
void unsigned_less_than_i64 (uint64_t *ui64);

void signed_less_than_i8  (int8_t  *i8 );
void signed_less_than_i16 (int16_t *i16);
void signed_less_than_i32 (int32_t *i32);
void signed_less_than_i64 (int64_t *i64);

void less_than_double(double *d);
void less_than_float (float  *f);

void unsigned_greater_than_i8  (uint8_t  *ui8 );
void unsigned_greater_than_i16 (uint16_t *ui16);
void unsigned_greater_than_i32 (uint32_t *ui32);
void unsigned_greater_than_i64 (uint64_t *ui64);

void signed_greater_than_i8  (int8_t  *i8 );
void signed_greater_than_i16 (int16_t *i16);
void signed_greater_than_i32 (int32_t *i32);
void signed_greater_than_i64 (int64_t *i64);

void greater_than_double(double *d);
void greater_than_float (float  *f);

void equal_double_and_mask(double  *d  , int *mask);
void equal_float_and_mask (float   *f  , int *mask);
void equal_i8_and_mask    (int8_t  *i8 , int *mask);
void equal_i16_and_mask   (int16_t *i16, int *mask);
void equal_i32_and_mask   (int32_t *i32, int *mask);
void equal_i64_and_mask   (int64_t *i64, int *mask);

void not_equal_double_and_mask(double  *d  , int *mask);
void not_equal_float_and_mask (float   *f  , int *mask);
void not_equal_i8_and_mask    (int8_t  *i8 , int *mask);
void not_equal_i16_and_mask   (int16_t *i16, int *mask);
void not_equal_i32_and_mask   (int32_t *i32, int *mask);
void not_equal_i64_and_mask   (int64_t *i64, int *mask);

void unsigned_less_equal_i8_and_mask  (uint8_t  *ui8 , int *mask);
void unsigned_less_equal_i16_and_mask (uint16_t *ui16, int *mask);
void unsigned_less_equal_i32_and_mask (uint32_t *ui32, int *mask);
void unsigned_less_equal_i64_and_mask (uint64_t *ui64, int *mask);

void signed_less_equal_i8_and_mask  (int8_t  *i8 , int *mask);
void signed_less_equal_i16_and_mask (int16_t *i16, int *mask);
void signed_less_equal_i32_and_mask (int32_t *i32, int *mask);
void signed_less_equal_i64_and_mask (int64_t *i64, int *mask);

void less_equal_double_and_mask(double *d, int *mask);
void less_equal_float_and_mask (float  *f, int *mask);

void unsigned_greater_equal_i8_and_mask  (uint8_t  *ui8 , int *mask);
void unsigned_greater_equal_i16_and_mask (uint16_t *ui16, int *mask);
void unsigned_greater_equal_i32_and_mask (uint32_t *ui32, int *mask);
void unsigned_greater_equal_i64_and_mask (uint64_t *ui64, int *mask);

void signed_greater_equal_i8_and_mask  (int8_t  *i8 , int *mask);
void signed_greater_equal_i16_and_mask (int16_t *i16, int *mask);
void signed_greater_equal_i32_and_mask (int32_t *i32, int *mask);
void signed_greater_equal_i64_and_mask (int64_t *i64, int *mask);

void greater_equal_double_and_mask(double *d, int *mask);
void greater_equal_float_and_mask (float  *f, int *mask);

void unsigned_less_than_i8_and_mask  (uint8_t  *ui8 , int *mask);
void unsigned_less_than_i16_and_mask (uint16_t *ui16, int *mask);
void unsigned_less_than_i32_and_mask (uint32_t *ui32, int *mask);
void unsigned_less_than_i64_and_mask (uint64_t *ui64, int *mask);

void signed_less_than_i8_and_mask  (int8_t  *i8 , int *mask);
void signed_less_than_i16_and_mask (int16_t *i16, int *mask);
void signed_less_than_i32_and_mask (int32_t *i32, int *mask);
void signed_less_than_i64_and_mask (int64_t *i64, int *mask);

void less_than_double_and_mask(double *d, int *mask);
void less_than_float_and_mask (float  *f, int *mask);

void unsigned_greater_than_i8_and_mask  (uint8_t  *ui8 , int *mask);
void unsigned_greater_than_i16_and_mask (uint16_t *ui16, int *mask);
void unsigned_greater_than_i32_and_mask (uint32_t *ui32, int *mask);
void unsigned_greater_than_i64_and_mask (uint64_t *ui64, int *mask);

void signed_greater_than_i8_and_mask  (int8_t  *i8 , int *mask);
void signed_greater_than_i16_and_mask (int16_t *i16, int *mask);
void signed_greater_than_i32_and_mask (int32_t *i32, int *mask);
void signed_greater_than_i64_and_mask (int64_t *i64, int *mask);

void greater_than_double_and_mask(double *d, int *mask);
void greater_than_float_and_mask (float  *f, int *mask);


void cast_bits_scalar_ui32_f(float    *f_u32);
void cast_bits_scalar_i32_f (float    *f_32);
void cast_bits_scalar_f_ui32(uint32_t *ui32);
void cast_bits_scalar_f_i32 (int32_t  *i32);
void cast_bits_scalar_ui64_d(double   *d_u64);
void cast_bits_scalar_i64_d (double   *d_64);
void cast_bits_scalar_d_ui64(uint64_t *ui64);
void cast_bits_scalar_d_i64 (int64_t  *i64);


void cast_bits_f_i32(int32_t *i32);
void cast_bits_i32_f(float   *f_32);
void cast_bits_d_i64(int64_t *i64);
void cast_bits_i64_d(double  *d_64);


void reduce_add_double(double  *d  );
void reduce_add_float (float   *f  );
void reduce_add_int8  (int8_t  *i8 );
void reduce_add_int16 (int16_t *i16);
void reduce_add_int32 (int32_t *i32);
void reduce_add_int64 (int64_t *i64);


void reduce_min_double (double  *d  );
void reduce_min_float  (float   *f  );
void reduce_min_int32  (int32_t *i32);
void reduce_min_uint32 (uint32_t *ui32);
void reduce_min_int64  (int64_t *i64);
void reduce_min_uint64 (uint64_t *ui64);

void reduce_max_double (double  *d  );
void reduce_max_float  (float   *f  );
void reduce_max_int32  (int32_t *i32);
void reduce_max_uint32 (uint32_t *ui32);
void reduce_max_int64  (int64_t *i64);
void reduce_max_uint64 (uint64_t *ui64);

void popcnt_int32(uint32_t *ui32);
void popcnt_int64(uint64_t *ui64);

void count_trailing_zeros_i32(uint32_t *ui32);
void count_trailing_zeros_i64(uint64_t *ui64);

void count_leading_zeros_i32(uint32_t *ui32);
void count_leading_zeros_i64(uint64_t *ui64);
int main () {
    printf ("Start\n");
// Prepare input data
    int      mask [16];
    double   d_8  [16];
    double   d_16 [16];
    double   d_32 [16];
    double   d_64 [16];
    float    f_8  [16];
    float    f_16 [16];
    float    f_32 [16];
    float    f_64 [16];
    double   d_u8 [16];
    double   d_u16[16];
    double   d_u32[16];
    double   d_u64[16];
    float    f_u8 [16];
    float    f_u16[16];
    float    f_u32[16];
    float    f_u64[16];
    bool     i1   [16];
    int8_t   i8   [16];
    int16_t  i16  [16];
    int32_t  i32  [16];
    int64_t  i64  [16];
    uint8_t  ui8  [16];
    uint16_t ui16 [16];
    uint32_t ui32 [16];
    uint64_t ui64 [16]; 
    
    mask[0] = 1;
    mask[1] = 1;

    d_u8[0] = UINT8_MAX;
    d_u8[1] = 0;

    d_u16[0] = UINT16_MAX;
    d_u16[1] = 0;

    d_u32[0] = UINT32_MAX;
    d_u32[1] = 0;

    d_u64[0] = UINT64_MAX;
    d_u64[1] = 0;

    f_u8[0] = UINT8_MAX;
    f_u8[1] = 0;

    f_u16[0] = UINT16_MAX;
    f_u16[1] = 0;

    f_u32[0] = UINT32_MAX;
    f_u32[1] = 0;

    f_u64[0] = UINT64_MAX;
    f_u64[1] = 0;

    d_8[0] = INT8_MAX;
    d_8[1] = INT8_MIN;

    d_16[0] = INT16_MAX;
    d_16[1] = INT16_MIN;

    d_32[0] = INT32_MAX;
    d_32[1] = INT32_MIN;

    d_64[0] = INT64_MAX;
    d_64[1] = INT64_MIN;

    f_8[0] = INT8_MAX;
    f_8[1] = INT8_MIN;

    f_16[0] = INT16_MAX;
    f_16[1] = INT16_MIN;

    f_32[0] = INT32_MAX;
    f_32[1] = INT32_MIN;

    f_64[0] = INT64_MAX;
    f_64[1] = INT64_MIN;

    i1[0] = true;
    i1[1] = false;

    i8[0] = INT8_MAX;
    i8[1] = INT8_MIN;

    i16[0] = INT16_MAX;
    i16[1] = INT16_MIN;

    i32[0] = INT32_MAX;
    i32[1] = INT32_MIN;

    i64[0] = INT64_MAX;
    i64[1] = INT64_MIN;

    ui8[0] = UINT8_MAX;
    ui8[1] = 0;

    ui16[0] = UINT16_MAX;
    ui16[1] = 0;

    ui32[0] = UINT32_MAX;
    ui32[1] = 0;

    ui64[0] = INT64_MAX;
    ui64[1] = 0;

    for (int i = 2; i < 16; i++) {
        d_u8 [i] = (i + 1) * 8;
        f_u8 [i] = (i + 1) * 8;
        d_u16[i] = (i + 1) * 16;
        f_u16[i] = (i + 1) * 16;
        d_u32[i] = (i + 1) * 32;
        f_u32[i] = (i + 1) * 32;
        d_u64[i] = (i + 1) * 64;
        f_u64[i] = (i + 1) * 64;
        ui8  [i] = (i + 1) * 8;
        ui16 [i] = (i + 1) * 16;
        ui32 [i] = (i + 1) * 32;
        ui64 [i] = (i + 1) * 64;    

        if (i % 3 == 0) {
            d_8  [i] = (i + 1) * 8;
            f_8  [i] = (i + 1) * 8;
            d_16 [i] = (i + 1) * 16;
            f_16 [i] = (i + 1) * 16;
            d_32 [i] = (i + 1) * 32;
            f_32 [i] = (i + 1) * 32;
            d_64 [i] = (i + 1) * 64;
            f_64 [i] = (i + 1) * 64;
            i8   [i] = (i + 1) * 8;
            i16  [i] = (i + 1) * 16;
            i32  [i] = (i + 1) * 32;
            i64  [i] = (i + 1) * 64;
            i1  [i] = false;
        }
        else {
            d_8  [i] = -1 * (i + 1) * 8;
            f_8  [i] = -1 * (i + 1) * 8;
            d_16 [i] = -1 * (i + 1) * 16;
            f_16 [i] = -1 * (i + 1) * 16;
            d_32 [i] = -1 * (i + 1) * 32;
            f_32 [i] = -1 * (i + 1) * 32;
            d_64 [i] = -1 * (i + 1) * 64;
            f_64 [i] = -1 * (i + 1) * 64;
            i8   [i] = -1 * (i + 1) * 8;
            i16  [i] = -1 * (i + 1) * 16;
            i32  [i] = -1 * (i + 1) * 32;
            i64  [i] = -1 * (i + 1) * 64;
            i1  [i] = true;
        }
    
        if (i % 2 == 0) 
            mask[i] = 0;
        else 
            mask[i] = 1;
    }

// variables, which can't cause overflow
        double   no_of_d_8  [16];
        double   no_of_d_16 [16];
        double   no_of_d_32 [16];
        double   no_of_d_64 [16];
        float    no_of_f_8  [16];
        float    no_of_f_16 [16];
        float    no_of_f_32 [16];
        float    no_of_f_64 [16];
        double   no_of_d_u8 [16];
        double   no_of_d_u16[16];
        double   no_of_d_u32[16];
        double   no_of_d_u64[16];
        float    no_of_f_u8 [16];
        float    no_of_f_u16[16];
        float    no_of_f_u32[16];
        float    no_of_f_u64[16];
        bool     no_of_i1   [16];
        int8_t   no_of_i8   [16];
        int16_t  no_of_i16  [16];
        int32_t  no_of_i32  [16];
        int64_t  no_of_i64  [16];
        uint8_t  no_of_ui8  [16];
        uint16_t no_of_ui16 [16];
        uint32_t no_of_ui32 [16];
        uint64_t no_of_ui64 [16];

        for (int i = 0; i < 16; i++) {
            no_of_d_8  [i] = i + 1;
            no_of_d_16 [i] = i + 1;
            no_of_d_32 [i] = i + 1;
            no_of_d_64 [i] = i + 1;
            no_of_f_8  [i] = i + 1;
            no_of_f_16 [i] = i + 1;
            no_of_f_32 [i] = i + 1;
            no_of_f_64 [i] = i + 1;
            no_of_d_u8 [i] = i + 1;
            no_of_d_u16[i] = i + 1;
            no_of_d_u32[i] = i + 1;
            no_of_d_u64[i] = i + 1;
            no_of_f_u8 [i] = i + 1;
            no_of_f_u16[i] = i + 1;
            no_of_f_u32[i] = i + 1;
            no_of_f_u64[i] = i + 1;
            no_of_i1   [i] = i + 1;
            no_of_i8   [i] = i + 1;
            no_of_i16  [i] = i + 1;
            no_of_i32  [i] = i + 1;
            no_of_i64  [i] = i + 1;
            no_of_ui8  [i] = i + 1;
            no_of_ui16 [i] = i + 1;
            no_of_ui32 [i] = i + 1;
            no_of_ui64 [i] = i + 1;
        }
        
#ifdef KNC_H
    printf ("Include knc.h\n");
#else
    printf ("Include knc-i1x16.h\n");
#endif
/*
    for(int i = 0; i < 16; i++) {
        printf("\n%d-----------------------------\n", i);
        printf("int     :%d\n", mask [i]);
        printf("double  :%f\n", d_8  [i]);
        printf("double  :%f\n", d_16 [i]);
        printf("double  :%f\n", d_32 [i]);
        printf("double  :%f\n", d_64 [i]);
        printf("float   :%f\n", f_8  [i]);
        printf("float   :%f\n", f_16 [i]);
        printf("float   :%f\n", f_32 [i]);
        printf("float   :%f\n", f_64 [i]);
        printf("double  :%f\n", d_u8 [i]);
        printf("double  :%f\n", d_u16[i]);
        printf("double  :%f\n", d_u32[i]);
        printf("double  :%f\n", d_u64[i]);
        printf("float   :%f\n", f_u8 [i]);
        printf("float   :%f\n", f_u16[i]);
        printf("float   :%f\n", f_u32[i]);
        printf("float   :%f\n", f_u64[i]);
        printf("bool    :%d\n", i1   [i]);
        printf("int8_t  :%d\n", i8   [i]);
        printf("int16_t :%d\n", i16  [i]);
        printf("int32_t :%d\n", i32  [i]);
        printf("int64_t :%d\n", i64  [i]);
        printf("uint8_t :%d\n", ui8  [i]);
        printf("uint16_t:%d\n", ui16 [i]);
        printf("uint32_t:%d\n", ui32 [i]);
        printf("uint64_t:%d\n", ui64 [i]);
    }
*/
    printf ("\n");
    
    /*
    gather32_double(d_32, mask);
    gather32_float(f_32, mask);   
    gather32_i8(i8, mask);
    gather32_i16(i16, mask);
    gather32_i32(i32, mask);
    gather32_i64(i64, mask);
     
    gather64_double(d_64, mask);
    gather64_float(f_64, mask);
    gather64_i8(i8, mask);
    gather64_i16(i16, mask);
    gather64_i32(i32, mask);
    gather64_i64(i64, mask);

    
    gather_base_offsets32_double(d_32, mask);
    gather_base_offsets32_float(f_32, mask);
    gather_base_offsets32_i8(i8, mask);
    gather_base_offsets32_i16(i16, mask); // modify define with type conversion(int32_t)
    gather_base_offsets32_i32(i32, mask);
    gather_base_offsets32_i64(i64, mask);

    gather_base_offsets64_double(d_64, mask);
    gather_base_offsets64_float(f_64, mask);
    gather_base_offsets64_i8(i8, mask);
    gather_base_offsets64_i16(i16, mask);
    gather_base_offsets64_i32(i32, mask);
    gather_base_offsets64_i64(i64, mask);
   
    
    scatter32_double(d_32, mask);
    scatter32_float(f_32, mask);
    scatter32_i8(i8, mask);
    scatter32_i16(i16, mask);
    scatter32_i32(i32, mask);
    scatter32_i64(i64, mask);

    scatter64_double(d_64, mask);
    scatter64_float(f_64, mask);
    scatter64_i8(i8, mask);
    scatter64_i16(i16, mask);
    scatter64_i32(i32, mask);
    scatter64_i64(i64, mask);
    

    scatter_base_offsets32_double(d_32, mask);
    scatter_base_offsets32_float(f_32, mask);
    scatter_base_offsets32_i8(i8, mask); // modify define with type conversion(int32_t)
    scatter_base_offsets32_i16(i16, mask); // modify define with type conversion(int32_t)
    scatter_base_offsets32_i32(i32, mask);
    scatter_base_offsets32_i64(i64, mask); // modify define with type conversion(int32_t)

    scatter_base_offsets64_double(d_64, mask);
    scatter_base_offsets64_float(f_64, mask);
    scatter_base_offsets64_i8(i8, mask);
    scatter_base_offsets64_i16(i16, mask);
    scatter_base_offsets64_i32(i32, mask);
    scatter_base_offsets64_i64(i64, mask);
        
    
    masked_load_double(d_32, mask);
    masked_load_float(f_32, mask);
    masked_load_i8(i8, mask);
    masked_load_i16(i16, mask);
    masked_load_i32(i32, mask);
    masked_load_i64(i64, mask);


    masked_store_double(d_32, mask);
    masked_store_float(f_32, mask);
    masked_store_i8(i8, mask);
    masked_store_i16(i16, mask);
    masked_store_i32(i32, mask);
    masked_store_i64(i64, mask);

    masked_store_blend_double(d_32, mask);
    masked_store_blend_float(f_32, mask);
    masked_store_blend_i8(i8, mask);
    masked_store_blend_i16(i16, mask);
    masked_store_blend_i32(i32, mask);
    masked_store_blend_i64(i64, mask);

    
    movmsk(mask);

    
    insert_extract_element_double(d_32);
    insert_extract_element_float(f_32);
    insert_extract_element_i8(i8);
    insert_extract_element_i16(i16);
    insert_extract_element_i32(i32);
    insert_extract_element_i64(i64);


    load_double(d_32);
    load_float(f_32);
    load_i8(i8);
    load_i16(i16);
    load_i32(i32);
    load_i64(i64);
    

    store_double(d_32);
    store_float(f_32);
    store_i8(i8);
    store_i16(i16);
    store_i32(i32);
    store_i64(i64);


    smear_double(d_32);
    smear_float(f_32);
    smear_i8(i8);
    smear_i16(i16);
    smear_i32(i32);
    smear_i64(i64);

    
    setzero_double();
    setzero_float();
    setzero_i8();
    setzero_i16();
    setzero_i32();
    setzero_i64();
    

    select_double(d_32, mask);
    select_float(f_32, mask);
    select_i8(i8, mask);
    select_i16(i16, mask);
    select_i32(i32, mask);
    select_i64(i64, mask);    
    
    select_double_cond(d_32, mask);
    select_float_cond(f_32, mask);
    select_i8_cond(i8, mask);
    select_i16_cond(i16, mask);
    select_i32_cond(i32, mask);
    select_i64_cond(i64, mask);
    

    broadcast_double(d_32);
    broadcast_float(f_32);
    broadcast_i8(i8);
    broadcast_i16(i16);
    broadcast_i32(i32);
    broadcast_i64(i64);
    

    rotate_double(d_32);
    rotate_float(f_32);
    rotate_i8(i8);
    rotate_i16(i16);
    rotate_i32(i32);
    rotate_i64(i64);


    shift_double(d_32);
    shift_float(f_32);
    shift_i8(i8);
    shift_i16(i16);
    shift_i32(i32);
    shift_i64(i64);
    

    shuffle_double(d_32);
    shuffle_float(f_32);
    shuffle_i8(i8);
    shuffle_i16(i16);
    shuffle_i32(i32);
    

    cast_i64_i32(i32);
    cast_i64_i16(i16);
    cast_i64_i8 (i8);
    cast_i64_i1 (i1);
    cast_i32_i16(i16);
    cast_i32_i8 (i8);
    cast_i32_i1 (i1);
    cast_i16_i8 (i8);
    cast_i16_i1 (i1);
    cast_i8_i1  (i1);

    cast_ui64_ui32(ui32);
    cast_ui64_ui16(ui16);
    cast_ui64_ui8 (ui8);
    cast_ui64_ui1 (i1);
    cast_ui32_ui16(ui16);
    cast_ui32_ui8 (ui8);
    cast_ui32_ui1 (i1);
    cast_ui16_ui8 (ui8);
    cast_ui16_ui1 (i1);
    cast_ui8_ui1  (i1);

    trunk_i32_i64(i64);
    trunk_i16_i64(i64);
    trunk_i8_i64 (i64);
    trunk_i16_i32(i32);
    trunk_i8_i32 (i32);
    trunk_i8_i16 (i16);
  
    cast_f_i8 (i8);
    cast_f_i16(i16);
    cast_f_i32(i32);
    cast_f_i64(i64);
    cast_d_i8 (i8);
    cast_d_i16(i16);
    cast_d_i32(i32);
    cast_d_i64(i64);

    cast_f_ui8 (ui8);
    cast_f_ui16(ui16);
    cast_f_ui32(ui32);
    cast_f_ui64(ui64);
    cast_d_ui8 (ui8);
    cast_d_ui16(ui16);
    cast_d_ui32(ui32);
    cast_d_ui64(ui64);

    cast_i8_f (f_8);
    cast_i16_f(f_16);
    cast_i32_f(f_32);
    cast_i64_f(f_64);
    cast_i8_d (d_8);
    cast_i16_d(d_16);
    cast_i32_d(d_32);
    cast_i64_d(d_64);
    
    cast_ui8_f (f_u8);
    cast_ui16_f(f_u16);
    cast_ui32_f(f_u32);
    cast_ui64_f(f_u64);
    cast_ui8_d (d_u8);
    cast_ui16_d(d_u16);
    cast_ui32_d(d_u32);
    cast_ui64_d(d_u64);
    

    cast_f_d(d_8);
    cast_f_d(d_16);
    cast_f_d(d_32);
    cast_f_d(d_64);
    cast_d_f(f_8);
    cast_d_f(f_16);
    cast_d_f(f_32);
    cast_d_f(f_64);
    */  

    add_double(no_of_d_32, no_of_d_32);
    add_float (no_of_f_32, no_of_f_32);
    add_i8    (no_of_i8  , no_of_i8);
    add_i16   (no_of_i16 , no_of_i16);
    add_i32   (no_of_i32 , no_of_i32);
    add_i64   (no_of_i64 , no_of_i64);

    sub_double(no_of_d_32, no_of_d_32);
    sub_float (no_of_f_32, no_of_f_32);
    sub_i8    (no_of_i8  , no_of_i8);
    sub_i16   (no_of_i16 , no_of_i16);
    sub_i32   (no_of_i32 , no_of_i32);
    sub_i64   (no_of_i64 , no_of_i64);

    mul_double(no_of_d_32, no_of_d_32);
    mul_float (no_of_f_32, no_of_f_32);
    mul_i8    (no_of_i8  , no_of_i8);
    mul_i16   (no_of_i16 , no_of_i16);
    mul_i32   (no_of_i32 , no_of_i32);
    mul_i64   (no_of_i64 , no_of_i64);

    or_i8 (no_of_i8  , no_of_i8);
    or_i16(no_of_i16 , no_of_i16);
    or_i32(no_of_i32 , no_of_i32);
    or_i64(no_of_i64 , no_of_i64);    
    
    and_i8 (no_of_i8  , no_of_i8);
    and_i16(no_of_i16 , no_of_i16);
    and_i32(no_of_i32 , no_of_i32);
    and_i64(no_of_i64 , no_of_i64); 

    xor_i8 (no_of_i8  , no_of_i8);
    xor_i16(no_of_i16 , no_of_i16);
    xor_i32(no_of_i32 , no_of_i32);
    xor_i64(no_of_i64 , no_of_i64);
   
    shl_i8 (no_of_i8  , no_of_i8);
    shl_i16(no_of_i16 , no_of_i16);
    shl_i32(no_of_i32 , no_of_i32);
    shl_i64(no_of_i64 , no_of_i64);
    
    udiv_ui8 (no_of_ui8  , no_of_ui8);
    udiv_ui16(no_of_ui16 , no_of_ui16);
    udiv_ui32(no_of_ui32 , no_of_ui32);
    udiv_ui64(no_of_ui64 , no_of_ui64);

    sdiv_i8 (no_of_i8  , no_of_i8);
    sdiv_i16(no_of_i16 , no_of_i16);
    sdiv_i32(no_of_i32 , no_of_i32);
    sdiv_i64(no_of_i64 , no_of_i64);
    
    div_double(no_of_d_32 , no_of_d_32);
    div_float (no_of_f_32 , no_of_f_32);

    urem_ui8 (no_of_ui8  , no_of_ui8);
    urem_ui16(no_of_ui16 , no_of_ui16);
    urem_ui32(no_of_ui32 , no_of_ui32);
    urem_ui64(no_of_ui64 , no_of_ui64);

    srem_i8 (no_of_i8  , no_of_i8);
    srem_i16(no_of_i16 , no_of_i16);
    srem_i32(no_of_i32 , no_of_i32);
    srem_i64(no_of_i64 , no_of_i64);

    lshr_ui8 (no_of_ui8  , no_of_ui8);
    lshr_ui16(no_of_ui16 , no_of_ui16);
    lshr_ui32(no_of_ui32 , no_of_ui32);
    lshr_ui64(no_of_ui64 , no_of_ui64);

    ashr_i8 (no_of_i8  , no_of_i8);
    ashr_i16(no_of_i16 , no_of_i16);
    ashr_i32(no_of_i32 , no_of_i32);
    ashr_i64(no_of_i64 , no_of_i64);

    lshr_ui8_uniform (no_of_ui8 , no_of_i32);
    lshr_ui16_uniform(no_of_ui16, no_of_i32);
    lshr_ui32_uniform(no_of_ui32, no_of_i32);
    lshr_ui64_uniform(no_of_ui64, no_of_i32);

    ashr_i8_uniform (no_of_i8 , no_of_i32);
    ashr_i16_uniform(no_of_i16, no_of_i32);
    ashr_i32_uniform(no_of_i32, no_of_i32);
    ashr_i64_uniform(no_of_i64, no_of_i32);
    /*
    equal_double(d_32);
    equal_float (f_32);
    equal_i8    (i8);
    equal_i16   (i16);
    equal_i32   (i32);
    equal_i64   (i64);

    not_equal_double(d_32);
    not_equal_float (f_32);
    not_equal_i8    (i8);
    not_equal_i16   (i16);
    not_equal_i32   (i32);
    not_equal_i64   (i64);

    unsigned_less_equal_i8  (ui8);
    unsigned_less_equal_i16 (ui16);
    unsigned_less_equal_i32 (ui32);
    unsigned_less_equal_i64 (ui64);

    signed_less_equal_i8  (i8);
    signed_less_equal_i16 (i16);
    signed_less_equal_i32 (i32);
    signed_less_equal_i64 (i64);

    less_equal_double(d_32);
    less_equal_float (f_32);

    unsigned_greater_equal_i8  (ui8);
    unsigned_greater_equal_i16 (ui16);
    unsigned_greater_equal_i32 (ui32);
    unsigned_greater_equal_i64 (ui64);

    signed_greater_equal_i8  (i8);
    signed_greater_equal_i16 (i16);
    signed_greater_equal_i32 (i32);
    signed_greater_equal_i64 (i64);

    greater_equal_double(d_32);
    greater_equal_float (f_32);

    unsigned_less_than_i8  (ui8);
    unsigned_less_than_i16 (ui16);
    unsigned_less_than_i32 (ui32);
    unsigned_less_than_i64 (ui64);

    signed_less_than_i8  (i8);
    signed_less_than_i16 (i16);
    signed_less_than_i32 (i32);
    signed_less_than_i64 (i64);

    less_than_double(d_32);
    less_than_float (f_32);

    unsigned_greater_than_i8  (ui8);
    unsigned_greater_than_i16 (ui16);
    unsigned_greater_than_i32 (ui32);
    unsigned_greater_than_i64 (ui64);

    signed_greater_than_i8  (i8);
    signed_greater_than_i16 (i16);
    signed_greater_than_i32 (i32);
    signed_greater_than_i64 (i64);

    greater_than_double(d_32);
    greater_than_float (f_32);
    
    equal_double_and_mask(d_32, mask);
    equal_float_and_mask (f_32, mask);
    equal_i8_and_mask    (i8, mask);
    equal_i16_and_mask   (i16, mask);
    equal_i32_and_mask   (i32, mask);
    equal_i64_and_mask   (i64, mask);

    not_equal_double_and_mask(d_32, mask);
    not_equal_float_and_mask (f_32, mask);
    not_equal_i8_and_mask    (i8, mask);
    not_equal_i16_and_mask   (i16, mask);
    not_equal_i32_and_mask   (i32, mask);
    not_equal_i64_and_mask   (i64, mask);

    unsigned_less_equal_i8_and_mask  (ui8, mask);
    unsigned_less_equal_i16_and_mask (ui16, mask);
    unsigned_less_equal_i32_and_mask (ui32, mask);
    unsigned_less_equal_i64_and_mask (ui64, mask);

    signed_less_equal_i8_and_mask  (i8, mask);
    signed_less_equal_i16_and_mask (i16, mask);
    signed_less_equal_i32_and_mask (i32, mask);
    signed_less_equal_i64_and_mask (i64, mask);

    less_equal_double_and_mask(d_32, mask);
    less_equal_float_and_mask (f_32, mask);

    unsigned_greater_equal_i8_and_mask  (ui8, mask);
    unsigned_greater_equal_i16_and_mask (ui16, mask);
    unsigned_greater_equal_i32_and_mask (ui32, mask);
    unsigned_greater_equal_i64_and_mask (ui64, mask);

    signed_greater_equal_i8_and_mask  (i8, mask);
    signed_greater_equal_i16_and_mask (i16, mask);
    signed_greater_equal_i32_and_mask (i32, mask);
    signed_greater_equal_i64_and_mask (i64, mask);

    greater_equal_double_and_mask(d_32, mask);
    greater_equal_float_and_mask (f_32, mask);

    unsigned_less_than_i8_and_mask  (ui8, mask);
    unsigned_less_than_i16_and_mask (ui16, mask);
    unsigned_less_than_i32_and_mask (ui32, mask);
    unsigned_less_than_i64_and_mask (ui64, mask);

    signed_less_than_i8_and_mask  (i8, mask);
    signed_less_than_i16_and_mask (i16, mask);
    signed_less_than_i32_and_mask (i32, mask);
    signed_less_than_i64_and_mask (i64, mask);

    less_than_double_and_mask(d_32, mask);
    less_than_float_and_mask (f_32, mask);

    unsigned_greater_than_i8_and_mask  (ui8, mask);
    unsigned_greater_than_i16_and_mask (ui16, mask);
    unsigned_greater_than_i32_and_mask (ui32, mask);
    unsigned_greater_than_i64_and_mask (ui64, mask);

    signed_greater_than_i8_and_mask  (i8, mask);
    signed_greater_than_i16_and_mask (i16, mask);
    signed_greater_than_i32_and_mask (i32, mask);
    signed_greater_than_i64_and_mask (i64, mask);

    greater_than_double_and_mask(d_32, mask);
    greater_than_float_and_mask (f_32, mask);
    

    cast_bits_scalar_ui32_f(f_u32);
    cast_bits_scalar_i32_f (f_32);
    cast_bits_scalar_f_ui32(ui32);
    cast_bits_scalar_f_i32 (i32);
    cast_bits_scalar_ui64_d(d_u64);
    cast_bits_scalar_i64_d (d_64);
    cast_bits_scalar_d_ui64(ui64);
    cast_bits_scalar_d_i64 (i64);
    

    cast_bits_f_i32(i32);
    cast_bits_i32_f(f_32);
    cast_bits_d_i64(i64);
    cast_bits_i64_d(d_64);
    

    reduce_add_double(no_of_d_32);
    reduce_add_float (no_of_f_32);
    reduce_add_int8  (no_of_i8);
    reduce_add_int16 (no_of_i16);
    reduce_add_int32 (no_of_i32);
    reduce_add_int64 (no_of_i64);
     
  
    reduce_min_double (d_32);
    reduce_min_float  (f_32);
    reduce_min_int32  (i32);
    reduce_min_uint32 (ui32);
    reduce_min_int64  (i64);
    reduce_min_uint64 (ui64);
    reduce_max_double (d_32);
    reduce_max_float  (f_32);
    reduce_max_int32  (i32);
    reduce_max_uint32 (ui32);
    reduce_max_int64  (i64);
    reduce_max_uint64 (ui64);
    

    popcnt_int32(ui32);
    popcnt_int64(ui64);
    

    count_trailing_zeros_i32(ui32);
    count_trailing_zeros_i64(ui64);

    count_leading_zeros_i32(ui32);
    count_leading_zeros_i64(ui64);
    */
return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Usefull stuff
/*
    uint32_t a_ = 65535;

  __vec16_i32 ptrs = __smear_i32<__vec16_i32>((int32_t)&a_);
  __vec16_i1 mask = __smear_i1<__vec16_i1>((int32_t)a_);
  __vec16_d res;
  res =  __gather32_double(ptrs, mask);

  __vec16_i8 smear = __smear_i8<__vec16_i8>((int8_t) -128);
  for (int i = 0; i < 16; ++i){
    printf("%d\n", smear[i]);
  }

  for (int i = 0; i < 16; ++i){
    printf("%d\n", ((uint*)&res)[i]);
  }

  __vec16_i64 smear_64 = __smear_i64<__vec16_i64>((int64_t) -65535);
  for (int i = 0; i < 16; ++i){
    printf("%d\n", smear_64[i]);
  }
    */
