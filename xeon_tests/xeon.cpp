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


int main () {
    printf ("Start\n");
// Prepare input data
    int mask [16];
    double  d_32[16];
    float   f_32[16];
    double  d_64[16];
    float   f_64[16];
    int8_t i8 [16];
    int16_t i16 [16];
    int32_t i32 [16];
    int64_t i64 [16];
    
    
    mask[0] = 1;
    mask[1] = 1;

    d_32[0] = INT32_MAX;
    d_32[1] = INT32_MIN;

    f_32[0] = INT32_MAX;
    f_32[1] = INT32_MIN;

    d_64[0] = INT64_MAX;
    d_64[1] = INT64_MIN;

    f_64[0] = INT64_MAX;
    f_64[1] = INT64_MIN;

    i8[0] = INT8_MAX;
    i8[1] = INT8_MIN;

    i16[0] = INT16_MAX;
    i16[1] = INT16_MIN;

    i32[0] = INT32_MAX;
    i32[1] = INT32_MIN;

    i64[0] = INT64_MAX;
    i64[1] = INT64_MIN;

    for (int i = 2; i < 16; i++) {
        d_32[i] = (i + 1) * 2;
        f_32[i] = (i + 1) * 2;
        d_64[i] = (i + 1) * 2;
        f_64[i] = (i + 1) * 2;
        i8[i] = (i + 1) * 2;
        i16[i] = (i + 1) * 2;
        i32[i] = (i + 1) * 2;
        i64[i] = (i + 1) * 2;
        if (i % 2 == 0)
            mask[i] = 0;
        else
            mask[i] = 1;
    }

#ifdef KNC_H
    printf ("Include knc.h\n");
#else
    printf ("Include knc-i1x16.h\n");
#endif

    printf ("\n");
    
    
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

return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Usefull staff
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
