//#include <stdio.h>
#define __STDC_LIMIT_MACROS // enable intN_t limits from stdint.h
#include <stdint.h>

#ifdef KNC_H
    #include "knc.h"
#else
    #include "knc-i1x16.h"
    #include <stdio.h>
#endif

template <typename T>
void allocator(T **array) {
    /* Cause seagfault    
    uint64_t seed = 123456789;
    int m = UINT32_MAX;
    int a = 1103515245;
    int c = c = 12345;
    T* tmp[4];
    seed = (a * seed + c) % m;    
    T* tmp1 = (T*) malloc(seed);

    for (int j = 0; j < 4; j++) {
        for (int i = 4 * j; i < 4 * (j + 1); i++) {
            array[i] = (T*) malloc(sizeof(T*));
            printf ("Array: %d\n", array[i]);
        }
        seed = (a * seed + c) % m;
        printf ("Seed %d\n", seed);
        tmp[j] = (T*) malloc(seed);
    }

    for (int j = 0; j < 4; j++)
        free(tmp[j]);
    */

    uint64_t seed = 123456789;
    int m = 100;
    int a = 1103515245;
    int c = 12345;
    T* tmp[4];
    seed = (a * seed + c) % m;
    T* tmp1 = (T*) malloc(seed);

    for (int j = 0; j < 4; j++) {
        for (int i = 4 * j; i < 4 * (j + 1); i++) {
            array[i] = (T*) malloc(sizeof(T*));
            //printf ("Array: %x\n", array[i]);
        }
        seed = (a * seed + c) % m;
        //printf ("Seed %d\n", seed);
        tmp[j] = (T*) malloc(seed * sizeof(T));
        //printf ("Tmp: %x\n", tmp[j]);
    }

    for (int j = 0; j < 4; j++)
        free(tmp[j]);
}

//Example: GATHER_VEC_TYPE=__vec16_i32, TYPE=double, VEC_TYPE=__vec16_d, TYPE_NAME-double, FUNC=__gather32_double
// TYPE=int16_t, VEC_TYPE=__vec16_i16, FUNC_NAME=gather32_i16, FUNC_CALL=__gather32_i16
#define GATHER(GATHER_VEC_TYPE, TYPE, VEC_TYPE, FUNC_NAME, FUNC_CALL)               \
void FUNC_NAME(TYPE *data, int *m) {                                                \
    printf (#FUNC_NAME, ":");                                                       \
                                                                                    \
    TYPE copy_data[16];                                                             \
    int copy_m[16];                                                                 \
    for (int i = 0; i < 16; i++) {                                                  \
        copy_data[i] = data[i];                                                     \
        copy_m[i] = m[i];                                                           \
    }                                                                               \
                                                                                    \
    GATHER_VEC_TYPE ptrs;                                                           \
    TYPE *b[16];                                                                    \
    allocator(b);                                                                   \
    for (int i = 0; i < 16; i++) {                                                  \
        *b[i] = (TYPE) copy_data[i];                                                \
        ptrs[i] = (int32_t) b[i];                                                   \
    }                                                                               \
                                                                                    \
    __vec16_i1 mask = __vec16_i1(copy_m[0],  copy_m[1],  copy_m[2],  copy_m[3],     \
                                 copy_m[4],  copy_m[5],  copy_m[6],  copy_m[7],     \
                                 copy_m[8],  copy_m[9],  copy_m[10], copy_m[11],    \
                                 copy_m[12], copy_m[13], copy_m[14], copy_m[15]);   \
                                                                                    \
    VEC_TYPE output;                                                                \
    for (int i = 0; i < 16; i ++)                                                   \
        output[i] = 0;                                                              \
    output = FUNC_CALL(ptrs, mask);                                                 \
                                                                                    \
    VEC_TYPE result;                                                                \
    for (int i = 0; i < 16; i ++)                                                   \
        if (m[i] == 0)                                                              \
            result[i] = 0;                                                          \
        else                                                                        \
            result[i] =  data[i];                                                   \
                                                                                    \
    int err_counter = 0;                                                            \
    for (int i = 0; i < 16; ++i){                                                   \
        if (m[i] != 0 && output[i] != result[i])                                    \
            err_counter++;                                                          \
        if (copy_data[i] != data[i])                                                \
            err_counter++;                                                          \
    }                                                                               \
    if (err_counter != 0)                                                           \
        printf(" errors %d\n", err_counter);                                        \
    else                                                                            \
         printf(" no fails\n");                                                     \
                                                                                    \
    for (int i = 0; i < 16; i++)                                                    \
        free(b[i]);                                                                 \
}

GATHER(__vec16_i32, double , __vec16_d  , gather32_double, __gather32_double)
GATHER(__vec16_i32, float  , __vec16_f  , gather32_float , __gather32_float)
GATHER(__vec16_i32, int16_t, __vec16_i16, gather32_i16   , __gather32_i16)
GATHER(__vec16_i32, int32_t, __vec16_i32, gather32_i32   , __gather32_i32)
GATHER(__vec16_i32, int64_t, __vec16_i64, gather32_i64   , __gather32_i64)

GATHER(__vec16_i64, double , __vec16_d  , gather64_double, __gather64_double)
GATHER(__vec16_i64, float  , __vec16_f  , gather64_float , __gather64_float)
GATHER(__vec16_i64, int16_t, __vec16_i16, gather64_i16   , __gather64_i16)
GATHER(__vec16_i64, int32_t, __vec16_i32, gather64_i32   , __gather64_i32)
GATHER(__vec16_i64, int64_t, __vec16_i64, gather64_i64   , __gather64_i64)
////////////////////////////////////////////////////////////////////////////////////
/*
void gather_base_offsets32_i8(uint8_t base, uint32_t scale, int *m) {
    printf ("gather_base_offsets32_i8");
    
    uint8_t copy_base = base;
    uint32_t copy_scale = scale;
    int copy_m[16];                                                                 
    for (int i = 0; i < 16; i++) {                                                  
        copy_m[i] = m[i];                                                           
    }                                                                               

    __vec16_i32 _offsets;                                                           
    int32_t *b[16];                                                                 
    for (int i = 0; i < 16; i++) {                                                  
        b[i] = (int32_t*) malloc(sizeof(int32_t));                                  
        _offsets[i] = (int32_t) ((b[i] - (int32_t*)&base) / scale);
    }                                                                               
                                                                                    
    __vec16_i1 mask = __vec16_i1(copy_m[0],  copy_m[1],  copy_m[2],  copy_m[3],      
                                 copy_m[4],  copy_m[5],  copy_m[6],  copy_m[7],     
                                 copy_m[8],  copy_m[9],  copy_m[10], copy_m[11],    
                                 copy_m[12], copy_m[13], copy_m[14], copy_m[15]);    
    __vec16_i8 output;
    for (int i = 0; i < 16; i++) {
        output[i] = 0;    
    }
    output = __gather_base_offsets32_i8(&base, scale, _offsets, mask);

    __vec16_i8 result;                                                                
    for (int i = 0; i < 16; i ++)                                                   
        if (m[i] == 0)                                                              
            result[i] = 0;                                                          
        else                                                                        
            result[i] = (int32_t) b[i];                                          
                                                                                    
    int err_counter = 0;                                                            
    for (int i = 0; i < 16; ++i){                                                   
        if (m[i] != 0 && output[i] != result[i])                                    
            err_counter++;                                                          
        if (m[i] == 0 && output[i] != 0)                                            
            err_counter++;                                                          
    }
    if (copy_base != base)
        err_counter++;
    if (copy_scale != scale)
        err_counter++;
                                                                               
    if (err_counter != 0)                                                           
        printf(" errors %d\n", err_counter);                                        
    else                                                                            
         printf(" no fails\n");                                                     
                                                                                    
                                                                                    
    printf("\noutput:\noutput     | result  |data  \n");                            
    for (int i = 0; i < 16; i++)                                                    
        printf("%10d | %10d | %10d\n", output[i], result[i], &b[i]);              
    printf("*********************************************************\n");         
}
*/ 
