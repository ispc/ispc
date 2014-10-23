#ifndef __DATA_CNTRL_H__
#define __DATA_CNTRL_H__

#define __STDC_LIMIT_MACROS // enable intN_t limits from stdint.h
#include <stdint.h>

struct InputData {
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

///////////////////////////////////////////////////////////////////////
   
    InputData(); 
};

#endif
