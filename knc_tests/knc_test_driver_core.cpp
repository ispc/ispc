#include "knc_test_driver_core.h"

InputData::InputData () {
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
    d_u64[2] = UINT32_MAX;
    d_u64[4] = 0;

    f_u8[0] = UINT8_MAX;
    f_u8[1] = 0;

    f_u16[0] = UINT16_MAX;
    f_u16[1] = 0;

    f_u32[0] = UINT32_MAX;
    f_u32[1] = 0;

    f_u64[0] = UINT64_MAX;
    f_u64[1] = 0;
    f_u64[2] = UINT32_MAX;
    f_u64[4] = 0;

    d_8[0] = INT8_MAX;
    d_8[1] = INT8_MIN;

    d_16[0] = INT16_MAX;
    d_16[1] = INT16_MIN;

    d_32[0] = INT32_MAX;
    d_32[1] = INT32_MIN;

    d_64[0] = INT64_MAX;
    d_64[1] = INT64_MIN;
    f_64[2] = INT32_MAX;
    f_64[4] = INT32_MIN;

    f_8[0] = INT8_MAX;
    f_8[1] = INT8_MIN;

    f_16[0] = INT16_MAX;
    f_16[1] = INT16_MIN;

    f_32[0] = INT32_MAX;
    f_32[1] = INT32_MIN;

    f_64[0] = INT64_MAX;
    f_64[1] = INT64_MIN;
    f_64[2] = INT32_MAX;
    f_64[4] = INT32_MIN;

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
    i64[2] = INT32_MAX;
    i64[4] = INT32_MIN;

    ui8[0] = UINT8_MAX;
    ui8[1] = 0;

    ui16[0] = UINT16_MAX;
    ui16[1] = 0;

    ui32[0] = UINT32_MAX;
    ui32[1] = 0;

    ui64[0] = INT64_MAX;
    ui64[1] = 0;
    ui64[2] = UINT32_MAX;
    ui64[4] = 0;

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
}
