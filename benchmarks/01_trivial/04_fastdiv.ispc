// Copyright (c) 2021-2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#define FASTDIVISPC(T_ISPC, DIV_VAL)                                                                                   \
    export void fastdiv_##T_ISPC##_##DIV_VAL(uniform T_ISPC *uniform src, uniform T_ISPC *uniform dst,                 \
                                             uniform int count) {                                                      \
        foreach (i = 0... count) {                                                                                     \
            dst[i] = src[i] / DIV_VAL;                                                                                 \
        }                                                                                                              \
    }

FASTDIVISPC(uint64, 13)
FASTDIVISPC(int64, 13)
FASTDIVISPC(uint32, 13)
FASTDIVISPC(int32, 13)
FASTDIVISPC(uint16, 13)
FASTDIVISPC(int16, 13)
FASTDIVISPC(uint8, 13)
FASTDIVISPC(int8, 13)
FASTDIVISPC(uint64, 16)
FASTDIVISPC(int64, 16)
FASTDIVISPC(uint32, 16)
FASTDIVISPC(int32, 16)
FASTDIVISPC(uint16, 16)
FASTDIVISPC(int16, 16)
FASTDIVISPC(uint8, 16)
FASTDIVISPC(int8, 16)

#define FASTDIVISPC_MASK(T_ISPC, DIV_VAL)                                                                              \
    export void fastdiv_mask_##T_ISPC##_##DIV_VAL(uniform T_ISPC *uniform src, uniform T_ISPC *uniform dst,            \
                                                  uniform int *uniform mask, uniform int count) {                      \
        foreach (i = 0... count) {                                                                                     \
            dst[i] = mask[i] ? src[i] / DIV_VAL : 0;                                                                   \
        }                                                                                                              \
    }

FASTDIVISPC_MASK(uint64, 13)
FASTDIVISPC_MASK(int64, 13)
FASTDIVISPC_MASK(uint32, 13)
FASTDIVISPC_MASK(int32, 13)
FASTDIVISPC_MASK(uint16, 13)
FASTDIVISPC_MASK(int16, 13)
FASTDIVISPC_MASK(uint8, 13)
FASTDIVISPC_MASK(int8, 13)
FASTDIVISPC_MASK(uint64, 16)
FASTDIVISPC_MASK(int64, 16)
FASTDIVISPC_MASK(uint32, 16)
FASTDIVISPC_MASK(int32, 16)
FASTDIVISPC_MASK(uint16, 16)
FASTDIVISPC_MASK(int16, 16)
FASTDIVISPC_MASK(uint8, 16)
FASTDIVISPC_MASK(int8, 16)
