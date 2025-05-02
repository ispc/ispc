// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
#include "tensor.isph"
export void tensor_mul_ISPC_int32(void *uniform A, void *uniform B, void *uniform C) { tensor_mul<int>(A, B, C); }
export void tensor_mul_ISPC_float(void *uniform A, void *uniform B, void *uniform C) { tensor_mul<float>(A, B, C); }
export void tensor_mul_ISPC_double(void *uniform A, void *uniform B, void *uniform C) { tensor_mul<double>(A, B, C); }