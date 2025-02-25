// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
#include "tensor.isph"
export void tensor_add_ISPC_int32(void *uniform A, void *uniform B, void *uniform C) { tensor_add<int>(A, B, C); }
export void tensor_add_ISPC_float(void *uniform A, void *uniform B, void *uniform C) { tensor_add<float>(A, B, C); }
export void tensor_add_ISPC_double(void *uniform A, void *uniform B, void *uniform C) { tensor_add<double>(A, B, C); }