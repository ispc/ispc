/*
  Copyright (c)2021-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/
export void do_add_ispc(uniform float A[], uniform float B[], uniform float C[]) {
    C[programIndex] = A[programIndex] + B[programIndex];
}
