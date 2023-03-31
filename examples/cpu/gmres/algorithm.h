/*
  Copyright (c) 2012-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__

#include "matrix.h"

/* Generalized Minimal Residual Method:
 * -----------------------------------
 * Takes a square matrix and an rhs and uses GMRES to find an estimate for x.
 * The specified error is relative.
 */
void gmres(const Matrix &A, const Vector &b, Vector &x, int num_iters, double err);

#endif
