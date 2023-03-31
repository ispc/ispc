/*
  Copyright (c) 2012-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/


/**************************************************************\
| General
\**************************************************************/
export void zero (uniform double data[],
                  uniform int size)
{
    foreach (i = 0 ... size)
        data[i] = 0.0;
}


/**************************************************************\
| Vector helpers
\**************************************************************/
export void vector_add (uniform double a[],
                        const uniform double b[],
                        const uniform int size)
{
    foreach (i = 0 ... size)
        a[i] += b[i];
}

export void vector_sub (uniform double a[],
                        const uniform double b[],
                        const uniform int size)
{
    foreach (i = 0 ... size)
        a[i] -= b[i];
}

export void vector_mult (uniform double a[],
                         const uniform double b,
                         const uniform int size)
{
    foreach (i = 0 ... size)
        a[i] *= b;
}

export void vector_div (uniform double a[],
                        const uniform double b,
                        const uniform int size)
{
    foreach (i = 0 ... size)
        a[i] /= b;
}

export void vector_add_ax (uniform double r[],
                           const uniform double a,
                           const uniform double x[],
                           const uniform int    size)
{
    foreach (i = 0 ... size)
        r[i] += a * x[i];
}

export uniform double vector_dot (const uniform double a[],
                                  const uniform double b[],
                                  const uniform int size)
{
    varying double sum = 0.0;
    foreach (i = 0 ... size)
        sum += a[i] * b[i];
    return reduce_add(sum);
}

/**************************************************************\
| Matrix helpers
\**************************************************************/
export void sparse_multiply (const uniform double entries[],
                             const uniform double columns[],
                             const uniform double row_offsets[],
                             const uniform int rows,
                             const uniform int cols,
                             const uniform int nonzeroes,
                             const uniform double v[],
                             uniform double r[])
{
    foreach (row = 0 ... rows) {
        int row_offset = row_offsets[row];
        int next_offset = ((row + 1 == rows) ? nonzeroes : row_offsets[row+1]);

        double sum = 0;
        for (int j = row_offset; j < next_offset; j++) {
            #pragma ignore warning(perf)
            sum += v[columns[j]] * entries[j];
        }
        r[row] = sum;
    }
}

