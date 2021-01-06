/*
  Copyright (c) 2012, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

