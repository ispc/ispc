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

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
/**************************************************************\
| Includes
\**************************************************************/
#include "matrix.h"
#include "matrix_ispc.h"

extern "C" {
#include "mmio.h"
}

/**************************************************************\
| DenseMatrix methods
\**************************************************************/
void DenseMatrix::multiply(const Vector &v, Vector &r) const {
    // Dimensionality check
    ASSERT(v.size() == cols());
    ASSERT(r.size() == rows());

    for (size_t i = 0; i < rows(); i++)
        r[i] = v.dot(entries + i * num_cols);
}

const Vector *DenseMatrix::row(size_t row) const { return new Vector(num_cols, entries + row * num_cols, true); }

void DenseMatrix::row(size_t row, Vector &r) {
    r.entries = entries + row * cols();
    r._size = cols();
}

void DenseMatrix::set_row(size_t row, const Vector &v) {
    ASSERT(v.size() == num_cols);
    memcpy(entries + row * num_cols, v.entries, num_cols * sizeof(double));
}

/**************************************************************\
| CRSMatrix Methods
\**************************************************************/
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

struct entry {
    int row;
    int col;
    double val;
};

bool compare_entries(struct entry i, struct entry j) {
    if (i.row < j.row)
        return true;
    if (i.row > j.row)
        return false;

    return i.col < j.col;
}

#define ERR_OUT(...)                                                                                                   \
    {                                                                                                                  \
        fprintf(stderr, __VA_ARGS__);                                                                                  \
        return NULL;                                                                                                   \
    }

CRSMatrix *CRSMatrix::matrix_from_mtf(char *path) {
    FILE *f;
    MM_typecode matcode;

    int m, n, nz;

    if ((f = fopen(path, "r")) == NULL)
        ERR_OUT("Error: %s does not name a valid/readable file.\n", path);

    if (mm_read_banner(f, &matcode) != 0)
        ERR_OUT("Error: Could not process Matrix Market banner.\n");

    if (mm_is_complex(matcode))
        ERR_OUT("Error: Application does not support complex numbers.\n")

    if (mm_is_dense(matcode))
        ERR_OUT("Error: supplied matrix is dense (should be sparse.)\n");

    if (!mm_is_matrix(matcode))
        ERR_OUT("Error: %s does not encode a matrix.\n", path)

    if (mm_read_mtx_crd_size(f, &m, &n, &nz) != 0)
        ERR_OUT("Error: could not read matrix size from file.\n");

    if (m != n)
        ERR_OUT("Error: Application does not support non-square matrices.");

    std::vector<struct entry> entries;
    entries.resize(nz);

    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %lg\n", &entries[i].row, &entries[i].col, &entries[i].val);
        // Adjust from 1-based to 0-based
        entries[i].row--;
        entries[i].col--;
    }

    sort(entries.begin(), entries.end(), compare_entries);

    CRSMatrix *M = new CRSMatrix(m, n, nz);
    int cur_row = -1;
    for (int i = 0; i < nz; i++) {
        while (entries[i].row > cur_row)
            M->row_offsets[++cur_row] = i;
        M->entries[i] = entries[i].val;
        M->columns[i] = entries[i].col;
    }

    return M;
}

Vector *Vector::vector_from_mtf(char *path) {
    FILE *f;
    MM_typecode matcode;

    int m, n, nz;

    if ((f = fopen(path, "r")) == NULL)
        ERR_OUT("Error: %s does not name a valid/readable file.\n", path);

    if (mm_read_banner(f, &matcode) != 0)
        ERR_OUT("Error: Could not process Matrix Market banner.\n");

    if (mm_is_complex(matcode))
        ERR_OUT("Error: Application does not support complex numbers.\n")

    if (mm_is_dense(matcode)) {
        if (mm_read_mtx_array_size(f, &m, &n) != 0)
            ERR_OUT("Error: could not read matrix size from file.\n");
    } else {
        if (mm_read_mtx_crd_size(f, &m, &n, &nz) != 0)
            ERR_OUT("Error: could not read matrix size from file.\n");
    }
    if (n != 1)
        ERR_OUT("Error: %s does not describe a vector.\n", path);

    Vector *x = new Vector(m);

    if (mm_is_dense(matcode)) {
        double val;
        for (int i = 0; i < m; i++) {
            fscanf(f, "%lg\n", &val);
            (*x)[i] = val;
        }
    } else {
        x->zero();
        double val;
        int row;
        int col;
        for (int i = 0; i < nz; i++) {
            fscanf(f, "%d %d %lg\n", &row, &col, &val);
            (*x)[row - 1] = val;
        }
    }
    return x;
}

#define ERR(...)                                                                                                       \
    {                                                                                                                  \
        fprintf(stderr, __VA_ARGS__);                                                                                  \
        exit(-1);                                                                                                      \
    }

void Vector::to_mtf(char *path) {
    FILE *f;
    MM_typecode matcode;

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_real(&matcode);
    mm_set_dense(&matcode);
    mm_set_general(&matcode);

    if ((f = fopen(path, "w")) == NULL)
        ERR("Error: cannot open/write to %s\n", path);

    mm_write_banner(f, matcode);
    mm_write_mtx_array_size(f, size(), 1);
    for (size_t i = 0; i < size(); i++)
        fprintf(f, "%lg\n", entries[i]);

    fclose(f);
}

void CRSMatrix::multiply(const Vector &v, Vector &r) const {
    ASSERT(v.size() == cols());
    ASSERT(r.size() == rows());

    for (size_t row = 0; row < rows(); row++) {
        int row_offset = row_offsets[row];
        int next_offset = ((row + 1 == rows()) ? _nonzeroes : row_offsets[row + 1]);

        double sum = 0;
        for (int i = row_offset; i < next_offset; i++) {
            sum += v[columns[i]] * entries[i];
        }
        r[row] = sum;
    }
}

void CRSMatrix::zero() {
    entries.clear();
    row_offsets.clear();
    columns.clear();
    _nonzeroes = 0;
}
