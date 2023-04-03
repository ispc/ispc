/*
  Copyright (c) 2012-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
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
    do {                                                                                                               \
        fprintf(stderr, __VA_ARGS__);                                                                                  \
        return nullptr;                                                                                                \
    } while(0)

#define ERR_OUT_WITH_CLOSE(file, ...)                                                                                  \
    do {                                                                                                               \
        fclose(file);                                                                                                  \
        ERR_OUT(__VA_ARGS__);                                                                                          \
    } while(0)

CRSMatrix *CRSMatrix::matrix_from_mtf(char *path) {
    FILE *f;
    MM_typecode matcode;

    int m, n, nz;

    if ((f = fopen(path, "r")) == nullptr)
        ERR_OUT("Error: %s does not name a valid/readable file.\n", path);

    if (mm_read_banner(f, &matcode) != 0)
        ERR_OUT_WITH_CLOSE(f, "Error: Could not process Matrix Market banner.\n");

    if (mm_is_complex(matcode))
        ERR_OUT_WITH_CLOSE(f, "Error: Application does not support complex numbers.\n");

    if (mm_is_dense(matcode))
        ERR_OUT_WITH_CLOSE(f, "Error: supplied matrix is dense (should be sparse.)\n");

    if (!mm_is_matrix(matcode))
        ERR_OUT_WITH_CLOSE(f, "Error: %s does not encode a matrix.\n", path);

    if (mm_read_mtx_crd_size(f, &m, &n, &nz) != 0)
        ERR_OUT_WITH_CLOSE(f, "Error: could not read matrix size from file.\n");

    if (m != n)
        ERR_OUT_WITH_CLOSE(f, "Error: Application does not support non-square matrices.");

    std::vector<struct entry> entries;
    entries.resize(nz);

    for (int i = 0; i < nz; i++) {
        if (3 != fscanf(f, "%d %d %lg\n", &entries[i].row, &entries[i].col, &entries[i].val)) {
            printf("Couldn't read input correctly\n");
            exit(-1);
        }
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

    fclose(f);
    return M;
}

Vector *Vector::vector_from_mtf(char *path) {
    FILE *f;
    MM_typecode matcode;

    int m, n, nz;

    if ((f = fopen(path, "r")) == nullptr)
        ERR_OUT("Error: %s does not name a valid/readable file.\n", path);

    if (mm_read_banner(f, &matcode) != 0)
        ERR_OUT_WITH_CLOSE(f, "Error: Could not process Matrix Market banner.\n");

    if (mm_is_complex(matcode))
        ERR_OUT_WITH_CLOSE(f, "Error: Application does not support complex numbers.\n");

    if (mm_is_dense(matcode)) {
        if (mm_read_mtx_array_size(f, &m, &n) != 0)
            ERR_OUT_WITH_CLOSE(f, "Error: could not read matrix size from file.\n");
    } else {
        if (mm_read_mtx_crd_size(f, &m, &n, &nz) != 0)
            ERR_OUT_WITH_CLOSE(f, "Error: could not read matrix size from file.\n");
    }
    if (n != 1)
        ERR_OUT_WITH_CLOSE(f, "Error: %s does not describe a vector.\n", path);

    Vector *x = new Vector(m);

    if (mm_is_dense(matcode)) {
        double val;
        for (int i = 0; i < m; i++) {
            if (1 != fscanf(f, "%lg\n", &val)) {
                printf("Couldn't read input correctly\n");
                exit(-1);
            }
            (*x)[i] = val;
        }
    } else {
        x->zero();
        double val;
        int row;
        int col;
        for (int i = 0; i < nz; i++) {
            if (3 != fscanf(f, "%d %d %lg\n", &row, &col, &val)) {
                printf("Couldn't read input correctly\n");
                exit(-1);
            }
            (*x)[row - 1] = val;
        }
    }
    fclose(f);
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

    if ((f = fopen(path, "w")) == nullptr)
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
