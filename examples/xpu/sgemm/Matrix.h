/*
 * Copyright (c) 2019-2020, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef Matrix_H
#define Matrix_H

#include "share.h"

typedef enum { RowMajor, ColMajor, Nd } storage_type_t;
static const char *storagename[Nd] = {"RowMajor", "ColMajor"};

#define CORRECTNESS_THRESHOLD 0.00002
struct alignas(4096) aligned_struct_t {
    float *data;

    float &operator[](int index) { return data[index]; }
};

class Matrix {
#ifndef ISPCRT
    float *M;
#else
    aligned_struct_t M;
#endif
    int _size_;
    int nrow;
    int ncol;
    int ld;
    storage_type_t st;
    char *mtxname;

  public:
    fptype &operator()(int r, int c) { return ((st == ColMajor) ? M[r + c * ld] : M[r * ld + c]); }

    Matrix(int nrow, int ncol, int ld, char *surfname, bool init, const char *mtxname, storage_type_t st = RowMajor) {
        if (st == ColMajor)
            assert(ld >= nrow);
        else {
            if (ld < ncol) {
                fprintf(stderr, "ld(%d) should be >= ncol(%d)\n", ld, ncol);
                exit(123);
            }
        }

        this->nrow = nrow;
        this->st = st;
        this->ncol = ncol;
        this->ld = ld;
#ifndef ISPCRT
        if (st == ColMajor)
            _size_ = (__int64)(sizeof(M[0]) * this->ncol * this->ld);
        else
            _size_ = (__int64)(sizeof(M[0]) * this->nrow * this->ld);

        M = (fptype *)CM_ALIGNED_MALLOC(_size_, 4096);
#else
        if (st == ColMajor)
            _size_ = (uint64_t)(sizeof(M[0]) * this->ncol * this->ld);
        else
            _size_ = (uint64_t)(sizeof(M[0]) * this->nrow * this->ld);

        M.data = (float *)malloc(_size_);
#endif
        this->mtxname = strdup(mtxname);
        for (int c = 0; c < this->ncol; c++) {
            for (int r = 0; r < this->nrow; r++) {
                (*this)(r, c) = (init == true) ? randData(0.0f, 1.0f) : 0.0f;
            }
        }
    }

    Matrix(Matrix &mat, const char *mtxname)
        : nrow(mat.nrow), ncol(mat.ncol), ld(mat.ld), st(mat.st), _size_(mat._size_) {

        this->mtxname = strdup(mtxname);
        // printf("Allocating %s \n", mtxname);
#ifndef ISPCRT
        M = (fptype *)CM_ALIGNED_MALLOC(mat._size_, 4096);
#else
        M.data = (float *)malloc(mat._size_);
#endif

        for (int c = 0; c < this->ncol; c++)
            for (int r = 0; r < this->nrow; r++) {
                (*this)(r, c) = mat(r, c);
            }
    }
    bool operator!=(Matrix &m) { return !(*this == m); }
    bool operator==(Matrix &m) {
        if (m.n_col() != this->n_col())
            return false;

        double max_relerror = 0.0;
        double max_abserror = 0.0;
        for (int c = 0; c < ncol; c++)
            for (int r = 0; r < nrow; r++) {

                // printf("I=%3d N=%3d  %08x  %08x\n", r, c, *(unsigned int*)&(*this)(r,c), *(unsigned int *)&m(r,c));

                double relerror = fabs((*this)(r, c) - m(r, c)) / max(fabs((*this)(r, c)), fabs(m(r, c)));
                double abserror = fabs((*this)(r, c) - m(r, c));

                max_relerror = max(max_relerror, relerror);
                max_abserror = max(max_abserror, abserror);

                if (relerror > CORRECTNESS_THRESHOLD) {
                    printf("Failure %f %f relerror: %lf at [%d, %d]\n", (*this)(r, c), m(r, c), relerror, r, c);
                    exit(-1);
                }
            }
        printf("max_relerror = %e  absolute error = %e\n", max_relerror, max_abserror);
        return (max_relerror > CORRECTNESS_THRESHOLD) ? false : true;
        return true;
    }

    // friend ostream& operator<<(ostream&, const Matrix&);
    void Print(const char *str = NULL) {
        if (str)
            printf("%s ", str);
        printf(" %d x %d\n", this->n_row(), this->l_dim());
        for (int i = 0; i < this->n_row(); i++) {
            for (int j = 0; j < this->n_col(); j++)
                printf("C(%d,%d)=%f \n", i, j, (*this)(i, j));
            printf("\n");
        }
    }

    int n_row() { return nrow; }
    int n_col() { return ncol; }
    int l_dim() { return ld; }
    ~Matrix() { /*printf("Deallocating %s \n", mtxname); */
#ifndef ISPCRT
        CM_ALIGNED_FREE(M);
#else
        free(M.data);
#endif
    }
};

// C := alpha*A*B + beta*C,
// A(m x k) , B(k x n) , C(m x n)
static int sgemmNxN(int m, int n, int k, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C,
                    int ldc, storage_type_t st = RowMajor) {

    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) {
            float tmp = 0.0f;
            if (st == ColMajor) {
                for (int t = 0; t < k; t++)
                    tmp += A[r + t * lda] * B[t + c * ldb];
                C[r + c * ldc] = alpha * tmp + beta * C[r + c * ldc];
            } else {
                for (int t = 0; t < k; t++)
                    tmp += A[r * lda + t] * B[t * ldb + c];
                C[r * ldc + c] = alpha * tmp + beta * C[r * ldc + c];
            }
        }

    return 1;
}

#endif
