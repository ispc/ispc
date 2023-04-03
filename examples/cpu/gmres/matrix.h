/*
  Copyright (c) 2012-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef __MATRIX_H__
#define __MATRIX_H__

/**************************************************************\
| Includes
\**************************************************************/
#include <cmath>   // sqrt
#include <cstdlib> // malloc, memcpy, etc.
#include <cstring> // size_t
#include <vector>

#include "debug.h"
#include "matrix_ispc.h"

class DenseMatrix;
/**************************************************************\
| Vector class
\**************************************************************/
class Vector {
  public:
    static Vector *vector_from_mtf(char *path);
    void to_mtf(char *path);

    Vector(size_t size, bool alloc_mem = true) {
        shared_ptr = false;
        _size = size;

        if (alloc_mem)
            entries = (double *)malloc(sizeof(double) * _size);
        else {
            shared_ptr = true;
            entries = nullptr;
        }
    }

    Vector(size_t size, double *content, bool share_ptr = false) {
        _size = size;
        if (share_ptr) {
            entries = content;
            shared_ptr = true;
        } else {
            shared_ptr = false;
            entries = (double *)malloc(sizeof(double) * _size);
            memcpy(entries, content, sizeof(double) * _size);
        }
    }

    ~Vector() {
        if (!shared_ptr)
            free(entries);
    }

    const double &operator[](size_t index) const {
        ASSERT(index < _size);
        return *(entries + index);
    }

    double &operator[](size_t index) {
        ASSERT(index < _size);
        return *(entries + index);
    }

    bool operator==(const Vector &v) const {
        if (v.size() != _size)
            return false;

        for (size_t i = 0; i < _size; i++)
            if (entries[i] != v[i])
                return false;

        return true;
    }

    size_t size() const { return _size; }

    double dot(const Vector &b) const {
        ASSERT(b.size() == this->size());
        return ispc::vector_dot(entries, b.entries, size());
    }

    double dot(const double *const b) const { return ispc::vector_dot(entries, b, size()); }

    void zero() { ispc::zero(entries, size()); }

    double norm() const { return sqrt(dot(entries)); }

    void normalize() { this->divide(this->norm()); }

    void add(const Vector &a) {
        ASSERT(size() == a.size());
        ispc::vector_add(entries, a.entries, size());
    }

    void subtract(const Vector &s) {
        ASSERT(size() == s.size());
        ispc::vector_sub(entries, s.entries, size());
    }

    void multiply(double scalar) { ispc::vector_mult(entries, scalar, size()); }

    void divide(double scalar) { ispc::vector_div(entries, scalar, size()); }

    // Note: x may be longer than *(this)
    void add_ax(double a, const Vector &x) {
        ASSERT(x.size() >= size());
        ispc::vector_add_ax(entries, a, x.entries, size());
    }

    // Note that copy only copies the first size() elements of the
    // supplied vector, i.e. the supplied vector can be longer than
    // this one.  This is useful in least squares calculations.
    void copy(const Vector &other) {
        ASSERT(other.size() >= size());
        memcpy(entries, other.entries, size() * sizeof(double));
    }

    friend class DenseMatrix;

  private:
    size_t _size;
    bool shared_ptr;
    double *entries;
};

/**************************************************************\
| Matrix base class
\**************************************************************/
class Matrix {
    friend class Vector;

  public:
    Matrix(size_t size_r, size_t size_c) {
        num_rows = size_r;
        num_cols = size_c;
    }
    ~Matrix() {}

    size_t rows() const { return num_rows; }
    size_t cols() const { return num_cols; }

    virtual void multiply(const Vector &v, Vector &r) const = 0;
    virtual void zero() = 0;

  protected:
    size_t num_rows;
    size_t num_cols;
};

/**************************************************************\
| DenseMatrix class
\**************************************************************/
class DenseMatrix : public Matrix {
    friend class Vector;

  public:
    DenseMatrix(size_t size_r, size_t size_c) : Matrix(size_r, size_c), shared_ptr(false) {
        entries = (double *)malloc(size_r * size_c * sizeof(double));
    }

    DenseMatrix(size_t size_r, size_t size_c, const double *content) : Matrix(size_r, size_c), shared_ptr(false) {
        entries = (double *)malloc(size_r * size_c * sizeof(double));
        memcpy(entries, content, size_r * size_c * sizeof(double));
    }

    ~DenseMatrix() {
        free(entries);
    }

    virtual void multiply(const Vector &v, Vector &r) const;

    double &operator()(unsigned int r, unsigned int c) { return *(entries + r * num_cols + c); }

    const double &operator()(unsigned int r, unsigned int c) const { return *(entries + r * num_cols + c); }

    const Vector *row(size_t row) const;
    void row(size_t row, Vector &r);
    void set_row(size_t row, const Vector &v);

    virtual void zero() { ispc::zero(entries, rows() * cols()); }

    void copy(const DenseMatrix &other) {
        ASSERT(rows() == other.rows());
        ASSERT(cols() == other.cols());
        memcpy(entries, other.entries, rows() * cols() * sizeof(double));
    }

  private:
    double *entries;
    bool shared_ptr;
};

/**************************************************************\
| CSRMatrix (compressed row storage, a sparse matrix format)
\**************************************************************/
class CRSMatrix : public Matrix {
  public:
    CRSMatrix(size_t size_r, size_t size_c, size_t nonzeroes) : Matrix(size_r, size_c) {
        _nonzeroes = nonzeroes;
        entries.resize(nonzeroes);
        columns.resize(nonzeroes);
        row_offsets.resize(size_r);
    }

    virtual void multiply(const Vector &v, Vector &r) const;

    virtual void zero();

    static CRSMatrix *matrix_from_mtf(char *path);

  private:
    unsigned int _nonzeroes;
    std::vector<double> entries;
    std::vector<int> row_offsets;
    std::vector<int> columns;
};

#endif
