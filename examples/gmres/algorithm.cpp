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


/*===========================================================================*\
|* Includes
\*===========================================================================*/
#include "algorithm.h"
#include "stdio.h"
#include "debug.h"


/*===========================================================================*\
|* GMRES
\*===========================================================================*/
/* upper_triangular_right_solve:
 * ----------------------------
 * Given upper triangular matrix R and rhs vector b, solve for
 * x.  This "solve" ignores the rows, columns of R that are greater than the
 * dimensions of x.
 */
void upper_triangular_right_solve (const DenseMatrix &R, const Vector &b, Vector &x) 
{
    // Dimensionality check
    ASSERT(R.rows() >= b.size());
    ASSERT(R.cols() >= x.size());
    ASSERT(b.size() >= x.size());

    int max_row = x.size() - 1;

    // first solve step:
    x[max_row] = b[max_row] / R(max_row, max_row);

    for (int row = max_row - 1; row >= 0; row--) {
        double xi = b[row];
        for (int col = max_row; col > row; col--)
            xi -= x[col] * R(row, col);
        x[row] = xi / R(row, row);
    }
}

/* create_rotation (used in gmres):
 * -------------------------------
 * Construct a Givens rotation to zero out the lowest non-zero entry in a partially
 * factored Hessenburg matrix.  Note that the previous Givens rotations should be
 * applied to this column before creating a new rotation.
 */
void create_rotation (const DenseMatrix &H, size_t col, Vector &Cn, Vector &Sn) 
{
    double a = H(col,     col);
    double b = H(col + 1, col);
    double r;

    if (b == 0) {
        Cn[col] = copysign(1, a);
        Sn[col] = 0;
    } 
    else if (a == 0) {
        Cn[col] = 0;
        Sn[col] = copysign(1, b);
    }
    else {
        r       = sqrt(a*a + b*b);
        Sn[col] = -b / r;
        Cn[col] =  a / r;
    }
}

/* Applies the 'col'th Givens rotation stored in vectors Sn and Cn to the 'col'th 
 * column of the DenseMatrix M.  (Previous columns don't need the rotation applied b/c
 * presumeably, the first col-1 columns are already upper triangular, and so their
 * entries in the col and col+1 rows are 0.)
 */
void apply_rotation (DenseMatrix &H, size_t col, Vector &Cn, Vector &Sn) 
{
    double c = Cn[col];
    double s = Sn[col];
    double tmp    = c * H(col, col) - s * H(col+1, col);
    H(col+1, col) = s * H(col, col) + c * H(col+1, col);
    H(col,   col) = tmp;
}

/* Applies the 'col'th Givens rotation to the vector.
 */
void apply_rotation (Vector &v, size_t col, Vector &Cn, Vector &Sn) 
{
    double a = v[col];
    double b = v[col + 1];

    double c = Cn[col];
    double s = Sn[col];

    v[col]     = c * a - s * b;
    v[col + 1] = s * a + c * b;
}

/* Applies the first 'col' Givens rotations to the newly-created column
 * of H.  (Leaves other columns alone.)
 */
void update_column (DenseMatrix &H, size_t col, Vector &Cn, Vector &Sn) 
{
    for (int i = 0; i < col; i++) {
        double c    = Cn[i];
        double s    = Sn[i];
        double t    = c * H(i,col) - s * H(i+1,col);
        H(i+1, col) = s * H(i,col) + c * H(i+1,col);
        H(i,   col) = t;
    }
}

/* After a new column has been added to the hessenburg matrix, factor it back into
 * an upper-triangular matrix by:
 * - applying the previous Givens rotations to the new column
 * - computing the new Givens rotation to make the column upper triangluar
 * - applying the new Givens rotation to the column, and
 * - applying the new Givens rotation to the solution vector
 */
void update_qr_decomp (DenseMatrix &H, Vector &s, size_t col, Vector &Cn, Vector &Sn)
{
    update_column(  H, col, Cn, Sn);
    create_rotation(H, col, Cn, Sn);
    apply_rotation( H, col, Cn, Sn);
    apply_rotation( s, col, Cn, Sn);
}

void gmres (const Matrix &A, const Vector &b, Vector &x, int num_iters, double max_err)  
{
    DEBUG_PRINT("gmres starting!\n");
    x.zero();

    ASSERT(A.rows() == A.cols());
    DenseMatrix Qstar(num_iters + 1, A.rows());
    DenseMatrix H(num_iters + 1, num_iters);

    // arrays for storing parameters of givens rotations
    Vector Sn(num_iters);
    Vector Cn(num_iters);

    // array for storing the rhs projected onto the hessenburg's column space
    Vector G(num_iters+1);
    G.zero();

    double beta = b.norm();
    G[0] = beta;

    // temp vector, stores Aqi
    Vector w(A.rows());

    w.copy(b);
    w.normalize();
    Qstar.set_row(0, w);

    int iter = 0;
    Vector temp(A.rows(), false);
    double rel_err;

    while (iter < num_iters) 
    {
        // w = Aqi
        Qstar.row(iter, temp);
        A.multiply(temp, w);

        // construct ith column of H, i+1th row of Qstar:        
        for (int row = 0; row <= iter; row++) {
            Qstar.row(row, temp);
            H(row, iter) = temp.dot(w);
            w.add_ax(-H(row, iter), temp);
        }

        H(iter+1, iter) = w.norm();
        w.divide(H(iter+1, iter));
        Qstar.set_row(iter+1, w);

        update_qr_decomp (H, G, iter, Cn, Sn);

        rel_err = fabs(G[iter+1] / beta);

        if (rel_err < max_err)
            break;

        if (iter % 100 == 0)
            DEBUG_PRINT("Iter %d: %f err\n", iter, rel_err);

        iter++;
    }

    if (iter == num_iters) {
        fprintf(stderr, "Error: gmres failed to converge in %d iterations (relative err: %f)\n", num_iters, rel_err);
        exit(-1);
    }

    // We've reached an acceptable solution (?):

    DEBUG_PRINT("gmres completed in %d iterations (rel. resid. %f, max %f)\n", num_iters, rel_err, max_err);
    Vector y(iter+1);
    upper_triangular_right_solve(H, G, y);
    for (int i = 0; i < iter + 1; i++) {
        Qstar.row(i, temp);
        x.add_ax(y[i], temp);
    }
}
