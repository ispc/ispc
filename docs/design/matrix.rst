===========
Matrix type
===========

Language design
---------------

A matrix type is an ISPC type with an underlying ISPC uniform basic type, a constant number of rows, and a constant number of columns.
A value of a matrix type includes storage for rows * columns values of the element type. The internal layout is raw-major layout.

Matrix types with the same basic type, rows, and columns are considered of the same type.
The maximum number of rows and columns supported is dependent on the specified target. If the maximum limit is exceeded,
ISPC will generate an error message.

Matrix types can't be specified with `uniform`/`varying` qualifiers. From ISPC perspective matrix type is always `uniform`.

.. code-block:: cpp

    matrix<uniform float, M, N> A; // create uninitialized matrix of floats MxN size
    matrix<uniform float, M, N> A(); // error
    matrix<varying unsigned int8, M, N> A; // error, no varying basic type is allowed
    uniform matrix<uniform unsigned int8, M, N> A; // error, uniform/varying qualifiers are not allowed with matrix types
    varying matrix<uniform unsigned int8, M, N> A; // error, uniform/varying qualifiers are not allowed with matrix types


Basic operators
---------------
To access the particular matrix element[s], the standard matrix access operator can be used: `A[X, Y]`, where X, Y are of integer type.

If `X` and `Y` are uniform, the result is a single matrix element.

If `X` or `Y` are varying, the result is a `varying`.

If `X` or `Y` are both varying, the result is a new matrix of size `programCount` x `programCount`.

If the index is varying and `programCount` is bigger than matrix dimension, ISPC will produce an error.

Consider an example:

.. code-block:: cpp

    /*
    Matrix A:
    a11 | a12 | a13 | a14 | a15 | a16 | a17 | a18
    a21 | a22 | a23 | a24 | a25 | a26 | a27 | a28
    a31 | a32 | a33 | a34 | a35 | a36 | a37 | a38
    a41 | a42 | a43 | a44 | a45 | a46 | a47 | a48
    */
    uniform int X = 1;
    uniform int Y = 2;
    uniform int xy = A[X, Y]; // returns a23

    varying int X = 1;
    uniform int Y = 2;
    varying int xy = A[X, Y]; // returns {a23, a23, a23, a23}

    uniform int X = 2;
    varying int Y = {2, 1, 3, 5};
    varying int xy = A[X, Y]; // returns {a33, a32, a34, a36}

    varying int X = {0, 1, 2, 3};
    varying int Y = {2, 1, 3, 5};
    // Returns
    // a13 | a11 | a14 | a16
    // a23 | a21 | a24 | a26
    // a33 | a31 | a34 | a36
    // a43 | a41 | a44 | a46
    matrix<int, programCount, programCount> XY = A[X, Y];


The following ISPC operators are supported for matrices of the same types or matrix and a scalar:

.. list-table:: Operators

  * - Symbols
    - Use
  * - ``=``
    - Assignment
  * - ``+``, ``-``
    - Element-wise addition and subtraction
  * - ``++``, ``--``
    - Element-wise increment/decrement
  * - ``+=``, ``-=``
    - Compound assignment operators

Operators ``+``, ``-``, ``*``, ``/`` can be used with matrix and uniform value, applying the operation to each element of the matrix.

Examples:

.. code-block:: cpp

    matrix<uniform int, M, N> A = 1; // init all matrix elements with 1
    matrix<uniform int, M, N> B = A + 2; // add 2 to each element of matrix A and init with result matrix B

Conversions
-----------
Matrix type can be converted to another matrix type if the number of rows and columns are the same and the value's elements can be converted to the element type of the result type.

.. code-block:: cpp

    matrix<uniform int16, M, N> A16;
    matrix<uniform float, M, N> AF = (matrix<uniform float, M, N>)A16;

Interoperability
----------------
Matrix is internal ISPC type. It can't be used as an argument to `export` or `extern "C"` functions. It can be used as an argument for internal ISPC functions.

Matrix stdlib functions
-----------------------

  * matrix<T, M, N> matrix_load(void* ptr, M, N) - loads matrix from memory to registers
  * void matrix_store(matrix<T, M, N> m, void* ptr) - stores matrix from registers to memory
  * matrix<T, M_pack, N_pack> matrix_horizontal_pack(matrix<T, M, N> m) - packs matrix horizontally into 32-bit elements
  * matrix<T, M_pack, N_pack> matrix_vertical_pack(matrix<T, M, N> m) - packs matrix vertically into 32-bit elements (VNNI packing)
  * uniform int matrix_size() - returns optimimal MxN size for current compilation target
  * matrix<T, N, M> matrix_transpose(matrix<T, M, N>) - transpose the matrix
  * matrix<T, M1, N2> matrix_mad(matrix<T, M1, N1> m1, matrix<T, M2, N2> m2) - matrix multiplications and add

`matrix<T, M1, N2> matrix_mad(matrix<T, M1, N1> m1, matrix<T, M2, N2> m2)` uses HW-specific intrinsics when available on the specified
platform. It may have limitations for sizes/types/data layout of input matrices. Exact supported combinations are implementation-defined.

There can be additional stdlib functions available for specific platforms only. For example it may be a function
to set a tile configuration on the platforms with Intel(R) AMX support.

MAD example in ISPC
-------------------

.. code-block:: cpp

  /*
         N                     K                     K
    ---     ---           ---     ---           ---     ---
    |         |           |         |           |         |
  M |         |   X     N |         |   =     M |         |
    |         |           |         |           |         |
    ---     ---           ---     ---           ---     ---
          A        X            B        =            C
  */

  struct Parameters {
    int *mC;
    int8 *mA;
    int8 *mB;
    unsigned int M;
    unsigned int N;
    unsigned int K;
  };

  task void matrix_example(void *uniform _p) {
    Parameters *uniform p = (Parameters * uniform) _p;
    matrix<int8, M, N> A = matrix_load(p->mA, M, N);
    matrix<int8, N, K> B = matrix_load(p->mB, N, K);
    matrix<int, M, K> C = 0;

    C = matrix_mad(matrix_vertical_pack(A), matrix_vertical_pack(B));

    matrix_store(C, p->mC);
  }


Implementation design
---------------------

Implementation details
----------------------


