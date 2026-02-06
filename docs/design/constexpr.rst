======================
Constexpr for ISPC (v1)
======================

Summary
-------

Add a ``constexpr`` specifier for variables and functions. ``constexpr`` values
are compile-time constants and can be used in constant-expression contexts such
as array sizes, short-vector lengths, switch case labels, and non-type template
arguments. ``constexpr`` functions are validated for constant-evaluation
compatibility at definition time (C++-style "constexpr-suitable" checks).

V1 supports both uniform and varying constexpr values. Aggregate constexpr
values (arrays, structs) and short vectors are supported. Pointer constants are
supported for null and addresses of globals/functions, but dereference is not
allowed in constexpr evaluation.


Goals
-----

- Provide a first-class, explicit compile-time constant feature.
- Allow constexpr function calls in constant-expression contexts.
- Support varying constexpr values (lane-wise compile-time constants).
- Validate constexpr functions at definition time.


Non-goals (v1)
-------------

- No constexpr evaluation of memory dereference or dynamic allocation.
- No constexpr support for tasks, exports, or external ABI functions.


Syntax
------

``constexpr`` is a declaration specifier.

Variables:

.. code-block:: c++

    constexpr uniform int Tile = 16;
    constexpr varying int lane = programIndex;

Functions:

.. code-block:: c++

    constexpr uniform int gcd(uniform int a, uniform int b) {
        while (b != 0) {
            uniform int t = a % b;
            a = b;
            b = t;
        }
        return a;
    }


Semantics
---------

constexpr variables
~~~~~~~~~~~~~~~~~~~

- ``constexpr`` implies ``const``.
- ``constexpr`` is not allowed on ``typedef`` declarations.
- The initializer must be a constant expression (constant-evaluable).
- Allowed v1 types:
  - atomic types (bool, int, float, double, etc.)
  - enum types
  - pointer types (value-only, no dereference in constexpr evaluation)
  - short vector types of allowed element types
  - arrays and structs of allowed element types
- Arrays must have fully specified sizes.
- Pointer values are limited to null or addresses of globals/functions.
- The value is available to constant-expression evaluation.
- For variables, storage class and linkage follow existing rules.

constexpr functions
~~~~~~~~~~~~~~~~~~~

- A constexpr function is compiled like a normal function and may also be
  evaluated at compile time when called in a constant-expression context.
- The selected overload must be ``constexpr`` for constant evaluation.
- ``constexpr`` functions must be "constexpr-suitable" at definition time.
- The return type can be any constexpr-supported type (including aggregates).
- ``constexpr`` is not allowed on function parameters.
- Linkage follows C++ constexpr function behavior:
  - non-``static`` constexpr functions are emitted with ODR linkage
    (``linkonce_odr``)
  - ``static`` constexpr functions have internal linkage

constexpr-suitable validation (definition time)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At definition time, the compiler validates that the function body only uses
constexpr-safe constructs. Examples of allowed and disallowed constructs are
listed below.

Allowed:

- Local variable declarations and assignments.
- Arithmetic, bitwise, and comparison operators.
- ``if``, ``switch``, ``for``, ``while``, ``do`` (with standard control flow).
- Uniform conditions for control flow.
- ``break`` and ``continue``.
- ``return`` with a value.
- Calls to other ``constexpr`` functions.
- ``sizeof``, ``alignof``, and type casts.
- Aggregate local variables with constant initializers.

Disallowed (v1):

- Writes to non-local storage (globals, captured references, or pointer
  dereference).
- Taking the address of local variables or parameters; only global/function
  addresses are allowed.
- ``new``, ``delete``, ``launch``, ``invoke_sycl``, ``sync``, ``print``,
  atomics, and any other side-effecting builtins.
- ``foreach*`` statements (``foreach``, ``foreach_active``, ``foreach_unique``).
- ``goto`` and labels.
- Calls to non-``constexpr`` functions.
- Return type ``void`` (v1 restriction).

These checks are purely structural; the function body is not executed at
definition time.

Constant-expression contexts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following contexts require constant expressions and can use constexpr
variables and constexpr function calls:

- Array sizes.
- Short vector lengths (``float<N>``).
- ``switch`` case labels.
- Enumerator values.
- Non-type template arguments.
- Default parameter values.

Existing constant expressions (literals, enum constants, and const variables
with constant initializers) remain valid. ``constexpr`` extends these contexts
to allow constexpr function calls.

Varying constexpr values
~~~~~~~~~~~~~~~~~~~~~~~~

``constexpr varying`` values are lane-wise compile-time constants. Constant
evaluation follows SPMD semantics and produces a compile-time vector of values
using the target width.

Examples:

.. code-block:: c++

    constexpr varying int lane = programIndex;
    constexpr varying int off = lane * 4;

These values are valid in constant expressions that accept varying results,
such as constant folding and builtins that operate on varying values. Contexts
that require uniform constants (array sizes, vector lengths, template args)
require the constexpr result to be uniform.

Target dependence
~~~~~~~~~~~~~~~~~

If constexpr evaluation depends on target-specific values (for example
``programCount`` or ``sizeof(varying T)``), the resulting value may differ
across targets. In multi-target compilation, such values are rejected where a
single, target-independent constant is required (global initializers, template
args, etc.). This mirrors existing behavior for constant expressions with
``sizeof``.

Forward references and deferral
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Global initializers and default parameter values may call constexpr functions
defined later in the same file. The compiler defers evaluation of such
initializers until after parsing. This deferral does not apply to array sizes,
short vector lengths, or template arguments; those contexts still require
definitions to be visible before use.


Examples
--------

Uniform constexpr variable for array size:

.. code-block:: c++

    constexpr uniform int Tile = 16;
    uniform float buf[Tile];

Varying constexpr values:

.. code-block:: c++

    constexpr varying int lane = programIndex;
    constexpr varying int lane2 = lane * 2 + 1;

constexpr function used in a constant context:

.. code-block:: c++

    constexpr uniform int gcd(uniform int a, uniform int b) {
        while (b != 0) {
            uniform int t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    uniform int buf[gcd(48, 18)];

constexpr function in template args:

.. code-block:: c++

    constexpr uniform int width() { return 8; }

    template <int N>
    void foo(float<N> v);

    void bar(float<width()> v) {
        foo<width()>(v);
    }

Aggregate constexpr values:

.. code-block:: c++

    struct Pair { int a; int b; };
    constexpr uniform Pair p = { 1, 2 };
    constexpr uniform int pb = p.b;

    constexpr uniform int arr[3] = { 10, 20, 30 };
    constexpr uniform int a2 = arr[2];

Pointer constexpr values:

.. code-block:: c++

    uniform int g;
    constexpr uniform int *getp() { return &g; }
    constexpr uniform int *pg = getp();

Errors (v1):

.. code-block:: c++

    // Error: side effects in constexpr function.
    constexpr uniform int bad(uniform int x) {
        print("x=%d", x);
        return x;
    }

    // Error: pointer dereference is not allowed in constexpr evaluation.
    constexpr uniform int bad_ptr(uniform int *p) {
        return *p;
    }


Proposed docs/ispc.rst section
------------------------------

Constexpr
~~~~~~~~~

``constexpr`` declares variables and functions that are usable in
constant-expression contexts.

``constexpr`` variables
^^^^^^^^^^^^^^^^^^^^^^^

- ``constexpr`` implies ``const``.
- ``constexpr`` is not allowed on ``typedef`` declarations.
- The initializer must be a constant expression.
- V1 supports atomic, enum, pointer, short vector, array, and struct types.
- Arrays must have fully specified sizes.
- Pointer values are limited to null or addresses of globals/functions.
- ``constexpr`` values may be ``uniform`` or ``varying``. Varying constexpr
  values are lane-wise compile-time constants.

Examples:

.. code-block:: c++

    constexpr uniform int Tile = 16;
    uniform float buf[Tile];

    constexpr varying int lane = programIndex;
    constexpr varying int off = lane * 4;

``constexpr`` functions
^^^^^^^^^^^^^^^^^^^^^^^

``constexpr`` functions can be evaluated at compile time when called in a
constant-expression context. The function is still generated as normal code
and can be called at runtime.

Requirements (v1):

- The function body must be constexpr-suitable at definition time.
- The function must return a value (no ``void`` return in v1).
- The function may only call other ``constexpr`` functions.
- ``constexpr`` is not allowed on function parameters.
- Control flow conditions must be uniform.
- Pointer dereference is not allowed; address-of is limited to
  globals/functions (not locals/parameters).
- ``task``, ``export``, and ``extern "C"/"SYCL"`` are not allowed on
  ``constexpr`` functions in v1.
- Linkage follows C++ constexpr function behavior:
  - non-``static`` constexpr functions are emitted with ODR linkage
    (``linkonce_odr``)
  - ``static`` constexpr functions have internal linkage

Examples:

.. code-block:: c++

    constexpr uniform int gcd(uniform int a, uniform int b) {
        while (b != 0) {
            uniform int t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    uniform int buf[gcd(48, 18)];

Constant-expression contexts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Constant expressions are required in:

- Array sizes.
- Short vector lengths.
- ``switch`` case labels.
- Enumerator values.
- Non-type template arguments.
- Default parameter values.

``constexpr`` extends constant expressions by allowing constexpr function
calls in these contexts.

Forward references
^^^^^^^^^^^^^^^^^^

Global initializers and default parameter values may call ``constexpr``
functions defined later in the same file. Other constant-expression contexts
(array sizes, short vector lengths, template arguments) still require
definitions to be visible before use.

Template non-type arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Update the existing rule in the Function Template section:

- Old: "Integral constants, enumeration constants and template parameters
  can be used as non-type template arguments. Constant expressions are not
  allowed."
- New: "Integral constants, enumeration constants, constexpr expressions,
  and template parameters can be used as non-type template arguments."
