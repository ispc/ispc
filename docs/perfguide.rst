=============================
Intel® ISPC Performance Guide
=============================

The SPMD programming model provided by ``ispc`` naturally delivers
excellent performance for many workloads thanks to efficient use of CPU
SIMD vector hardware.  This guide provides more details about how to get
the most out of ``ispc`` in practice.

* `Key Concepts`_

  + `Efficient Iteration With "foreach"`_
  + `Improving Control Flow Coherence With "foreach_tiled"`_
  + `Using Coherent Control Flow Constructs`_
  + `Use "uniform" Whenever Appropriate`_
  + `Use "Structure of Arrays" Layout When Possible`_

* `Tips and Techniques`_

  + `Understanding Gather and Scatter`_
  + `Avoid 64-bit Addressing Calculations When Possible`_
  + `Avoid Computation With 8 and 16-bit Integer Types`_
  + `Implementing Reductions Efficiently`_
  + `Using "foreach_active" Effectively`_
  + `Using Low-level Vector Tricks`_
  + `The "Fast math" Option`_
  + `"inline" Aggressively`_
  + `Avoid The System Math Library`_
  + `Declare Variables In The Scope Where They're Used`_
  + `Instrumenting Intel® ISPC Programs To Understand Runtime Behavior`_
  + `Choosing A Target Vector Width`_

* `Notices & Disclaimers`_

Key Concepts
============

This section describes the four most important concepts to understand and
keep in mind when writing high-performance ``ispc`` programs.  It assumes
good familiarity with the topics covered in the ``ispc`` `Users Guide`_.

.. _Users Guide: ispc.html

Efficient Iteration With "foreach"
----------------------------------

The ``foreach`` parallel iteration construct is semantically equivalent to
a regular ``for()`` loop, though it offers meaningful performance benefits.
(See the `documentation on "foreach" in the Users Guide`_ for a review of
its syntax and semantics.)  As an example, consider this simple function
that iterates over some number of elements in an array, doing computation
on each one:

.. _documentation on "foreach" in the Users Guide: ispc.html#parallel-iteration-statements-foreach-and-foreach-tiled

::

    export void foo(uniform int a[], uniform int count) {
        for (int i = programIndex; i < count; i += programCount) {
            // do some computation on a[i]
        }
    }

Depending on the specifics of the computation being performed, the code
generated for this function could likely be improved by modifying the code 
so that the loop only goes as far through the data as is possible to pack
an entire gang of program instances with computation each time through the
loop.  Doing so enables the ``ispc`` compiler to generate more efficient
code for cases where it knows that the execution mask is "all on".  Then,
an ``if`` statement at the end handles processing the ragged extra bits of
data that didn't fully fill a gang.

::

    export void foo(uniform int a[], uniform int count) {
        // First, just loop up to the point where all program instances
        // in the gang will be active at the loop iteration start
        uniform int countBase = count & ~(programCount-1);
        for (uniform int i = 0; i < countBase; i += programCount) {
            int index = i + programIndex;
            // do some computation on a[index]
        }
        // Now handle the ragged extra bits at the end
        int index = countBase + programIndex;
        if (index < count) {
            // do some computation on a[index]
        }
    }

While the performance of the above code will likely be better than the
first version of the function, the loop body code has been duplicated (or
has been forced to move into a separate utility function).

Using the ``foreach`` looping construct as below provides all of the
performance benefits of the second version of this function, with the
compactness of the first.

::

    export void foo(uniform int a[], uniform int count) {
        foreach (i = 0 ... count) {
            // do some computation on a[i]
        }
    }

Improving Control Flow Coherence With "foreach_tiled"
-----------------------------------------------------

Depending on the computation being performed, ``foreach_tiled`` may give
better performance than ``foreach``.  (See the `documentation in the Users
Guide`_ for the syntax and semantics of ``foreach_tiled``.)  Given a
multi-dimensional iteration like:

.. _documentation in the Users Guide: ispc.html#parallel-iteration-statements-foreach-and-foreach-tiled

::

    foreach (i = 0 ... width, j = 0 ... height) {
        // do computation on element (i,j)
    }

if the ``foreach`` statement is used, elements in the gang of program
instances will be mapped to values of ``i`` and ``j`` by taking spans of
``programCount`` elements across ``i`` with a single value of ``j``.  For
example, the ``foreach`` statement above roughly corresponds to:

::

    for (uniform int j = 0; j < height; ++j)
        for (int i = 0; i < width; i += programCount) {
            // do computation 
    }

When a multi-dimensional domain is being iterated over, ``foreach_tiled``
statement maps program instances to data in a way that tries to select
square n-dimensional segments of the domain.  For example, on a compilation
target with 8-wide gangs of program instances, it generates code that
iterates over the domain the same way as the following code (though more
efficiently):

::

    for (int j = programIndex/4; j < height; j += 2)
        for (int i = programIndex%4; i < width; i += 4) {
            // do computation 
    }

Thus, each gang of program instances operates on a 2x4 tile of the domain.
With higher-dimensional iteration and different gang sizes, a similar
mapping is performed--e.g. for 2D iteration with a 16-wide gang size, 4x4
tiles are iterated over; for 4D iteration with a 8-gang, 1x2x2x2 tiles are
processed, and so forth.  

Performance benefit can come from using ``foreach_tiled`` in that it
essentially optimizes for the benefit of iterating over *compact* regions
of the domain (while ``foreach`` iterates over the domain in a way that
generally allows linear memory access.)  There are two benefits from
processing compact regions of the domain.  

First, it's often the case that the control flow coherence of the program
instances in the gang is improved; if data-dependent control flow decisions
are related to the values of the data in the domain being processed, and if
the data values have some coherence, iterating with compact regions will
improve control flow coherence.

Second, processing compact regions may mean that the data accessed by
program instances in the gang is more coherent, leading to performance
benefits from better cache hit rates.

As a concrete example, for the ray tracer example in the ``ispc``
distribution (in the ``examples/rt`` directory), performance is 20% better
when the pixels are iterated over using ``foreach_tiled`` than ``foreach``,
because more coherent regions of the scene are accessed by the set of rays
in the gang of program instances.


Using Coherent Control Flow Constructs
--------------------------------------

Recall from the ``ispc`` Users Guide, in the `SPMD-on-SIMD Execution Model
section`_ that ``if`` statements with a ``uniform`` test compile to more
efficient code than ``if`` tests with varying tests.  The coherent ``cif``
statement can provide many benefits of ``if`` with a uniform test in the
case where the test is actually varying.

.. _SPMD-on-SIMD Execution Model section: ispc.html#the-spmd-on-simd-execution-model

In this case, the code the compiler generates for the ``if``
test is along the lines of the following pseudo-code:

::

   bool expr = /* evaluate cif condition */
   if (all(expr)) {
       // run "true" case of if test only
   } else if (!any(expr)) {
       // run "false" case of if test only
   } else {
       // run both true and false cases, updating mask appropriately
   }

For ``if`` statements where the different running SPMD program instances
don't have coherent values for the boolean ``if`` test, using ``cif``
introduces some additional overhead from the ``all`` and ``any`` tests as
well as the corresponding branches.  For cases where the program
instances often do compute the same boolean value, this overhead is
worthwhile.  If the control flow is in fact usually incoherent, this
overhead only costs performance.

In a similar fashion, ``ispc`` provides ``cfor``, ``cwhile``, and ``cdo``
statements.  These statements are semantically the same as the
corresponding non-"c"-prefixed functions.

Use "uniform" Whenever Appropriate
----------------------------------

For any variable that will always have the same value across all of the
program instances in a gang, declare the variable with the  ``uniform``
qualifier.  Doing so enables the ``ispc`` compiler to emit better code in
many different ways.

As a simple example, consider a ``for`` loop that always does the same
number of iterations:

::

    for (int i = 0; i < 10; ++i)
        // do something ten times

If this is written with ``i`` as a ``varying`` variable, as above, there's
additional overhead in the code generated for the loop as the compiler
emits instructions to handle the possibility of not all program instances
following the same control flow path (as might be the case if the loop
limit, 10, was itself a ``varying`` value.)

If the above loop is instead written with ``i`` ``uniform``, as:

::

    for (uniform int i = 0; i < 10; ++i)
        // do something ten times

Then better code can be generated (and the loop possibly unrolled).

In some cases, the compiler may be able to detect simple cases like these,
but it's always best to provide the compiler with as much help as possible
to understand the actual form of your computation.


Use "Structure of Arrays" Layout When Possible
----------------------------------------------

In general, memory access performance (for both reads and writes) is best
when the running program instances access a contiguous region of memory; in
this case efficient vector load and store instructions can often be used
rather than gathers and scatters.  As an example of this issue, consider an
array of a simple point datatype laid out and accessed in conventional
"array of structures" (AOS) layout:

::

    struct Point { float x, y, z; };
    uniform Point pts[...];
    float v = pts[programIndex].x;

In the above code, the access to ``pts[programIndex].x`` accesses
non-sequential memory locations, due to the ``y`` and ``z`` values between
the desired ``x`` values in memory.  A "gather" is required to get the
value of ``v``, with a corresponding decrease in performance.

If ``Point`` was defined as a "structure of arrays" (SOA) type, the access
can be much more efficient:

::

    struct Point8 { float x[8], y[8], z[8]; };
    uniform Point8 pts8[...];
    int majorIndex = programIndex / 8;
    int minorIndex = programIndex % 8;
    float v = pts8[majorIndex].x[minorIndex];

In this case, each ``Point8`` has 8 ``x`` values contiguous in memory
before 8 ``y`` values and then 8 ``z`` values.  If the gang size is 8 or
less, the access for ``v`` will have the same value of ``majorIndex`` for
all program instances and will access consecutive elements of the ``x[8]``
array with a vector load.  (For larger gang sizes, two 8-wide vector loads
would be issues, which is also quite efficient.)

However, the syntax in the above code is messy; accessing SOA data in this
fashion is much less elegant than the corresponding code for accessing the
data with AOS layout.  The ``soa`` qualifier in ``ispc`` can be used to
cause the corresponding transformation to be made to the ``Point`` type,
while preserving the clean syntax for data access that comes with AOS
layout:

::

    soa<8> Point pts[...]; 
    float v = pts[programIndex].x;

Thanks to having SOA layout a first-class concept in the language's type
system, it's easy to write functions that convert data between the
layouts.  For example, the ``aos_to_soa`` function below converts ``count``
elements of the given ``Point`` type from AOS to 8-wide SOA layout.  (It
assumes that the caller has pre-allocated sufficient space in the
``pts_soa`` output array.

::

    void aos_to_soa(uniform Point pts_aos[], uniform int count,
                    soa<8> pts_soa[]) {
         foreach (i = 0 ... count)
             pts_soa[i] = pts_aos[i];
    }

Analogously, a function could be written to convert back from SOA to AOS if
needed.


Tips and Techniques
===================

This section introduces a number of additional techniques that are worth
keeping in mind when writing ``ispc`` programs.

Understanding Gather and Scatter
--------------------------------

Memory reads and writes from the program instances in a gang that access
irregular memory locations (rather than a consecutive set of locations, or
a single location) can be relatively inefficient.  As an example, consider
the "simple" array indexing calculation below:

::

    int i = ....;
    uniform float x[10] = { ... };
    float f = x[i];

Since the index ``i`` is a varying value, the program instances in the gang
will in general be reading different locations in the array ``x``.  Because
not all CPUs have a "gather" instruction, the ``ispc`` compiler has to
serialize these memory reads, performing a separate memory load for each
running program instance, packing the result into ``f``.  (The analogous
case happens for a write into ``x[i]``.)

In many cases, gathers like these are unavoidable; the program instances
just need to access incoherent memory locations.  However, if the array
index ``i`` actually has the same value for all of the program instances or
if it represents an access to a consecutive set of array locations, much
more efficient load and store instructions can be generated instead of
gathers and scatters, respectively.

In many cases, the ``ispc`` compiler is able to deduce that the memory
locations accessed by a varying index are either all the same or are
uniform.  For example, given:

::

  uniform int x = ...;
  int y = x;
  return array[y];

The compiler is able to determine that all of the program instances are
loading from the same location, even though ``y`` is not a ``uniform``
variable.  In this case, the compiler will transform this load to a regular
vector load, rather than a general gather.

Sometimes the running program instances will access a linear sequence of
memory locations; this happens most frequently when array indexing is done
based on the built-in ``programIndex`` variable.  In many of these cases,
the compiler is also able to detect this case and then do a vector load.
For example, given:

::

    for (int i = programIndex; i < count; i += programCount)
      // process array[i];

Regular vector loads and stores are issued for accesses to ``array[i]``.

Both of these cases have been ones where the compiler is able to determine
statically that the index has the same value at compile-time.  It's 
often the case that this determination can't be made at compile time, but
this is often the case at run time.  The ``reduce_equal()`` function from
the standard library can be used in this case; it checks to see if the
given value is the same across over all of the running program instances,
returning true and its ``uniform`` value if so.

The following function shows the use of ``reduce_equal()`` to check for an
equal index at execution time and then either do a scalar load and
broadcast or a general gather.

::

    uniform float array[..] = { ... };
    float value;
    int i = ...;
    uniform int ui;
    if (reduce_equal(i, &ui) == true)
        value = array[ui]; // scalar load + broadcast
    else
        value = array[i];  // gather

For a simple case like the one above, the overhead of doing the
``reduce_equal()`` check is likely not worthwhile compared to just always
doing a gather.  In more complex cases, where a number of accesses are done
based on the index, it can be worth doing.  See the example
``examples/volume_rendering`` in the ``ispc`` distribution for the use of
this technique in an instance where it is beneficial to performance.

Understanding Memory Read Coalescing
------------------------------------

XXXX todo


Avoid 64-bit Addressing Calculations When Possible
--------------------------------------------------

Even when compiling to a 64-bit architecture target, ``ispc`` does many of
the addressing calculations in 32-bit precision by default--this behavior
can be overridden with the ``--addressing=64`` command-line argument.  This
option should only be used if it's necessary to be able to address over 4GB
of memory in the ``ispc`` code, as it essentially doubles the cost of
memory addressing calculations in the generated code.

Avoid Computation With 8 and 16-bit Integer Types
-------------------------------------------------

The code generated for 8 and 16-bit integer types is generally not as
efficient as the code generated for 32-bit integer types.  It is generally
worthwhile to use 32-bit integer types for intermediate computations, even
if the final result will be stored in a smaller integer type.

Implementing Reductions Efficiently
-----------------------------------

It's often necessary to compute a reduction over a data set--for example,
one might want to add all of the values in an array, compute their minimum,
etc.  ``ispc`` provides a few capabilities that make it easy to efficiently
compute reductions like these.  However, it's important to use these
capabilities appropriately for best results.

As an example, consider the task of computing the sum of all of the values
in an array.  In C code, we might have:

::

    /* C implementation of a sum reduction */
    float sum(const float array[], int count) {
        float sum = 0;
        for (int i = 0; i < count; ++i)
            sum += array[i];
        return sum;
    } 

Exactly this computation could also be expressed as a purely uniform
computation in ``ispc``, though without any benefit from vectorization:

::

    /* inefficient ispc implementation of a sum reduction */
    uniform float sum(const uniform float array[], uniform int count) {
        uniform float sum = 0;
        for (uniform int i = 0; i < count; ++i)
            sum += array[i];
        return sum;
    } 

As a first try, one might try using the ``reduce_add()`` function from the
``ispc`` standard library; it takes a ``varying`` value and returns the sum
of that value across all of the active program instances.

::

    /* inefficient ispc implementation of a sum reduction */
    uniform float sum(const uniform float array[], uniform int count) {
        uniform float sum = 0;
        foreach (i = 0 ... count)
            sum += reduce_add(array[i]);
        return sum;
    } 

This implementation loads a gang's worth of values from the array, one for
each of the program instances, and then uses ``reduce_add()`` to reduce
across the program instances and then update the sum.  Unfortunately this
approach loses most benefit from vectorization, as it does more work on the
cross-program instance ``reduce_add()`` call than it saves from the vector
load of values.

The most efficient approach is to do the reduction in two phases: rather
than using a ``uniform`` variable to store the sum, we maintain a varying
value, such that each program instance is effectively computing a local
partial sum on the subset of array values that it has loaded from the
array.  When the loop over array elements concludes, a single call to
``reduce_add()`` computes the final reduction across each of the program
instances' elements of ``sum``.  This approach effectively compiles to a
single vector load and a single vector add for each loop iteration's of
values--very efficient code in the end.

::

    /* good ispc implementation of a sum reduction */
    uniform float sum(const uniform float array[], uniform int count) {
        float sum = 0;
        foreach (i = 0 ... count)
            sum += array[i];
        return reduce_add(sum);
    } 

Using "foreach_active" Effectively
----------------------------------

For high-performance code,

For example, consider this segment of code, from the introduction of
``foreach_active`` in the ispc User's Guide:

::

    uniform float array[...] = { ... };    
    int index = ...;
    foreach_active (i) {
        ++array[index];
    }  

Here, ``index`` was assumed to possibly have the same value for multiple
program instances, so the updates to ``array[index]`` are serialized by the
``foreach_active`` statement in order to not have undefined results when
``index`` values do collide.

The code generated by the compiler can be improved  in this case by making
it clear that only a single element of the array is accessed by
``array[index]`` and that thus a general gather or scatter isn't required.
Specifically, by using the ``extract()`` function from the standard library
to extract the current program instance's value of ``index`` into a
``uniform`` variable and then using that to index into ``array``, as below,
more efficient code is generated.

::

    foreach_active (instanceNum) {
        uniform int unifIndex = extract(index, instanceNum);
        ++array[unifIndex];
    }


Using Low-level Vector Tricks
-----------------------------

Many low-level Intel® SSE and AVX coding constructs can be implemented in
``ispc`` code.  The ``ispc`` standard library functions ``intbits()`` and
``floatbits()`` are often useful in this context.  Recall that
``intbits()`` takes a ``float`` value and returns it as an integer where
the bits of the integer are the same as the bit representation in memory of
the ``float``.  (In other words, it does *not* perform an integer to
floating-point conversion.)  ``floatbits()``, then, performs the inverse
computation.

As an example of the use of these functions, the following code efficiently
reverses the sign of the given values.

::

  float flipsign(float a) {
      unsigned int i = intbits(a);
      i ^= 0x80000000;
      return floatbits(i);
  }

This code compiles down to a single XOR instruction.

The "Fast math" Option
----------------------

``ispc`` has a ``--opt=fast-math`` command-line flag that enables a number of
optimizations that may be undesirable in code where numerical precision is
critically important.  For many graphics applications, for example, the
approximations introduced may be acceptable, however.  The following two
optimizations are performed when ``--opt=fast-math`` is used.  By default, the
``--opt=fast-math`` flag is off.

* Expressions like ``x / y``, where ``y`` is a compile-time constant, are
  transformed to ``x * (1./y)``, where the inverse value of ``y`` is
  precomputed at compile time.

* Expressions like ``x / y``, where ``y`` is not a compile-time constant,
  are transformed to ``x * rcp(y)``, where ``rcp()`` maps to the
  approximate reciprocal instruction from the ``ispc`` standard library.


"inline" Aggressively
---------------------

Inlining functions aggressively is generally beneficial for performance
with ``ispc``.  Definitely use the ``inline`` qualifier for any short
functions (a few lines long), and experiment with it for longer functions.

Avoid The System Math Library
-----------------------------

The default math library for transcendentals and the like in ``ispc`` has
higher error than the system's math library, though is much more efficient
due to being vectorized across the program instances and due to the fact
that the functions can be inlined in the final code.  (It generally has
errors in the range of 10ulps, while the system math library generally has
no more than 1ulp of error for transcendentals.)

If the ``--math-lib=system`` command-line option is used when compiling an
``ispc`` program, then calls to the system math library will be generated
instead.  This option should only be used if the higher precision is
absolutely required as the performance impact of using it can be
significant.

Declare Variables In The Scope Where They're Used
-------------------------------------------------

Performance is slightly improved by declaring variables at the same block
scope where they are first used.  For example, in code like the
following, if the lifetime of ``foo`` is only within the scope of the
``if`` clause, write the code like this:  

::

    float func() {
        ....
        if (x < y) {
            float foo;
            ... use foo ...
        }
    }

Try not to write code as:

::

    float func() {
        float foo;
        ....
        if (x < y) {
            ... use foo ...
        }
    }

Doing so can reduce the amount of masked store instructions that the
compiler needs to generate.

Instrumenting Intel® ISPC Programs To Understand Runtime Behavior
-----------------------------------------------------------------

``ispc`` has an optional instrumentation feature that can help you
understand performance issues.  If a program is compiled using the
``--instrument`` flag, the compiler emits calls to a function with the
following signature at various points in the program (for
example, at interesting points in the control flow, when scatters or
gathers happen.)

::

    extern "C" {
        void ISPCInstrument(const char *fn, const char *note, 
                            int line, uint64_t mask);
    }

This function is passed the file name of the ``ispc`` file running, a short
note indicating what is happening, the line number in the source file, and
the current mask of active program instances in the gang.  You must provide an
implementation of this function and link it in with your application.

For example, when the ``ispc`` program runs, this function might be called
as follows:

::

   ISPCInstrument("foo.ispc", "function entry", 55, 0xfull);

This call indicates that at the currently executing program has just
entered the function defined at line 55 of the file ``foo.ispc``, with a
mask of all lanes currently executing (assuming a four-wide gang size
target machine).

For a fuller example of the utility of this functionality, see
``examples/aobench_instrumented`` in the ``ispc`` distribution.  This
example includes an implementation of the ``ISPCInstrument()`` function
that collects aggregate data about the program's execution behavior.

When running this example, you will want to direct to the ``ao`` executable
to generate a low resolution image, because the instrumentation adds
substantial execution overhead.  For example:

::

    % ./ao 1 32 32

After the ``ao`` program exits, a summary report along the following lines
will be printed.  In the first few lines, you can see how many times a few
functions were called, and the average percentage of SIMD lanes that were
active upon function entry.

:: 

    ao.ispc(0067) - function entry: 342424 calls (0 / 0.00% all off!), 95.86% active lanes
    ao.ispc(0067) - return: uniform control flow: 342424 calls (0 / 0.00% all off!), 95.86% active lanes
    ao.ispc(0071) - function entry: 1122 calls (0 / 0.00% all off!), 97.33% active lanes
    ao.ispc(0075) - return: uniform control flow: 1122 calls (0 / 0.00% all off!), 97.33% active lanes
    ao.ispc(0079) - function entry: 10072 calls (0 / 0.00% all off!), 45.09% active lanes
    ao.ispc(0088) - function entry: 36928 calls (0 / 0.00% all off!), 97.40% active lanes
    ...


Choosing A Target Vector Width
------------------------------

By default, ``ispc`` compiles to the natural vector width of the target
instruction set.  For example, for SSE2 and SSE4, it compiles four-wide,
and for AVX, it complies 8-wide.  For some programs, higher performance may
be seen if the program is compiled to a doubled vector width--8-wide for
SSE and 16-wide for AVX.  

For workloads that don't require many of registers, this method can lead to
significantly more efficient execution thanks to greater instruction level
parallelism and amortization of various overhead over more program
instances.  For other workloads, it may lead to a slowdown due to higher
register pressure; trying both approaches for key kernels may be
worthwhile.

This option is only available for each of the SSE2, SSE4 and AVX targets.
It is selected with the ``--target=sse2-x2``, ``--target=sse4-x2`` and
``--target=avx-x2`` options, respectively.


Notices & Disclaimers
=====================

Performance varies by use, configuration and other factors. Learn more at
www.intel.com/PerformanceIndex.
