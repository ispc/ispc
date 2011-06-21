==============================
Intel(r) SPMD Program Compiler
==============================

Welcome to the Intel(r) SPMD Program Compiler (ispc)!  

ispc is a new compiler for "single program, multiple data" (SPMD)
programs. Under the SPMD model, the programmer writes a program that mostly
appears to be a regular serial program, though the execution model is
actually that a number of program instances execute in parallel on the
hardware. ispc compiles a C-based SPMD programming language to run on the
SIMD units of CPUs; it frequently provides a a 3x or more speedup on CPUs
with 4-wide SSE units, without any of the difficulty of writing intrinsics
code.

ispc is an open source compiler under the BSD license; see the file
LICENSE.txt.  ispc supports Windows, Mac, and Linux, with both x86 and
x86-64 targets. It currently supports the SSE2 and SSE4 instruction sets,
though support for AVX should be available soon.

For more information and examples, as well as a wiki and the bug database,
see the ispc distribution site, http://ispc.github.com.
