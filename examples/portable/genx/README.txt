====================
ISPC Examples README
====================

This directory has a number of sample ispc programs ported to GEN. Before building them,
install the appropriate ispc compiler binary and runtime into a directory in your path.
Add ISPC binary to your PATH. Then, do the following:
mkdir build
cd build
cmake ../

Some of the benchmarks are running ispc for CPU/GEN and then regular serial C++ implementations,
printing out execution time.

Simple
======

This is the most basic example. It executes a simple kernel on target device
(which can be a GEN GPU or CPU) and demonstrates basics concepts
of ISPC Runtime API (such as device, module, kernel, memory view).
It uses C++ API of ispcrt.

If no command line arguments are provided, the example chooses device
to execute on automatically. It is possible to force usage of concrete
device using command line options:

simple [ --cpu | --gpu ]


AOBench
=======

This is an ISPC implementation of the "AO bench" benchmark
(http://syoyo.wordpress.com/2009/01/26/ao-bench-is-evolving/).
The command line arguments are:

ao (num iterations) (x resolution) (y resolution)

This examples also demontrates usage of C interface of ispcrt so you can see how to
execute the same ISPC kernel on CPU and GPU in a semaless way.

It executes the program for the given number of iterations, rendering an
(xres x yres) image each time and measuring the computation time with
serial and ispc implementations on CPU and GEN.


Mandelbrot
==========

Mandelbrot set generation.  This example is extensively documented at the
http://ispc.github.com/example.html page. The comamnd line arguments are:
mandelbrot [--scale=<factor>] [tasks iterations] [serial iterations]

This examples also demontrates usage of C++ interface of ispcrt so you can see how to
execute the same ISPC kernel on CPU and GPU in a semaless way.

It executes the program for the given number of iterations, rendering an
image of fixed size each time and measuring the computation time with
serial and ispc implementations on CPU and GEN.
You can change scale of the image with --scale option.


Noise
=====

This example has an implementation of Ken Perlin's procedural "noise"
function, as described in his 2002 "Improving Noise" SIGGRAPH paper. The command
line arguments are:

noise [niterations] [group threads width] [group threads height]

This examples also demontrates usage of C++ interface of ispcrt so you can see how to
execute the same ISPC kernel on CPU and GPU in a semaless way.

It executes the program for the given number of iterations in particular
thread space, rendering an image of fixed size each time and measuring the
computation time with serial and ispc implementations on CPU and GEN.


SGEMM
=====
This program uses ISPC to implement naive version of matrix multiply. It also contains
CM implementation so if you have CM compiler installed you can compare ISPC/CM performance.

The command line arguments are:
sgemm (optional)[num iterations] (optional)[group threads width] (optional)[group threads height]

This example demonstrate usage of pure Level 0.


Simple-DPCPP
======================================
This simple example demonstrates a basic scanerio of interoperability between ISPC
and the oneAPI DPC++ Compiler. It runs an ISPC kernel in a Level Zero context and then
a corresponding SYCL kernel in SYCL context created from the same Level Zero context.
Then the results are compared to check if those are identical.

It requires oneAPI DPC++ Compiler.

To enable this example please configure the build of ISPC examples using the following
command line:

cmake -DCMAKE_C_COMPILER=<dpcpp_path>/bin/clang -DCMAKE_CXX_COMPILER=<dpcpp_path>/bin/clang++ \
      -DISPC_INCLUDE_DPCPP_EXAMPLES=ON <examples source dir>

Running this example may require setting the LD_LIBRARY_PATH environmental variable to include
oneAPI DPC++ Compiler libraries.


Pipeline-DPCPP
======================================
This example demonstrates how to create a pipeline of kernels in the ISPC
and the oneAPI DPC++ Compiler that cooperate working on a single problem represented
by a memory region. The memory region is shared between the kernels, but it also
is shared between the CPU and the GPU. The Level Zero runtime takes care
of the necessary data movements in an efficent way and the user does not need
to manage copying data to/from the GPU.

This example requires the oneAPI DPC++ Compiler.

To enable this example please configure the build of ISPC examples using the following
command line:

cmake -DCMAKE_C_COMPILER=<dpcpp_path>/bin/clang -DCMAKE_CXX_COMPILER=<dpcpp_path>/bin/clang++ \
      -DISPC_INCLUDE_DPCPP_EXAMPLES=ON <examples source dir>

Running this example may require setting the LD_LIBRARY_PATH environmental variable to include
oneAPI DPC++ Compiler libraries.
