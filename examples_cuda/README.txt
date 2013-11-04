====================
ISPC Examples README
====================

This directory has a number of sample ispc programs.  Before building them
(on an system), install the appropriate ispc compiler binary into a
directory in your path.  Then, if you're running Windows, open the
"examples.sln" file and built from there.  For building under Linux/OSX,
there are makefiles in each directory that build the examples individually.

Almost all of them benchmark ispc implementations of the given computation
against regular serial C++ implementations, printing out a comparison of
the runtimes and the speedup delivered by ispc.  It may be instructive to
do a side-by-side diff of the C++ and ispc implementations of these
algorithms to learn more about wirting ispc code.

 
AOBench
=======

This is an ISPC implementation of the "AO bench" benchmark
(http://syoyo.wordpress.com/2009/01/26/ao-bench-is-evolving/).  The command
line arguments are:

ao (num iterations) (x res) (yres)

It executes the program for the given number of iterations, rendering an
(xres x yres) image each time and measuring the computation time with both
serial and ispc implementations.


AOBench_Instrumented
====================

This version of AO Bench is compiled with the --instrument ispc compiler
flag.  This causes the compiler to emit calls to a (user-supplied)
ISPCInstrument() function at interesting places in the compiled code.  An
example implementation of this function that counts the number of times the
callback is made and records some statistics about control flow coherence
is provided in the instrument.cpp file.


Deferred
========

This example shows an extensive example of using ispc for efficient
deferred shading of scenes with thousands of lights; it's an implementation
of the algorithm that Johan Andersson described at SIGGRAPH 2009,
implemented by Andrew Lauritzen and Jefferson Montgomery.  The basic idea
is that a pre-rendered G-buffer is partitioned into tiles, and in each
tile, the set of lights that contribute to the tile is first computed.
Then, the pixels in the tile are then shaded using just those light
sources. (See slides 19-29 of
http://s09.idav.ucdavis.edu/talks/04-JAndersson-ParallelFrostbite-Siggraph09.pdf
for more details on the algorithm.)

This directory includes three implementations of the algorithm:

- An ispc implementation that first does a static partitioning of the
  screen into tiles to parallelize across the CPU cores.  Within each tile
  ispc kernels provide highly efficient implementations of the light
  culling and shading calculations.
- A "best practices" serial C++ implementation.  This implementation does a
  dynamic partitioning of the screen, refining tiles with significant Z
  depth complexity (these tiles often have a large number of lights that
  affect them).  Within each final tile, the pixels are shaded using
  regular C++ code.
- If the Cilk extensions are available in your compiler, an ispc
  implementation that uses Cilk will also be built.
  (See http://software.intel.com/en-us/articles/intel-cilk-plus/).  Like 
  the "best practices" serial implementation, this version does dynamic
  tile partitioning for better load balancing and then uses ispc for the
  light culling and shading.


GMRES
=====

An implementation of the generalized minimal residual method for solving
sparse matrix equations.
(http://en.wikipedia.org/wiki/Generalized_minimal_residual_method)


Mandelbrot
==========

Mandelbrot set generation.  This example is extensively documented at the
http://ispc.github.com/example.html page.


Mandelbrot_tasks
================

Implementation of Mandelbrot set generation that also parallelizes across
cores using tasks.  Under Windows, a simple task system built on
Microsoft's Concurrency Runtime is used (see tasks_concrt.cpp).  On OSX, a
task system based on Grand Central Dispatch is used (tasks_gcd.cpp), and on
Linux, a pthreads-based task system is used (tasks_pthreads.cpp).  When
using tasks with ispc, no task system is mandated; the user is free to plug
in any task system they want, for ease of interoperating with existing task
systems.


Noise
=====

This example has an implementation of Ken Perlin's procedural "noise"
function, as described in his 2002 "Improving Noise" SIGGRAPH paper.

 
Options
=======

This program implements both the Black-Scholes and Binomial options pricing
models in both ispc and regular serial C++ code.


Perfbench
=========

This runs a number of microbenchmarks to measure system performance and
code generation quality.


RT
==

This is a simple ray tracer; it reads in camera parameters and a bounding
volume hierarchy and renders the scene from the given viewpoint.  The
command line arguments are:

rt <scene name base>

Where <scene base name> is one of "cornell", "teapot", or "sponza".

The implementation originally derives from the bounding volume hierarchy
and triangle intersection code from pbrt; see the pbrt source code and/or
"Physically Based Rendering" book for more about the basic algorithmic
details.


Simple
======

This is a simple "hello world" type program that shows a ~10 line
application program calling out to a ~5 line ispc program to do a simple
computation.

Sort
====
This is a bucket sort of 32 bit unsigned integers.
By default 1000000 random elements get sorted.
Call ./sort N in order to sort N elements instead.

Volume
======

Ray-marching volume rendering, with single scattering lighting model.  To
run it, specify a camera parameter file and a volume density file, e.g.:

volume camera.dat density_highres.vol

(See, e.g. Chapters 11 and 16 of "Physically Based Rendering" for
information about the algorithm implemented here.)  The volume data set
included here was generated by the example implementation of the "Wavelet
Turbulence for Fluid Simulation" SIGGRAPH 2008 paper by Kim et
al. (http://www.cs.cornell.edu/~tedkim/WTURB/)
