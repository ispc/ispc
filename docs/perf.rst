===========
Performance
===========

The SPMD programming model that ``ispc`` makes it easy to harness the
computational power available in SIMD vector units on modern CPUs, while
its basis in C makes it easy for programmers to adopt and use
productively.  This page summarizes the performance of ``ispc`` with the
workloads in the ``examples/`` directory of the ``ispc`` distribution.

These results were measured on a 4-core Apple iMac with a 4-core 3.4GHz
Intel® Core-i7 processor using the Intel® AVX instruction set.  The basis
for comparison is a reference C++ implementation compiled with gcc 4.2.1,
the version distributed with OS X 10.7.2.  (The reference implementation is
also included in the ``examples/`` directory.)

.. list-table:: Performance of ``ispc`` with a variety of the workloads
   from the ``examples/`` directory of the ``ispc`` distribution, compared
   a reference C++ implementation compiled with gcc 4.2.1.

  * - Workload
    - ``ispc``, 1 core
    - ``ispc``, 4 cores
  * - `AOBench`_ (512 x 512 resolution)
    - 6.19x
    - 28.06x
  * - `Binomial Options`_ (128k options)
    - 7.94x
    - 33.43x
  * - `Black-Scholes Options`_ (128k options)
    - 8.45x
    - 32.48x
  * - `Deferred Shading`_ (1280p)
    - 5.02x
    - 23.06x
  * - `Mandelbrot Set`_
    - 6.21x
    - 20.28x
  * - `Perlin Noise Function`_
    - 5.37x
    - n/a
  * - `Ray Tracer`_ (Sponza dataset)
    - 4.31x
    - 20.29x
  * - `3D Stencil`_
    - 4.05x
    - 15.53x
  * - `Volume Rendering`_
    - 3.60x
    - 17.53x


.. _AOBench: https://github.com/ispc/ispc/tree/master/examples/aobench
.. _Binomial Options: https://github.com/ispc/ispc/tree/master/examples/options
.. _Black-Scholes Options: https://github.com/ispc/ispc/tree/master/examples/options
.. _Deferred Shading: https://github.com/ispc/ispc/tree/master/examples/deferred
.. _Mandelbrot Set: https://github.com/ispc/ispc/tree/master/examples/mandelbrot_tasks
.. _Ray Tracer: https://github.com/ispc/ispc/tree/master/examples/rt
.. _Perlin Noise Function: https://github.com/ispc/ispc/tree/master/examples/noise
.. _3D Stencil: https://github.com/ispc/ispc/tree/master/examples/stencil
.. _Volume Rendering: https://github.com/ispc/ispc/tree/master/examples/volume_rendering


The following table shows speedups for a number of the examples on a
2.40GHz, 40-core Intel® Xeon E7-8870 system with the Intel® SSE4
instruction set, running Microsoft Windows Server 2008 Enterprise.  Here,
the serial C/C++ baseline code was compiled with MSVC 2010.
 
.. list-table:: Performance of ``ispc`` with a variety of the workloads
   from the ``examples/`` directory of the ``ispc`` distribution, on 
   system with 40 CPU cores.

  * - Workload
    - ``ispc``, 40 cores
  * - AOBench (2048 x 2048 resolution)
    - 182.36x
  * - Binomial Options (2m options)
    - 63.85x
  * - Black-Scholes Options (2m options)
    - 83.97x
  * - Ray Tracer (Sponza dataset)
    - 195.67x
  * - Volume Rendering
    - 243.18x


Notices & Disclaimers
=====================

Software and workloads used in performance tests may have been optimized for
performance only on Intel microprocessors.

Performance tests, such as SYSmark and MobileMark, are measured using specific
computer systems, components, software, operations and functions.  Any change
to any of those factors may cause the results to vary.  You should consult
other information and performance tests to assist you in fully evaluating your
contemplated purchases, including the performance of that product when combined
with other products.   For more complete information visit
www.intel.com/benchmarks.

Performance results are based on testing as of dates shown in configurations and
may not reflect all publicly available updates.  See backup for configuration
details.  No product or component can be absolutely secure.

Your costs and results may vary.

Intel technologies may require enabled hardware, software or service activation.

© Intel Corporation.  Intel, the Intel logo, and other Intel marks are
trademarks of Intel Corporation or its subsidiaries.  Other names and brands may
be claimed as the property of others.


Optimization Notice
===================

Intel's compilers may or may not optimize to the same degree for non-Intel
microprocessors for optimizations that are not unique to Intel microprocessors.
These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other
optimizations. Intel does not guarantee the availability, functionality, or
effectiveness of any optimization on microprocessors not manufactured by Intel.
Microprocessor-dependent optimizations in this product are intended for use with
Intel microprocessors. Certain optimizations not specific to Intel
microarchitecture are reserved for Intel microprocessors. Please refer to the
applicable product User and Reference Guides for more information regarding the
specific instruction sets covered by this notice.

