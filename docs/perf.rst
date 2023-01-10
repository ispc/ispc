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


.. _AOBench: https://github.com/ispc/ispc/tree/main/examples/cpu/aobench
.. _Binomial Options: https://github.com/ispc/ispc/tree/main/examples/cpu/options
.. _Black-Scholes Options: https://github.com/ispc/ispc/tree/main/examples/cpu/options
.. _Deferred Shading: https://github.com/ispc/ispc/tree/main/examples/cpu/deferred
.. _Mandelbrot Set: https://github.com/ispc/ispc/tree/main/examples/cpu/mandelbrot_tasks
.. _Ray Tracer: https://github.com/ispc/ispc/tree/main/examples/cpu/rt
.. _Perlin Noise Function: https://github.com/ispc/ispc/tree/main/examples/cpu/noise
.. _3D Stencil: https://github.com/ispc/ispc/tree/main/examples/cpu/stencil
.. _Volume Rendering: https://github.com/ispc/ispc/tree/main/examples/cpu/volume_rendering


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

Performance varies by use, configuration and other factors. Learn more at
www.intel.com/PerformanceIndex.
