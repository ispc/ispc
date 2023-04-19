ISPC Examples README
====================

This directory has a number of sample ISPC programs ported to Xe. Before building them, install the appropriate `ispc`
compiler binary and runtime into a directory in your path.  Add ISPC binary to your PATH. Then, do the following:

```
mkdir build
cd build
cmake ../
```

Some of the benchmarks are running `ispc` for CPU/Xe and then regular serial C++ implementations, printing out execution
time.

Simple
======

This is the most basic example. It executes a simple kernel on target device (which can be a Xe GPU or CPU) and
demonstrates basics concepts of ISPC Runtime API (such as device, module, kernel, memory view).  It uses C++ API of ISPC
Runtime.

If no command line arguments are provided, the example chooses device to execute on automatically. It is possible to
force usage of concrete device using command line options:

`simple [ --cpu | --gpu ]`

Simple-USM
==========

This example corresponds to the Simple example, but uses shared memory mechanisms. The shared memory functionality in
Level Zero allows for allocating memory that is shared between the CPU and the GPU and forms Unified Shared Memory
(pointers valid on the CPU are also valid on the GPU). There is no need to explicitly copy data between the host and the
device. This is handled by the Level Zero.

The ISPC Runtime enables using the USM via Array type and provides an allocator that can be used in standard C++
containers, such as `std::vector`.


Simple-fence
============

This example shows how one can use ISPCRT to asynchronously compute something on GPU with other work being
done on CPU in parallel. It is derived from `Simple` example. The key difference is that `TaskQueue` object is not used
here. `CommandQueue` and `CommandList` objects are explicitly constructed here instead. Commands are submitted with
`copyToDevice`, `launch` and `copyToHost` methods of the command list object. It is important to notice that barriers
should be inserted explicitly if needed between computations or memory copying. It can be done with `barrier` method
of `CommandList` object. After filling the command list, `submit` method is called. It instructs GPU to execute
the submitted commands. This method returns a `Fence` object. It has two states: unsignalled and signalled. GPU change
the fence object state to signalled when execution is completed. Fence state is checked via `status` method. After
submission, host CPU thread computes the validation result effectively in parallel with GPU computation. After that,
fence object is waited in the loop until being signalled.

`host_simple-fence [--gpu | --cpu ]`


AOBench
=======

This is an ISPC implementation of the "AO bench" benchmark
(http://syoyo.wordpress.com/2009/01/26/ao-bench-is-evolving/).  The command line arguments are:

`ao (num iterations) (x resolution) (y resolution)`

This examples also demontrates usage of C interface of ISPC Runtime so you can see how to execute the same ISPC kernel
on CPU and GPU in a seamless way.

It executes the program for the given number of iterations, rendering an (xres x yres) image each time and measuring the
computation time with serial and ISPC implementations on CPU and Xe.


Mandelbrot
==========

Mandelbrot set generation. This example is extensively documented at the https://ispc.github.io/example.html page. The
comamnd line arguments are:

`mandelbrot [--scale=<factor>] [tasks iterations] [serial iterations]`

This examples also demonstrates usage of C++ interface of ISPC Runtime so you can see how to execute the same ISPC
kernel on CPU and GPU in a seamless way.

It executes the program for the given number of iterations, rendering an image of fixed size each time and measuring the
computation time with serial and ISPC implementations on CPU and Xe.  You can change scale of the image with `--scale`
option.


Noise
=====

This example has an implementation of Ken Perlin's procedural "noise" function, as described in his 2002 "Improving
Noise" SIGGRAPH paper. The command line arguments are:

`noise [niterations] [group threads width] [group threads height]`

This examples also demonstrates usage of C++ interface of ISPC Runtime so you can see how to execute the same ISPC
kernel on CPU and GPU in a seamless way.

It executes the program for the given number of iterations in particular thread space, rendering an image of fixed size
each time and measuring the computation time with serial and ISPC implementations on CPU and Xe.


SGEMM
=====

This program uses ISPC to implement naive version of matrix multiply.

The command line arguments are:

`sgemm (optional)[num iterations] (optional)[group threads width] (optional)[group threads height]`

This example demonstrate usage of pure Level Zero.


Simple-DPCPP
============

This simple example demonstrates a basic scenario of interoperability between ISPC and the oneAPI DPC++ Compiler. It
runs an ISPC kernel using ISPC Runtime and then creates a SYCL context using native Level Zero handles obtained from
ISPCRT.  Then it runs a corresponding SYCL kernel in SYCL. The results are compared to confirm that those are identical.

It requires oneAPI DPC++ Compiler.

To enable this example please configure the build of ISPC examples using the following command line:

```
cmake -DCMAKE_C_COMPILER=<dpcpp_path>/bin/clang -DCMAKE_CXX_COMPILER=<dpcpp_path>/bin/clang++ \
      -DISPC_INCLUDE_DPCPP_EXAMPLES=ON <examples source dir>
```

Running this example may require setting the `LD_LIBRARY_PATH` environmental variable to include oneAPI DPC++ Compiler
libraries.


Simple-DPCPP-L0
===============

This simple example demonstrates a basic scenario of interoperability between ISPC and the oneAPI DPC++ Compiler. It
runs an ISPC kernel in a Level Zero context and then a corresponding SYCL kernel in SYCL context created from the same
Level Zero context.  Then the results are compared to check if those are identical.  The key difference between this and
the previous example is that this one uses native Level Zero API then the previous one uses ISPCRT.

It requires oneAPI DPC++ Compiler.

To enable this example please configure the build of ISPC examples using the following command line:

```
cmake -DCMAKE_C_COMPILER=<dpcpp_path>/bin/clang -DCMAKE_CXX_COMPILER=<dpcpp_path>/bin/clang++ \
      -DISPC_INCLUDE_DPCPP_EXAMPLES=ON <examples source dir>
```

Running this example may require setting the `LD_LIBRARY_PATH` environmental variable to include oneAPI DPC++ Compiler
libraries.

Pipeline-DPCPP
==============

This example demonstrates how to create a pipeline of kernels in the ISPC and the oneAPI DPC++ Compiler that cooperate
working on a single problem represented by a memory region. The memory region is shared between the kernels, but it also
is shared between the CPU and the GPU. The Level Zero runtime takes care of the necessary data movements in an efficent
way and the user does not need to manage copying data to/from the GPU.

This example requires the oneAPI DPC++ Compiler.

To enable this example please configure the build of ISPC examples using the following command line:

```
cmake -DCMAKE_C_COMPILER=<dpcpp_path>/bin/clang -DCMAKE_CXX_COMPILER=<dpcpp_path>/bin/clang++ \
      -DISPC_INCLUDE_DPCPP_EXAMPLES=ON <examples source dir>
```

Running this example may require setting the `LD_LIBRARY_PATH` environmental variable to include oneAPI DPC++ Compiler
libraries.

Simple-ESIMD
============

This simple example demonstrates a basic scenario of interoperability between ISPC and Explicit SIMD SYCL* Extension. It
uses ISPC Runtime and runs an ISPC kernel which calls to ESIMD function.

It is required to use include interop.cmake file to your CMakeLists.txt if you want to use ISPC/ESIMD interoperability
feature:

`include(${ISPCRT_DIR}/interop.cmake)`

It requires oneAPI DPC++ Compiler.

To enable this example please configure the build of ISPC examples using the following command line:

```
cmake -DCMAKE_C_COMPILER=<dpcpp_path>/bin/clang -DCMAKE_CXX_COMPILER=<dpcpp_path>/bin/clang++ \
      -DISPC_INCLUDE_DPCPP_EXAMPLES=ON <examples source dir>
```

Running this example may require setting the `LD_LIBRARY_PATH` environmental variable to include oneAPI DPC++ Compiler
libraries.

vadd-esimd
==========

This vector add example demonstrates a basic scenario of interoperability between Explicit SIMD SYCL* Extension and
ISPC. It uses SYCL Runtime and runs an ESIMD kernel which calls to ISPC function.

It is required to use include interop.cmake file to your CMakeLists.txt if you want to use ISPC/ESIMD interoperability
feature:

`include(${ISPCRT_DIR}/interop.cmake)`

It requires oneAPI DPC++ Compiler.

To enable this example please configure the build of ISPC examples using the following command line:

```
cmake -DCMAKE_C_COMPILER=<dpcpp_path>/bin/clang -DCMAKE_CXX_COMPILER=<dpcpp_path>/bin/clang++ \
      -DISPC_INCLUDE_DPCPP_EXAMPLES=ON <examples source dir>
```

Running this example may require setting the `LD_LIBRARY_PATH` environmental variable to include oneAPI DPC++ Compiler
libraries.

callback-esimd
==============

This example demonstrates usage of callbacks between ISPC and Explicit SIMD SYCL* Extension by passing a pointer to an
ISPC function to ESIMD and calling that function from ESIMD.

It is required to use include interop.cmake file to your CMakeLists.txt if you want to use ISPC/ESIMD interoperability
feature:

`include(${ISPCRT_DIR}/interop.cmake)`

It requires oneAPI DPC++ Compiler.

To enable this example please configure the build of ISPC examples using the following command line:

```
cmake -DCMAKE_C_COMPILER=<dpcpp_path>/bin/clang -DCMAKE_CXX_COMPILER=<dpcpp_path>/bin/clang++ \
      -DISPC_INCLUDE_DPCPP_EXAMPLES=ON <examples source dir>
```

Running this example may require setting the `LD_LIBRARY_PATH` environmental variable to include oneAPI DPC++ Compiler
libraries.

invoke-sycl-aobench
===================

This is simple aobench-like example demonstrating call of SYCL from ISPC using binary and vISA linking.  It has host
part written in ISPC Runtime and uses `invoke_sycl` in ISPC code to call SYCL function.

The CMake provided will build both an ISPC-only version (`invoke_sycl_aobench_ispc`), SYCL reference version
(`aobench_sycl_bin`), and a ISPC/SYCL version (`invoke_sycl_aobench_ispc_sycl_bin` for binary linking and
`invoke_sycl_aobench_ispc_sycl_visa` for vISA linking)

It is required to use include interop.cmake file to your CMakeLists.txt if you want to use ISPC/SYCL interoperability
feature:

`include(${ISPCRT_DIR}/interop.cmake)`

It requires oneAPI DPC++ Compiler.

To enable this example please configure the build of ISPC examples using the following command line:

```
cmake -DCMAKE_C_COMPILER=<dpcpp_path>/bin/clang -DCMAKE_CXX_COMPILER=<dpcpp_path>/bin/clang++ \
      -DISPC_INCLUDE_DPCPP_EXAMPLES=ON <examples source dir>
```

Running the example requires setting of one environment variable:

```
IGC_ForceOCLSIMDWidth=<ISPC SIMD width> (which is set to 16 in CMakeLists.txt)
```

Running this example may also require setting the `LD_LIBRARY_PATH` environmental variable to include oneAPI DPC++
Compiler libraries.

ISPC/SYCL interop tests are target Gen12+ HW.

invoke-simd-vadd
================

This is simple vector_add example. It has host part written in SYCL and uses `invoke_simd` to call ISPC function with
simple SIMD CF for vector addition/substraction.

It is required to use include interop.cmake file to your CMakeLists.txt if you want to use ISPC/SYCL interoperability
feature:

`include(${ISPCRT_DIR}/interop.cmake)`

It requires oneAPI DPC++ Compiler.

To enable this example please configure the build of ISPC examples using the following command line:

```
cmake -DCMAKE_C_COMPILER=<dpcpp_path>/bin/clang -DCMAKE_CXX_COMPILER=<dpcpp_path>/bin/clang++ \
      -DISPC_INCLUDE_DPCPP_EXAMPLES=ON <examples source dir>
```

Running the example requires setting of two environment variables:

```
IGC_VCSaveStackCallLinkage=1
IGC_ForceOCLSIMDWidth=<ISPC SIMD width> (which is set to 16 in CMakeLists.txt)
```

Running this example may also require setting the `LD_LIBRARY_PATH` environmental variable to include oneAPI DPC++
Compiler libraries.

ISPC/SYCL interop tests are target Gen12+ HW.
