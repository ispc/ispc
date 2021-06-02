===================
Intel® ISPC for GEN
===================

The Intel® Implicit SPMD Program Compiler (Intel® ISPC) is actively developed to support
latest Intel GPUs. The compilation for a GPU is pretty straightforward from
the user's point of view, but managing the execution of code on a GPU may add
complexity. You can use a low-level API `oneAPI Level Zero
<https://spec.oneapi.com/level-zero/latest/index.html>`_ to manage available GPU
devices, memory transfers between CPU and GPU, code execution, and
synchronization. Another possibility is to use `ISPC Run Time (ISPCRT)`_,
which is part of the ISPC package, to manage that complexity and create
a unified abstraction for executing tasks on CPU and GPU.

Contents:

* `Using The ISPC Compiler`_

  + `Environment`_
  + `Basic Command-line Options`_

* `ISPC Run Time (ISPCRT)`_

  + `ISPCRT Objects`_
  + `Execution Model`_
  + `Configuration`_

* `Compiling and Running Simple ISPC Program`_

* `Language Limitations and Known Issues`_

* `Performance`_

  + `Performance Guide for GPU Programming`_

* `FAQ`_

  + `How to Get an Assembly File from SPIR-V?`_
  + `How to Debug on GPU?`_

Using The ISPC Compiler
=======================

The output from ``ispc`` for GEN targets is SPIR-V file by default. It is used
when one of ``genx`` targets is selected:

::

   ispc foo.ispc --target=genx-x8 -o foo.spv

The SPIR-V file is consumed by the runtime for further compilation and execution
on GPU.

You can also generate L0 binary using ``--emit-zebin`` flag. Please note that
currently SPIR-V format is more stable but feel free to experiment with L0 binary.
L0 binary format is supported now on Linux platform only.

Environment
-----------
``Intel® ISPC for GEN`` is supported on Linux for quite a while (recommended
and tested Linux distribution is Ubuntu 20.04) and it's got Windows support since
v1.16.0.

You need to have a system with ``Intel(R) Processor Graphics Gen9`` or later.

For the execution of ISPC programs on GPU, please install `Intel(R)
Graphics Compute Runtime <https://github.com/intel/compute-runtime/releases>`_
and `Level Zero Loader <https://github.com/oneapi-src/level-zero/releases>`_.

To use ISPC Run Time for CPU you need to have ``OpenMP runtime`` installed on
your system. Consult your Linux distribution documentation for the installation
of OpenMP runtime instructions.


Basic Command-line Options
--------------------------

Two new targets were introduced for GPU support: ``genx-x8`` and ``genx-x16``.

If the ``-o`` flag is given, ``ispc`` will generate a SPIR-V output file.
Optionally you can use ``--emit-spirv`` flag:

::

   ispc --target=genx-x8 --emit-spirv foo.ispc -o foo.spv

To generate L0 binary, use ``--emit-zebin`` flag. When you use L0 binary you may
want to pass some additional options to the vector backend. You can do this using
``--vc-options`` flag.

Also two new ``arch`` options were introduced: ``genx32`` and ``genx64``.
``genx64`` is default and corresponds to 64-bit host and has 64-bit pointer size,
``genx32`` corresponds to 32-bit host and has 32-bit pointer size.

To generate LLVM bitcode, use the ``--emit-llvm`` flag.
To generate LLVM bitcode in textual form, use the ``--emit-llvm-text`` flag.

Optimizations are on by default; they can be turned off with ``-O0``.

Generating a text assembly file using ``--emit-asm`` is not supported yet.
See `How to Get an Assembly File from SPIR-V?`_ section about how to get the
assembly from SPIR-V file.

By default, 64-bit addressing is used. You can change it to 32-bit addressing by
using ``--addressing=32`` or ``--arch=genx32`` however pointer size should be
the same for host and device code so 32-bit addressing will only work with
32-bit host programs.

ISPC Run Time (ISPCRT)
======================

``ISPC Run Time (ISPCRT)`` unifies execution models for CPU and GPU targets. It
is a high-level abstraction on the top of `oneAPI Level Zero
<https://spec.oneapi.com/level-zero/latest/index.html>`_. You can continue
using ISPC for CPU without this runtime and alternatively use pure ``oneAPI
Level Zero`` for GPU. However, we strongly encourage you to try ``ISPCRT``
and give us feedback!
The ``ISPCRT`` provides C and C++ APIs which are documented in the header files
(see ``ispcrt.h`` and ``ispcrt.hpp``) and distributed as a library that you can
link to.
Examples in ``ispc/examples/xpu`` directory demonstrate how to use
this API to run SPMD programs on CPU or GPU. You can see how to use
``oneAPI Level Zero`` runtime in ``sgemm`` example.
It is also possible to run ISPC kernels and DPCPP kernels written with ``oneAPI
DPC++ Compiler`` using ``oneAPI Level Zero`` from the same process and share data
between them. Try ``Simple-DPCPP`` and ``Pipeline-DPCPP`` examples to learn
more about this possibility. Please keep in mind though that this
feature is experimental.

ISPCRT Objects
--------------

The ``ISPC Run Time`` uses the following abstractions to manage code execution:

* ``Device`` - represents a CPU or a GPU that can execute SPMD program and has
  some operational memory available. The user may select a particular type of
  device (CPU or GPU) or allow the runtime to decide which device will be used.

* ``Memory view`` - represents data that need to be accessed by different
  ``devices``. For example, input data for code running on GPU must be firstly
  prepared by a CPU in its memory, then transferred to a GPU memory to perform
  computations on. ``Memory view`` can also represent memory allocated using
  a Unified Shared Memory mechanism provided by ``oneAPI Level Zero``. Pointers
  to data allocated in the USM are valid both on the host and on the device.
  Also, there is no need to explicitly handle data movement between the CPU
  and the GPU. This is handled automatically by the ``oneAPI Level Zero`` runtime.

* ``Task queue`` - Each ``device`` has a task (command) queue and executes
  commands from it. The execution may be asynchronous, which means that subsequent
  commands can begin executing before the previous ones complete. There are
  synchronization primitives available to make the execution synchronous.

* ``Barrier`` - synchronization primitive that can be inserted into
  a ``task queue`` to make sure that all tasks previously inserted into this
  queue have completed execution.

* ``Module`` - represents a set of ``kernels`` that are compiled together and
  thus can share some common code. In this sense, SPIR-V file produced by ``ispc``
  is a ``module`` for the ``ISPCRT``.

* ``Kernel`` - is a function that is an entry point to a ``module`` and can be
  called by inserting kernel execution command into a ``task queue``. A kernel
  has one parameter - a pointer to a structure of actual kernel parameters.

* ``Future`` - can be treated as a promise that at some point ``kernel``
  execution connected to this object will be completed and the object will become
  valid.
  ``Futures`` are returned when a ``kernel`` invocation is inserted into
  a ``task queue``. When the ``task queue`` is executed on a device, the
  ``future`` object becomes valid and can be used to retrieve information about
  the ``kernel`` execution.

* ``Array`` - Conveniently wraps up memory view objects and allows for easy
  allocation of memory on the device or in the Unified Shared Memory (USM).
  The ISPCRT also provides an example allocator that makes it even more simple
  to allocate data in the USM and a SharedVector class that serves the same
  purpose. See XPU examples and documentation for more details.

All ``ISPCRT`` objects support reference counting, which means that it is not
necessary to perform detailed memory management. The objects will be released
once they are not used.

Execution Model
---------------

The idea of `ISPC tasks
<https://ispc.github.io/ispc.html#task-parallelism-launch-and-sync-statements>`_
has been extended to support the execution of kernels on a GPU. Each kernel
execution command inserted into a task queue is parametrized with the number
of tasks (threads) that should be launched on a GPU. Each task must decide
on which part of the problem it should work, exactly the same as it happens
in the CPU case. Within tasks, the program executes in SPMD manner (again
the regular ISPC execution model is copied). All built-in variables used for
that purpose (such as ``taskIndex``, ``taskCount``, ``programIndex``,
``programCount``) are available for use on GPU.

Configuration
-------------

The behavior of ``ISPCRT`` can be configured using the following environment
variables:

* ``ISPCRT_USE_ZEBIN`` - use experimental L0 native binary format.
  Unlike SPIR-V files, zebin files are not portable between different GPU types.

* ``ISPCRT_IGC_OPTIONS`` - ``ISPCRT`` is using an Intel® Graphics Compiler (IGC)
  to produce binary code that can be executed on the GPU. ``ISPCRT`` allows
  for passing certain options to the IGC via ``ISPCRT_IGC_OPTIONS`` variable.
  The content of this variable should be prefixed with ``+`` or ``=`` sign.
  ``+`` means that the content of the variable should be added to the default
  IGC options already passsed by the ``ISPCRT``, while ``=`` tells the ``ISPCRT``
  to replace the default options with the content of the environment variable.

* ``ISPCRT_GPU_DEVICE`` - if more than one supported GPU is present in the system,
  the user can select the GPU device to be used by the ``ISPCRT`` using ``ISPCRT_GPU_DEVICE``
  variable. It should be set to a number of a device as enumerated
  by the Level Zero runtime. For example, in a system with two GPUs present,
  the variable can be set to ``0`` or ``1``.

* ``ISPCRT_MAX_KERNEL_LAUNCHES`` - there is a limit of the maximum number of enqueued
  kernel launches in a given task queue. If the limit is reached, sync() method
  needs to be called to submit the queue for execution. The limit is currently
  set to 100000, but can be lowered (for example for testing) using this environmental variable.
  Please note that the limit cannot be set to more than 100000. If a greater value is provided,
  the ``ISPCRT`` will set the limit to the default value and display a warning message.

Compiling and Running Simple ISPC Program
=========================================
The directory ``examples/xpu/simple`` in the ``ispc`` distribution
includes a simple example of how to use ``ispc`` with a short C++ program for
CPU and GPU targets with ISPC Run Time. See the file ``simple.ispc`` in that
directory (also reproduced here.)

::

  struct Parameters {
      float *vin;
      float *vout;
      int    count;
  };

  task void simple_ispc(void *uniform _p) {
      Parameters *uniform p = (Parameters * uniform) _p;

      foreach (index = 0 ... p->count) {
          // Load the appropriate input value for this program instance.
          float v = p->vin[index];

          // Do an arbitrary little computation, but at least make the
          // computation dependent on the value being processed
          if (v < 3.)
              v = v * v;
          else
              v = sqrt(v);

          // And write the result to the output array.
          p->vout[index] = v;
      }
  }

  #include "ispcrt.isph"
  DEFINE_CPU_ENTRY_POINT(simple_ispc)

There are several differences in comparison with CPU-only version of this
example located in ``examples/simple``. The first thing to notice
in this program is the usage of the ``task`` keyword in the function definition
instead of ``export``; this indicates that this function is a ``kernel`` so it
can be called from the host.

The second thing to notice is ``DEFINE_CPU_ENTRY_POINT`` which tells ``ISPCRT`` what
function is an entry point for CPU. If you look into the definition of
``DEFINE_CPU_ENTRY_POINT``, it is just simple ``launch`` call:

::

  launch[dim0, dim1, dim2] fcn_name(parameters);

It is used to set up thread space for CPU and GPU targets in a seamless way
in host code. If you don't plan to use ``ISPCRT`` on CPU, you don't need to use
``DEFINE_CPU_ENTRY_POINT`` in ISPC program. Otherwise, you should have
``DEFINE_CPU_ENTRY_POINT`` for each function you plan to call from ``ISPCRT``.

The final thing to notice is that instead of using real parameters for the
kernel ``void * uniform`` is used and later it is cast to ``struct Parameters``.
This approach is used to set up parameters for the kernel in a seamless way
for CPU and GPU on the host side.

Now let's look into ``simple.cpp``. It executes the ISPC kernel on CPU or GPU
depending on an input parameter. The device type is managed by
``ISPCRTDeviceType`` which can be set to ``ISPCRT_DEVICE_TYPE_CPU``,
``ISPCRT_DEVICE_TYPE_GPU`` or ``ISPCRT_DEVICE_TYPE_AUTO`` (tries to use GPU, but
fallback to CPU if no GPUs found).

The program starts with including ``ISPCRT`` header:
::

  #include "ispcrt.hpp"

After that ``ISPCRT`` device is created:
::

  ispcrt::Device device(device_type)

Then we're setting up parameters for ISPC kernel:
::

    // Setup input array
    ispcrt::Array<float> vin_dev(device, vin);

    // Setup output array
    ispcrt::Array<float> vout_dev(device, vout);

    // Setup parameters structure
    Parameters p;

    p.vin = vin_dev.devicePtr();
    p.vout = vout_dev.devicePtr();
    p.count = SIZE;

    auto p_dev = ispcrt::Array<Parameters>(device, p);

Notice that all reference types like arrays and structures should be wrapped up
into ``ispcrt::Array`` for correct passing to ISPC kernel.

Then we set up module and kernel to execute:
::

    ispcrt::Module module(device, "genx_simple");
    ispcrt::Kernel kernel(device, module, "simple_ispc");

The name of the module must correspond to the name of output from ISPC compilation
without extension. So in this example ``simple.ispc`` will be compiled to
``genx_simple.spv`` for GPU and to ``libgenx_simple.so`` for CPU so we use
``genx_simple`` as the module name.
The name of the kernel is just the name of the required ``task`` function from
the ISPC kernel.

The rest of the program creates ``ispcrt::TaskQueue``, fills it with required
steps and executes it:
::

    ispcrt::TaskQueue queue(device);

    // ispcrt::Array objects which used as inputs for ISPC kernel should be
    // explicitly copied to device from host
    queue.copyToDevice(p_dev);
    queue.copyToDevice(vin_dev);

    // Make sure that input arrays were copied
    queue.barrier();

    // Launch the kernel on the device using 1 thread
    queue.launch(kernel, p_dev, 1);

    // Make sure that execution completed
    queue.barrier();

    // ispcrt::Array objects which used as outputs of ISPC kernel should be
    // explicitly copied to host from device
    queue.copyToHost(vout_dev);

    // Make sure that input arrays were copied
    queue.barrier();

    // Execute queue and sync
    queue.sync();


To build and run examples go to ``examples/xpu`` and create
``build`` folder. Run ``cmake -DISPC_EXECUTABLE=<path_to_ispc_binary>
-Dispcrt_DIR=<path_to_ispcrt_cmake> ../`` from ``build`` folder. Or add path
to ``ispc`` to your PATH and just run ``cmake ../``. On Windows you also need
to pass ``-DLEVEL_ZERO_ROOT=<path_lo_level_zero>`` with PATH to ``oneAPI Level Zero``
on the system. Build examples using ``make`` or using ``Visual Studio`` solution.
Go to ``simple`` folder and see what files were generated:

* ``genx_simple.spv`` contains SPIR-V representation. This file is passed
  by ``ISPCRT`` to ``Intel(R) Graphics Compute Runtime`` for execution on GPU.

* ``libgenx_simple.so`` on Linux / ``genx_simple.dll`` on Windows incorporates
  object files produced from ISPC kernel for different targets (you can find
  them in ``local_ispc`` subfolder). This library is loaded from host application
  ``host_simple`` and is used for execution on CPU.

* ``simple_ispc_<target>.h`` files include the declaration for the C-callable
  functions. They are not really used and produced just for the reference.

* ``host_simple`` is the main executable. When it runs, it generates
  the expected output:

::

    Executed on: Auto
    0: simple(0.000000) = 0.000000
    1: simple(1.000000) = 1.000000
    2: simple(2.000000) = 4.000000
    3: simple(3.000000) = 1.732051
    4: simple(4.000000) = 2.000000
    ...

To set up all compilation/link commands in your application we strongly
recommend using ``add_ispc_kernel`` CMake function from CMake module included
into ISPC distribution package.

So the complete ``CMakeFile.txt`` to build ``simple`` example extracted from ISPC
build system is the following:

::

  cmake_minimum_required(VERSION 3.14)
  project(simple)
  find_package(ispcrt REQUIRED)
  add_executable(host_simple simple.cpp)
  add_ispc_kernel(genx_simple simple.ispc "")
  target_link_libraries(host_simple PRIVATE ispcrt::ispcrt)


And you can configure and build it using:
::

  cmake ../ -DISPC_EXECUTABLE_GPU=/home/ispc_package/bin/ispc && make


You can also run separate compilation commands to achieve the same result.
Here are example commands for Linux:

* Compile ISPC kernel for GPU:
  ::

    ispc -I /home/ispc_package/include/ispcrt -DISPC_GPU --target=genx-x8 --woff
    -o /home/ispc_package/examples/xpu/simple/genx_simple.spv
    /home/ispc_package/examples/xpu/simple/simple.ispc

* Compile ISPC kernel for CPU:
  ::

    ispc -I /home/ispc_package/include/ispcrt --arch=x86-64
    --target=sse4-i32x4,avx1-i32x8,avx2-i32x8,avx512knl-i32x16,avx512skx-i32x16
    --woff --pic --opt=disable-assertions
    -h /home/ispc_package/examples/xpu/simple/simple_ispc.h
    -o /home/ispc_package/examples/xpu/simple/simple.dev.o
    /home/ispc_package/examples/xpu/simple/simple.ispc

* Produce a library from object files:
  ::

    /usr/bin/c++ -fPIC -shared -Wl,-soname,libgenx_simple.so -o libgenx_simple.so
    simple.dev*.o

* Compile and link host code:
  ::

    /usr/bin/c++ -DISPCRT -isystem /home/ispc_package/include/ispcrt -fPIE
    -o /home/ispc_package/examples/xpu/simple/host_simple
    /home/ispc_package/examples/xpu/simple/simple.cpp -lispcrt -L/home/ispc_package/lib
    -Wl,-rpath,/home/ispc_package/lib

By default, examples use SPIR-V format. You can try them with L0 binary format:
  ::

    cd examples/xpu/build
    cmake -DISPC_GENX_FORMAT=zebin ../ && make
    export ISPCRT_USE_ZEBIN=y
    cd simple && ./host_simple --gpu

Language Limitations and Known Issues
=====================================

The current release of ``Intel® ISPC for GEN`` is still in Beta stage so you may face
some issues. However, it is actively developed so we expect to fix the remaining
issues in the future releases.
Below is the list of known limitations:

* Limited function pointers support
* Limited stack calls support. We recommend inlining functions as much as you can
  by marking them ``inline``.
* Double math functions like ``sin``, ``cos``, ``log`` etc. are extremely slow.
* Integer fast division is not fast yet especially for unsigned types.
* Float precision is slightly different on CPU and GPU, GPU is more precise.
  Please consider it when designing your algorithms.
* ``print`` doesn't work perfectly especially in deep control flow statements.
  Also, ``print`` is not supported with L0 binary format.


There are several features that we do not plan to implement for GPU:

* ``launch`` and ``sync`` keywords are not supported for GPU in ISPC program
  since kernel execution is managed in the host code now.

* ``new`` and ``delete`` keywords are not expected to be supported in ISPC
  program for GEN target. We expect all memory to be set up on the host side.

* ``export`` functions must return ``void`` for GEN targets.


Performance
===========
The performance of ``Intel® ISPC for GEN`` was significantly improved in this release
but still has room for improvements and we're working hard to make it better for
the next release. Here are our results for ``mandelbrot`` which were obtained on
Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz with Intel(R) Gen9 HD Graphics
(max compute units 24):

* @time of CPU run:			[9.285] milliseconds
* @time of GPU run:			[10.886] milliseconds
* @time of serial run:			[569] milliseconds

Talking about real-world workloads, usually we demonstrate good performance on GPU
that is on par with CPU.

Performance Guide for GPU Programming
----------------------------------------

There are several rules for GPU programming which can bring you better performance.

**Reduce register pressure**

The first guidance is to reduce number of local variables. All variables are stored
in GPU registers, and in the case when number of variables exceeds the number of
registers, time-costly ``register spill`` occurs.

For example, Intel(R) Gen9 register file size is 128x8x32bit. Each 32-bit
varying value takes 8x32bit in SIMD-8, and 16x32bit in SIMD-16.

To reduce number of local variables you can follow these simple rules:

* Use uniform instead of varyings wherever it is possible. This practice
  is good for both CPU and GPU but on GPU it is essential.

::

  // Good example
  for(uniform int j=0;  j<3; j++) {
      do_something();
  }

::

  // Bad example
  for(int j=0;  j<3; j++) {
      do_something();
  }


* Avoid nested code with a lot of local variables. It is more effective
  to split kernel into stages with separate variable scopes.

* Avoid returning complex structures from functions. Instead of operation that
  may need work on structure copy, consider to use reference or pointer. We're
  working to make such optimization automatically for future release:

::

  // Instead of this:
  struct ExampleStructure
  {
    //...
  }

  ExampleStructure createExampleStructure()
  {
    ExampleStructure retVal;
    //... initialize
    return retVal;
  }

  int test()
  {
    ExampleStructure s;
    s = createExampleStructure();
  }

::

  // Consider using pointer:
  struct ExampleStructure
  {
    //...
  }

  void initExampleStructure(ExampleStructure* init)
  {
    //... initialize
  }

  int test()
  {
    ExampleStructure s;
    initExampleStructure( &s );
  }


* Avoid recursion.

* Use SIMD-8 where it is impossible to fit in the available register number.
  If you see the warning message below during runtime, consider compiling your code
  for SIMD-8 target (``--target=genx-x8``).

::

  Spill memory used = 32 bytes for kernel kernel_name___vyi


**Code Branching**

The second set of rules is related to code branching.

* Use ``select`` instead of branching:

::

  if (x > 0)
    a = x;
  else
    a = 7;


::

  // May be implemented without branch:
  a = (x > 0)? x : 7;


When using ``select``, try to simplify it as much as possible:

::

  // Not optimized version:
  varying int K;
  uniform bool Constant;
  ...
  return bConstant == true ? inParam[0] : InParam[K];


::

  // Optimized version
  return InParam[bConstant == true ? 0 : K];

* Keep branches as small as possible. Common operations should be moved outside the branch.
  In case when large code branches are necessary, consider changing your algorithm to group
  data processed by one task to follow the same path in the branch.

::

  // Both branches execute memory access to 'array'. In the case of split branch between
  // different lanes, two memory access instructions would be executed.
  if (x > 0)
    a = array[x];
  else
    a = array[0];


::

  // Instead move common part outside of the branch:
  int i;
  if (x > 0)
    i = x;
  else
    i = 0;
  a = array[i];


Similar situation with loops:

::

  // Good example
  foreach (i = 0 ... WIDTH) {
    p->output[i + WIDTH * taskIndex] = 0;
    int temp = p->output[i + WIDTH * taskIndex];
    for (int j = 0; j < DEPTH; j++) {
      temp += N;
      temp += M;
    }
    p->output[i + WIDTH * taskIndex] = temp;
  }

::

  // Bad example
  foreach (i = 0 ... WIDTH) {
    p->output[i + WIDTH * taskIndex] = 0;
    for (int j = 0; j < DEPTH; j++) {
      p->output[i + WIDTH * taskIndex] += N;
      p->output[i + WIDTH * taskIndex] += M;
    }
  }

**Memory Operations**

Remember that memory operations on GPU are expensive. We do not support dynamic
memory allocations in kernel code for GPU so use fixed-size buffers preallocated
by the host.

We have several memory optimizations for GPU like gather/scatter coalescing. However
current implementation covers only limited number of cases and we expect to improve it
for the next release.


FAQ
====

How to Get an Assembly File from SPIR-V?
----------------------------------------

Use ``ocloc`` tool installed as part of intel-ocloc package:
::

  // Create binary first
  ocloc compile -file file.spv -spirv_input -options "-vc-codegen" -device <name>

::

  // Then disassemble it
  ocloc disasm -file file_Gen9core.bin -device <name> -dump <FOLDER_TO_DUMP>

You will get ``.asm`` files for each kernel in <FOLDER_TO_DUMP>.

How to Debug on GPU?
----------------------------------------

To debug your application, you can use oneAPI Debugger as described here:
`Get Started with GDB* for oneAPI on Linux* OS Host
<https://software.intel.com/get-started-with-debugging-dpcpp-linux>`_.
Debugger support is quite limited at this time but you can set breakpoints
in kernel code, do step-by-step execution and print variables.
