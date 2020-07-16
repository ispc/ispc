========================
Intel® ISPC for GEN
========================

The Intel® Implicit SPMD Program Compiler (Intel® ISPC) got initial support for
Intel GPUs recently. The compilation for a GPU is pretty straightforward from
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

* `Compiling and Running Simple ISPC Program`_

* `Language Limitations and Known Issues`_

* `Performance`_

* `FAQ`_

  + `How to Get an Assembly File from SPIR-V?`_

Using The ISPC Compiler
=======================

The output from ``ispc`` for GEN targets is SPIR-V file. SPIR-V output format
is selected by default when one of ``genx`` targets is used:

::

   ispc foo.ispc --target=genx-x8 -o foo.spv

The SPIR-V file is consumed by the runtime for further compilation and execution
on GPU.


Environment
-----------
Currently, ``Intel® ISPC for GEN`` is supported on Linux only but it is planned
to be extended for Windows as well in a future release.
Recommended and tested Linux distribution is Ubuntu 18.04.

You need to have a system with ``Intel(R) Processor Graphics Gen9``.

For the execution of ISPC programs on GPU, please install `Intel(R)
Graphics Compute Runtime <https://github.com/intel/compute-runtime/releases>`_
and `Level Zero Loader <https://github.com/oneapi-src/level-zero/releases>`_.

To use ISPC Run Time for CPU you need to have ``OpenMP runtime`` installed on
your system. Consult your Linux distribution documentation for installation
of OpenMP runtime instructions.


Basic Command-line Options
--------------------------

Two new targets were introduced for GPU support: ``genx-x8`` and ``genx-x16``.

If the ``-o`` flag is given, ``ispc`` will generate a SPIR-V output file.
Optionally you can use ``--emit-spirv`` flag:

::

   ispc --target=genx-x8 --emit-spirv foo.ispc -o foo.spv

Also two new ``arch`` options were introduced: ``genx32`` and ``genx64``.
``genx64`` is default and corresponds to 64-bit host and has 64-bit pointer size,
``genx32`` corresponds to 32-bit host and has 32-bit pointer size.

To generate LLVM bitcode, use the ``--emit-llvm`` flag.
To generate LLVM bitcode in textual form, use the ``--emit-llvm-text`` flag.


Optimizations are on by default; they can be turned off with ``-O0``. However,
for the current release, it is not recommended to use ``-O0``, there are several
known issues.

Generating a text assembly file using ``--emit-asm`` is not supported yet.
See `How to Get an Assembly File from SPIR-V?`_ section about how to get the
assembly from SPIR-V file.

By default, 64-bit addressing is used. You can change it to 32-bit addressing by
using ``--addressing=32`` or ``--arch=genx32`` however pointer size should be
the same for host and device code so 32-bit addressing will only work with
32-bit host programs.


ISPC Run Time (ISPCRT)
=======================

``ISPC Run Time (ISPCRT)`` unifies execution models for CPU and GPU targets. It
is a high-level abstraction on the top of `oneAPI Level Zero
<https://spec.oneapi.com/level-zero/latest/index.html>`_. You can continue
using ISPC for CPU without this runtime and alternatively use pure ``oneAPI
Level Zero`` for GPU. However, we strongly encourage you to try ``ISPCRT``
and give us feedback!
The ``ISPCRT`` provides C and C++ APIs which are documented in the header files
(see ``ispcrt.h`` and ``ispcrt.hpp``) and distributed as a library that you can
link to.
Examples in ``ispc/examples/portable/genx/`` directory demonstrate how to use
this API to run SPMD programs on CPU or GPU. You can see how to use
``oneAPI Level Zero`` runtime in ``sgemm`` example.


ISPCRT Objects
---------------

The ``ISPC Run Time`` uses the following abstractions to manage code execution:

* ``Device`` - represents a CPU or a GPU that can execute SPMD program and has
  some operational memory available. The user may select a particular type of
  device (CPU or GPU) or allow the runtime to decide which device will be used.

* ``Memory view`` - represents data that need to be accessed by different
  ``devices``. For example, input data for code running on GPU must be firstly
  prepared by a CPU in its memory, then transferred to a GPU memory to perform
  computations on.

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


Compiling and Running Simple ISPC Program
=========================================
The directory ``examples/portable/genx/simple`` in the ``ispc`` distribution
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

Second thing to notice is ``DEFINE_CPU_ENTRY_POINT`` which tells ``ISPCRT`` what
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

The prograam starts with including ``ISPCRT`` header:
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


To build and run examples go to ``examples/portable/genx`` and create
``build`` folder. Run ``cmake -DISPC_EXECUTABLE=<path_to_ispc_binary>
-Dispcrt_DIR=<path_to_ispcrt_cmake> ../`` from ``build`` folder. Or add path
to ``ispc`` to your PATH and just run ``cmake ../``. Build examples using ``make``.
Go to ``simple`` folder and see what files were generated:

* ``genx_simple.spv`` contains SPIR-V representation. This file is passed
  by ``ISPCRT`` to ``Intel(R) Graphics Compute Runtime`` for execution on GPU.

* ``libgenx_simple.so`` incorporates object files produced from ISPC kernel
  for different targets (you can find them in ``local_ispc`` subfolder).
  This library is loaded from host application ``host_simple`` and is used for
  execution on CPU.

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
  find_package(ispcrt REQUIRED)
  add_executable(host_simple simple.cpp)
  add_ispc_kernel(genx_simple simple.ispc "")
  target_link_libraries(host_simple PRIVATE ispcrt::ispcrt)


And you can configure and build it using:
::

  cmake ../ -DISPC_EXECUTABLE_GPU=/home/install/bin/ispc && make


You can also run separate compilation commands to achive the same result:

* Compile ISPC kernel for GPU:
  ::

    ispc -I /home/install/include/ispcrt -DISPC_GPU --target=genx-x8 --woff
    -o /home/ispc/examples/portable/genx/simple/genx_simple.spv
    /home/ispc/examples/portable/genx/simple/simple.ispc

* Compile ISPC kernel for CPU:
  ::

    ispc -I /home/install/include/ispcrt --arch=x86-64
    --target=sse4-i32x4,avx1-i32x8,avx2-i32x8,avx512knl-i32x16,avx512skx-i32x16
    --woff --pic --opt=disable-assertions
    -h /home/ispc/examples/portable/genx/simple/simple_ispc.h
    -o /home/ispc/examples/portable/genx/simple/simple.dev.o
    /home/ispc/examples/portable/genx/simple/simple.ispc

* Produce a library from object files:
  ::

    /usr/bin/c++ -fPIC -shared -Wl,-soname,libgenx_simple.so -o libgenx_simple.so
    simple.dev*.o

* Compile and link host code:
  ::

    /usr/bin/c++  -DISPCRT -isystem /home/install/include/ispcrt -fPIE
    -o /home/ispc/examples/portable/genx/simple/host_simple
    /home/ispc/examples/portable/genx/simple/simple.cpp -L/usr/local/lib
    -Wl,-rpath,/home/install/lib /home/install/lib/libispcrt.so.1.13.0


Language Limitations and Known Issues
=====================================

The current release of ``Intel® ISPC for GEN`` is in alpha state so not all
functionality is implemented yet. However, it is actively developed so we
expect to cover missing features in the nearest future releases.
Below is the list of known limitations:

* No function pointers support
* No recursion support
* No ``prefetch`` support
* No global atomics
* No double precision math functions
* Foreach inside varying control flow is not supported by default. However you
  can enable it by using experimental flag ``--opt=enable-genx-foreach-varying``
  but performance will degrade with this flag.

There are several features which we do not plan to implement for GPU:

* ``launch`` and ``sync`` keywords are not supported for GPU in ISPC program
  since kernel execution is managed in the host code now.

* ``New`` and ``delete`` keywords are not expected to be supported in ISPC
  program for GEN target. We expect all memory to be set up on the host side.

* ``export`` functions must return ``void`` for GEN targets.


Performance
===========
The performance of ``Intel® ISPC for GEN`` is not the goal of the current release.
It has a room for improvements and we're working hard to make it better for
the next release. Here are our results for ``mandelbrot`` which were obtained on
Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz with Intel(R) Gen9 HD Graphics
(max compute units 24):

* @time of CPU run:			[16.343] milliseconds
* @time of GPU run:			[29.583] milliseconds
* @time of serial run:			[566] milliseconds


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
