# ISPC Run Time (ISPCRT)

## Introduction

Originally, the Intel® Implicit SPMD Program Compiler (Intel® ISPC) was a compiler for writing SPMD (single program multiple data) programs to run on the CPU. The range of supported architectures has grown and now Intel® ISPC allows compilation of SPMD programs for Intel GPUs.
The compilation for a GPU is pretty straightforward from the user's point of view, but managing execution of code on a GPU may add complexity. The user can use a specific API (such as the [oneAPI Level Zero](https://spec.oneapi.com/level-zero/latest/index.html)) to manage available GPU devices, memory transfers between CPU and GPU, code execution, and synchronization.
Onother possibility is to use **ISPC Run Time (ISPCRT)** library, which is part of ISPC package, to manage that complexity and create unified abstraction for executing tasks on CPU and GPU. The ISPCRT provides C and C++ APIs (see `ispcrt.h` and `ispcrt.hpp` header files) implemented in a library that the user can link to. The same execution model, C/C++ code, and ISPC code can be used to run programs on both CPU and GPU.

## ISPCRT Objects

The ISPC Run Time uses the following abstractions to manage code execution:

* **Device** - represents a CPU or a GPU that can execute SPMD program and has some operational memory available. The user may select particular type of a device (CPU or GPU) or allow the runtime to decide which device will be used.
* **Memory view** - represents data that need to be accessed by different *devices*. For example, input data for code running on GPU must be firstly prepared by a CPU in its memory, then transferred to a GPU memory to perform computations on.
* **Task queue** - Each *device* has a task (command) queue and executes commands from it. The execution may be asynchronous, which means that subseqent commands can begin executing before the previous ones complete. There are synchronization primitives avilable to make the execution synchronous.
* **Barrier** - synchronization primitive that can be inserted into a *task queue* to make sure that all tasks previously inserted into this queue has completed execution.
* **Module** - represents a set of *kernels* that are compiled together and thus can share some common code.
* **Kernel** - is a function that is an entry point to a *module* and can be called by inserting kernel execution command into a *task queue*. A kernel has one parameter - a pointer to a structure of actual kernel parameters.
* **Future** - can be treated as a promise that at some point *kernel* execution connected to this object will be completed and the object will become valid. *Futures* are returned when a *kernel* invocation is inserted into a *task queue*. When the *task queue* is executed on a device, the *future* object becomes valid and can be used to retrieve information about the *kernel* execution.

All ISPCRT objects support reference counting, which means that it is not necessary to perform detailed memory management. The objects will be released once they are not used.

## Execution model

The idea of [ISPC tasks](https://ispc.github.io/ispc.html#task-parallelism-launch-and-sync-statements) has been extended to support execution of kernels on a GPU. Each kernel execution command inserted into a task queue is parametrized with the number of tasks (threads) that should be launched on a GPU. Each task must decide on which part of the problem it should work, exactly the same as it happens in the CPU case. Within tasks, program executes in SPMD manner (again the regular ISPC execution model is copied). All built-in variables used for tha purpose (such as `taskIndex`, `taskCount`, `programIndex`, `programCount`) are avaialable for use on GPU.

## More details

Functions provided by the ISPCRT API are documented in the header files for C (`ispcrt.h`) and C++ (`ispcrt.hpp`) languages. Examples in `ispc/examples/portable/genx/` directory demonstrate how to use this API to run SPMD programs on CPU or GPU.