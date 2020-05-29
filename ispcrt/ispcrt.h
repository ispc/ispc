// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types ////////////////////////////////////////////////////////

#ifdef __cplusplus
struct _ISPCRTDevice;
struct _ISPCRTMemoryView;
struct _ISPCRTTaskQueue;
struct _ISPCRTModule;
struct _ISPCRTKernel;
struct _ISPCRTFuture;

typedef _ISPCRTDevice *ISPCRTDevice;
typedef _ISPCRTMemoryView *ISPCRTMemoryView;
typedef _ISPCRTTaskQueue *ISPCRTTaskQueue;
typedef _ISPCRTModule *ISPCRTModule;
typedef _ISPCRTKernel *ISPCRTKernel;
typedef _ISPCRTFuture *ISPCRTFuture;
#else
typedef void *ISPCRTDevice;
typedef void *ISPCRTMemoryView;
typedef void *ISPCRTTaskQueue;
typedef void *ISPCRTModule;
typedef void *ISPCRTKernel;
typedef void *ISPCRTFuture;
#endif

// NOTE: ISPCRTGenericHandle usage implies compatibility with any of the above
//       handle types.
typedef void *ISPCRTGenericHandle;

// Error handling /////////////////////////////////////////////////////////////

typedef enum {
    ISPCRT_NO_ERROR = 0,
    ISPCRT_UNKNOWN_ERROR = 1,
    ISPCRT_INVALID_ARGUMENT = 2,
    ISPCRT_INVALID_OPERATION = 3
} ISPCRTError;

typedef void (*ISPCRTErrorFunc)(ISPCRTError, const char *errorMessage);

void ispcrtSetErrorFunc(ISPCRTErrorFunc);

// Object lifetime ////////////////////////////////////////////////////////////

void ispcrtRelease(ISPCRTGenericHandle);
void ispcrtRetain(ISPCRTGenericHandle);

// Device initialization //////////////////////////////////////////////////////

typedef enum {
    ISPCRT_DEVICE_TYPE_CPU,
    ISPCRT_DEVICE_TYPE_GPU,
    ISPCRT_DEVICE_TYPE_AUTO // try 'GPU', but fallback to 'CPU' if no GPUs found
} ISPCRTDeviceType;

ISPCRTDevice ispcrtGetDevice(ISPCRTDeviceType);
uint64_t     ispcrtGetDeviceTimeTickResolution();

// MemoryViews ////////////////////////////////////////////////////////////////

ISPCRTMemoryView ispcrtNewMemoryView(ISPCRTDevice, void *appMemory, size_t numBytes);

void *ispcrtHostPtr(ISPCRTMemoryView);
void *ispcrtDevicePtr(ISPCRTMemoryView);

size_t ispcrtSize(ISPCRTMemoryView);

// Kernels ////////////////////////////////////////////////////////////////////

ISPCRTModule ispcrtLoadModule(ISPCRTDevice, const char *moduleFile);
ISPCRTKernel ispcrtNewKernel(ISPCRTDevice, ISPCRTModule, const char *name);

// Task queues ////////////////////////////////////////////////////////////////

ISPCRTTaskQueue ispcrtNewTaskQueue(ISPCRTDevice);

void ispcrtDeviceBarrier(ISPCRTTaskQueue);

void ispcrtCopyToDevice(ISPCRTTaskQueue, ISPCRTMemoryView);
void ispcrtCopyToHost(ISPCRTTaskQueue, ISPCRTMemoryView);

// NOTE: 'params' can be a NULL handle (NULL will get passed to the ISPC task as the function parameter)
ISPCRTFuture ispcrtLaunch1D(ISPCRTTaskQueue, ISPCRTKernel, ISPCRTMemoryView params, size_t dim0);
ISPCRTFuture ispcrtLaunch2D(ISPCRTTaskQueue, ISPCRTKernel, ISPCRTMemoryView params, size_t dim0, size_t dim1);
ISPCRTFuture ispcrtLaunch3D(ISPCRTTaskQueue, ISPCRTKernel, ISPCRTMemoryView params, size_t dim0, size_t dim1, size_t dim2);

void ispcrtSync(ISPCRTTaskQueue);

// Futures and task timing ////////////////////////////////////////////////////

uint64_t ispcrtFutureGetTimeNs(ISPCRTFuture);

bool ispcrtFutureIsValid(ISPCRTFuture);

#ifdef __cplusplus
} // extern "C"
#endif
