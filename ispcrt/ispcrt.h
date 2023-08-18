// Copyright 2020-2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types ////////////////////////////////////////////////////////

#ifdef __cplusplus
struct _ISPCRTContext;
struct _ISPCRTDevice;
struct _ISPCRTMemoryView;
struct _ISPCRTTaskQueue;
struct _ISPCRTModule;
struct _ISPCRTModuleOptions;
struct _ISPCRTKernel;
struct _ISPCRTFuture;
struct _ISPCRTFence;
struct _ISPCRTCommandQueue;
struct _ISPCRTCommandList;

typedef _ISPCRTContext *ISPCRTContext;
typedef _ISPCRTDevice *ISPCRTDevice;
typedef _ISPCRTMemoryView *ISPCRTMemoryView;
typedef _ISPCRTTaskQueue *ISPCRTTaskQueue;
typedef _ISPCRTModule *ISPCRTModule;
typedef _ISPCRTModuleOptions *ISPCRTModuleOptions;
typedef _ISPCRTKernel *ISPCRTKernel;
typedef _ISPCRTFuture *ISPCRTFuture;
typedef _ISPCRTFence *ISPCRTFence;
typedef _ISPCRTCommandQueue *ISPCRTCommandQueue;
typedef _ISPCRTCommandList *ISPCRTCommandList;
#else
typedef void *ISPCRTContext;
typedef void *ISPCRTDevice;
typedef void *ISPCRTMemoryView;
typedef void *ISPCRTTaskQueue;
typedef void *ISPCRTModule;
typedef void *ISPCRTModuleOptions;
typedef void *ISPCRTKernel;
typedef void *ISPCRTFuture;
typedef void *ISPCRTFence;
typedef void *ISPCRTCommandQueue;
typedef void *ISPCRTCommandList;
#endif

// NOTE: ISPCRTGenericHandle usage implies compatibility with any of the above
//       handle types.
typedef void *ISPCRTGenericHandle;

// Error handling /////////////////////////////////////////////////////////////

typedef enum {
    ISPCRT_NO_ERROR = 0,
    ISPCRT_UNKNOWN_ERROR = 1,
    ISPCRT_INVALID_ARGUMENT = 2,
    ISPCRT_INVALID_OPERATION = 3,
    ISPCRT_OUT_OF_MEMORY = 5,
    ISPCRT_UNINITIALIZED = 6,
    ISPCRT_UNSUPPORTED = 7,
    ISPCRT_DEVICE_LOWER_POWER = 8,
    ISPCRT_DEVICE_RESET = 9,
    ISPCRT_DEVICE_LOST = 10

} ISPCRTError;

typedef void (*ISPCRTErrorFunc)(ISPCRTError, const char *errorMessage);

void ispcrtSetErrorFunc(ISPCRTErrorFunc);

// Tasking ////////////////////////////////////////////////////////////////////
// Tasking is CPU specific part of API.

// Callback types definition.
typedef void (*ISPCRTTaskingLaunchFType)(void **, void *, void *, int, int, int);
typedef void *(*ISPCRTTaskingAllocFType)(void **, int64_t, int32_t);
typedef void (*ISPCRTTaskingSyncFType)(void *);

// Applications can provide their own implementation of ISPCLaunch/ISPCAlloc/ISPCSync tasking API.
void ispcrtSetTaskingCallbacks(ISPCRTTaskingLaunchFType, ISPCRTTaskingAllocFType, ISPCRTTaskingSyncFType);

// Object lifetime ////////////////////////////////////////////////////////////

long long ispcrtUseCount(ISPCRTGenericHandle);
void ispcrtRelease(ISPCRTGenericHandle);
void ispcrtRetain(ISPCRTGenericHandle);

// Device initialization //////////////////////////////////////////////////////

typedef enum {
    ISPCRT_DEVICE_TYPE_CPU,
    ISPCRT_DEVICE_TYPE_GPU,
    ISPCRT_DEVICE_TYPE_AUTO // try 'GPU', but fallback to 'CPU' if no GPUs found
} ISPCRTDeviceType;

typedef struct {
    uint32_t vendorId;
    uint32_t deviceId;
} ISPCRTDeviceInfo;

// deviceIdx is an index of the device in the list of supported devices
// The list of the supported devices can be obtained with a ispcrtGetDeviceCount call
// and a series of ispcrtGetDeviceInfo calls
ISPCRTDevice ispcrtGetDevice(ISPCRTDeviceType, uint32_t deviceIdx);
ISPCRTDevice ispcrtGetDeviceFromContext(ISPCRTContext, uint32_t deviceIdx);
// Alternatively ISPCRTDevice can be constructed from device native handler
ISPCRTDevice ispcrtGetDeviceFromNativeHandle(ISPCRTContext context, ISPCRTGenericHandle d);

ISPCRTDeviceType ispcrtGetDeviceType(ISPCRTDevice d);

uint32_t ispcrtGetDeviceCount(ISPCRTDeviceType);
void ispcrtGetDeviceInfo(ISPCRTDeviceType, uint32_t deviceIdx, ISPCRTDeviceInfo *);

// Context initialization //////////////////////////////////////////////////////

ISPCRTContext ispcrtNewContext(ISPCRTDeviceType);
// Alternatively ISPCRTContext can be constructed from context native handler
ISPCRTContext ispcrtGetContextFromNativeHandle(ISPCRTDeviceType, ISPCRTGenericHandle c);

// MemoryViews ////////////////////////////////////////////////////////////////

// Choose allocation type
typedef enum {
    // Allocate memory on the device (and associate appMemory with this allocation)
    ISPCRT_ALLOC_TYPE_DEVICE = 0,
    // Allocate in the memory shared between the host and the device (ignore appMemory ptr)
    ISPCRT_ALLOC_TYPE_SHARED,
    // The following allocation types are not used for allocation of ISPCRuntime objects,
    // but required to match possible L0 allocation types.
    ISPCRT_ALLOC_TYPE_HOST,
    ISPCRT_ALLOC_TYPE_UNKNOWN,
} ISPCRTAllocationType;

// Choose shared memory allocation flags
typedef enum {
    ISPCRT_SM_HOST_DEVICE_READ_WRITE = 0,
    ISPCRT_SM_HOST_WRITE_DEVICE_READ,
    ISPCRT_SM_HOST_READ_DEVICE_WRITE,
    ISPCRT_SM_APPLICATION_MANAGED_DEVICE,
    ISPCRT_SM_UNKNOWN,
} ISPCRTSharedMemoryAllocationHint;

typedef struct {
    ISPCRTAllocationType allocType;
    ISPCRTSharedMemoryAllocationHint smHint;
} ISPCRTNewMemoryViewFlags;

ISPCRTMemoryView ispcrtNewMemoryView(ISPCRTDevice, void *appMemory, size_t numBytes, ISPCRTNewMemoryViewFlags *flags);
ISPCRTMemoryView ispcrtNewMemoryViewForContext(ISPCRTContext c, void *appMemory, size_t numBytes,
                                               ISPCRTNewMemoryViewFlags *flags);

void *ispcrtHostPtr(ISPCRTMemoryView);
void *ispcrtDevicePtr(ISPCRTMemoryView);
void *ispcrtSharedPtr(ISPCRTMemoryView);

size_t ispcrtSize(ISPCRTMemoryView);

ISPCRTAllocationType ispcrtGetMemoryViewAllocType(ISPCRTMemoryView);
ISPCRTAllocationType ispcrtGetMemoryAllocType(ISPCRTDevice d, void *memBuffer);

// Modules ////////////////////////////////////////////////////////////////////
typedef enum {
    // Module using IGC VC backend
    ISPCRT_VECTOR_MODULE = 0,
    // Module using IGC scalar backend
    ISPCRT_SCALAR_MODULE,
} ISPCRTModuleType;

ISPCRTModuleOptions ispcrtNewModuleOptionsEmpty(ISPCRTDevice);
ISPCRTModuleOptions ispcrtNewModuleOptions(ISPCRTDevice, ISPCRTModuleType moduleType, bool libraryCompilation = false,
                                           uint32_t stackSize = 0);
uint32_t ispcrtModuleOptionsGetStackSize(ISPCRTModuleOptions);
bool ispcrtModuleOptionsGetLibraryCompilation(ISPCRTModuleOptions);
ISPCRTModuleType ispcrtModuleOptionsGetModuleType(ISPCRTModuleOptions);
void ispcrtModuleOptionsSetStackSize(ISPCRTModuleOptions, uint32_t);
void ispcrtModuleOptionsSetLibraryCompilation(ISPCRTModuleOptions, bool);
void ispcrtModuleOptionsSetModuleType(ISPCRTModuleOptions, ISPCRTModuleType);

ISPCRTModule ispcrtLoadModule(ISPCRTDevice, const char *moduleFile);
ISPCRTModule ispcrtLoadModuleWithOptions(ISPCRTDevice, const char *moduleFile, ISPCRTModuleOptions);
void ispcrtDynamicLinkModules(ISPCRTDevice, ISPCRTModule *modules, uint32_t numModules);
ISPCRTModule ispcrtStaticLinkModules(ISPCRTDevice, ISPCRTModule *modules, uint32_t numModules);
void *ispcrtFunctionPtr(ISPCRTModule, const char *name);
ISPCRTKernel ispcrtNewKernel(ISPCRTDevice, ISPCRTModule, const char *name);

// Command lists //////////////////////////////////////////////////////////////
void ispcrtCommandListBarrier(ISPCRTCommandList);

ISPCRTFuture ispcrtCommandListCopyToDevice(ISPCRTCommandList, ISPCRTMemoryView);
ISPCRTFuture ispcrtCommandListCopyToHost(ISPCRTCommandList, ISPCRTMemoryView);
ISPCRTFuture ispcrtCommandListCopyMemoryView(ISPCRTCommandList, ISPCRTMemoryView, ISPCRTMemoryView, const size_t size);

// NOTE: 'params' can be a nullptr handle (nullptr will get passed to the ISPC task as the function parameter)
ISPCRTFuture ispcrtCommandListLaunch1D(ISPCRTCommandList, ISPCRTKernel, ISPCRTMemoryView params, size_t dim0);
ISPCRTFuture ispcrtCommandListLaunch2D(ISPCRTCommandList, ISPCRTKernel, ISPCRTMemoryView params, size_t dim0,
                                       size_t dim1);
ISPCRTFuture ispcrtCommandListLaunch3D(ISPCRTCommandList, ISPCRTKernel, ISPCRTMemoryView params, size_t dim0,
                                       size_t dim1, size_t dim2);

void ispcrtCommandListClose(ISPCRTCommandList);
ISPCRTFence ispcrtCommandListSubmit(ISPCRTCommandList);
void ispcrtCommandListReset(ISPCRTCommandList);
void ispcrtCommandListEnableTimestamps(ISPCRTCommandList);

// Command queues /////////////////////////////////////////////////////////////
ISPCRTCommandQueue ispcrtNewCommandQueue(ISPCRTDevice, uint32_t ordinal);

ISPCRTCommandList ispcrtCommandQueueCreateCommandList(ISPCRTCommandQueue);
void ispcrtCommandQueueSync(ISPCRTCommandQueue);

// Task queues ////////////////////////////////////////////////////////////////

ISPCRTTaskQueue ispcrtNewTaskQueue(ISPCRTDevice);

void ispcrtDeviceBarrier(ISPCRTTaskQueue);

void ispcrtCopyToDevice(ISPCRTTaskQueue, ISPCRTMemoryView);
void ispcrtCopyToHost(ISPCRTTaskQueue, ISPCRTMemoryView);
void ispcrtCopyMemoryView(ISPCRTTaskQueue, ISPCRTMemoryView, ISPCRTMemoryView, const size_t size);

// NOTE: 'params' can be a nullptr handle (nullptr will get passed to the ISPC task as the function parameter)
ISPCRTFuture ispcrtLaunch1D(ISPCRTTaskQueue, ISPCRTKernel, ISPCRTMemoryView params, size_t dim0);
ISPCRTFuture ispcrtLaunch2D(ISPCRTTaskQueue, ISPCRTKernel, ISPCRTMemoryView params, size_t dim0, size_t dim1);
ISPCRTFuture ispcrtLaunch3D(ISPCRTTaskQueue, ISPCRTKernel, ISPCRTMemoryView params, size_t dim0, size_t dim1,
                            size_t dim2);

void ispcrtSync(ISPCRTTaskQueue);

// Fence //////////////////////////////////////////////////////////////////////
typedef enum {
    ISPCRT_FENCE_UNSIGNALED = 0,
    ISPCRT_FENCE_SIGNALED = 1,
} ISPCRTFenceStatus;

void ispcrtFenceSync(ISPCRTFence);
ISPCRTFenceStatus ispcrtFenceStatus(ISPCRTFence);
void ispcrtFenceReset(ISPCRTFence);

// Futures and task timing ////////////////////////////////////////////////////

uint64_t ispcrtFutureGetTimeNs(ISPCRTFuture);
bool ispcrtFutureIsValid(ISPCRTFuture);

// Access to objects of native runtime ///////////////////////////////////////

ISPCRTGenericHandle ispcrtPlatformNativeHandle(ISPCRTDevice);
ISPCRTGenericHandle ispcrtDeviceNativeHandle(ISPCRTDevice);
ISPCRTGenericHandle ispcrtDeviceContextNativeHandle(ISPCRTDevice);
ISPCRTGenericHandle ispcrtContextNativeHandle(ISPCRTContext);
ISPCRTGenericHandle ispcrtCommandListNativeHandle(ISPCRTCommandList);
ISPCRTGenericHandle ispcrtCommandQueueNativeHandle(ISPCRTCommandQueue);
ISPCRTGenericHandle ispcrtTaskQueueNativeHandle(ISPCRTTaskQueue);
ISPCRTGenericHandle ispcrtFenceNativeHandle(ISPCRTFence);

#ifdef __cplusplus
} // extern "C"
#endif
