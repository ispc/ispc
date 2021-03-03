// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ze_mock.h"

#include <cstring>
#include <iostream>

namespace ispcrt {
namespace testing {
namespace mock {
namespace driver {

#define MOCK_RET return Config::getRetValue(__FUNCTION__)
#define MOCK_SHOULD_SUCCEED (Config::getRetValue(__FUNCTION__) == ZE_RESULT_SUCCESS)
#define MOCK_CNT_CALL CallCounters::inc(__FUNCTION__)

static unsigned MockHandleHandle = 1;

template <typename HT> struct MockHandle {
    HT get() { return handle; }
    MockHandle() { handle = reinterpret_cast<HT>(MockHandleHandle++); }

  private:
    HT handle;
};

MockHandle<ze_device_handle_t> DeviceHandle;
MockHandle<ze_context_handle_t> ContextHandle;
MockHandle<ze_module_handle_t> ModuleHandle;
MockHandle<ze_kernel_handle_t> KernelHandle;
MockHandle<ze_command_list_handle_t> CmdListHandle;
MockHandle<ze_command_queue_handle_t> CmdQueueHandle;
MockHandle<ze_event_pool_handle_t> EventPoolHandle;
MockHandle<ze_event_handle_t> LaunchEventHandle;

bool ExpectedDevice(ze_device_handle_t hDevice) {
    auto dp = reinterpret_cast<DeviceProperties*>(hDevice);
    return dp == Config::getDevicePtr(Config::getExpectedDevice());
}

bool ValidDevice(ze_device_handle_t hDevice) {
    auto dp = reinterpret_cast<DeviceProperties*>(hDevice);
    for (int i = 0; i < Config::getDeviceCount(); i++)
        if (dp == Config::getDevicePtr(i))
            return true;
    return false;
}

ze_result_t zeInit(ze_init_flags_t flags) { MOCK_RET; }

ze_result_t zeDriverGet(uint32_t *pCount, ze_driver_handle_t *phDrivers) {
    MOCK_CNT_CALL;
    if (*pCount == 0) {
        *pCount = 1;
    }
    // Should always succeed - error handling is done by Level Zero loader
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeDeviceGet(ze_driver_handle_t hDriver, uint32_t *pCount, ze_device_handle_t *phDevices) {
    MOCK_CNT_CALL;
    *pCount = Config::getDeviceCount();
    if (phDevices) {
        for (int i = 0; i < *pCount; i++) {
            phDevices[i] = reinterpret_cast<ze_device_handle_t>(Config::getDevicePtr(i));
        }
    }
    MOCK_RET;
}

ze_result_t zeDeviceGetProperties(ze_device_handle_t hDevice, ze_device_properties_t *pDeviceProperties) {
    MOCK_CNT_CALL;
    if (!ValidDevice(hDevice) || pDeviceProperties == nullptr)
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;

    auto dp = reinterpret_cast<DeviceProperties*>(hDevice);

    pDeviceProperties->stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    pDeviceProperties->type = ZE_DEVICE_TYPE_GPU;
    pDeviceProperties->deviceId = dp->deviceId;
    pDeviceProperties->vendorId = dp->vendorId;
    std::strcpy(pDeviceProperties->name, "ISPCRT Mock Device");

    MOCK_RET;
}

ze_result_t zeContextCreate(ze_driver_handle_t hDriver, const ze_context_desc_t *desc, ze_context_handle_t *phContext) {
    MOCK_CNT_CALL;
    *phContext = ContextHandle.get();
    MOCK_RET;
}

ze_result_t zeContextDestroy(ze_context_handle_t hContext) {
    MOCK_CNT_CALL;
    if (hContext != ContextHandle.get())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    MOCK_RET;
}

ze_result_t zeCommandQueueCreate(ze_context_handle_t hContext, ze_device_handle_t hDevice,
                                 const ze_command_queue_desc_t *desc, ze_command_queue_handle_t *phCommandQueue) {
    MOCK_CNT_CALL;
    if (!ExpectedDevice(hDevice) || hContext != ContextHandle.get() || desc == nullptr ||
        phCommandQueue == nullptr)
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    *phCommandQueue = CmdQueueHandle.get();
    MOCK_RET;
}

ze_result_t zeCommandQueueDestroy(ze_command_queue_handle_t hCommandQueue) {
    MOCK_CNT_CALL;
    if (hCommandQueue != CmdQueueHandle.get())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    MOCK_RET;
}

ze_result_t zeCommandQueueExecuteCommandLists(ze_command_queue_handle_t hCommandQueue, uint32_t numCommandLists,
                                              ze_command_list_handle_t *phCommandLists, ze_fence_handle_t hFence) {
    MOCK_CNT_CALL;
    if (hCommandQueue != CmdQueueHandle.get() || !Config::isCmdListClosed())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    MOCK_RET;
}

ze_result_t zeCommandQueueSynchronize(ze_command_queue_handle_t hCommandQueue, uint64_t timeout) {
    MOCK_CNT_CALL;
    if (hCommandQueue != CmdQueueHandle.get() || !Config::isCmdListClosed())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    MOCK_RET;
}

ze_result_t zeCommandListCreate(ze_context_handle_t hContext, ze_device_handle_t hDevice,
                                const ze_command_list_desc_t *desc, ze_command_list_handle_t *phCommandList) {
    MOCK_CNT_CALL;
    if (!ExpectedDevice(hDevice) || hContext != ContextHandle.get() || desc == nullptr || phCommandList == nullptr)
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    *phCommandList = CmdListHandle.get();
    MOCK_RET;
}

ze_result_t zeCommandListDestroy(ze_command_list_handle_t hCommandList) {
    MOCK_CNT_CALL;
    if (hCommandList != CmdListHandle.get())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    MOCK_RET;
}

ze_result_t zeCommandListClose(ze_command_list_handle_t hCommandList) {
    MOCK_CNT_CALL;
    if (hCommandList != CmdListHandle.get())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    Config::closeCmdList();
    MOCK_RET;
}

ze_result_t zeCommandListReset(ze_command_list_handle_t hCommandList) {
    MOCK_CNT_CALL;
    if (hCommandList != CmdListHandle.get())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    Config::resetCmdList();
    MOCK_RET;
}

ze_result_t zeCommandListAppendBarrier(ze_command_list_handle_t hCommandList, ze_event_handle_t hSignalEvent,
                                       uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents) {
    MOCK_CNT_CALL;
    if (hCommandList != CmdListHandle.get() || Config::isCmdListClosed())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    if (MOCK_SHOULD_SUCCEED)
        Config::addToCmdList(CmdListElem::Barrier);
    MOCK_RET;
}

ze_result_t zeCommandListAppendMemoryCopy(ze_command_list_handle_t hCommandList, void *dstptr, const void *srcptr,
                                          size_t size, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
                                          ze_event_handle_t *phWaitEvents) {
    MOCK_CNT_CALL;
    if (hCommandList != CmdListHandle.get() || Config::isCmdListClosed())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    if (MOCK_SHOULD_SUCCEED)
        Config::addToCmdList(CmdListElem::MemoryCopy);
    MOCK_RET;
}

ze_result_t zeEventPoolCreate(ze_context_handle_t hContext, const ze_event_pool_desc_t *desc, uint32_t numDevices,
                              ze_device_handle_t *phDevices, ze_event_pool_handle_t *phEventPool) {
    MOCK_CNT_CALL;
    if (hContext != ContextHandle.get() || numDevices == 0 || !phDevices || !phEventPool)
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    *phEventPool = EventPoolHandle.get();
    MOCK_RET;
}

ze_result_t zeEventPoolDestroy(ze_event_pool_handle_t hEventPool) {
    MOCK_CNT_CALL;
    if (hEventPool != EventPoolHandle.get())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    MOCK_RET;
}

ze_result_t zeEventCreate(ze_event_pool_handle_t hEventPool, const ze_event_desc_t *desc, ze_event_handle_t *phEvent) {
    MOCK_CNT_CALL;
    if (hEventPool != EventPoolHandle.get() || !desc || !phEvent)
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    *phEvent = LaunchEventHandle.get();
    MOCK_RET;
}

ze_result_t zeEventDestroy(ze_event_handle_t hEvent) {
    MOCK_CNT_CALL;
    MOCK_RET;
}

ze_result_t zeEventQueryKernelTimestamp(ze_event_handle_t hEvent, ze_kernel_timestamp_result_t *dstptr) {
    MOCK_CNT_CALL;
    MOCK_RET;
}

ze_result_t zeMemAllocDevice(ze_context_handle_t hContext, const ze_device_mem_alloc_desc_t *device_desc, size_t size,
                             size_t alignment, ze_device_handle_t hDevice, void **pptr) {
    MOCK_CNT_CALL;
    if (hContext != ContextHandle.get())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    if (MOCK_SHOULD_SUCCEED)
        *pptr = new uint8_t[size];
    MOCK_RET;
}

ze_result_t zeMemFree(ze_context_handle_t hContext, void *ptr) {
    MOCK_CNT_CALL;
    if (hContext != ContextHandle.get() || !ptr)
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    delete[](uint8_t *) ptr;
    MOCK_RET;
}

ze_result_t zeModuleCreate(ze_context_handle_t hContext, ze_device_handle_t hDevice, const ze_module_desc_t *desc,
                           ze_module_handle_t *phModule, ze_module_build_log_handle_t *phBuildLog) {
    MOCK_CNT_CALL;
    if (hContext != ContextHandle.get() || !ExpectedDevice(hDevice))
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;

    *phModule = ModuleHandle.get();

    MOCK_RET;
}

ze_result_t zeModuleDestroy(ze_module_handle_t hModule) {
    MOCK_CNT_CALL;
    if (hModule != ModuleHandle.get())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;

    MOCK_RET;
}

ze_result_t zeKernelCreate(ze_module_handle_t hModule, const ze_kernel_desc_t *desc, ze_kernel_handle_t *phKernel) {
    MOCK_CNT_CALL;
    if (hModule != ModuleHandle.get())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;

    *phKernel = KernelHandle.get();

    MOCK_RET;
}

ze_result_t zeKernelDestroy(ze_kernel_handle_t hKernel) {
    MOCK_CNT_CALL;
    if (hKernel != KernelHandle.get())
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;

    MOCK_RET;
}

ze_result_t zeKernelSetArgumentValue(ze_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
                                     const void *pArgValue) {
    MOCK_CNT_CALL;
    MOCK_RET;
}

ze_result_t zeKernelSetIndirectAccess(ze_kernel_handle_t hKernel, ze_kernel_indirect_access_flags_t) {
    MOCK_CNT_CALL;
    MOCK_RET;
}

ze_result_t zeCommandListAppendLaunchKernel(ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel,
                                            const ze_group_count_t *pLaunchFuncArgs, ze_event_handle_t hSignalEvent,
                                            uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents) {
    MOCK_CNT_CALL;
    if (hCommandList != CmdListHandle.get() || hKernel != KernelHandle.get() || !pLaunchFuncArgs)
        return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    if (MOCK_SHOULD_SUCCEED)
        Config::addToCmdList(CmdListElem::KernelLaunch);
    MOCK_RET;
}
} // namespace driver
} // namespace mock
} // namespace testing
} // namespace ispcrt

#if defined(__cplusplus)
extern "C" {
#endif

ze_result_t zeGetGlobalProcAddrTable(ze_api_version_t version, ze_global_dditable_t *pDdiTable) {
    pDdiTable->pfnInit = ispcrt::testing::mock::driver::zeInit;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeGetDriverProcAddrTable(ze_api_version_t version, ze_driver_dditable_t *pDdiTable) {
    pDdiTable->pfnGet = ispcrt::testing::mock::driver::zeDriverGet;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeGetDeviceProcAddrTable(ze_api_version_t version, ze_device_dditable_t *pDdiTable) {
    pDdiTable->pfnGet = ispcrt::testing::mock::driver::zeDeviceGet;
    pDdiTable->pfnGetProperties = ispcrt::testing::mock::driver::zeDeviceGetProperties;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeGetContextProcAddrTable(ze_api_version_t version, ze_context_dditable_t *pDdiTable) {
    pDdiTable->pfnCreate = ispcrt::testing::mock::driver::zeContextCreate;
    pDdiTable->pfnDestroy = ispcrt::testing::mock::driver::zeContextDestroy;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeGetCommandQueueProcAddrTable(ze_api_version_t version, ze_command_queue_dditable_t *pDdiTable) {
    pDdiTable->pfnCreate = ispcrt::testing::mock::driver::zeCommandQueueCreate;
    pDdiTable->pfnDestroy = ispcrt::testing::mock::driver::zeCommandQueueDestroy;
    pDdiTable->pfnExecuteCommandLists = ispcrt::testing::mock::driver::zeCommandQueueExecuteCommandLists;
    pDdiTable->pfnSynchronize = ispcrt::testing::mock::driver::zeCommandQueueSynchronize;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeGetCommandListProcAddrTable(ze_api_version_t version, ze_command_list_dditable_t *pDdiTable) {
    pDdiTable->pfnCreate = ispcrt::testing::mock::driver::zeCommandListCreate;
    pDdiTable->pfnDestroy = ispcrt::testing::mock::driver::zeCommandListDestroy;
    pDdiTable->pfnClose = ispcrt::testing::mock::driver::zeCommandListClose;
    pDdiTable->pfnReset = ispcrt::testing::mock::driver::zeCommandListReset;
    pDdiTable->pfnAppendBarrier = ispcrt::testing::mock::driver::zeCommandListAppendBarrier;
    pDdiTable->pfnAppendMemoryCopy = ispcrt::testing::mock::driver::zeCommandListAppendMemoryCopy;
    pDdiTable->pfnAppendLaunchKernel = ispcrt::testing::mock::driver::zeCommandListAppendLaunchKernel;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeGetEventProcAddrTable(ze_api_version_t version, ze_event_dditable_t *pDdiTable) {
    pDdiTable->pfnCreate = ispcrt::testing::mock::driver::zeEventCreate;
    pDdiTable->pfnDestroy = ispcrt::testing::mock::driver::zeEventDestroy;
    pDdiTable->pfnQueryKernelTimestamp = ispcrt::testing::mock::driver::zeEventQueryKernelTimestamp;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeGetEventPoolProcAddrTable(ze_api_version_t version, ze_event_pool_dditable_t *pDdiTable) {
    pDdiTable->pfnCreate = ispcrt::testing::mock::driver::zeEventPoolCreate;
    pDdiTable->pfnDestroy = ispcrt::testing::mock::driver::zeEventPoolDestroy;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeGetKernelProcAddrTable(ze_api_version_t version, ze_kernel_dditable_t *pDdiTable) {
    pDdiTable->pfnCreate = ispcrt::testing::mock::driver::zeKernelCreate;
    pDdiTable->pfnDestroy = ispcrt::testing::mock::driver::zeKernelDestroy;
    pDdiTable->pfnSetArgumentValue = ispcrt::testing::mock::driver::zeKernelSetArgumentValue;
    pDdiTable->pfnSetIndirectAccess = ispcrt::testing::mock::driver::zeKernelSetIndirectAccess;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeGetMemProcAddrTable(ze_api_version_t version, ze_mem_dditable_t *pDdiTable) {
    pDdiTable->pfnAllocDevice = ispcrt::testing::mock::driver::zeMemAllocDevice;
    pDdiTable->pfnFree = ispcrt::testing::mock::driver::zeMemFree;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeGetModuleProcAddrTable(ze_api_version_t version, ze_module_dditable_t *pDdiTable) {
    pDdiTable->pfnCreate = ispcrt::testing::mock::driver::zeModuleCreate;
    pDdiTable->pfnDestroy = ispcrt::testing::mock::driver::zeModuleDestroy;
    return ZE_RESULT_SUCCESS;
}

#define MOCK_DDI_FUN(Fn, TT)                                                                                           \
    ze_result_t Fn(ze_api_version_t version, TT *pDdiTable) { return ZE_RESULT_SUCCESS; }

MOCK_DDI_FUN(zeGetFenceProcAddrTable, ze_fence_dditable_t)
MOCK_DDI_FUN(zeGetImageProcAddrTable, ze_image_dditable_t)
MOCK_DDI_FUN(zeGetModuleBuildLogProcAddrTable, ze_module_build_log_dditable_t)
MOCK_DDI_FUN(zeGetPhysicalMemProcAddrTable, ze_physical_mem_dditable_t)
MOCK_DDI_FUN(zeGetSamplerProcAddrTable, ze_sampler_dditable_t)
MOCK_DDI_FUN(zeGetVirtualMemProcAddrTable, ze_virtual_mem_dditable_t)

// ZES
MOCK_DDI_FUN(zesGetDeviceProcAddrTable, zes_device_dditable_t)
MOCK_DDI_FUN(zesGetDriverProcAddrTable, zes_driver_dditable_t)
MOCK_DDI_FUN(zesGetDiagnosticsProcAddrTable, zes_diagnostics_dditable_t)
MOCK_DDI_FUN(zesGetEngineProcAddrTable, zes_engine_dditable_t)
MOCK_DDI_FUN(zesGetFabricPortProcAddrTable, zes_fabric_port_dditable_t)
MOCK_DDI_FUN(zesGetFanProcAddrTable, zes_fan_dditable_t)
MOCK_DDI_FUN(zesGetFirmwareProcAddrTable, zes_firmware_dditable_t)
MOCK_DDI_FUN(zesGetFrequencyProcAddrTable, zes_frequency_dditable_t)
MOCK_DDI_FUN(zesGetLedProcAddrTable, zes_led_dditable_t)
MOCK_DDI_FUN(zesGetMemoryProcAddrTable, zes_memory_dditable_t)
MOCK_DDI_FUN(zesGetPerformanceFactorProcAddrTable, zes_performance_factor_dditable_t)
MOCK_DDI_FUN(zesGetPowerProcAddrTable, zes_power_dditable_t)
MOCK_DDI_FUN(zesGetPsuProcAddrTable, zes_psu_dditable_t)
MOCK_DDI_FUN(zesGetRasProcAddrTable, zes_ras_dditable_t)
MOCK_DDI_FUN(zesGetSchedulerProcAddrTable, zes_scheduler_dditable_t)
MOCK_DDI_FUN(zesGetStandbyProcAddrTable, zes_standby_dditable_t)
MOCK_DDI_FUN(zesGetTemperatureProcAddrTable, zes_temperature_dditable_t)

// ZET
MOCK_DDI_FUN(zetGetDeviceProcAddrTable, zet_device_dditable_t)
MOCK_DDI_FUN(zetGetContextProcAddrTable, zet_context_dditable_t)
MOCK_DDI_FUN(zetGetCommandListProcAddrTable, zet_command_list_dditable_t)
MOCK_DDI_FUN(zetGetKernelProcAddrTable, zet_kernel_dditable_t)
MOCK_DDI_FUN(zetGetModuleProcAddrTable, zet_module_dditable_t)
MOCK_DDI_FUN(zetGetDebugProcAddrTable, zet_debug_dditable_t)
MOCK_DDI_FUN(zetGetMetricProcAddrTable, zet_metric_dditable_t)
MOCK_DDI_FUN(zetGetMetricGroupProcAddrTable, zet_metric_group_dditable_t)
MOCK_DDI_FUN(zetGetMetricQueryProcAddrTable, zet_metric_query_dditable_t)
MOCK_DDI_FUN(zetGetMetricQueryPoolProcAddrTable, zet_metric_query_pool_dditable_t)
MOCK_DDI_FUN(zetGetMetricStreamerProcAddrTable, zet_metric_streamer_dditable_t)
MOCK_DDI_FUN(zetGetTracerExpProcAddrTable, zet_tracer_exp_dditable_t)

#undef MOCK_DDI_FUN

#if defined(__cplusplus)
}
#endif
