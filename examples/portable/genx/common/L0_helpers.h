/*
  Copyright (c) 2020, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef L0_HELPERS_H 
#define L0_HELPERS_H 

#include <level_zero/ze_api.h>
#include <level_zero/zet_metric.h>
#include <cassert>
#include <fstream>

#include "common_helpers.h"

#define L0_SAFE_CALL(call)                                                                                             \
    {                                                                                                                  \
        auto status = (call);                                                                                          \
        if (status != 0) {                                                                                             \
            fprintf(stderr, "%s:%d: L0 error %d\n", __FILE__, __LINE__, (int)status);                                  \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }


namespace hostutil {
   void L0InitContext(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                          ze_command_queue_handle_t &hCommandQueue, const char* filename) {
        L0_SAFE_CALL(zeInit(ZE_INIT_FLAG_NONE));

        // Discover all the driver instances
        uint32_t driverCount = 0;
        L0_SAFE_CALL(zeDriverGet(&driverCount, nullptr));

        ze_driver_handle_t *allDrivers = (ze_driver_handle_t *)malloc(driverCount * sizeof(ze_driver_handle_t));
        L0_SAFE_CALL(zeDriverGet(&driverCount, allDrivers));

        // Find a driver instance with a GPU device
        for (uint32_t i = 0; i < driverCount; ++i) {
            uint32_t deviceCount = 0;
            hDriver = allDrivers[i];
            L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, nullptr));
            ze_device_handle_t *allDevices = (ze_device_handle_t *)malloc(deviceCount * sizeof(ze_device_handle_t));
            L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, allDevices));
            for (uint32_t d = 0; d < deviceCount; ++d) {
                ze_device_properties_t device_properties;
                L0_SAFE_CALL(zeDeviceGetProperties(allDevices[d], &device_properties));
                if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
                    hDevice = allDevices[d];
                    break;
                }
            }
            free(allDevices);
            if (nullptr != hDevice) {
                break;
            }
        }
        free(allDrivers);
        assert(hDriver);
        assert(hDevice);
        // Create a command queue
        ze_command_queue_desc_t commandQueueDesc = {ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT, ZE_COMMAND_QUEUE_FLAG_NONE,
                                                    ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL, 0};
        L0_SAFE_CALL(zeCommandQueueCreate(hDevice, &commandQueueDesc, &hCommandQueue));

        std::ifstream ins;
        // FIXME
        std::string fn = filename;
        ins.open(fn, std::ios::binary);
        if (!ins.good()) {
            fprintf(stderr, "Open %s failed\n", fn.c_str());
            return;
        }

        ins.seekg(0, std::ios::end);
        size_t codeSize = ins.tellg();
        ins.seekg(0, std::ios::beg);

        if (codeSize == 0) {
            return;
        }

        unsigned char *codeBin = new unsigned char[codeSize];
        if (!codeBin) {
            return;
        }

        ins.read((char *)codeBin, codeSize);
        ins.close();

        ze_module_desc_t moduleDesc = {ZE_MODULE_DESC_VERSION_CURRENT, //
                                       ZE_MODULE_FORMAT_IL_SPIRV,      //
                                       codeSize,                       //
                                       codeBin,                        //
                                       "-cmc"};
        L0_SAFE_CALL(zeModuleCreate(hDevice, &moduleDesc, &hModule, nullptr));
    }
};

void L0Create_Kernel(ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                            ze_command_list_handle_t &hCommandList, ze_kernel_handle_t &hKernel, const char *name) {
    // Create a command list
    ze_command_list_desc_t commandListDesc = {ZE_COMMAND_LIST_DESC_VERSION_CURRENT, ZE_COMMAND_LIST_FLAG_NONE};
    L0_SAFE_CALL(zeCommandListCreate(hDevice, &commandListDesc, &hCommandList));
    ze_kernel_desc_t kernelDesc = {ZE_KERNEL_DESC_VERSION_CURRENT, //
                                   ZE_KERNEL_FLAG_NONE,            //
                                   name};

    L0_SAFE_CALL(zeKernelCreate(hModule, &kernelDesc, &hKernel));
}

ze_result_t FindMetricGroup( ze_device_handle_t hDevice,
                               const char* pMetricGroupName,
                               uint32_t desiredSamplingType,
                               zet_metric_group_handle_t* phMetricGroup )
{
    // Obtain available metric groups for the specific device
    uint32_t metricGroupCount = 0;
    L0_SAFE_CALL(zetMetricGroupGet( hDevice, &metricGroupCount, nullptr ));
    zet_metric_group_handle_t* phMetricGroups = (zet_metric_group_handle_t*)malloc(metricGroupCount * sizeof(zet_metric_group_handle_t));
    L0_SAFE_CALL(zetMetricGroupGet( hDevice, &metricGroupCount, phMetricGroups ));
    // Iterate over all metric groups available
    for( uint32_t i = 0; i < metricGroupCount; i++ )
    {
        // Get metric group under index 'i' and its properties
        zet_metric_group_properties_t metricGroupProperties;
        L0_SAFE_CALL(zetMetricGroupGetProperties( phMetricGroups[i], &metricGroupProperties ));
        printf("Metric Group: %s\n", metricGroupProperties.name);
        // Check whether the obtained metric group supports the desired sampling type
        if((metricGroupProperties.samplingType & desiredSamplingType) == desiredSamplingType)
        {
            // Check whether the obtained metric group has the desired name
            if( strcmp( pMetricGroupName, metricGroupProperties.name ) == 0 )
            {
                *phMetricGroup = phMetricGroups[i];
                break;
            }
        }
    }
    free(phMetricGroups);
}

ze_result_t CalculateMetricsExample( zet_metric_group_handle_t hMetricGroup,
                                       size_t rawSize, uint8_t* rawData )
{
    // Calculate metric data
    uint32_t numMetricValues = 0;
    L0_SAFE_CALL(zetMetricGroupCalculateMetricValues( hMetricGroup, rawSize, rawData, &numMetricValues, nullptr ));
    zet_typed_value_t* metricValues = (zet_typed_value_t*)malloc( numMetricValues * sizeof(zet_typed_value_t) );
    L0_SAFE_CALL(zetMetricGroupCalculateMetricValues( hMetricGroup, rawSize, rawData, &numMetricValues, metricValues ));
    // Obtain available metrics for the specific metric group
    uint32_t metricCount = 0;
    L0_SAFE_CALL(zetMetricGet( hMetricGroup, &metricCount, nullptr ));
    zet_metric_handle_t* phMetrics = (zet_metric_handle_t*)malloc(metricCount * sizeof(zet_metric_handle_t));
    L0_SAFE_CALL(zetMetricGet( hMetricGroup, &metricCount, phMetrics ));
    // Print metric results
    uint32_t numReports = numMetricValues / metricCount;
    for( uint32_t report = 0; report < numReports; ++report )
    {
        printf("Report: %d\n", report);
        for( uint32_t metric = 0; metric < metricCount; ++metric )
        {
            zet_typed_value_t data = metricValues[report * metricCount + metric];
            zet_metric_properties_t metricProperties;
            L0_SAFE_CALL(zetMetricGetProperties( phMetrics[ metric ], &metricProperties ));
            printf("Metric: %s\n", metricProperties.name );
            switch( data.type )
            {
            case ZET_VALUE_TYPE_UINT32:
                printf(" Value: %lu\n", data.value.ui32 );
                break;
            case ZET_VALUE_TYPE_UINT64:
                printf(" Value: %llu\n", data.value.ui64 );
                break;
            case ZET_VALUE_TYPE_FLOAT32:
                printf(" Value: %f\n", data.value.fp32 );
                break;
            case ZET_VALUE_TYPE_FLOAT64:
                printf(" Value: %f\n", data.value.fp64 );
                break;
            case ZET_VALUE_TYPE_BOOL8:
                if( data.value.ui32 )
                    printf(" Value: true\n" );
                else
                    printf(" Value: false\n" );
                break;
            default:
                break;
            };
        }
    }
    free(metricValues);
    free(phMetrics);
}

#endif
