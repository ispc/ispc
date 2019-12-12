#ifndef L0_HELPERS_H 
#define L0_HELPERS_H 

#include <level_zero/ze_api.h>
#include <cassert>
#include <fstream>

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
                          ze_command_queue_handle_t &hCommandQueue) {
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
        std::string fn = "test.spv";
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

template <class T, size_t N> struct alignas(4096) PageAlignedArray { T data[N]; };

#endif
