// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <cstring>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <level_zero/ze_ddi.h>
#include <level_zero/zes_ddi.h>
#include <level_zero/zet_ddi.h>

namespace ispcrt {
namespace testing {
namespace mock {

enum class CmdListElem { MemoryCopy, KernelLaunch, Barrier };

namespace VendorId {
  constexpr uint32_t Intel  = 0x8086;
  constexpr uint32_t Nvidia = 0x10DE;
  constexpr uint32_t AMD    = 0x1002;
} // namespace VendorId

namespace DeviceId {
  constexpr uint32_t Gen9   = 0x9BCA;
  constexpr uint32_t Gen12  = 0xAABB; // some fake number

  constexpr uint32_t GenericNvidia = 0x1234;
  constexpr uint32_t GenericAMD    = 0x5678;
} // namespace DeviceIdIntel

struct DeviceProperties {
    uint32_t vendorId;
    uint32_t deviceId;

    DeviceProperties() = default;
    DeviceProperties(uint32_t vendorId, uint32_t deviceId) : vendorId(vendorId), deviceId(deviceId) {}
};

struct CallCounters {
    static void inc(const std::string& fun);
    static int  get(const std::string& fun);
    static void resetAll();
    static void resetOne(const std::string& fun);
  private:
    static std::unordered_map<std::string, int> counters;
};

class Config {
  public:
    static void setRetValue(const std::string &fun, ze_result_t result);
    static ze_result_t getRetValue(const std::string &fun);
    static void cleanup();
    static void addToCmdList(CmdListElem cle);
    static void resetCmdList();
    static void closeCmdList();
    static bool isCmdListClosed();
    static bool checkCmdList(const std::vector<CmdListElem>& expected);
    static void setDeviceCount(uint32_t count);
    static DeviceProperties* getDevicePtr(uint32_t deviceIdx);
    static uint32_t getDeviceCount();
    static void setExpectedDevice(uint32_t deviceIdx);
    static uint32_t getExpectedDevice();
    static void setDeviceProperties(uint32_t deviceIdx, const DeviceProperties& dp);

  private:
    static std::unordered_map<std::string, ze_result_t> resultsMap;
    static std::vector<CmdListElem> cmdList;
    static bool cmdListOpened;
    static std::vector<DeviceProperties> devices;
    static uint32_t expectedDevice;
};

} // namespace mock
} // namespace testing
} // namespace ispcrt
