// Copyright 2020-2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ze_mock.h"

namespace ispcrt {
namespace testing {
namespace mock {

// Counters

std::unordered_map<std::string, int> CallCounters::counters;

void CallCounters::inc(const std::string &fun) { counters[fun]++; }

int CallCounters::get(const std::string &fun) { return counters[fun]; }
void CallCounters::resetAll() { counters.clear(); }

void CallCounters::resetOne(const std::string &fun) { counters[fun] = 0; }

// Config

std::unordered_map<std::string, ze_result_t> Config::resultsMap;
std::vector<CmdListElem> Config::cmdList;
bool Config::cmdListOpened = true;
uint32_t Config::expectedDevice = 0;

const DeviceProperties DefaultGpuDevice(VendorId::Intel, DeviceId::Gen9);

std::vector<DeviceProperties> Config::devices(1, DefaultGpuDevice);

void Config::setRetValue(const std::string &fun, ze_result_t result) { resultsMap[fun] = result; }

ze_result_t Config::getRetValue(const std::string &fun) {
    return resultsMap.count(fun) == 0 ? ZE_RESULT_SUCCESS : resultsMap[fun];
}

void Config::cleanup() {
    setDeviceCount(1);
    setExpectedDevice(0);
    resetCmdList();
    resultsMap.clear();
}

void Config::addToCmdList(CmdListElem cle) { cmdList.push_back(cle); }

void Config::resetCmdList() {
    cmdList.clear();
    cmdListOpened = true;
}

void Config::closeCmdList() { cmdListOpened = false; }

bool Config::isCmdListClosed() { return !cmdListOpened; }

bool Config::checkCmdList(const std::vector<CmdListElem> &expected) { return expected == cmdList; }

void Config::setDeviceCount(uint32_t count) {
    devices.clear();
    devices.resize(count, DefaultGpuDevice);
}

void Config::setExpectedDevice(uint32_t deviceIdx) { expectedDevice = deviceIdx; }

uint32_t Config::getExpectedDevice() { return expectedDevice; }

uint32_t Config::getDeviceCount() { return devices.size(); }

DeviceProperties *Config::getDevicePtr(uint32_t deviceIdx) { return &devices[deviceIdx]; }

void Config::setDeviceProperties(uint32_t deviceIdx, const DeviceProperties &dp) {
    if (deviceIdx >= devices.size())
        throw std::runtime_error("Config::setDeviceProperties: invalid device number");
    devices[deviceIdx] = dp;
}

} // namespace mock
} // namespace testing
} // namespace ispcrt
