// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ze_mock.h"

#include <cstring>

namespace ispcrt {
namespace testing {
namespace mock {

std::unordered_map<std::string, ze_result_t> Config::resultsMap;
std::vector<CmdListElem> Config::cmdList;
bool Config::cmdListOpened = true;

void Config::setRetValue(const std::string &fun, ze_result_t result) { resultsMap[fun] = result; }

ze_result_t Config::getRetValue(const std::string &fun) {
    return resultsMap.count(fun) == 0 ? ZE_RESULT_SUCCESS : resultsMap[fun];
}

void Config::cleanup() {
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

bool Config::checkCmdList(std::vector<CmdListElem> expected) { return expected == cmdList; }

} // namespace mock
} // namespace testing
} // namespace ispcrt