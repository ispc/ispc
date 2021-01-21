// Copyright Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <stdlib.h>
#include <unordered_map>
#include <vector>

#include <level_zero/ze_ddi.h>
#include <level_zero/zes_ddi.h>
#include <level_zero/zet_ddi.h>

namespace ispcrt {
namespace testing {
namespace mock {

enum class CmdListElem { MemoryCopy, KernelLaunch, Barrier };

class Config {
  public:
    static void setRetValue(const std::string &fun, ze_result_t result);
    static ze_result_t getRetValue(const std::string &fun);
    static void cleanup();
    static void addToCmdList(CmdListElem cle);
    static void resetCmdList();
    static void closeCmdList();
    static bool isCmdListClosed();
    static bool checkCmdList(std::vector<CmdListElem> expected);

  private:
    static std::unordered_map<std::string, ze_result_t> resultsMap;
    static std::vector<CmdListElem> cmdList;
    static bool cmdListOpened;
};

} // namespace mock
} // namespace testing
} // namespace ispcrt
