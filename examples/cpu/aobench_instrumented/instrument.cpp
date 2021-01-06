/*
  Copyright (c) 2010-2011, Intel Corporation
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

#include "instrument.h"
#include <assert.h>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdio.h>
#include <string>

struct CallInfo {
    CallInfo() { count = laneCount = allOff = 0; }
    int count;
    int laneCount;
    int allOff;
};

static std::map<std::string, CallInfo> callInfo;

int countbits(uint64_t i) {
    int ret = 0;
    while (i) {
        if (i & 0x1)
            ++ret;
        i >>= 1;
    }
    return ret;
}

// Callback function that ispc compiler emits calls to when --instrument
// command-line flag is given while compiling.
void ISPCInstrument(const char *fn, const char *note, int line, uint64_t mask) {
    std::stringstream s;
    s << fn << "(" << std::setfill('0') << std::setw(4) << line << ") - " << note;

    // Find or create a CallInfo instance for this callsite.
    CallInfo &ci = callInfo[s.str()];

    // And update its statistics...
    ++ci.count;
    if (mask == 0)
        ++ci.allOff;
    ci.laneCount += countbits(mask);
}

void ISPCPrintInstrument() {
    // When program execution is done, go through the stats and print them
    // out.  (This function is called by ao.cpp).
    std::map<std::string, CallInfo>::iterator citer = callInfo.begin();
    while (citer != callInfo.end()) {
        CallInfo &ci = citer->second;
        float activePct = 100.f * ci.laneCount / (4.f * ci.count);
        float allOffPct = 100.f * ci.allOff / ci.count;
        printf("%s: %d calls (%d / %.2f%% all off!), %.2f%% active lanes\n", citer->first.c_str(), ci.count, ci.allOff,
               allOffPct, activePct);
        ++citer;
    }
}
