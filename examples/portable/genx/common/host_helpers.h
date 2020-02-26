/*
  Copyright (c) 2019, Intel Corporation
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

#ifndef HOST_HELPERS_H
#define HOST_HELPERS_H

#include <cstdlib>

#include "cm_rt_helpers.h"
#include "common_helpers.h"
#include "isa_helpers.h"

#include <chrono>

namespace hostutil {

void CMInitContext(CmDevice *&device, CmKernel *&kernel, CmProgram *&program, const char *isa_file_name,
                   const char *func_name) {
    unsigned int version = 0;
    cm_result_check(::CreateCmDevice(device, version));
    if (version < CM_1_0) {
        std::cerr << "The runtime API version is later than runtime DLL version\n";
        exit(1);
    }

    std::string isa_code = cm::util::isa::loadFile(isa_file_name);
    if (isa_code.size() == 0) {
        std::cerr << "Error: empty ISA binary.\n";
        exit(1);
    }

    cm_result_check(device->LoadProgram(const_cast<char *>(isa_code.data()), isa_code.size(), program));
    cm_result_check(device->CreateKernel(program, func_name, kernel));
}

Timings execute(CmDevice *device, CmKernel *kernel, int threadSpaceWidth = 1, int threadSpaceHeight = 1,
                unsigned int niter = 1, bool flush = false, unsigned int timeout = 2000) {
    printf("Thread-group setting: %d x %d \n", threadSpaceWidth, threadSpaceHeight);

    CmQueue *queue = nullptr;
    cm_result_check(device->CreateQueue(queue));
    if (flush)
        cm_result_check(device->InitPrintBuffer());

    CmTask *task = nullptr;
    cm_result_check(device->CreateTask(task));
    cm_result_check(task->AddSync());
    cm_result_check(task->AddKernel(kernel));

    CmThreadGroupSpace *pts = nullptr;
    device->CreateThreadGroupSpace(1, 1, threadSpaceWidth, threadSpaceHeight, pts);

    CmEvent *sync = nullptr;

    UINT64 kernel_ns = 0;
    UINT64 host_ns = 0;

    for (int i = 0; i < niter; i++) {
        UINT64 execution_time = 0;
        auto wct = std::chrono::system_clock::now();

        cm_result_check(queue->EnqueueWithGroup(task, sync, pts));

        auto dur = (std::chrono::system_clock::now() - wct);
        auto secs = std::chrono::duration_cast<std::chrono::nanoseconds>(dur);

        cm_result_check(sync->WaitForTaskFinished(timeout));
        cm_result_check(sync->GetExecutionTime(execution_time));

        host_ns += secs.count();
        kernel_ns += execution_time;
    }

    if (flush)
        cm_result_check(device->FlushPrintBuffer());

    return Timings(kernel_ns, host_ns);
}

} // namespace hostutil

#endif HOST_HELPERS_H
