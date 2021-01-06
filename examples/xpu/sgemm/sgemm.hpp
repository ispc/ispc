/*
 * Copyright (c) 2019-2020, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <level_zero/ze_api.h>

class SGEMMApp {
  public:
    SGEMMApp() = default;
    SGEMMApp(bool verbose) : m_verbose{verbose} {};

    struct RunResult {
        uint64_t cpuTime;
        uint64_t gpuTime;
        bool valid;

        RunResult() = default;
    };

    void initialize();
    void run(RunResult &result, int m, int niter, int gx, int gy, bool validate);
    void cleanup();

  private:
    bool initialized{false};
    bool m_verbose{true};
    ze_driver_handle_t m_driver{nullptr};
    ze_device_handle_t m_device{nullptr};
    uint64_t m_timestamp_freq{0};
    ze_event_pool_handle_t m_pool{nullptr};
    ze_event_handle_t m_event{nullptr};
    ze_module_handle_t m_module{nullptr};
    ze_kernel_handle_t m_kernel{nullptr};
    ze_context_handle_t m_context{nullptr};
    ze_command_list_handle_t m_command_list{nullptr};
    ze_command_queue_handle_t m_command_queue{nullptr};
    void *m_device_ptr{nullptr};
};
