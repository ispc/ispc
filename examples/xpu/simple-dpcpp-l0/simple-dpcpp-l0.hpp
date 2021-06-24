/*
 * Copyright (c) 2020-2021, Intel Corporation
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

class DpcppApp {
  public:
    DpcppApp() = default;

    void initialize();
    bool run();
    void cleanup();
    std::vector<float> transformIspc(const std::vector<float> &in);
    std::vector<float> transformDpcpp(const std::vector<float> &in);

  private:
    bool initialized{false};
    ze_driver_handle_t m_driver{nullptr};
    ze_device_handle_t m_device{nullptr};
    ze_module_handle_t m_module{nullptr};
    ze_kernel_handle_t m_kernel{nullptr};
    ze_context_handle_t m_context{nullptr};
    ze_command_list_handle_t m_command_list{nullptr};
    ze_command_queue_handle_t m_command_queue{nullptr};
};
