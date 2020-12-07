/*
 * Copyright (c) 2020, Intel Corporation
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

    // Transformation passes
    void transformStage1(const std::vector<float>& in); // ISPC
    void transformStage2(); // DPC++
    std::vector<float> transformStage3(); // ISPC

    // Validation is done on the CPU
    std::vector<float> transformCpu(const std::vector<float>& in);

  private:
    bool m_initialized{false};
    unsigned m_count{0};
    float *m_shared_data{nullptr};
    ze_driver_handle_t m_driver{nullptr};
    ze_device_handle_t m_device{nullptr};
    ze_module_handle_t m_module{nullptr};
    ze_kernel_handle_t m_kernel1{nullptr};
    ze_kernel_handle_t m_kernel2{nullptr};
    ze_context_handle_t m_context{nullptr};
    ze_command_list_handle_t m_command_list{nullptr};
    ze_command_queue_handle_t m_command_queue{nullptr};
};
