// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ispcrt.hpp"
#include "ze_mock.h"

#include "gtest/gtest.h"

#include <stdlib.h>

namespace ispcrt {
namespace testing {
namespace mock {

// Base fixture for mock tests
class MockTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ResetError();
        Config::cleanup();
        setenv("ISPCRT_MOCK_DEVICE", "y", 1);
        // hijak ispcrt errors - we need it to test error handling
        ispcrtSetErrorFunc([](ISPCRTError e, const char *m) { sm_rt_error = e; });
        EXPECT_EQ(m_device, 0);
        m_device = Device(ISPCRT_DEVICE_TYPE_GPU);
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        EXPECT_NE(m_device, 0);
    }

    void TearDown() override {
        ResetError();
        Config::cleanup();
        // Make sure we can still recreate a device object
        ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        ResetError();
    }

    void ResetError() { sm_rt_error = ISPCRT_NO_ERROR; }

    ispcrt::Device m_device;
    // TODO: not great it's a static, but for now the ISPCRT error reporting
    // does not support any kind of context allowing to pass 'this' pointer
    static ISPCRTError sm_rt_error;
};

ISPCRTError MockTest::sm_rt_error;

class MockTestWithModule : public MockTest {
  protected:
    void SetUp() override {
        MockTest::SetUp();
        EXPECT_EQ(m_module, 0);
        m_module = Module(m_device, "");
        EXPECT_NE(m_module, 0);
    }

    ispcrt::Module m_module;
};

class MockTestWithModuleQueueKernel : public MockTestWithModule {
  protected:
    void SetUp() override {
        MockTestWithModule::SetUp();
        EXPECT_EQ(m_task_queue, 0);
        EXPECT_EQ(m_kernel, 0);
        m_task_queue = TaskQueue(m_device);
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        EXPECT_NE(m_task_queue, 0);
        m_kernel = Kernel(m_device, m_module, "");
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        EXPECT_NE(m_kernel, 0);
    }

    ispcrt::TaskQueue m_task_queue;
    ispcrt::Kernel m_kernel;
};

/////////////////////////////////////////////////////////////////////
// Device tests

TEST_F(MockTest, Device_Constructor_zeInit) {
    // Make sure we can re-create a device even if first try failed
    Config::setRetValue("zeInit", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTest, Device_Constructor_zeDeviceGet) {
    // Make sure we can re-create a device even if first try failed
    Config::setRetValue("zeDeviceGet", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTest, Device_Constructor_zeDeviceGetProperties) {
    // Make sure we can re-create a device even if first try failed
    Config::setRetValue("zeDeviceGetProperties", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTest, Device_Constructor_zeContextCreate) {
    // Make sure we can re-create a device even if first try failed
    Config::setRetValue("zeContextCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

/////////////////////////////////////////////////////////////////////
// Module tests

TEST_F(MockTest, Module_Constructor) {
    // Simply create a module
    ispcrt::Module m(m_device, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTest, Module_Constructor_zeModuleCreate) {
    // Check if error is reported from module constructor
    Config::setRetValue("zeModuleCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Module m(m_device, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

/////////////////////////////////////////////////////////////////////
// Kernel tests

TEST_F(MockTestWithModule, Kernel_Constructor) {
    // Simply create a kernel
    ispcrt::Kernel k(m_device, m_module, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTestWithModule, Kernel_Constructor_zeKernelCreate) {
    // Check if error is reported from kernel constructor
    Config::setRetValue("zeKernelCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Kernel k(m_device, m_module, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

/////////////////////////////////////////////////////////////////////
// Memory allocation tests
TEST_F(MockTest, ArrayObj) {
    // Simply create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    // devicePtr() does actual allocation
    auto dev_buf_ptr = buf_dev.devicePtr();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTest, ArrayObj_zeMemAllocDevice) {
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    Config::setRetValue("zeMemAllocDevice", ZE_RESULT_ERROR_DEVICE_LOST);
    auto dev_buf_ptr = buf_dev.devicePtr();
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    // Check that nullptr is returned
    ASSERT_EQ(dev_buf_ptr, nullptr);
}

/////////////////////////////////////////////////////////////////////
// TaskQueue tests

TEST_F(MockTest, TaskQueue_Constructor) {
    // Simply create a task queue
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTest, TaskQueue_Constructor_zeEventPoolCreate) {
    Config::setRetValue("zeEventPoolCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTest, TaskQueue_Constructor_zeCommandListCreate) {
    Config::setRetValue("zeCommandListCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTest, TaskQueue_Constructor_zeCommandQueueCreate) {
    Config::setRetValue("zeCommandQueueCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTest, TaskQueue_CopyToDevice) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy"
    tq.copyToDevice(buf_dev);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy, CmdListElem::Barrier}));
}

TEST_F(MockTest, TaskQueue_CopyToDevice_zeCommandListAppendMemoryCopy) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy", but fail
    Config::setRetValue("zeCommandListAppendMemoryCopy", ZE_RESULT_ERROR_DEVICE_LOST);
    tq.copyToDevice(buf_dev);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    ASSERT_TRUE(Config::checkCmdList({}));
}

TEST_F(MockTest, TaskQueue_Barrier_zeCommandListAppendBarrier) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    Config::setRetValue("zeCommandListAppendBarrier", ZE_RESULT_ERROR_DEVICE_LOST);
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    ASSERT_TRUE(Config::checkCmdList({}));
}

// Normal kernel launch (plus a few memory transfers) - but no waiting on future
TEST_F(MockTestWithModuleQueueKernel, TaskQueue_FullKernelLaunchNoFuture) {
    auto tq = m_task_queue;
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy"
    tq.copyToDevice(buf_dev);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy, CmdListElem::Barrier}));
    tq.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy, CmdListElem::Barrier, CmdListElem::KernelLaunch}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList(
        {CmdListElem::MemoryCopy, CmdListElem::Barrier, CmdListElem::KernelLaunch, CmdListElem::Barrier}));
    tq.sync();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({}));
}

// Normal kernel launch (plus a few memory transfers)
TEST_F(MockTestWithModuleQueueKernel, TaskQueue_FullKernelLaunch) {
    auto tq = m_task_queue;
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy"
    tq.copyToDevice(buf_dev);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy, CmdListElem::Barrier}));
    auto f = tq.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy, CmdListElem::Barrier, CmdListElem::KernelLaunch}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList(
        {CmdListElem::MemoryCopy, CmdListElem::Barrier, CmdListElem::KernelLaunch, CmdListElem::Barrier}));
    tq.sync();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({}));
    ASSERT_TRUE(f.valid());
}

TEST_F(MockTestWithModuleQueueKernel, TaskQueue_KernelLaunchNoSync) {
    auto tq = m_task_queue;
    auto f = tq.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::KernelLaunch}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::KernelLaunch, CmdListElem::Barrier}));
    // Future should not be signaled
    ASSERT_FALSE(f.valid());
}

TEST_F(MockTestWithModuleQueueKernel, TaskQueue_Launch_zeKernelSetArgumentValue) {
    Config::setRetValue("zeKernelSetArgumentValue", ZE_RESULT_ERROR_DEVICE_LOST);
    m_task_queue.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    ASSERT_TRUE(Config::checkCmdList({}));
}

TEST_F(MockTestWithModuleQueueKernel, DISABLED_TaskQueue_Launch_zeCommandListAppendLaunchKernel) {
    Config::setRetValue("zeCommandListAppendLaunchKernel", ZE_RESULT_ERROR_DEVICE_LOST);
    m_task_queue.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    ASSERT_TRUE(Config::checkCmdList({}));
}


} // namespace mock
} // namespace testing
} // namespace ispcrt