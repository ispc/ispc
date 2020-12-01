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
    Config::setRetValue("zeInit", ZE_RESULT_SUCCESS);
    ResetError();
    ispcrt::Device d2(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTest, Device_Constructor_zeDeviceGet) {
    // Make sure we can re-create a device even if first try failed
    Config::setRetValue("zeDeviceGet", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    Config::setRetValue("zeDeviceGet", ZE_RESULT_SUCCESS);
    ResetError();
    ispcrt::Device d2(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTest, Device_Constructor_zeDeviceGetProperties) {
    // Make sure we can re-create a device even if first try failed
    Config::setRetValue("zeDeviceGetProperties", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    Config::setRetValue("zeDeviceGetProperties", ZE_RESULT_SUCCESS);
    ResetError();
    ispcrt::Device d2(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTest, Device_Constructor_zeContextCreate) {
    // Make sure we can re-create a device even if first try failed
    Config::setRetValue("zeContextCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    Config::setRetValue("zeContextCreate", ZE_RESULT_SUCCESS);
    ResetError();
    ispcrt::Device d2(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

/////////////////////////////////////////////////////////////////////
// Module tests

TEST_F(MockTest, Module_Constructor) {
    // Simply create a module
    ispcrt::Module m(m_device, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTest, Module_Constructor_zeModuleCreate) {
    // Check if it's possible to create a module after first try failed
    Config::setRetValue("zeModuleCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Module m(m_device, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    Config::setRetValue("zeModuleCreate", ZE_RESULT_SUCCESS);
    ResetError();
    ispcrt::Module m2(m_device, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

/////////////////////////////////////////////////////////////////////
// Kernel tests

TEST_F(MockTestWithModule, Kernel_Constructor) {
    // Simply create a kernel
    ispcrt::Kernel k(m_device, m_module, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTestWithModule, Kernel_Constructor_zeKernelCreate) {
    // Check if it's possible to create a kernel after the first try failed
    Config::setRetValue("zeKernelCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Kernel k(m_device, m_module, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    Config::setRetValue("zeKernelCreate", ZE_RESULT_SUCCESS);
    ResetError();
    ispcrt::Kernel k2(m_device, m_module, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
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
    Config::setRetValue("zeMemAllocDevice", ZE_RESULT_SUCCESS);
    ResetError();
    // Reset error and try again
    dev_buf_ptr = buf_dev.devicePtr();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_NE(dev_buf_ptr, nullptr);
}

TEST_F(MockTest, ArrayObj_zeMemAllocDevice_DiffObjs) {
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    Config::setRetValue("zeMemAllocDevice", ZE_RESULT_ERROR_DEVICE_LOST);
    auto dev_buf_ptr = buf_dev.devicePtr();
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    // Check that nullptr is returned
    ASSERT_EQ(dev_buf_ptr, nullptr);
    Config::setRetValue("zeMemAllocDevice", ZE_RESULT_SUCCESS);
    ResetError();
    // Reset error and try again, but create different object
    ispcrt::Array<float> buf_dev2(m_device, buf);
    auto dev_buf_ptr2 = buf_dev2.devicePtr();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_EQ(dev_buf_ptr, nullptr);
    ASSERT_NE(dev_buf_ptr2, nullptr);
}

/////////////////////////////////////////////////////////////////////
// TaskQueue tests

TEST_F(MockTest, TaskQueue_Constructor) {
    // Simply create a task queue
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

// TODO: Enable the test when new Level Zero loader release is available
TEST_F(MockTest, DISABLED_TaskQueue_Constructor_zeEventPoolCreate) {
    // Check if it's possible to create a task queue after the first try failed
    Config::setRetValue("zeEventPoolCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    Config::setRetValue("zeEventPoolCreate", ZE_RESULT_SUCCESS);
    ResetError();
    ispcrt::TaskQueue tq2(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTest, TaskQueue_Constructor_zeCommandListCreate) {
    // Check if it's possible to create a task queue after the first try failed
    Config::setRetValue("zeCommandListCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    Config::setRetValue("zeCommandListCreate", ZE_RESULT_SUCCESS);
    ResetError();
    ispcrt::TaskQueue tq2(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTest, TaskQueue_Constructor_zeCommandQueueCreate) {
    // Check if it's possible to create a task queue after the first try failed
    Config::setRetValue("zeCommandQueueCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    Config::setRetValue("zeCommandQueueCreate", ZE_RESULT_SUCCESS);
    ResetError();
    ispcrt::TaskQueue tq2(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
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
    Config::setRetValue("zeCommandListAppendMemoryCopy", ZE_RESULT_SUCCESS);
    ResetError();
    tq.copyToDevice(buf_dev);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
}

// TODO: Write test that will check if command list is cleaned up when zeCommandQueueCreate fails

// TODO: Enable the test when new Level Zero loader release is available
// Normal kernel launch (plus a few memory transfers) - but no waiting on future
TEST_F(MockTestWithModuleQueueKernel, DISABLED_TaskQueue_FullKernelLaunchNoFuture) {
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

// TODO: Enable the test when new Level Zero loader release is available
// Normal kernel launch (plus a few memory transfers)
TEST_F(MockTestWithModuleQueueKernel, DISABLED_TaskQueue_FullKernelLaunch) {
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

// The test is disabled due to crash in EventPool destructor
// The fix for this crash is to properly deallocate all events
// that has been created in ::launch method and should be cleaned up
// in sync (but sync never happens due to - for example program exit)
TEST_F(MockTestWithModuleQueueKernel, DISABLED_TaskQueue_KernelLaunchNoSync) {
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

// TODO: Enable the test when new Level Zero loader release is available
TEST_F(MockTestWithModuleQueueKernel, DISABLED_TaskQueue_FullKernelLaunch1) {
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

} // namespace mock
} // namespace testing
} // namespace ispcrt