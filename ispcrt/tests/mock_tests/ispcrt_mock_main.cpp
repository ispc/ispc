// Copyright 2020-2021 Intel Corporation
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
        CallCounters::resetAll();
        setenv("ISPCRT_MOCK_DEVICE", "y", 1);
        // hijak ispcrt errors - we need it to test error handling
        ispcrtSetErrorFunc([](ISPCRTError e, const char *m) { sm_rt_error = e; });
    }

    void TearDown() override {
        ResetError();
        Config::cleanup();
        CallCounters::resetAll();
        ResetError();
    }

    void ResetError() { sm_rt_error = ISPCRT_NO_ERROR; }

    // TODO: not great it's a static, but for now the ISPCRT error reporting
    // does not support any kind of context allowing to pass 'this' pointer
    static ISPCRTError sm_rt_error;
};

ISPCRTError MockTest::sm_rt_error;

class MockTestWithDevice : public MockTest {
  protected:
    void SetUp() override {
        MockTest::SetUp();
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

    ispcrt::Device m_device;
};


class MockTestWithModule : public MockTestWithDevice {
  protected:
    void SetUp() override {
        MockTestWithDevice::SetUp();
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

    void testMultipleKernelLaunches(unsigned launchCnt, bool expectError = false, unsigned errorIter = 0) {
        ispcrt::TaskQueue tq(m_device); // use local queue to grab env variables setup
        std::vector<CmdListElem> expectedCmdList;
        std::vector<ispcrt::Future> futures;
        for (unsigned i = 0; i < launchCnt; i++) {
            auto f = tq.launch(m_kernel, 0);
            if (expectError && i >= errorIter) {
                ASSERT_NE(sm_rt_error, ISPCRT_NO_ERROR);
                return;
            }
            else {
                ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
            }
            futures.push_back(f);
            expectedCmdList.push_back(CmdListElem::KernelLaunch);
            ASSERT_TRUE(Config::checkCmdList(expectedCmdList));
        }
        tq.sync();
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        ASSERT_TRUE(Config::checkCmdList({}));
        for (const auto& f : futures) {
            ASSERT_TRUE(f.valid());
        }
    }

    ispcrt::TaskQueue m_task_queue;
    ispcrt::Kernel m_kernel;
};

/////////////////////////////////////////////////////////////////////
// Device tests

TEST_F(MockTest, Device_Constructor_zeInit) {
    Config::setRetValue("zeInit", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTest, Device_Constructor_zeDeviceGet) {
    Config::setRetValue("zeDeviceGet", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTest, Device_Constructor_zeDeviceGetProperties) {
    Config::setRetValue("zeDeviceGetProperties", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTest, Device_Constructor_zeContextCreate) {
    Config::setRetValue("zeContextCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Device d(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

/////////////////////////////////////////////////////////////////////
// Module tests

TEST_F(MockTestWithDevice, Module_Constructor) {
    // Simply create a module
    ASSERT_NE(m_device, 0);
    ispcrt::Module m(m_device, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTestWithDevice, Module_Constructor_zeModuleCreateWithOptions) {
    // Create module with options
    ISPCRTModuleOptions opts = {};
    ispcrt::Module m(m_device, "", opts);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTestWithDevice, Module_Constructor_zeModuleCreateWithStackSize) {
    // Create module with stack size
    ISPCRTModuleOptions opts;
    opts.stackSize = 32000;
    ispcrt::Module m(m_device, "", opts);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTestWithDevice, Module_Constructor_zeModuleCreate) {
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

TEST_F(MockTestWithModule, Kernel_Constructor_zeKernelCreateSetIndirectAccess) {
    Config::setRetValue("zeKernelSetIndirectAccess", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::Kernel k(m_device, m_module, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

/////////////////////////////////////////////////////////////////////
// Memory allocation tests
TEST_F(MockTestWithDevice, ArrayObj) {
    // Simply create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    // devicePtr() does actual allocation
    auto dev_buf_ptr = buf_dev.devicePtr();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTestWithDevice, ArrayObj_zeMemAllocDevice) {
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

TEST_F(MockTestWithDevice, TaskQueue_Constructor) {
    // Simply create a task queue
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
}

TEST_F(MockTestWithDevice, TaskQueue_Constructor_zeEventPoolCreate) {
    Config::setRetValue("zeEventPoolCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTestWithDevice, TaskQueue_Constructor_zeCommandListCreate) {
    Config::setRetValue("zeCommandListCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTestWithDevice, TaskQueue_Constructor_zeCommandQueueCreate) {
    Config::setRetValue("zeCommandQueueCreate", ZE_RESULT_ERROR_DEVICE_LOST);
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTestWithDevice, TaskQueue_CopyToDevice) {
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

TEST_F(MockTestWithDevice, TaskQueue_CopyToDevice_zeCommandListAppendMemoryCopy) {
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

TEST_F(MockTestWithDevice, TaskQueue_CopyToHost) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy"
    tq.copyToHost(buf_dev);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy, CmdListElem::Barrier}));
}

TEST_F(MockTestWithDevice, TaskQueue_CopyToHost_zeCommandListAppendMemoryCopy) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy", but fail
    Config::setRetValue("zeCommandListAppendMemoryCopy", ZE_RESULT_ERROR_DEVICE_LOST);
    tq.copyToHost(buf_dev);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTestWithDevice, TaskQueue_CopyArray) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ispcrt::Array<float> buf_copy(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy"
    tq.copyArray(buf_copy, buf_dev, buf_dev.size());
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
}

TEST_F(MockTestWithDevice, TaskQueue_CopyArray_zeCommandListAppendMemoryCopy) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_copy(m_device, buf);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy", but fail
    Config::setRetValue("zeCommandListAppendMemoryCopy", ZE_RESULT_ERROR_DEVICE_LOST);
    tq.copyArray(buf_copy, buf_dev, buf_dev.size());
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTestWithDevice, TaskQueue_CopyArray_InvalidSize) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ispcrt::Array<float> buf_copy(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // copy command should return error since the requested size if biffer than buffer size
    tq.copyArray(buf_copy, buf_dev, buf_dev.size() * 2);
    ASSERT_EQ(sm_rt_error, ISPCRT_UNKNOWN_ERROR);
}

TEST_F(MockTestWithDevice, TaskQueue_Barrier_zeCommandListAppendBarrier) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    Config::setRetValue("zeCommandListAppendBarrier", ZE_RESULT_ERROR_DEVICE_LOST);
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
    ASSERT_TRUE(Config::checkCmdList({}));
}

TEST_F(MockTestWithDevice, TaskQueue_CopyToDevice_Events) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy"
    tq.copyToDevice(buf_dev);
    // Event should be created before appending memory copy
    ASSERT_EQ(CallCounters::get("zeEventCreate"), 1);
    ASSERT_EQ(CallCounters::get("zeCommandListAppendMemoryCopy"), 1);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
    // "sync"
    tq.sync();
    ASSERT_EQ(CallCounters::get("zeCommandListClose"), 1);
    ASSERT_EQ(CallCounters::get("zeCommandQueueExecuteCommandLists"), 1);
    ASSERT_EQ(CallCounters::get("zeCommandListReset"), 1);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({}));
}

TEST_F(MockTestWithDevice, TaskQueue_CopyToDevice_Reuse_Events) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy"
    tq.copyToDevice(buf_dev);
    // Event should be created before appending memory copy
    ASSERT_EQ(CallCounters::get("zeEventCreate"), 1);
    ASSERT_EQ(CallCounters::get("zeCommandListAppendMemoryCopy"), 1);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
    // "sync"
    tq.sync();
    ASSERT_EQ(CallCounters::get("zeCommandListClose"), 1);
    ASSERT_EQ(CallCounters::get("zeCommandQueueExecuteCommandLists"), 1);
    ASSERT_EQ(CallCounters::get("zeCommandListReset"), 1);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({}));
    // "copy"
    tq.copyToDevice(buf_dev);
    // Event should be reused so the count of zeEventCreate is still 1
    ASSERT_EQ(CallCounters::get("zeEventCreate"), 1);
    ASSERT_EQ(CallCounters::get("zeEventQueryStatus"), 1);
    ASSERT_EQ(CallCounters::get("zeEventHostReset"), 1);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
    // "sync"
    tq.sync();
    ASSERT_EQ(CallCounters::get("zeCommandListClose"), 2);
    ASSERT_EQ(CallCounters::get("zeCommandQueueExecuteCommandLists"), 2);
    ASSERT_EQ(CallCounters::get("zeCommandListReset"), 2);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({}));
}

TEST_F(MockTestWithDevice, TaskQueue_CopyToHost_Events) {
    ispcrt::TaskQueue tq(m_device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy"
    tq.copyToHost(buf_dev);
    // No events should be created
    ASSERT_EQ(CallCounters::get("zeEventCreate"), 0);
    ASSERT_EQ(CallCounters::get("zeCommandListAppendMemoryCopy"), 1);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
    // "sync"
    tq.sync();
    ASSERT_EQ(CallCounters::get("zeCommandListClose"), 1);
    ASSERT_EQ(CallCounters::get("zeCommandQueueExecuteCommandLists"), 1);
    ASSERT_EQ(CallCounters::get("zeCommandListReset"), 1);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
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
    tq.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy, CmdListElem::KernelLaunch}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList(
        {CmdListElem::MemoryCopy, CmdListElem::KernelLaunch, CmdListElem::Barrier}));
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
    auto f = tq.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy, CmdListElem::KernelLaunch}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList(
        {CmdListElem::MemoryCopy, CmdListElem::KernelLaunch, CmdListElem::Barrier}));
    tq.sync();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({}));
    ASSERT_TRUE(f.valid());
}

// Try to submit a lot of kernel launches
TEST_F(MockTestWithModuleQueueKernel, TaskQueue_MultipleKernelLaunchesBasic) {
    testMultipleKernelLaunches(1000);
}

// Check some other sizes
TEST_F(MockTestWithModuleQueueKernel, TaskQueue_MultipleKernelLaunchesAdvanced) {
    auto launches = std::vector<unsigned>({100, 1000, 10000, 50000});
    for (auto l : launches) {
        testMultipleKernelLaunches(l);
    }
}

// Test the limit of number of launches
TEST_F(MockTestWithModuleQueueKernel, TaskQueue_MultipleKernelLaunchesLimit) {
    constexpr unsigned LIMIT = 100000;
    // OK to add LIMIT kernel launches
    testMultipleKernelLaunches(LIMIT);
    // But not OK to add more
    testMultipleKernelLaunches(LIMIT + 1, true, LIMIT);
    ResetError();
    Config::resetCmdList();
    // Double check that we still can enqueue correct amount of events;
    testMultipleKernelLaunches(LIMIT);

}

// Check if setting the expected maximum of kernel launches with env var works
TEST_F(MockTestWithModuleQueueKernel, TaskQueue_MultipleKernelLaunchesEnvLimitCap) {
    auto limits = std::vector<unsigned>({100, 1000, 10000});
    for (auto limit : limits) {
        setenv("ISPCRT_MAX_KERNEL_LAUNCHES", std::to_string(limit).c_str(), 1);
        testMultipleKernelLaunches(limit);
        testMultipleKernelLaunches(limit + 1, true, limit);
        ResetError();
        Config::resetCmdList();
        testMultipleKernelLaunches(limit);
        unsetenv("ISPCRT_MAX_KERNEL_LAUNCHES");
    }
}

// Check that capping the value of kernel launches limit works
// and produces expected warning
TEST_F(MockTestWithModuleQueueKernel, TaskQueue_MultipleKernelLaunchesEnvLimit) {
    constexpr const char* EXPECTED_WARNING =
        "[ISPCRT][WARNING] ISPCRT_MAX_KERNEL_LAUNCHES value too large, using 100000 instead.\n";
    // Set the limit to 200000
    setenv("ISPCRT_MAX_KERNEL_LAUNCHES", "200000", 1);
    // Check that it's OK to enqueue 100000 launches...
    ::testing::internal::CaptureStderr();
    testMultipleKernelLaunches(100000);
    EXPECT_STREQ(::testing::internal::GetCapturedStderr().c_str(), EXPECTED_WARNING);
    // ... but really the limit is 100000
    ::testing::internal::CaptureStderr();
    testMultipleKernelLaunches(100001, true, 100000);
    EXPECT_STREQ(::testing::internal::GetCapturedStderr().c_str(), EXPECTED_WARNING);
    ResetError();
    Config::resetCmdList();
    unsetenv("ISPCRT_MAX_KERNEL_LAUNCHES");
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

TEST_F(MockTestWithModuleQueueKernel, TaskQueue_Sync) {
    auto tq = m_task_queue;
    auto f = tq.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::KernelLaunch}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::KernelLaunch, CmdListElem::Barrier}));
    // Future should not be signaled
    ASSERT_FALSE(f.valid());
    ASSERT_EQ(CallCounters::get("zeCommandQueueExecuteCommandLists"), 0);
    tq.sync();
    // Now future should be valid
    ASSERT_TRUE(f.valid());
    // Execute and synchronize should be called
    ASSERT_EQ(CallCounters::get("zeCommandQueueExecuteCommandLists"), 1);
    ASSERT_EQ(CallCounters::get("zeCommandQueueSynchronize"), 1);
}

TEST_F(MockTestWithModuleQueueKernel, TaskQueue_Launch_zeKernelSetArgumentValue) {
    Config::setRetValue("zeKernelSetArgumentValue", ZE_RESULT_ERROR_DEVICE_LOST);
    // We need an argument to a kernel to make sure zeKernelSetArgumentValue is called
    std::vector<float> buf(64);
    ispcrt::Array<float> buf_dev(m_device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    m_task_queue.launch(m_kernel, buf_dev, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTestWithModuleQueueKernel, TaskQueue_Launch_zeCommandListAppendLaunchKernel) {
    Config::setRetValue("zeCommandListAppendLaunchKernel", ZE_RESULT_ERROR_DEVICE_LOST);
    m_task_queue.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTestWithModuleQueueKernel, TaskQueue_Launch_zeKernelSuggestGroupSize) {
    Config::setRetValue("zeKernelSuggestGroupSize", ZE_RESULT_ERROR_DEVICE_LOST);
    m_task_queue.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTestWithModuleQueueKernel, TaskQueue_Launch_zeKernelSetGroupSize) {
    Config::setRetValue("zeKernelSetGroupSize", ZE_RESULT_ERROR_DEVICE_LOST);
    m_task_queue.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTestWithModuleQueueKernel, TaskQueue_KernelLaunchGroupSize) {
    m_task_queue.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_EQ(CallCounters::get("zeKernelSuggestGroupSize"), 1);
    ASSERT_EQ(CallCounters::get("zeKernelSetGroupSize"), 1);
    ASSERT_EQ(CallCounters::get("zeCommandListAppendLaunchKernel"), 1);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::KernelLaunch}));
}

TEST_F(MockTestWithModuleQueueKernel, TaskQueue_Sync_zeCommandQueueSynchronize) {
    auto tq = m_task_queue;
    auto f = tq.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::KernelLaunch}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::KernelLaunch, CmdListElem::Barrier}));
    Config::setRetValue("zeCommandQueueSynchronize", ZE_RESULT_ERROR_DEVICE_LOST);
    m_task_queue.sync();
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTestWithModuleQueueKernel, TaskQueue_Sync_zeCommandListReset) {
    auto tq = m_task_queue;
    auto f = tq.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::KernelLaunch}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::KernelLaunch, CmdListElem::Barrier}));
    Config::setRetValue("zeCommandListReset", ZE_RESULT_ERROR_DEVICE_LOST);
    m_task_queue.sync();
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

TEST_F(MockTestWithModuleQueueKernel, TaskQueue_Sync_zeEventQueryKernelTimestamp) {
    auto tq = m_task_queue;
    auto f = tq.launch(m_kernel, 0);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::KernelLaunch}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::KernelLaunch, CmdListElem::Barrier}));
    Config::setRetValue("zeEventQueryKernelTimestamp", ZE_RESULT_ERROR_DEVICE_LOST);
    m_task_queue.sync();
    ASSERT_EQ(sm_rt_error, ISPCRT_DEVICE_LOST);
}

/// C API
TEST_F(MockTest, C_API_DeviceCount1) {
    // CPU
    uint32_t devCnt = ispcrtGetDeviceCount(ISPCRT_DEVICE_TYPE_CPU);
    ASSERT_EQ(devCnt, 1);
    // GPU
    Config::setDeviceCount(1);
    devCnt = ispcrtGetDeviceCount(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(devCnt, Config::getDeviceCount());
}

TEST_F(MockTest, C_API_DeviceCount2) {
    // CPU
    uint32_t devCnt = ispcrtGetDeviceCount(ISPCRT_DEVICE_TYPE_CPU);
    ASSERT_EQ(devCnt, 1);
    // GPU
    Config::setDeviceCount(2);
    devCnt = ispcrtGetDeviceCount(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(devCnt, Config::getDeviceCount());
}

TEST_F(MockTest, C_API_DeviceCountNonIntel) {
    // Mix in some non-Intel GPUs
    Config::setDeviceCount(4);
    Config::setDeviceProperties(1, DeviceProperties(VendorId::Nvidia, DeviceId::GenericNvidia));
    Config::setDeviceProperties(2, DeviceProperties(VendorId::AMD, DeviceId::GenericAMD));
    auto devCnt = ispcrtGetDeviceCount(ISPCRT_DEVICE_TYPE_GPU);
    // Only two Intel devices
    ASSERT_EQ(devCnt, 2);
}

TEST_F(MockTest, C_API_DeviceInfoCPU) {
    auto devCnt = ispcrtGetDeviceCount(ISPCRT_DEVICE_TYPE_CPU);
    ASSERT_EQ(devCnt, 1);
    ISPCRTDeviceInfo di;
    ispcrtGetDeviceInfo(ISPCRT_DEVICE_TYPE_CPU, 0, &di);
    ASSERT_EQ(0, di.deviceId);
    ASSERT_EQ(0, di.vendorId);
}

TEST_F(MockTest, C_API_DeviceInfoGPU) {
    std::vector<DeviceProperties> dps = {
        DeviceProperties(VendorId::Intel, DeviceId::Gen9),
        DeviceProperties(VendorId::Nvidia, DeviceId::GenericNvidia),
        DeviceProperties(VendorId::AMD, DeviceId::GenericAMD),
        DeviceProperties(VendorId::Intel, DeviceId::Gen12)
    };
    constexpr uint devices = 4;
    Config::setDeviceCount(devices);
    for (int d = 0; d < devices; d++)
        Config::setDeviceProperties(d, dps[d]);

    auto devCnt = ispcrtGetDeviceCount(ISPCRT_DEVICE_TYPE_GPU);
    // Only two Intel devices
    ASSERT_EQ(devCnt, 2);

    for (int d = 0; d < devCnt; d++) {
        ISPCRTDeviceInfo di;
        ispcrtGetDeviceInfo(ISPCRT_DEVICE_TYPE_GPU, d, &di);
        ASSERT_EQ(dps[d == 0?0:3].deviceId, di.deviceId);
        ASSERT_EQ(dps[d == 0?0:3].vendorId, di.vendorId);
    }
}

/// C++ Device API
TEST_F(MockTest, Device_DeviceCount1) {
    // CPU
    uint32_t devCnt = ispcrt::Device::deviceCount(ISPCRT_DEVICE_TYPE_CPU);
    ASSERT_EQ(devCnt, 1);
    // GPU
    Config::setDeviceCount(1);
    devCnt = ispcrt::Device::deviceCount(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(devCnt, Config::getDeviceCount());
}

TEST_F(MockTest, Device_DeviceCount2) {
    // CPU
    uint32_t devCnt = ispcrt::Device::deviceCount(ISPCRT_DEVICE_TYPE_CPU);
    ASSERT_EQ(devCnt, 1);
    // GPU
    Config::setDeviceCount(2);
    devCnt = ispcrt::Device::deviceCount(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(devCnt, Config::getDeviceCount());
}

TEST_F(MockTest, Device_DeviceCountNonIntel) {
    // Mix in some non-Intel GPUs
    Config::setDeviceCount(4);
    Config::setDeviceProperties(1, DeviceProperties(VendorId::Nvidia, DeviceId::GenericNvidia));
    Config::setDeviceProperties(2, DeviceProperties(VendorId::AMD, DeviceId::GenericAMD));
    auto devCnt = ispcrt::Device::deviceCount(ISPCRT_DEVICE_TYPE_GPU);
    // Only two Intel devices
    ASSERT_EQ(devCnt, 2);
}

TEST_F(MockTest, Device_DeviceInfoCPU) {
    auto devCnt = ispcrt::Device::deviceCount(ISPCRT_DEVICE_TYPE_CPU);
    ASSERT_EQ(devCnt, 1);
    auto di = ispcrt::Device::deviceInformation(ISPCRT_DEVICE_TYPE_CPU, 0);
    ASSERT_EQ(0, di.deviceId);
    ASSERT_EQ(0, di.vendorId);
}

TEST_F(MockTest, Device_DeviceInfoGPU) {
    std::vector<DeviceProperties> dps = {
        DeviceProperties(VendorId::Intel, DeviceId::Gen9),
        DeviceProperties(VendorId::Nvidia, DeviceId::GenericNvidia),
        DeviceProperties(VendorId::AMD, DeviceId::GenericAMD),
        DeviceProperties(VendorId::Intel, DeviceId::Gen12)
    };
    constexpr uint devices = 4;
    Config::setDeviceCount(devices);
    for (int d = 0; d < devices; d++)
        Config::setDeviceProperties(d, dps[d]);

    auto devCnt = ispcrt::Device::deviceCount(ISPCRT_DEVICE_TYPE_GPU);
    for (int d = 0; d < devCnt; d++) {
        auto di = ispcrt::Device::deviceInformation(ISPCRT_DEVICE_TYPE_GPU, d);
        ASSERT_EQ(dps[d == 0?0:3].deviceId, di.deviceId);
        ASSERT_EQ(dps[d == 0?0:3].vendorId, di.vendorId);
    }
}

TEST_F(MockTest, Device_DeviceInfoAllGPUs) {
    std::vector<DeviceProperties> dps = {
        DeviceProperties(VendorId::Intel, DeviceId::Gen9),
        DeviceProperties(VendorId::Nvidia, DeviceId::GenericNvidia),
        DeviceProperties(VendorId::AMD, DeviceId::GenericAMD),
        DeviceProperties(VendorId::Intel, DeviceId::Gen12)
    };
    constexpr uint devices = 4;
    Config::setDeviceCount(devices);
    for (int d = 0; d < devices; d++)
        Config::setDeviceProperties(d, dps[d]);

    auto di = ispcrt::Device::allDevicesInformation(ISPCRT_DEVICE_TYPE_GPU);
    ASSERT_EQ(di.size(), 2);
    for (int d = 0; d < di.size(); d++) {
        ASSERT_EQ(dps[d == 0?0:3].deviceId, di[d].deviceId);
        ASSERT_EQ(dps[d == 0?0:3].vendorId, di[d].vendorId);
    }
}

TEST_F(MockTest, Device_SecondGPU) {
    std::vector<DeviceProperties> dps = {
        DeviceProperties(VendorId::Intel, DeviceId::Gen9),
        DeviceProperties(VendorId::Nvidia, DeviceId::GenericNvidia),
        DeviceProperties(VendorId::AMD, DeviceId::GenericAMD),
        DeviceProperties(VendorId::Intel, DeviceId::Gen12)
    };
    constexpr uint devices = 4;
    Config::setDeviceCount(devices);
    for (int d = 0; d < devices; d++)
        Config::setDeviceProperties(d, dps[d]);
    Config::setExpectedDevice(3);

    ispcrt::Device device(ISPCRT_DEVICE_TYPE_GPU, 1);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    EXPECT_NE(device, 0);

    auto module = Module(device, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    EXPECT_NE(module, 0);
    auto tq = TaskQueue(device);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    EXPECT_NE(tq, 0);
    auto kernel = Kernel(device, module, "");
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    EXPECT_NE(kernel, 0);

    // Create an allocation
    std::vector<float> buf(64 * 1024);
    ispcrt::Array<float> buf_dev(device, buf);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    // "copy"
    tq.copyToDevice(buf_dev);
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
    tq.barrier();
    ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
    ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy, CmdListElem::Barrier}));
    auto f = tq.launch(kernel, 0);
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

TEST_F(MockTest, Device_ManyGPUs) {
    // Have 4 Gen12s and run on each of them
    std::vector<DeviceProperties> dps = {
        DeviceProperties(VendorId::Intel, DeviceId::Gen12),
        DeviceProperties(VendorId::Intel, DeviceId::Gen12),
        DeviceProperties(VendorId::Intel, DeviceId::Gen12),
        DeviceProperties(VendorId::Intel, DeviceId::Gen12)
    };
    Config::setDeviceCount(dps.size());
    for (int d = 0; d < dps.size(); d++)
        Config::setDeviceProperties(d, dps[d]);

    auto run = [&](uint32_t deviceNo) {
        ispcrt::Device device(ISPCRT_DEVICE_TYPE_GPU, deviceNo);
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        EXPECT_NE(device, 0);

        auto module = Module(device, "");
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        EXPECT_NE(module, 0);
        auto tq = TaskQueue(device);
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        EXPECT_NE(tq, 0);
        auto kernel = Kernel(device, module, "");
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        EXPECT_NE(kernel, 0);
        // Create an allocation
        std::vector<float> buf(64 * 1024);
        ispcrt::Array<float> buf_dev(device, buf);
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        // "copy"
        tq.copyToDevice(buf_dev);
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
        tq.barrier();
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy, CmdListElem::Barrier}));
        auto f = tq.launch(kernel, 0);
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
        Config::resetCmdList();
    };

    for (int d = 0; d < dps.size(); d++) {
        Config::setExpectedDevice(d);
        run(d);
    }
}

TEST_F(MockTest, Device_ManyGPUs_EnvOverride) {
    // Have 4 Gen12s and run on just one of them using env variable override
    std::vector<DeviceProperties> dps = {
        DeviceProperties(VendorId::Intel, DeviceId::Gen12),
        DeviceProperties(VendorId::Intel, DeviceId::Gen12),
        DeviceProperties(VendorId::Intel, DeviceId::Gen12),
        DeviceProperties(VendorId::Intel, DeviceId::Gen12)
    };
    Config::setDeviceCount(dps.size());
    for (int d = 0; d < dps.size(); d++)
        Config::setDeviceProperties(d, dps[d]);

    auto run = [&](uint32_t deviceNo) {
        ispcrt::Device device(ISPCRT_DEVICE_TYPE_GPU, deviceNo);
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        EXPECT_NE(device, 0);

        auto module = Module(device, "");
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        EXPECT_NE(module, 0);
        auto tq = TaskQueue(device);
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        EXPECT_NE(tq, 0);
        auto kernel = Kernel(device, module, "");
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        EXPECT_NE(kernel, 0);
        // Create an allocation
        std::vector<float> buf(64 * 1024);
        ispcrt::Array<float> buf_dev(device, buf);
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        // "copy"
        tq.copyToDevice(buf_dev);
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy}));
        tq.barrier();
        ASSERT_EQ(sm_rt_error, ISPCRT_NO_ERROR);
        ASSERT_TRUE(Config::checkCmdList({CmdListElem::MemoryCopy, CmdListElem::Barrier}));
        auto f = tq.launch(kernel, 0);
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
        Config::resetCmdList();
    };

    setenv("ISPCRT_GPU_DEVICE", "1", 1);
    for (int d = 0; d < dps.size(); d++) {
        Config::setExpectedDevice(1);
        run(d);
    }
    unsetenv("ISPCRT_GPU_DEVICE");
}

/// Compilation tests
TEST_F(MockTest, Compilation_SharedArray) {
    auto d = Device(ISPCRT_DEVICE_TYPE_CPU);
    struct Parameters { int i; };
    auto pmv = ispcrt::Array<Parameters, ispcrt::AllocType::Shared>(d);
    auto p = pmv.sharedPtr();
    p->i = 1234;
    auto pmv2 = ispcrt::Array<Parameters, ispcrt::AllocType::Shared>(d, 2);
    p = pmv.sharedPtr();
    p->i = 1234;
    ispcrt::SharedMemoryAllocator<float> sma(d);
    ispcrt::SharedVector<float> v(16, sma);
}

} // namespace mock
} // namespace testing
} // namespace ispcrt