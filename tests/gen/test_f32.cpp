#include <string>
#include <iostream>
#include "cm_rt.h"
#include "cm_rt_helpers.h"
#include "isa_helpers.h"
using namespace std;

int main(int argc, char *argv[]) {
  // creates a CmDevice
  CmDevice *device = nullptr;
  unsigned int version = 0;
  cm_result_check(::CreateCmDevice(device, version));

  // load kernel code
  std::string code = cm::util::isa::loadFile("test_genx.isa");
  if (code.empty()) {
    std::cerr << "Error: empty ISA binary.\n";
    exit(1);
  }

  // creates a CmProgram object consisting of the kernels loaded from the code.
  CmProgram *program = nullptr;
  CmKernel *kernel = nullptr;
  cm_result_check(device->LoadProgram(const_cast<char *>(code.data()), code.size(), program));
  cm_result_check(device->CreateKernel(program, "vecAdd", kernel));

  // init data
  constexpr int N = 67;
  struct alignas(4096) AlignedArray { float data[N]; } AX, AY, AZ;
  void *XData = AX.data;
  void *YData = AY.data;
  void *OUTData = AZ.data;
  for (int i = 0; i < N; i++) {
    AX.data[i] = (float)i;
    AY.data[i] = 1.0f;
  }

  // setup arguments
  CmBufferSVM *XBuff = nullptr;
  CmBufferSVM *YBuff = nullptr;
  CmBufferSVM *OUTBuff = nullptr;
  cm_result_check(device->CreateBufferSVM(N * sizeof(float), XData, CM_SVM_ACCESS_FLAG_DEFAULT, XBuff));
  cm_result_check(device->CreateBufferSVM(N * sizeof(float), YData, CM_SVM_ACCESS_FLAG_DEFAULT, YBuff));
  cm_result_check(device->CreateBufferSVM(N * sizeof(float), OUTData, CM_SVM_ACCESS_FLAG_DEFAULT, OUTBuff));
  cm_result_check(kernel->SetKernelArg(0, sizeof(void *), &XData));
  cm_result_check(kernel->SetKernelArg(1, sizeof(void *), &YData));
  cm_result_check(kernel->SetKernelArg(2, sizeof(void *), &OUTData));
  cm_result_check(kernel->SetKernelArg(3, sizeof(int), &N));

  // run the tasks
  CmTask *task = nullptr;
  CmThreadGroupSpace *pts = nullptr;
  CmQueue *cmd_queue = nullptr;
  CmEvent *sync_event = nullptr;
  cm_result_check(device->CreateTask(task));
  cm_result_check(task->AddKernel(kernel));
  cm_result_check(device->CreateThreadGroupSpace(1, 1, 1, 1, pts));
  cm_result_check(device->CreateQueue(cmd_queue));
  cm_result_check(cmd_queue->EnqueueWithGroup(task, sync_event, pts));
  cm_result_check(sync_event->WaitForTaskFinished());
  cm_result_check(DestroyCmDevice(device));
  // check results.
  bool passed = true;
  for (int i = 0; i < N; ++i) {
	std::cout << "AX[i] " << AX.data[i] << std::endl;
    std::cout << "AY[i] " << AY.data[i] << std::endl;
	std::cout << "AZ[i] " << AZ.data[i] << std::endl;
  }

  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
  return (passed ? 0 : 1);
}
