// This test verifies that JIT compilation with --cpu and no --target selects
// the target from the CPU, as the command line does, and does not check the
// CPU against the target detected from the host.

// RUN: %{cxx} -x c++ -std=c++17 -I%{ispc_include} %s -L%{ispc_lib} -lispc -o %t.bin
// RUN: env LD_LIBRARY_PATH=%{ispc_lib} %t.bin %t.ispc 2>&1 | FileCheck %s

// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST && LLVM_22_0+
// REQUIRES: ISPC_LIBRARY_JIT && !ASAN_RUN

// CHECK: --cpu=dmr without --target: SUCCESS
// CHECK-NEXT: Error: The requested device (dmr) is incompatible with the device required for avx10.2nvl-x8 target (novalake)
// CHECK: --target=host --cpu=dmr: FAILED (expected)
// CHECK-NOT: FAILED (unexpected)

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ispc/ispc.h"

static int lCompile(const std::string &file, std::vector<std::string> args) {
    args.insert(args.begin(), file);
    args.push_back("--nowrap");
    auto engine = ispc::ISPCEngine::CreateFromArgs(args);
    return engine ? engine->CompileFromFileToJit(file) : -1;
}

int main(int argc, char **argv) {
    if (argc != 2 || !ispc::Initialize()) {
        return 1;
    }
    const std::string file = argv[1];
    std::ofstream(file) << "export uniform int answer() { return 42; }\n";

    // The mocked host ISA (Nova Lake) is incompatible with --cpu=dmr, so the
    // compilation only succeeds if --cpu selects the target.
    int result = lCompile(file, {"--cpu=dmr", "--mock-host-target=avx10.2nvl-x8"});
    std::cout << "--cpu=dmr without --target: " << (result == 0 ? "SUCCESS" : "FAILED (unexpected)") << std::endl;

    // With --target=host the host still selects the target, as on the command
    // line, so the same mismatch is an error. std::endl orders stdout with the
    // diagnostic on stderr.
    result = lCompile(file, {"--target=host", "--cpu=dmr", "--mock-host-target=avx10.2nvl-x8"});
    std::cout << "--target=host --cpu=dmr: " << (result != 0 ? "FAILED (expected)" : "SUCCESS (unexpected)")
              << std::endl;

    ispc::Shutdown();
    return 0;
}
