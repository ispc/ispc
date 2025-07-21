// This test verifies that API for ISPC C++ library works correctly
// and without state corruption.

// RUN: %{cxx} -x c++ -std=c++17 -I%{ispc_include} %s -L%{ispc_lib} -lispc -o %t.bin
// RUN: env LD_LIBRARY_PATH=%{ispc_lib} %t.bin | FileCheck %s

// REQUIRES: LINUX_HOST

// CHECK: ISPC C++ API Test Starting
// CHECK: Basic compilation test: SUCCESS
// CHECK: Multiple compilation test: SUCCESS
// CHECK: Engine isolation test: SUCCESS
// CHECK: All tests completed successfully

// REQUIRES: ISPC_LIBRARY && !ASAN_RUN

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include "ispc/ispc.h"

int main() {
    std::cout << "ISPC C++ API Test Starting\n";

    // Initialize ISPC
    if (!ispc::Initialize()) {
        std::cerr << "Failed to initialize ISPC\n";
        return 1;
    }

    bool all_tests_passed = true;

    // Test 1: Basic compilation - create a simple ISPC program in memory and compile it
    // We'll write a simple ISPC file first
    {
        // Create a simple test ISPC file
        std::ofstream test_file("simple_test.ispc");
        test_file << "export void test_func(uniform float vin[], uniform float vout[], uniform int count) {\n";
        test_file << "    foreach (i = 0 ... count) {\n";
        test_file << "        vout[i] = vin[i] * 2.0f;\n";
        test_file << "    }\n";
        test_file << "}\n";
        test_file.close();

        std::vector<std::string> args = {"ispc", "simple_test.ispc", "--target=host", "-o", "test1.o"};
        int result = ispc::CompileFromArgs(args);
        if (result == 0) {
            std::cout << "Basic compilation test: SUCCESS\n";
        } else {
            std::cout << "Basic compilation test: FAILED\n";
            all_tests_passed = false;
        }
    }

    // Test 2: Multiple compilations with different options
    {
        std::vector<std::string> args1 = {"ispc", "simple_test.ispc", "--target=host", "-O2", "-o", "test2a.o"};
        std::vector<std::string> args2 = {"ispc", "simple_test.ispc", "--target=host", "-O0", "--emit-asm", "-o", "test2b.s"};

        int result1 = ispc::CompileFromArgs(args1);
        int result2 = ispc::CompileFromArgs(args2);

        if (result1 == 0 && result2 == 0) {
            std::cout << "Multiple compilation test: SUCCESS\n";
        } else {
            std::cout << "Multiple compilation test: FAILED\n";
            all_tests_passed = false;
        }
    }

    // Test 3: Engine isolation test
    {
        std::vector<std::string> engine_args1 = {"ispc", "simple_test.ispc", "--target=host", "-O2", "-o", "engine1.o"};
        std::vector<std::string> engine_args2 = {"ispc", "simple_test.ispc", "--target=host", "-O0", "-o", "engine2.o"};
        std::vector<std::string> engine_args3 = {"ispc", "simple_test.ispc", "--target=host", "--emit-asm", "-o", "engine3.s"};

        auto engine1 = ispc::ISPCEngine::CreateFromArgs(engine_args1);
        auto engine2 = ispc::ISPCEngine::CreateFromArgs(engine_args2);
        auto engine3 = ispc::ISPCEngine::CreateFromArgs(engine_args3);

        bool engines_created = engine1 && engine2 && engine3;
        bool engines_executed = false;

        if (engines_created) {
            int r1 = engine1->Execute();
            int r2 = engine2->Execute();
            int r3 = engine3->Execute();
            engines_executed = (r1 == 0 && r2 == 0 && r3 == 0);
        }

        if (engines_created && engines_executed) {
            std::cout << "Engine isolation test: SUCCESS\n";
        } else {
            std::cout << "Engine isolation test: FAILED\n";
            all_tests_passed = false;
        }
    }

    ispc::Shutdown();

    if (all_tests_passed) {
        std::cout << "All tests completed successfully\n";
        return 0;
    } else {
        std::cout << "Some tests failed\n";
        return 1;
    }
}
