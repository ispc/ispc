// This test verifies that JIT API for ISPC C++ library works correctly
// and can compile and execute ISPC code at runtime.

// RUN: %{cxx} -x c++ -std=c++17 -I%{ispc_include} %s -L%{ispc_lib} -lispc -o %t.bin
// RUN: env LD_LIBRARY_PATH=%{ispc_lib} %t.bin | FileCheck %s

// REQUIRES: LINUX_HOST

// CHECK: ISPC JIT API Test Starting
// CHECK: Basic JIT compilation test: SUCCESS
// CHECK: JIT function execution test: SUCCESS
// CHECK: Multiple JIT compilation test: SUCCESS
// CHECK: All JIT tests completed successfully

// REQUIRES: ISPC_LIBRARY_JIT && !ASAN_RUN

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include "ispc/ispc.h"

// Function pointer type for our test function
typedef void (*test_func_t)(float vin[], float vout[], int count);

int main() {
    std::cout << "ISPC JIT API Test Starting\n";

    // Initialize ISPC
    if (!ispc::Initialize()) {
        std::cerr << "Failed to initialize ISPC\n";
        return 1;
    }

    bool all_tests_passed = true;

    // Test 1: Basic JIT compilation - create a simple ISPC program and compile it to JIT
    {
        // Create a simple test ISPC file
        std::ofstream test_file("jit_test.ispc");
        test_file << "export void double_array(uniform float vin[], uniform float vout[], uniform int count) {\n";
        test_file << "    foreach (i = 0 ... count) {\n";
        test_file << "        vout[i] = vin[i] * 2.0f;\n";
        test_file << "    }\n";
        test_file << "}\n";
        test_file.close();

        std::vector<std::string> args = {"jit_test.ispc"};
        auto engine = ispc::ISPCEngine::CreateFromArgs(args);

        if (engine) {
            int result = engine->CompileFromFileToJit("jit_test.ispc");
            if (result == 0) {
                std::cout << "Basic JIT compilation test: SUCCESS\n";
            } else {
                std::cout << "Basic JIT compilation test: FAILED (compilation error)\n";
                all_tests_passed = false;
            }
        } else {
            std::cout << "Basic JIT compilation test: FAILED (engine creation)\n";
            all_tests_passed = false;
        }
    }

    // Test 2: JIT function execution - compile and execute a function
    {
        std::vector<std::string> args = {"jit_test.ispc"};
        auto engine = ispc::ISPCEngine::CreateFromArgs(args);

        if (engine && engine->CompileFromFileToJit("jit_test.ispc") == 0) {
            // Get the function pointer
            auto func_ptr = engine->GetJitFunction("double_array");

            if (func_ptr) {
                // Cast to our function type and test it
                test_func_t test_func = reinterpret_cast<test_func_t>(func_ptr);

                // Prepare test data
                const int count = 4;
                float input[count] = {1.0f, 2.0f, 3.0f, 4.0f};
                float output[count] = {0.0f, 0.0f, 0.0f, 0.0f};

                // Execute the JIT-compiled function
                test_func(input, output, count);

                // Verify results
                bool results_correct = true;
                for (int i = 0; i < count; i++) {
                    if (output[i] != input[i] * 2.0f) {
                        results_correct = false;
                        break;
                    }
                }

                if (results_correct) {
                    std::cout << "JIT function execution test: SUCCESS\n";
                } else {
                    std::cout << "JIT function execution test: FAILED (incorrect results)\n";
                    all_tests_passed = false;
                }
            } else {
                std::cout << "JIT function execution test: FAILED (function not found)\n";
                all_tests_passed = false;
            }
        } else {
            std::cout << "JIT function execution test: FAILED (compilation failed)\n";
            all_tests_passed = false;
        }
    }

    // Test 3: Multiple JIT compilations with different functions
    {
        // Create second test file
        std::ofstream test_file2("jit_test2.ispc");
        test_file2 << "export void add_constant(uniform float vin[], uniform float vout[], uniform int count, uniform float constant) {\n";
        test_file2 << "    foreach (i = 0 ... count) {\n";
        test_file2 << "        vout[i] = vin[i] + constant;\n";
        test_file2 << "    }\n";
        test_file2 << "}\n";
        test_file2.close();

        std::vector<std::string> args = {};
        auto engine = ispc::ISPCEngine::CreateFromArgs(args);

        if (engine) {
            int result1 = engine->CompileFromFileToJit("jit_test.ispc");
            int result2 = engine->CompileFromFileToJit("jit_test2.ispc");

            if (result1 == 0 && result2 == 0) {
                // Check that both functions are available
                auto func1 = engine->GetJitFunction("double_array");
                auto func2 = engine->GetJitFunction("add_constant");

                if (func1 && func2) {
                    std::cout << "Multiple JIT compilation test: SUCCESS\n";
                } else {
                    std::cout << "Multiple JIT compilation test: FAILED (functions not found)\n";
                    all_tests_passed = false;
                }
            } else {
                std::cout << "Multiple JIT compilation test: FAILED (compilation error)\n";
                all_tests_passed = false;
            }
        } else {
            std::cout << "Multiple JIT compilation test: FAILED (engine creation)\n";
            all_tests_passed = false;
        }
    }

    ispc::Shutdown();

    if (all_tests_passed) {
        std::cout << "All JIT tests completed successfully\n";
        return 0;
    } else {
        std::cout << "Some JIT tests failed\n";
        return 1;
    }
}