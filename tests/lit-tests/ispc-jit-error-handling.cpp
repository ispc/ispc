// This test verifies comprehensive error handling in JIT API for ISPC C++ library.
// It tests all error conditions from ispc_impl.cpp to ensure proper error reporting.

// RUN: %{cxx} -x c++ -std=c++17 -I%{ispc_include} %s -L%{ispc_lib} -lispc -o %t.bin
// RUN: env LD_LIBRARY_PATH=%{ispc_lib} %t.bin | FileCheck %s

// REQUIRES: LINUX_HOST

// CHECK: JIT Error Handling Test Starting
// CHECK: Test 1 - Input file validation errors: SUCCESS
// CHECK: Test 2 - Multiple targets error: SUCCESS
// CHECK: Test 3 - JIT compilation errors: SUCCESS
// CHECK: Test 4 - Function retrieval errors: SUCCESS
// CHECK: Test 5 - Runtime function registration errors: SUCCESS
// CHECK: All JIT error handling tests completed successfully

// REQUIRES: ISPC_LIBRARY_JIT && !ASAN_RUN

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <cstdio>
#include "ispc/ispc.h"

// Capture stderr to verify error messages
class StderrCapture {
public:
    StderrCapture() {
        old_stderr = stderr;
        stderr = tmpfile();
    }

    ~StderrCapture() {
        if (stderr != old_stderr) {
            fclose(stderr);
            stderr = old_stderr;
        }
    }

    std::string getOutput() {
        if (stderr == old_stderr) return "";

        fflush(stderr);
        long size = ftell(stderr);
        if (size <= 0) return "";

        rewind(stderr);
        std::string result(size, '\0');
        fread(&result[0], 1, size, stderr);
        return result;
    }

private:
    FILE* old_stderr;
};

int main() {
    std::cout << "JIT Error Handling Test Starting\n";

    // Initialize ISPC
    if (!ispc::Initialize()) {
        std::cerr << "Failed to initialize ISPC\n";
        return 1;
    }

    bool all_tests_passed = true;

    // Test 1: Input file validation errors
    {
        bool test_passed = true;

        // Test 1a: Empty filename
        {
            std::vector<std::string> args = {"--target=host"};
            auto engine = ispc::ISPCEngine::CreateFromArgs(args);

            if (engine) {
                StderrCapture capture;
                int result = engine->CompileFromFileToJit("");
                std::string error_output = capture.getOutput();

                if (result != 0 && error_output.find("No input file specified") != std::string::npos) {
                    // Expected error for empty filename
                } else {
                    test_passed = false;
                }
            } else {
                test_passed = false;
            }
        }

        // Test 1b: Non-existent file
        {
            std::vector<std::string> args = {"--target=host"};
            auto engine = ispc::ISPCEngine::CreateFromArgs(args);

            if (engine) {
                StderrCapture capture;
                int result = engine->CompileFromFileToJit("non_existent_file.ispc");
                std::string error_output = capture.getOutput();

                if (result != 0 && error_output.find("does not exist") != std::string::npos) {
                    // Expected error for non-existent file
                } else {
                    test_passed = false;
                }
            } else {
                test_passed = false;
            }
        }

        // Test 1c: Directory instead of file
        {
            // Create a directory
            system("mkdir -p test_dir_for_jit");

            std::vector<std::string> args = {"--target=host"};
            auto engine = ispc::ISPCEngine::CreateFromArgs(args);

            if (engine) {
                StderrCapture capture;
                int result = engine->CompileFromFileToJit("test_dir_for_jit");
                std::string error_output = capture.getOutput();

                if (result != 0 && error_output.find("is a directory") != std::string::npos) {
                    // Expected error for directory
                } else {
                    test_passed = false;
                }
            } else {
                test_passed = false;
            }

            system("rmdir test_dir_for_jit");
        }

        if (test_passed) {
            std::cout << "Test 1 - Input file validation errors: SUCCESS\n";
        } else {
            std::cout << "Test 1 - Input file validation errors: FAILED\n";
            all_tests_passed = false;
        }
    }

    // Test 2: Multiple targets error
    {
        bool test_passed = false;

        std::vector<std::string> args = {"--target=sse4-i32x4,avx2-i32x8"};
        auto engine = ispc::ISPCEngine::CreateFromArgs(args);

        if (engine) {
            // Create a simple test file
            std::ofstream test_file("simple_jit_test.ispc");
            test_file << "export void simple() { }\n";
            test_file.close();

            StderrCapture capture;
            int result = engine->CompileFromFileToJit("simple_jit_test.ispc");
            std::string error_output = capture.getOutput();

            if (result != 0 && error_output.find("only supports single target compilation") != std::string::npos) {
                test_passed = true;
            }

            remove("simple_jit_test.ispc");
        }

        if (test_passed) {
            std::cout << "Test 2 - Multiple targets error: SUCCESS\n";
        } else {
            std::cout << "Test 2 - Multiple targets error: FAILED\n";
            all_tests_passed = false;
        }
    }

    // Test 3: JIT compilation errors
    {
        bool test_passed = false;

        std::vector<std::string> args = {"--target=host"};
        auto engine = ispc::ISPCEngine::CreateFromArgs(args);

        if (engine) {
            // Create a file with invalid ISPC syntax
            std::ofstream invalid_file("invalid_syntax.ispc");
            invalid_file << "this is not valid ISPC syntax at all!\n";
            invalid_file << "definitely_not_a_function() { invalid_stuff; }\n";
            invalid_file.close();

            StderrCapture capture;
            int result = engine->CompileFromFileToJit("invalid_syntax.ispc");

            if (result != 0) {
                test_passed = true; // Expected compilation failure
            }

            remove("invalid_syntax.ispc");
        }

        if (test_passed) {
            std::cout << "Test 3 - JIT compilation errors: SUCCESS\n";
        } else {
            std::cout << "Test 3 - JIT compilation errors: FAILED\n";
            all_tests_passed = false;
        }
    }

    // Test 4: Function retrieval errors
    {
        bool test_passed = true;

        std::vector<std::string> args = {"--target=host"};
        auto engine = ispc::ISPCEngine::CreateFromArgs(args);

        if (engine) {
            // Test 4a: GetJitFunction without JIT mode active
            {
                StderrCapture capture;
                void* func = engine->GetJitFunction("some_function");
                std::string error_output = capture.getOutput();

                if (func == nullptr && error_output.find("JIT mode is not active") != std::string::npos) {
                    // Expected error
                } else {
                    test_passed = false;
                }
            }

            // Compile a simple program first
            std::ofstream test_file("func_test.ispc");
            test_file << "export void test_func() { }\n";
            test_file.close();

            if (engine->CompileFromFileToJit("func_test.ispc") == 0) {
                // Test 4b: Empty function name
                {
                    StderrCapture capture;
                    void* func = engine->GetJitFunction("");
                    std::string error_output = capture.getOutput();

                    if (func == nullptr && error_output.find("Function name cannot be empty") != std::string::npos) {
                        // Expected error
                    } else {
                        test_passed = false;
                    }
                }

                // Test 4c: Non-existent function
                {
                    StderrCapture capture;
                    void* func = engine->GetJitFunction("non_existent_function");
                    std::string error_output = capture.getOutput();

                    if (func == nullptr && error_output.find("not found in JIT") != std::string::npos) {
                        // Expected error
                    } else {
                        test_passed = false;
                    }
                }
            } else {
                test_passed = false;
            }

            remove("func_test.ispc");
        } else {
            test_passed = false;
        }

        if (test_passed) {
            std::cout << "Test 4 - Function retrieval errors: SUCCESS\n";
        } else {
            std::cout << "Test 4 - Function retrieval errors: FAILED\n";
            all_tests_passed = false;
        }
    }

    // Test 5: Runtime function registration errors
    {
        bool test_passed = true;

        std::vector<std::string> args = {"--target=host"};
        auto engine = ispc::ISPCEngine::CreateFromArgs(args);

        if (engine) {
            // Test 5a: Empty function name
            {
                StderrCapture capture;
                bool result = engine->SetJitRuntimeFunction("", (void*)0x12345);
                std::string error_output = capture.getOutput();

                if (!result && error_output.find("Runtime function name cannot be empty") != std::string::npos) {
                    // Expected error
                } else {
                    test_passed = false;
                }
            }

            // Test 5b: Null function pointer
            {
                StderrCapture capture;
                bool result = engine->SetJitRuntimeFunction("ISPCLaunch", nullptr);
                std::string error_output = capture.getOutput();

                if (!result && error_output.find("Runtime function pointer cannot be null") != std::string::npos) {
                    // Expected error
                } else {
                    test_passed = false;
                }
            }

        } else {
            test_passed = false;
        }

        if (test_passed) {
            std::cout << "Test 5 - Runtime function registration errors: SUCCESS\n";
        } else {
            std::cout << "Test 5 - Runtime function registration errors: FAILED\n";
            all_tests_passed = false;
        }
    }

    ispc::Shutdown();

    if (all_tests_passed) {
        std::cout << "All JIT error handling tests completed successfully\n";
        return 0;
    } else {
        std::cout << "Some JIT error handling tests failed\n";
        return 1;
    }
}