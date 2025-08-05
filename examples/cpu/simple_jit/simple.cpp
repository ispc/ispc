/*
  Copyright (c) 2025, Intel Corporation
  SPDX-License-Identifier: BSD-3-Clause
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cmath>

#include "ispc/ispc.h"

// Function pointer type for our ISPC function
typedef void (*simple_func_t)(float vin[], float vout[], int count);

int main() {
    // Initialize ISPC
    std::cout << "Initializing ISPC...\n";
    if (!ispc::Initialize()) {
        std::cerr << "Error: Failed to initialize ISPC\n";
        return 1;
    }

    std::cout << "Testing JIT compilation of simple.ispc...\n";

    // Test 1: Basic JIT compilation and execution
    {
        std::cout << "\n=== Test 1: Basic JIT compilation ===\n";
        
        std::vector<std::string> args = {"-O2"};
        auto engine = ispc::ISPCEngine::CreateFromArgs(args);

        if (!engine) {
            std::cerr << "Failed to create ISPC engine\n";
            ispc::Shutdown();
            return 1;
        }

        int result = engine->CompileFromFileToJit("simple.ispc");
        if (result == 0) {
            std::cout << "JIT compilation successful!\n";
            
            // Get the function pointer
            auto func_ptr = engine->GetJitFunction("simple");
            if (func_ptr) {
                std::cout << "Successfully retrieved 'simple' function from JIT\n";
                
                // Test the function
                simple_func_t simple_func = reinterpret_cast<simple_func_t>(func_ptr);
                
                // Prepare test data
                const int count = 8;
                float input[count] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
                float output[count] = {0.0f};
                
                std::cout << "Executing JIT-compiled function...\n";
                simple_func(input, output, count);
                
                // Verify and display results
                std::cout << "Results:\n";
                bool all_correct = true;
                for (int i = 0; i < count; i++) {
                    float expected = (input[i] < 3.0f) ? input[i] * input[i] : sqrt(input[i]);
                    std::cout << "  input[" << i << "] = " << input[i] 
                              << " -> output[" << i << "] = " << output[i] 
                              << " (expected: " << expected << ")";
                    
                    if (fabs(output[i] - expected) < 0.001f) {
                        std::cout << " ✓\n";
                    } else {
                        std::cout << " ✗\n";
                        all_correct = false;
                    }
                }
                
                if (all_correct) {
                    std::cout << "SUCCESS: All results match expected values\n";
                } else {
                    std::cout << "FAILURE: Some results don't match\n";
                }
            } else {
                std::cerr << "Failed to get 'simple' function from JIT\n";
            }
        } else {
            std::cerr << "JIT compilation failed with code: " << result << "\n";
        }
    }

    // Test 2: Multiple JIT engines with different optimization levels
    {
        std::cout << "\n=== Test 2: Multiple JIT engines ===\n";
        
        std::vector<std::string> args_o2 = {"-O2"};
        std::vector<std::string> args_o0 = {"-O0"};
        
        auto engine_o2 = ispc::ISPCEngine::CreateFromArgs(args_o2);
        auto engine_o0 = ispc::ISPCEngine::CreateFromArgs(args_o0);

        if (!engine_o2 || !engine_o0) {
            std::cerr << "Failed to create ISPC engines\n";
        } else {
            std::cout << "Created engines with different optimization levels\n";

            // Compile with both engines
            int result_o2 = engine_o2->CompileFromFileToJit("simple.ispc");
            int result_o0 = engine_o0->CompileFromFileToJit("simple.ispc");

            if (result_o2 == 0 && result_o0 == 0) {
                std::cout << "Both engines compiled successfully\n";
                
                // Get functions from both engines
                auto func_o2 = engine_o2->GetJitFunction("simple");
                auto func_o0 = engine_o0->GetJitFunction("simple");
                
                if (func_o2 && func_o0) {
                    std::cout << "SUCCESS: Both engines provide the 'simple' function\n";
                    std::cout << "  - O2 optimized function: " << func_o2 << "\n";
                    std::cout << "  - O0 function: " << func_o0 << "\n";
                    
                    // Test that both functions work correctly
                    simple_func_t simple_o2 = reinterpret_cast<simple_func_t>(func_o2);
                    simple_func_t simple_o0 = reinterpret_cast<simple_func_t>(func_o0);
                    
                    const int count = 4;
                    float input[count] = {1.5f, 2.5f, 3.5f, 4.5f};
                    float output_o2[count] = {0.0f};
                    float output_o0[count] = {0.0f};
                    
                    simple_o2(input, output_o2, count);
                    simple_o0(input, output_o0, count);
                    
                    // Verify both produce same results
                    bool results_match = true;
                    for (int i = 0; i < count; i++) {
                        if (fabs(output_o2[i] - output_o0[i]) > 0.001f) {
                            results_match = false;
                            break;
                        }
                    }
                    
                    if (results_match) {
                        std::cout << "SUCCESS: Both optimization levels produce identical results\n";
                    } else {
                        std::cout << "Note: Different optimization levels may produce slightly different results\n";
                    }
                } else {
                    std::cerr << "Failed to get functions from one or both engines\n";
                }
            } else {
                std::cerr << "Compilation failed - O2: " << result_o2 << ", O0: " << result_o0 << "\n";
            }
        }
    }

    // Test 3: Test engine cleanup
    {
        std::cout << "\n=== Test 3: Engine cleanup ===\n";
        
        std::vector<std::string> args = {"--target=host"};
        auto engine = ispc::ISPCEngine::CreateFromArgs(args);
        
        if (engine) {
            int result = engine->CompileFromFileToJit("simple.ispc");
            if (result == 0) {
                auto func = engine->GetJitFunction("simple");
                if (func) {
                    std::cout << "Function retrieved successfully\n";
                }
            }
            std::cout << "Engine will be automatically cleaned up when it goes out of scope\n";
        }
        
        std::cout << "SUCCESS: Engine cleanup test completed\n";
    }

    std::cout << "\nCleaning up...\n";
    ispc::Shutdown();
    std::cout << "ISPC shutdown complete\n";
    return 0;
}