/*
  Copyright (c) 2025, Intel Corporation
  SPDX-License-Identifier: BSD-3-Clause
*/

#include <filesystem>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "ispc/ispc.h"

int main() {
    // Initialize ISPC
    std::cout << "Initializing ISPC...\n";
    if (!ispc::Initialize()) {
        std::cerr << "Error: Failed to initialize ISPC\n";
        return 1;
    }

    std::cout << "Compiling simple.ispc using library mode...\n";

    std::vector<std::string> args1 = {"simple.ispc", "--target=host", "-O2", "-o",
                                      "simple_ispc.o", "-h", "simple_ispc.h"};

    int result = ispc::CompileFromArgs(args1);

    if (result == 0) {
        std::cout << "ISPC compilation successful!\n";
    } else {
        std::cerr << "ISPC compilation failed with code: " << result << "\n";
        ispc::Shutdown();
        return result;
    }

    std::cout << "Compiling simple.ispc using library mode with different options...\n";

    // Set up a second compilation with different options
    std::vector<std::string> args2 = {"simple.ispc", "--target=host",  "-O0", "--emit-asm", "-o",
                                      "simple_debug.s", "-h", "simple_debug.h", "-g"};

    // Execute second compilation
    int result2 = ispc::CompileFromArgs(args2);

    if (result2 == 0) {
        std::cout << "Second compilation successful with different options\n";

        // Verify that files with different extensions were created
        if (std::filesystem::exists("simple_ispc.o") && std::filesystem::exists("simple_debug.s")) {
            std::cout << "SUCCESS: Both compilations produced different output files:\n";
            std::cout << "  - First compilation:  simple_ispc.o\n";
            std::cout << "  - Second compilation: simple_debug.s\n";
        } else {
            std::cout << "Note: Output files may not be visible in current directory\n";
        }
    } else {
        std::cerr << "Second ISPC compilation failed with code: " << result2 << "\n";
    }

    std::cout << "\nTesting multiple engines...\n";

    // Create multiple engines with different targets that would conflict
    std::vector<std::string> engineArgs1 = {"simple.ispc", "--target=sse2-i32x4", "-O2", "-o",
                                            "simple_sse2.o", "-h", "simple_sse2.h"};

    std::vector<std::string> engineArgs2 = {"simple.ispc", "--target=avx2-i32x8", "-O0", "-o",
                                            "simple_avx2.o", "-h", "simple_avx2.h"};

    std::vector<std::string> engineArgs3 = {"simple.ispc", "--target=host", "--emit-asm", "-o",
                                            "simple_host.s", "-h", "simple_host.h"};

    // Create engines but don't execute yet - this tests target state isolation
    auto engine1 = ispc::ISPCEngine::CreateFromArgs(engineArgs1);
    auto engine2 = ispc::ISPCEngine::CreateFromArgs(engineArgs2);
    auto engine3 = ispc::ISPCEngine::CreateFromArgs(engineArgs3);

    if (!engine1 || !engine2 || !engine3) {
        std::cerr << "Failed to create one or more ISPC engines\n";
    } else {
        std::cout << "Created 3 engines with different targets\n";

        // Execute engines in sequence
        std::cout << "Executing engine 1 (SSE2)...\n";
        int result1 = engine1->Execute();

        std::cout << "Executing engine 2 (AVX2)...\n";
        int result2 = engine2->Execute();

        std::cout << "Executing engine 3 (Host ASM)...\n";
        int result3 = engine3->Execute();

        if (result1 == 0 && result2 == 0 && result3 == 0) {
            std::cout << "SUCCESS\n";
            std::cout << "  - SSE2 compilation: simple_sse2.o\n";
            std::cout << "  - AVX2 compilation: simple_avx2.o\n";
            std::cout << "  - Host ASM compilation: simple_host.s\n";
        } else {
            std::cerr << "FAILURE\n";
            std::cerr << "  Engine 1 result: " << result1 << "\n";
            std::cerr << "  Engine 2 result: " << result2 << "\n";
            std::cerr << "  Engine 3 result: " << result3 << "\n";
        }
    }

    std::cout << "\nCleaning up...\n";
    ispc::Shutdown();
    std::cout << "ISPC shutdown complete\n";
    return 0;
}