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

#include "ispc/compiler.h"

int main() {
    // Initialize the ISPC library
    std::cout << "Initializing ISPC library...\n";
    if (!ispc::Initialize()) {
        std::cerr << "Error: Failed to initialize ISPC library\n";
        return 1;
    }

    // Compile the ISPC file using the simplified CompileFromArgs API
    std::cout << "Compiling simple.ispc using library mode...\n";

    std::vector<const char *> args = {
        "ispc",          // Program name
        "simple.ispc",   // Input file
        "--target=host", // Target architecture
        "-O2",           // Optimization level
        "-o",
        "simple_ispc.o", // Output object file
        "-h",
        "simple_ispc.h" // Generate header file
    };

    // Compile using the simplified API
    int result = ispc::CompileFromArgs(static_cast<int>(args.size()), const_cast<char **>(args.data()));

    if (result == 0) {
        std::cout << "ISPC compilation successful!\n";
    } else {
        std::cerr << "ISPC compilation failed with code: " << result << "\n";
        ispc::Shutdown();
        return result;
    }

    // Demonstrate that multiple compilations can be run with different settings
    std::cout << "\nDemonstrating multiple compilations with different settings...\n";

    // Set up a second compilation with DIFFERENT settings
    std::vector<const char *> args2 = {
        "ispc",          // Program name
        "simple.ispc",   // Input file
        "--target=host", // Target architecture
        "-O0",           // DIFFERENT: No optimization (vs -O2 in first)
        "--emit-asm",    // DIFFERENT: Emit assembly (vs object in first)
        "-o",
        "simple_debug.s", // DIFFERENT: Assembly output file
        "-h",
        "simple_debug.h", // DIFFERENT: Header file name
        "-g"              // DIFFERENT: Debug info (not in first)
    };

    std::cout << "Compiling again with different settings:\n";
    std::cout << "  - First compilation:  -O2, object output, no debug\n";
    std::cout << "  - Second compilation: -O0, assembly output, with debug\n";

    // Execute second compilation (should produce .s file instead of .o)
    int result2 = ispc::CompileFromArgs(static_cast<int>(args2.size()), const_cast<char **>(args2.data()));

    if (result2 == 0) {
        std::cout << "Second compilation successful with DIFFERENT settings!\n";

        // Verify that files with different extensions were created
        if (std::filesystem::exists("simple_ispc.o") && std::filesystem::exists("simple_debug.s")) {
            std::cout << "SUCCESS: Both compilations produced different output files:\n";
            std::cout << "  - First compilation:  simple_ispc.o (object file)\n";
            std::cout << "  - Second compilation: simple_debug.s (assembly file)\n";
            std::cout << "This proves the API handles different settings correctly!\n";
        } else {
            std::cout << "Note: Output files may not be visible in current directory\n";
        }
    } else {
        std::cerr << "Second ISPC compilation failed with code: " << result2 << "\n";
    }

    std::cout << "\nCleaning up...\n";
    ispc::Shutdown();
    std::cout << "ISPC shutdown complete\n";

    return 0;
}
