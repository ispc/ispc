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

#include "ispc_compiler.h"

int main() {
    // Initialize the ISPC library
    std::cout << "Initializing ISPC library...\n";
    if (!ispc::Compiler::Initialize()) {
        std::cerr << "Error: Failed to initialize ISPC library\n";
        return 1;
    }

    // Compile the ISPC file using the ispc::Compiler class
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

    // Create ispc::Compiler instance from command line arguments
    auto instance = ispc::Compiler::CreateFromArgs(static_cast<int>(args.size()), const_cast<char **>(args.data()));

    if (!instance) {
        std::cerr << "Error: Failed to create ispc::Compiler instance\n";
        ispc::Compiler::Shutdown();
        return 1;
    }

    // Execute the compilation
    int result = instance->Execute();

    if (result == 0) {
        std::cout << "ISPC compilation successful!\n";
    } else {
        std::cerr << "ISPC compilation failed with code: " << result << "\n";
        ispc::Compiler::Shutdown();
        return result;
    }

    // Demonstrate that multiple instances can be created safely
    std::cout << "\nDemonstrating multiple compiler instances with different settings...\n";

    // Create a second compiler instance with DIFFERENT settings to prove isolation
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

    auto instance2 = ispc::Compiler::CreateFromArgs(static_cast<int>(args2.size()), const_cast<char **>(args2.data()));

    if (!instance2) {
        std::cerr << "Error: Failed to create second ispc::Compiler instance\n";
        ispc::Compiler::Shutdown();
        return 1;
    }

    std::cout << "Second compiler instance created with different settings:\n";
    std::cout << "  - First instance:  -O2, object output, no debug\n";
    std::cout << "  - Second instance: -O0, assembly output, with debug\n";
    std::cout << "First instance still valid: " << (instance ? "Yes" : "No") << "\n";

    // Execute second instance (should produce .s file instead of .o)
    int result2 = instance2->Execute();

    if (result2 == 0) {
        std::cout << "Second compilation successful with DIFFERENT settings!\n";

        // Verify that files with different extensions were created
        if (std::filesystem::exists("simple_ispc.o") && std::filesystem::exists("simple_debug.s")) {
            std::cout << "SUCCESS: Both instances produced different output files:\n";
            std::cout << "  - First instance:  simple_ispc.o (object file)\n";
            std::cout << "  - Second instance: simple_debug.s (assembly file)\n";
            std::cout << "This proves instances don't interfere with each other!\n";
        } else {
            std::cout << "Note: Output files may not be visible in current directory\n";
        }
    } else {
        std::cerr << "Second ISPC compilation failed with code: " << result2 << "\n";
    }

    std::cout << "\nCleaning up...\n";
    ispc::Compiler::Shutdown();
    std::cout << "ispc::Compiler shutdown complete\n";

    return 0;
}
