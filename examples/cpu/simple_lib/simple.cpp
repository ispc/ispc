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

    std::vector<std::string> args1 = {"ispc", "simple.ispc", "--target=host", "-O2", "-o",
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
    std::vector<std::string> args2 = {"ispc", "simple.ispc", "--target=host",  "-O0", "--emit-asm", "-o",
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

    std::cout << "\nCleaning up...\n";
    ispc::Shutdown();
    std::cout << "ISPC shutdown complete\n";

    return 0;
}