/*
  Copyright (c) 2025, Intel Corporation
  SPDX-License-Identifier: BSD-3-Clause
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include "ispc_compiler.h"

int main() {
   
    // Compile the ISPC file using the ispc::Compiler class
    std::cout << "Compiling simple.ispc using library mode...\n";
    
    std::vector<const char*> args = {
        "ispc",                 // Program name
        "simple.ispc",          // Input file  
        "--target=host",        // Target architecture
        "-O2",                  // Optimization level
        "-o", "simple_ispc.o",  // Output object file
        "-h", "simple_ispc.h"   // Generate header file
    };
    
    // Create ispc::Compiler instance from command line arguments
    auto instance = ispc::Compiler::CreateFromArgs(static_cast<int>(args.size()), 
                                                   const_cast<char**>(args.data()));
    
    if (!instance) {
        std::cerr << "Error: Failed to create ispc::Compiler instance\n";
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
           
    std::cout << "Cleaning up...\n";
    ispc::Compiler::Shutdown();
    std::cout << "ispc::Compiler shutdown complete\n";
    
    return 0;
}

