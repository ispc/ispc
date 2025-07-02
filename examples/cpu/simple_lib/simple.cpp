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

// Include our Driver header
#include "driver.h"

int main() {
   
    // Compile the ISPC file using the Driver
    std::cout << "1. Compiling simple.ispc using Driver library...\n";
    
    // Simple and clean argument setup
    std::vector<const char*> args = {
        "ispc",                 // Program name
        "simple.ispc",          // Input file  
        "--target=sse2",        // Target architecture
        "-O2",                  // Optimization level
        "-o", "simple_ispc.o",  // Output object file
        "-h", "simple_ispc.h"   // Generate header file
    };
    
    // Create Driver instance from command line arguments
    auto driver = ispc::Driver::CreateFromArgs(static_cast<int>(args.size()), 
                                               const_cast<char**>(args.data()));
    
    if (!driver) {
        std::cerr << "Error: Failed to create Driver instance\n";
        return 1;
    }
    
    // Execute the compilation
    int result = driver->Execute();
    
    if (result == 0) {
        std::cout << "✓ ISPC compilation successful!\n";
    } else {
        std::cerr << "✗ ISPC compilation failed with code: " << result << "\n";
        ispc::Driver::Shutdown();
        return result;
    }
           
    // Clean up
    std::cout << "3. Cleaning up...\n";
    ispc::Driver::Shutdown();
    std::cout << "✓ Driver shutdown complete\n";
    
    return 0;
}

