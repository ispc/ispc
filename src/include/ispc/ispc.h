/*
  Copyright (c) 2025, Intel Corporation
  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ispc {

/**
 * @brief Initializes the ISPC library.
 * This must be called once before creating any ISPCEngine instances.
 * Initializes LLVM targets and creates global state.
 *
 * @return true on success, false on failure.
 */
bool Initialize();

/**
 * @brief Shuts down the ISPC library and releases global resources.
 * This should be called once when the consumer is finished using the library.
 * After calling this, Initialize() must be called again before creating new instances.
 */
void Shutdown();

/**
 * @brief Compiles ISPC code from command-line arguments.
 * This function parses command-line arguments and executes the appropriate action
 * (compile, link, or help). Initialize() must be called successfully before using this function.
 *
 * @param args Vector of command-line arguments. The first argument should be the program name
 *             or any dummy string (it will be ignored eventually).
 * @return 0 on success, non-zero on failure.
 */
int CompileFromArgs(const std::vector<std::string> &args);

/**
 * @brief Compiles ISPC code from command-line arguments.
 * This function parses command-line arguments and executes the appropriate action
 * (compile, link, or help). Initialize() must be called successfully before using this function.
 *
 * @param argc Argument count.
 * @param argv Argument vector. The first argument should be the program name
 *             or any dummy string (it will be ignored eventually).
 * @return 0 on success, non-zero on failure.
 */
int CompileFromCArgs(int argc, char *argv[]);

class ISPCEngine {
  public:
    /**
     * @brief Factory method to create a ISPCEngine instance from command-line arguments.
     * Initialize() must be called successfully before using this method.
     *
     * @param args Vector of command-line arguments. The first argument should be the program name
     *             or any dummy string (it will be ignored eventually).
     * @return A unique_ptr to a ISPCEngine instance, or nullptr on failure.
     */
    static std::unique_ptr<ISPCEngine> CreateFromArgs(const std::vector<std::string> &args);

    /**
     * @brief Factory method for C-style argc/argv arguments.
     * @param argc Argument count.
     * @param argv Argument vector. The first argument should be the program name
     *             or any dummy string (it will be ignored eventually).
     * @return A unique_ptr to a ISPCEngine instance, or nullptr on failure.
     */
    static std::unique_ptr<ISPCEngine> CreateFromCArgs(int argc, char *argv[]);

    /**
     * @brief Executes the appropriate action based on the driver state (link, help, or compile).
     * @return 0 on success, non-zero on failure.
     */
    int Execute();

    /**
     * @brief Compiles ISPC code from a file using JIT compilation.
     * @param filename Path to the ISPC source file.
     * @return 0 on success, non-zero on failure.
     */
    int CompileFromFileToJit(const std::string &filename);

    /**
     * @brief Retrieves a function pointer from JIT-compiled code.
     * @param functionName Name of the exported function.
     * @return Function pointer or nullptr if not found.
     */
    void *GetJitFunction(const std::string &functionName);

    /**
     * @brief Sets a user-provided runtime function for JIT compilation.
     * Runtime functions must be provided before calling CompileFromFileToJit().
     * @param functionName Name of the runtime function (e.g., "ISPCLaunch", "ISPCSync", "ISPCAlloc")
     * @param functionPtr Pointer to the user-provided function implementation
     * @return true on success, false if functionName is invalid or functionPtr is null
     */
    bool SetJitRuntimeFunction(const std::string &functionName, void *functionPtr);

    /**
     * @brief Clears a specific runtime function.
     * @param functionName Name of the function to clear
     */
    void ClearJitRuntimeFunction(const std::string &functionName);

    /**
     * @brief Clears all runtime functions.
     */
    void ClearJitRuntimeFunctions();

    /**
     * @brief Clears all JIT-compiled code.
     */
    void ClearJitCode();

    ~ISPCEngine();

  private:
    ISPCEngine();

    /**
     * @brief Checks if the engine is in JIT mode.
     * @return true if JIT mode is active, false otherwise.
     */
    bool IsJitMode() const;

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ispc