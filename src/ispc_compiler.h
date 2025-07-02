/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include <memory>

namespace ispc {

class Compiler {
  public:
    /**
     * @brief Initializes the ISPC library.
     * This must be called once before creating any Compiler instances.
     * Initializes LLVM targets and creates global state.
     *
     * @return true on success, false on failure.
     */
    static bool Initialize();

    /**
     * @brief Shuts down the ISPC library and releases global resources.
     * This should be called once when the consumer is finished using the library.
     * After calling this, Initialize() must be called again before creating new instances.
     */
    static void Shutdown();

    /**
     * @brief Factory method to create a Compiler instance from command-line arguments.
     * Initialize() must be called successfully before using this method.
     *
     * @param argc Argument count.
     * @param argv Argument vector.
     * @return A unique_ptr to a Compiler instance, or nullptr on failure.
     */
    static std::unique_ptr<Compiler> CreateFromArgs(int argc, char *argv[]);

    /**
     * @brief Checks if the link mode was requested.
     */
    bool IsLinkMode() const;

    /**
     * @brief Executes the compilation process based on the parsed arguments.
     * @return 0 on success, non-zero on failure.
     */
    int Compile();

    /**
     * @brief Executes the linking process based on the parsed arguments.
     * @return 0 on success, non-zero on failure.
     */
    int Link();

    /**
     * @brief Executes the appropriate action based on the driver state (link, help, or compile).
     * @return 0 on success, non-zero on failure.
     */
    int Execute();

    ~Compiler();

  private:
    Compiler();

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ispc