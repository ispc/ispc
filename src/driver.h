/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include <memory>

namespace ispc {

class Driver {
  public:
    /**
     * @brief Factory method to create a Driver instance from command-line arguments.
     * This method encapsulates LLVM initialization, creation of globals, and argument parsing.
     *
     * @param argc Argument count.
     * @param argv Argument vector.
     * @return A unique_ptr to a Driver instance, or nullptr on failure (excluding --help).
     */
    static std::unique_ptr<Driver> CreateFromArgs(int argc, char *argv[]);

    /**
     * @brief Shuts down the driver and releases global resources.
     * This should be called once when the consumer is finished using the library.
     */
    static void Shutdown();

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

    ~Driver();

  private:
    Driver();

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ispc