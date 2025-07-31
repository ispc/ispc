/*
  Copyright (c) 2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file args.h
    @brief Command line argument parsing interface for ispc
*/

#pragma once

#include <string>
#include <vector>

// Include necessary headers
#include "module.h"
#include "target_enums.h"

namespace ispc {

enum class ArgsParseResult : int { success = 0, failure = 1, help_requested = 2 };

/** We take arguments from both the command line as well as from the
 *  ISPC_ARGS environment variable - and each of these can include a file containing
 *  additional arguments using @<filename>. This function returns a new set of
 *  arguments representing the ones from all these sources merged together.
 */
void GetAllArgs(int Argc, char *Argv[], std::vector<char *> &argv);

/** Free all dynamically allocated argument strings */
void FreeArgv(std::vector<char *> &argv);

/** Parse command line arguments and populate configuration
 *  @param argc Number of command line arguments
 *  @param argv Array of command line argument strings
 *  @param file Output parameter for input file name
 *  @param arch Output parameter for target architecture
 *  @param cpu Output parameter for target CPU
 *  @param targets Output parameter for target specifications
 *  @param output Output parameter for output configuration
 *  @param linkFileNames Output parameter for link file names
 *  @param isLinkMode Output parameter indicating if in link mode
 *  @return ArgsParseResult indicating success, failure, or help requested
 */
ArgsParseResult ParseCommandLineArgs(int argc, char *argv[], std::string &file, Arch &arch, std::string &cpu,
                                     std::vector<ISPCTarget> &targets, struct Module::Output &output,
                                     std::vector<std::string> &linkFileNames, bool &isLinkMode);

/** Validate input file name
 *  @param filename Input file name to validate
 *  @param allowStdin Whether to allow stdin ("-") as input (default: true)
 *  @return true if input is valid, false otherwise
 */
bool ValidateInput(const std::string &filename, bool allowStdin = true);

/** Validate output configuration and warn if no output specified
 *  @param output Output configuration to validate
 */
void ValidateOutput(const Module::Output &output);

} // namespace ispc