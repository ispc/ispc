/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file util.h

    @brief
*/

#pragma once

#include "ispc.h"

#ifdef ISPC_HOST_IS_WINDOWS
#include <stdarg.h>
#endif

#ifdef __GNUG__
#define PRINTF_FUNC __attribute__((format(printf, 2, 3)))
#else
#define PRINTF_FUNC
#endif // __GNUG__

// for cross-platform compatibility
#ifdef ISPC_HOST_IS_WINDOWS
int vasprintf(char **sptr, const char *fmt, va_list argv);
int asprintf(char **sptr, const char *fmt, ...);
#endif

// LLVM 19.0+ uses a different API for insertion points, which is related to debug info representation. See more details
// here: https://llvm.org/docs/RemoveDIsDebugInfo.html#how-to-update-existing-code
#if ISPC_LLVM_VERSION >= ISPC_LLVM_19_0
#define ISPC_INSERTION_POINT_ITERATOR(iterator) iterator
// Our utility functions that create llvm instructions are built with the assumption that insertBefore argument may be
// null, which means that instruction needs not to be inserted into the IR. When this macro is removed, the interface of
// these utility functions will need to be changed to accept either llvm::BasicBlock::iterator or llvm::InsertionPoint
// instead of llvm::Instruction*, so null value may be passed without any trouble.
#define ISPC_INSERTION_POINT_INSTRUCTION(instruction)                                                                  \
    ((instruction) ? ((llvm::InsertPosition)instruction->getIterator()) : (llvm::InsertPosition) nullptr)
#else
#define ISPC_INSERTION_POINT_ITERATOR(iterator) &*iterator
#define ISPC_INSERTION_POINT_INSTRUCTION(instruction) instruction
#endif

namespace ispc {

struct SourcePos;

/** Rounds up the given value to the next power of two, if it isn't a power
    of two already. */
inline uint32_t RoundUpPow2(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

/** Prints a debugging message.  These messages are only printed if
    g->debugPrint is \c true.  In addition to a program source code
    position to associate with the message, a printf()-style format string
    is passed along with any values needed for items in the format
    string.
*/
void Debug(SourcePos p, const char *format, ...) PRINTF_FUNC;

/** Prints a warning about an issue found during compilation.  Compilation
    can still continue after warnings; they are purely informative for the
    user.  In addition to a program source code position to associate with
    the message, a printf()-style format string is passed along with any
    values needed for items in the format string.
*/
void Warning(SourcePos p, const char *format, ...) PRINTF_FUNC;

/** Prints an error message.  It is assumd that compilation can not be
    successfully completed after an error has been issued, though the
    system tries to continue compiling as much as possible, in order to be
    able to issue any subsequent error messages.  In addition to a program
    source code position to associate with the message, a printf()-style
    format string is passed along with any values needed for items in the
    format string.
*/
void Error(SourcePos p, const char *format, ...) PRINTF_FUNC;

/** Prints a message about a potential performance issue in the user's
    code.  These messages are purely informative and don't affect the
    completion of compilation.  In addition to a program source code
    position to associate with the message, a printf()-style format string
    is passed along with any values needed for items in the format
    string.
*/
void PerformanceWarning(SourcePos p, const char *format, ...) PRINTF_FUNC;

/** Reports that unreachable location is reached. This is a kind of fatal error
    that causes the program to terminate.
 */
#define UNREACHABLE() FatalError(__FILE__, __LINE__, "unreachable code")

/** Reports a fatal error that causes the program to terminate.  This
    should only be used for cases where there is an internal error in the
    compiler.
 */
#define FATAL(message) FatalError(__FILE__, __LINE__, message)

/** This function generally shouldn't be called directly, but should be
    used via the FATAL macro, which includes the file and line number where
    the error was issued.
 */
[[noreturn]] void FatalError(const char *file, int line, const char *message);

/** Asserts that expr parameter is not equal to zero. Otherwise the program is
    terminated with propper error message and with file and line number where
    the assertion happend.
 */
#define Assert(expr) ((void)((expr) ? 0 : ((void)DoAssert(__FILE__, __LINE__, #expr), 0)))

/** This function generally shouldn't be called directly, but should be
    used via the Assert macro, which includes the file and line number where
    the assertion happens.
    Note: avoid adding [[noreturn]] as VS2017 treats Assert macros as never returning.
 */
void DoAssert(const char *file, int line, const char *expr);

/** Asserts that expr parameter is not equal to zero. Otherwise the program is
    terminated with propper error message and with file and line number where
    the assertion happend and the information about source position in the user
    program, which has triggered the problem.
 */
#define AssertPos(pos, expr) ((void)((expr) ? 0 : ((void)DoAssertPos(pos, __FILE__, __LINE__, #expr), 0)))

/** This function generally shouldn't be called directly, but should be
    used via the AssertPos macro, which includes the file and line number where
    the assertion happens.
    Note: avoid adding [[noreturn]] as VS2017 treats AssertPos macros as never returning.
 */
void DoAssertPos(SourcePos pos, const char *file, int line, const char *expr);

/** Returns the number of single-character edits needed to transform
    between the two strings.

    @param str1    First string
    @param str2    Second string
    @param maxDist Maximum number of single-character edits allowed
    @return        Number of single-character edits to transform from str1
                   to str2, or maxDist+1 if it's not psosible to do so
                   in fewer than maxDist steps
*/
int StringEditDistance(const std::string &str1, const std::string &str2, int maxDist);

/** Given a string and a set of candidate strings, returns the set of
    candidates that are "close" to the given string, where distance is
    measured by the number of single-character changes needed to transform
    between the two.  An empty vector may be returned if none of the
    options is close to \c str.
 */
std::vector<std::string> MatchStrings(const std::string &str, const std::vector<std::string> &options);

/** Given the current working directory and a filename relative to that
    directory, this function returns the final directory that the resulting
    file is in and the base name of the file itself. */
std::pair<std::string, std::string> GetDirectoryAndFileName(const std::string &currentDir,
                                                            const std::string &relativeName);

/** Verification routine, which ensures that DataLayout of the module being
    compiled is compatible with DataLayout of the library. At the moment we
    allow the library DataLayout to a subset of the module DataLayout (and
    extra floating point and vector types to be defined for module) or
    empty library DataLayout.
 */
bool VerifyDataLayoutCompatibility(const std::string &module_dl, const std::string &lib_dl);

/** Print the given string to the given FILE, assuming the given output
    column width.  Break words as needed to avoid words spilling past the
    last column.  */
void PrintWithWordBreaks(const char *buf, int indent, int columnWidth, FILE *out);

/** Returns the width of the terminal where the compiler is running.
    Finding this out may fail in a variety of reasonable situations (piping
    compiler output to 'less', redirecting output to a file, running the
    compiler under a debuffer; in this case, just return a reasonable
    default.
 */
int TerminalWidth();

/** Returns true is the filepath represents stdin, otherwise false.
 */
bool IsStdin(const char *);
} // namespace ispc

/** Global variable for yacc/bison parser debugging.
 *  Defined in the generated parser for main builds.
 */
extern int yydebug;
