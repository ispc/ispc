/*
  Copyright (c) 2010-2021, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
void GetDirectoryAndFileName(const std::string &currentDir, const std::string &relativeName, std::string *directory,
                             std::string *filename);

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
