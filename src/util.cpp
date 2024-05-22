/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file util.cpp
    @brief Various small utility routines.
*/

#include "util.h"
#include "module.h"

#ifdef ISPC_HOST_IS_LINUX
#include <alloca.h>
#include <unistd.h>
#elif defined(ISPC_HOST_IS_WINDOWS)
#include <malloc.h>
#include <shlwapi.h>
#ifndef __MINGW32__
#define alloca _alloca
#endif // __MINGW32__
#endif // ISPC_HOST_IS_LINUX
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef ISPC_HOST_IS_WINDOWS
#include <direct.h>
#include <io.h>
#include <windows.h>
#else
#include <errno.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif // ISPC_HOST_IS_WINDOWS
#include <algorithm>
#include <set>

#include <llvm/IR/DataLayout.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

#ifdef ISPC_HOST_IS_APPLE
#if ISPC_LLVM_VERSION >= ISPC_LLVM_17_0
// Provide own definition of std::__libcpp_verbose_abort to avoid missing
// symbols error in macOS builds.
// See https://libcxx.llvm.org/UsingLibcxx.html#overriding-the-default-termination-handler
// It is not quite clear why and where this symbol is used.
void std::__libcpp_verbose_abort(char const *format, ...) {
    va_list list;
    va_start(list, format);
    vfprintf(stderr, format, list);
    va_end(list);

    abort();
}
#endif // ISPV_LLVM_17_0
#endif // ISPC_HOST_IS_APPLE

using namespace ispc;

/** Returns the width of the terminal where the compiler is running.
    Finding this out may fail in a variety of reasonable situations (piping
    compiler output to 'less', redirecting output to a file, running the
    compiler under a debuffer; in this case, just return a reasonable
    default.
 */
int ispc::TerminalWidth() {
    if (g->disableLineWrap)
        return 1 << 30;

#if defined(ISPC_HOST_IS_WINDOWS)
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE || h == nullptr)
        return 80;
    CONSOLE_SCREEN_BUFFER_INFO bufferInfo = {{0}};
    GetConsoleScreenBufferInfo(h, &bufferInfo);
    return bufferInfo.dwSize.X;
#else
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) < 0)
        return 80;
    return w.ws_col;
#endif // ISPC_HOST_IS_WINDOWS
}

static bool lHaveANSIColors() {
    static bool r = (getenv("TERM") != nullptr && strcmp(getenv("TERM"), "dumb") != 0);
#ifndef ISPC_HOST_IS_WINDOWS
    r &= (bool)isatty(2);
    r |= g->forceColoredOutput;
#endif // !ISPC_HOST_IS_WINDOWS
    return r;
}

static const char *lStartBold() {
    if (lHaveANSIColors())
        return "\033[1m";
    else
        return "";
}

static const char *lStartRed() {
    if (lHaveANSIColors())
        return "\033[31m";
    else
        return "";
}

static const char *lStartBlue() {
    if (lHaveANSIColors())
        return "\033[34m";
    else
        return "";
}

static const char *lResetColor() {
    if (lHaveANSIColors())
        return "\033[0m";
    else
        return "";
}

/** Given a pointer into a string, find the end of the current word and
    return a pointer to its last character.
*/
static const char *lFindWordEnd(const char *buf) {
    while (*buf != '\0' && !isspace(*buf))
        ++buf;
    return buf;
}

/** When printing error messages, we sometimes want to include the source
    file line for context.  This function print the line(s) of the file
    corresponding to the provided SourcePos and underlines the range of the
    SourcePos with '^' symbols.
*/
static void lPrintFileLineContext(SourcePos p) {
    if (p.first_line == 0)
        return;

    FILE *f = fopen(p.name, "r");
    if (!f)
        return;

    int c, curLine = 1;
    while ((c = fgetc(f)) != EOF) {
        // Don't print more than three lines of context.  (More than that,
        // and we're probably doing the wrong thing...)
        if (curLine >= std::max(p.first_line, p.last_line - 2) && curLine <= p.last_line) {
            if (c == '\t')
                c = ' ';

            fputc(c, stderr);
        }
        if (c == '\n')
            ++curLine;
        if (curLine > p.last_line)
            break;
    }

    int i = 1;
    for (; i < p.first_column; ++i)
        fputc(' ', stderr);
    fputc('^', stderr);
    ++i;
    for (; i < p.last_column; ++i)
        fputc('^', stderr);
    fputc('\n', stderr);
    fputc('\n', stderr);

    fclose(f);
}

/** Counts the number of characters into the buf at which the numColons
    colon character is found.  Skips over ANSI escape sequences and doesn't
    include their characters in the final count.
 */
static int lFindIndent(int numColons, const char *buf) {
    int indent = 0;
    while (*buf != '\0') {
        if (*buf == '\033') {
            while (*buf != '\0' && *buf != 'm')
                ++buf;
            if (*buf == 'm')
                ++buf;
        } else {
            if (*buf == ':') {
                if (--numColons == 0)
                    break;
            }
            ++indent;
            ++buf;
        }
    }
    return indent + 2;
}

/** Print the given string to the given FILE, assuming the given output
    column width.  Break words as needed to avoid words spilling past the
    last column.  */
void ispc::PrintWithWordBreaks(const char *buf, int indent, int columnWidth, FILE *out) {
#ifdef ISPC_HOST_IS_WINDOWS
    fputs(buf, out);
    fputs("\n", out);
#else
    int column = 0;
    int width = std::max(40, columnWidth - 2);

    // Collect everything into a string and print it all at once at the end
    // -> try to avoid troubles with mangled error messages with
    // multi-threaded builds.
    std::string outStr;

    const char *msgPos = buf;
    while (true) {
        if (*msgPos == '\033') {
            // handle ANSI color escape: copy it to the output buffer
            // without charging for the characters it uses
            do {
                outStr.push_back(*msgPos++);
            } while (*msgPos != '\0' && *msgPos != 'm');
            continue;
        } else if (*msgPos == '\n') {
            // Handle newlines cleanly
            column = indent;
            outStr.push_back('\n');
            for (int i = 0; i < indent; ++i)
                outStr.push_back(' ');
            // Respect spaces after newlines
            ++msgPos;
            while (*msgPos == ' ') {
                outStr.push_back(' ');
                ++msgPos;
            }
            continue;
        }

        while (*msgPos != '\0' && isspace(*msgPos))
            ++msgPos;
        if (*msgPos == '\0')
            break;

        const char *wordEnd = lFindWordEnd(msgPos);
        if (column > indent && column + wordEnd - msgPos > width) {
            // This word would overflow, so start a new line
            column = indent;
            outStr.push_back('\n');
            // Indent to the same column as the ":" at the start of the
            // message.
            for (int i = 0; i < indent; ++i)
                outStr.push_back(' ');
        }

        // Finally go and copy the word
        while (msgPos != wordEnd) {
            outStr.push_back(*msgPos++);
            ++column;
        }
        outStr.push_back(' ');
        ++column;
    }
    outStr.push_back('\n');
    fputs(outStr.c_str(), out);
#endif
}

#ifdef ISPC_HOST_IS_WINDOWS
// we cover for the lack vasprintf and asprintf on windows (also covers mingw)
int vasprintf(char **sptr, const char *fmt, va_list argv) {
    int wanted = vsnprintf(*sptr = nullptr, 0, fmt, argv);
    if ((wanted < 0) || ((*sptr = (char *)malloc(1 + wanted)) == nullptr))
        return -1;

    return vsprintf(*sptr, fmt, argv);
}

int asprintf(char **sptr, const char *fmt, ...) {
    int retval;
    va_list argv;
    va_start(argv, fmt);
    retval = vasprintf(sptr, fmt, argv);
    va_end(argv);
    return retval;
}
#endif

/** Helper function for Error(), Warning(), etc.

    @param type   The type of message being printed (e.g. "Warning")
    @param p      Position in source file that is connected to the message
                  being printed
    @param fmt    printf()-style format string
    @param args   Arguments with values for format string % entries
*/
static void lPrint(const char *type, bool isError, SourcePos p, const char *fmt, va_list args) {
    char *errorBuf, *formattedBuf;
    if (vasprintf(&errorBuf, fmt, args) == -1) {
        fprintf(stderr, "vasprintf() unable to allocate memory!\n");
        exit(1);
    }

    int indent = 0;
    if (p.first_line == 0) {
        // We don't have a valid SourcePos, so create a message without it
        if (asprintf(&formattedBuf, "%s%s%s%s%s: %s%s", lStartBold(), isError ? lStartRed() : lStartBlue(), type,
                     lResetColor(), lStartBold(), errorBuf, lResetColor()) == -1) {
            fprintf(stderr, "asprintf() unable to allocate memory!\n");
            exit(1);
        }
        indent = lFindIndent(1, formattedBuf);
    } else {
        // Create an error message that includes the file and line number
        if (asprintf(&formattedBuf, "%s%s:%d:%d: %s%s%s%s: %s%s", lStartBold(), p.name, p.first_line, p.first_column,
                     isError ? lStartRed() : lStartBlue(), type, lResetColor(), lStartBold(), errorBuf,
                     lResetColor()) == -1) {
            fprintf(stderr, "asprintf() unable to allocate memory!\n");
            exit(1);
        }
        indent = lFindIndent(3, formattedBuf);
    }
    // Don't indent too much with long filenames
    indent = std::min(indent, 8);

    // Now that we've done all that work, see if we've already printed the
    // exact same error message.  If so, return, so we don't redundantly
    // print it and annoy the user.
    static std::set<std::string> printed;
    if (printed.find(formattedBuf) != printed.end()) {
        free(errorBuf);
        free(formattedBuf);
        return;
    }
    printed.insert(formattedBuf);

    PrintWithWordBreaks(formattedBuf, indent, TerminalWidth(), stderr);
    lPrintFileLineContext(p);

    free(errorBuf);
    free(formattedBuf);
}

void ispc::Error(SourcePos p, const char *fmt, ...) {
    if (m != nullptr) {
        ++m->errorCount;
        if ((g->errorLimit != -1) && (g->errorLimit <= m->errorCount - 1))
            return;
    }
    if (g->quiet)
        return;

    va_list args;
    va_start(args, fmt);
    lPrint("Error", true, p, fmt, args);
    va_end(args);
}

void ispc::Debug(SourcePos p, const char *fmt, ...) {
    if (!g->debugPrint || g->quiet)
        return;

    va_list args;
    va_start(args, fmt);
    lPrint("Debug", false, p, fmt, args);
    va_end(args);
}

void ispc::Warning(SourcePos p, const char *fmt, ...) {

    std::map<std::pair<int, std::string>, bool>::iterator turnOffWarnings_it =
        g->turnOffWarnings.find(std::pair<int, std::string>(p.last_line, std::string(p.name)));
    if ((turnOffWarnings_it != g->turnOffWarnings.end()) && (turnOffWarnings_it->second == false))
        return;

    if (g->warningsAsErrors && m != nullptr)
        ++m->errorCount;

    if (g->disableWarnings || g->quiet)
        return;

    va_list args;
    va_start(args, fmt);
    lPrint(g->warningsAsErrors ? "Error" : "Warning", g->warningsAsErrors, p, fmt, args);
    va_end(args);
}

void ispc::PerformanceWarning(SourcePos p, const char *fmt, ...) {
    std::string stdlibFile = "stdlib.ispc";
    std::string sourcePosName = p.name;
    if (!g->emitPerfWarnings ||
        (sourcePosName.length() >= stdlibFile.length() &&
         sourcePosName.compare(sourcePosName.length() - stdlibFile.length(), stdlibFile.length(), stdlibFile) == 0) ||
        g->quiet)
        return;

    std::map<std::pair<int, std::string>, bool>::iterator turnOffWarnings_it =
        g->turnOffWarnings.find(std::pair<int, std::string>(p.last_line, p.name));
    if (turnOffWarnings_it != g->turnOffWarnings.end())
        return;

    if (g->warningsAsErrors && m != nullptr)
        ++m->errorCount;

    va_list args;
    va_start(args, fmt);
    lPrint("Performance Warning", false, p, fmt, args);
    va_end(args);
}

static void lPrintBugText() {
    static bool printed = false;
    if (printed)
        return;

    printed = true;
    fprintf(stderr, "***\n"
                    "*** Please file a bug report at https://github.com/ispc/ispc/issues\n"
                    "*** (Including as much information as you can about how to "
                    "reproduce this error).\n"
                    "*** You have apparently encountered a bug in the compiler that we'd "
                    "like to fix!\n***\n");
}

[[noreturn]] void ispc::FatalError(const char *file, int line, const char *message) {
    fprintf(stderr, "%s(%d): FATAL ERROR: %s\n", file, line, message);
    lPrintBugText();
    abort();
}

void ispc::DoAssert(const char *file, int line, const char *expr) {
    fprintf(stderr, "%s:%u: Assertion failed: \"%s\".\n", file, line, expr);
    lPrintBugText();
    abort();
}

void ispc::DoAssertPos(SourcePos pos, const char *file, int line, const char *expr) {
    Error(pos, "Assertion failed (%s:%u): \"%s\".", file, line, expr);
    lPrintBugText();
    abort();
}

///////////////////////////////////////////////////////////////////////////

// http://en.wikipedia.org/wiki/Levenshtein_distance
int ispc::StringEditDistance(const std::string &str1, const std::string &str2, int maxDist) {
    // Small hack: don't return 0 if the strings are the same; if we've
    // gotten here, there's been a parsing error, and suggesting the same
    // string isn't going to actually help things.
    if (str1 == str2)
        return maxDist;

    int n1 = (int)str1.size(), n2 = (int)str2.size();
    int nmax = std::max(n1, n2);

    int *current = (int *)alloca((nmax + 1) * sizeof(int));
    int *previous = (int *)alloca((nmax + 1) * sizeof(int));

    for (int i = 0; i <= n2; ++i)
        previous[i] = i;

    for (int y = 1; y <= n1; ++y) {
        current[0] = y;
        int rowBest = y;

        for (int x = 1; x <= n2; ++x) {
            current[x] = std::min(previous[x - 1] + (str1[y - 1] == str2[x - 1] ? 0 : 1),
                                  std::min(current[x - 1], previous[x]) + 1);
            rowBest = std::min(rowBest, current[x]);
        }

        if (maxDist != 0 && rowBest > maxDist)
            return maxDist + 1;

        std::swap(current, previous);
    }

    return previous[n2];
}

std::vector<std::string> ispc::MatchStrings(const std::string &str, const std::vector<std::string> &options) {
    if (str.size() == 0 || (str.size() == 1 && !isalpha(str[0])))
        // don't even try...
        return std::vector<std::string>();

    const int maxDelta = 2;
    std::vector<std::string> matches[maxDelta + 1];

    // For all of the options that are up to maxDelta edit distance, store
    // them in the element of matches[] that corresponds to their edit
    // distance.
    for (int i = 0; i < (int)options.size(); ++i) {
        int dist = StringEditDistance(str, options[i], maxDelta + 1);
        if (dist <= maxDelta)
            matches[dist].push_back(options[i]);
    }

    // And return the first one of them, if any, that has at least one
    // match.
    for (int i = 0; i <= maxDelta; ++i) {
        if (matches[i].size())
            return matches[i];
    }
    return std::vector<std::string>();
}

std::pair<std::string, std::string> ispc::GetDirectoryAndFileName(const std::string &currentDirectory,
                                                                  const std::string &relativeName) {
    llvm::SmallString<256> fullPath;

    if (llvm::sys::path::is_absolute(relativeName)) {
        // We may actually get not relative path but absolute, then just use it.
        fullPath = relativeName;
    } else {
        fullPath = currentDirectory;
        llvm::sys::path::append(fullPath, relativeName);
    }

    // Extract the directory and file name from the full path
    std::string filename = llvm::sys::path::filename(fullPath).str();
    std::string dirname = llvm::sys::path::parent_path(fullPath).str();

    return std::make_pair(dirname, filename);
}

static std::set<std::string> lGetStringArray(const std::string &str) {
    std::set<std::string> result;

    Assert(str.find('-') != str.npos);

    size_t pos_prev = 0, pos;
    do {
        pos = str.find('-', pos_prev);
        std::string substr = str.substr(pos_prev, pos - pos_prev);
        result.insert(substr);
        pos_prev = pos;
        pos_prev++;
    } while (pos != str.npos);

    return result;
}

bool VerifyDataLayoutCompatibility(const std::string &module_dl, const std::string &lib_dl) {
    if (lib_dl.empty()) {
        // This is the case for most of library pre-compiled .ll files.
        return true;
    }

    // Get "canonical" form. Instead of looking at "raw" DataLayout string, we
    // look at the actual representation, as DataLayout class understands it.
    // In the most cases there's no difference. But on x86 Windows (i686-pc-win32),
    // clang generates a DataLayout string, which contains two definitions of f80,
    // which contradic: f80:128:128 followed by f80:32:32. This is a bug, but
    // correct thing to do is to interpret this exactly how LLVM would treat it,
    // so we create a DataLayout class and take its string representation.

    llvm::DataLayout d1(module_dl);
    llvm::DataLayout d2(lib_dl);

    std::string module_dl_canonic = d1.getStringRepresentation();
    std::string lib_dl_canonic = d2.getStringRepresentation();

    // Break down DataLayout strings to separate type definitions.
    std::set<std::string> module_dl_set = lGetStringArray(module_dl_canonic);
    std::set<std::string> lib_dl_set = lGetStringArray(lib_dl_canonic);

    // For each element in library data layout, find matching module element.
    // If no match is found, then we are in trouble and the library can't be used.
    for (std::set<std::string>::iterator it = lib_dl_set.begin(); it != lib_dl_set.end(); ++it) {
        // We use the simplest possible definition of "match", which is match exactly.
        // Ideally it should be relaxed and for triples [p|i|v|f|a|s]<size>:<abi>:<pref>
        // we should allow <pref> part (preferred alignment) to not match.
        // But this seems to have no practical value at this point.
        std::set<std::string>::iterator module_match = std::find(module_dl_set.begin(), module_dl_set.end(), *it);
        if (module_match == module_dl_set.end()) {
            // No match for this piece of library DataLayout was found,
            // return false.
            return false;
        }
        // Remove matching piece from Module set.
        module_dl_set.erase(module_match);
    }

    // We allow extra types to be defined in the Module, but we should check
    // that it's something that we expect. And we expect vectors and floats.
    for (std::set<std::string>::iterator it = module_dl_set.begin(); it != module_dl_set.end(); ++it) {
        if ((*it)[0] == 'v' || (*it)[0] == 'f') {
            continue;
        }
        return false;
    }

    return true;
}

bool ispc::IsStdin(const char *filepath) {
    Assert(filepath != nullptr);
    if (!strcmp(filepath, "-")) {
        return true;
    } else {
        return false;
    }
}
