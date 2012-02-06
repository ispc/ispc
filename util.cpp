/*
  Copyright (c) 2010-2011, Intel Corporation
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

/** @file util.cpp
    @brief Various small utility routines.
*/

#include "util.h"
#include "module.h"
#ifdef ISPC_IS_WINDOWS
#include <shlwapi.h>
#ifdef __MINGW32__
#include <malloc.h> // for alloca()
#endif
#else
#include <alloca.h>
#endif
#include <stdio.h>

#include <stdio.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#ifdef ISPC_IS_WINDOWS
#include <io.h>
#include <direct.h>
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#include <errno.h>
#endif // ISPC_IS_WINDOWS
#include <set>

/** Returns the width of the terminal where the compiler is running.
    Finding this out may fail in a variety of reasonable situations (piping
    compiler output to 'less', redirecting output to a file, running the
    compiler under a debuffer; in this case, just return a reasonable
    default.
 */
static int
lTerminalWidth() {
    if (g->disableLineWrap)
        return 1<<30;

#if defined(ISPC_IS_WINDOWS)
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE || h == NULL)
        return 80;
    CONSOLE_SCREEN_BUFFER_INFO bufferInfo = { {0} };
    GetConsoleScreenBufferInfo(h, &bufferInfo);
    return bufferInfo.dwSize.X;
#else
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) < 0)
        return 80;
    return w.ws_col;
#endif // ISPC_IS_WINDOWS
}


/** Given a pointer into a string, find the end of the current word and
    return a pointer to its last character. 
*/
static const char *
lFindWordEnd(const char *buf) {
    while (*buf != '\0' && !isspace(*buf))
        ++buf;
    return buf;
}

/** When printing error messages, we sometimes want to include the source
    file line for context.  This function print the line(s) of the file
    corresponding to the provided SourcePos and underlines the range of the
    SourcePos with '^' symbols.
*/
static void
lPrintFileLineContext(SourcePos p) {
    if (p.first_line == 0)
        return;

    FILE *f = fopen(p.name, "r");
    if (!f)
        return;

    int c, curLine = 1;
    while ((c = fgetc(f)) != EOF) {
        // Don't print more than three lines of context.  (More than that,
        // and we're probably doing the wrong thing...)
        if (curLine >= std::max(p.first_line, p.last_line-2) && 
            curLine <= p.last_line)
            fputc(c, stderr);
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

/** Print the given string to the given FILE, assuming the given output
    column width.  Break words as needed to avoid words spilling past the
    last column.  */
static void
lPrintWithWordBreaks(const char *buf, int columnWidth, FILE *out) {
#ifdef ISPC_IS_WINDOWS
    fputs(buf, out);
#else
    int column = 0;
    Assert(strchr(buf, ':') != NULL);
    int indent = strchr(buf, ':') - buf + 2;
    int width = std::max(40, columnWidth - 2);

    // Collect everything into a string and print it all at once at the end
    // -> try to avoid troubles with mangled error messages with
    // multi-threaded builds.
    std::string outStr;

    const char *msgPos = buf;
    while (true) {
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
            // message
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


#ifdef ISPC_IS_WINDOWS
// we cover for the lack vasprintf and asprintf on windows (also covers mingw)
int
vasprintf(char **sptr, const char *fmt, va_list argv)
{
    int wanted = vsnprintf(*sptr = NULL, 0, fmt, argv);
    if((wanted < 0) || ((*sptr = (char*)malloc( 1 + wanted )) == NULL))
        return -1;

    return vsprintf(*sptr, fmt, argv);
}


int
asprintf(char **sptr, const char *fmt, ...)
{
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
static void
lPrint(const char *type, SourcePos p, const char *fmt, va_list args) {
    char *errorBuf, *formattedBuf;
    if (vasprintf(&errorBuf, fmt, args) == -1) {
        fprintf(stderr, "vasprintf() unable to allocate memory!\n");
        abort();
    }
    if (p.first_line == 0) {
        // We don't have a valid SourcePos, so create a message without it
        if (asprintf(&formattedBuf, "%s: %s\n", type, errorBuf) == -1) {
            fprintf(stderr, "asprintf() unable to allocate memory!\n");
            exit(1);
        }
    }
    else {
        // Create an error message that includes the file and line number
        if (asprintf(&formattedBuf, "%s:%d:%d: %s: %s\n", p.name, 
                     p.first_line, p.first_column, type, errorBuf) == -1) {
            fprintf(stderr, "asprintf() unable to allocate memory!\n");
            exit(1);
        }
    }

    // Now that we've done all that work, see if we've already printed the
    // exact same error message.  If so, return, so we don't redundantly
    // print it and annoy the user.
    static std::set<std::string> printed;
    if (printed.find(formattedBuf) != printed.end())
        return;
    printed.insert(formattedBuf);

    lPrintWithWordBreaks(formattedBuf, lTerminalWidth(), stderr);
    lPrintFileLineContext(p);

    free(errorBuf);
    free(formattedBuf);
}


void
Error(SourcePos p, const char *fmt, ...) {
    if (m != NULL) ++m->errorCount;
    if (g->quiet)
        return;

    va_list args;
    va_start(args, fmt);
    lPrint("Error", p, fmt, args);
    va_end(args);
}


void
Debug(SourcePos p, const char *fmt, ...) {
    if (!g->debugPrint || g->quiet)
        return;

    va_list args;
    va_start(args, fmt);
    lPrint("Debug", p, fmt, args);
    va_end(args);
}


void
Warning(SourcePos p, const char *fmt, ...) {
    if (g->warningsAsErrors && m != NULL)
        ++m->errorCount;

    if (g->disableWarnings || g->quiet)
        return;

    va_list args;
    va_start(args, fmt);
    lPrint(g->warningsAsErrors ? "Error" : "Warning", p, fmt, args);
    va_end(args);
}


void
PerformanceWarning(SourcePos p, const char *fmt, ...) {
    if (!g->emitPerfWarnings || strcmp(p.name, "stdlib.ispc") == 0 ||
        g->quiet)
        return;

    va_list args;
    va_start(args, fmt);
    lPrint("Performance Warning", p, fmt, args);
    va_end(args);
}


void
FatalError(const char *file, int line, const char *message) {
    fprintf(stderr, "%s(%d): FATAL ERROR: %s\n", file, line, message);
    fprintf(stderr, "***\n"
            "*** Please file a bug report at https://github.com/ispc/ispc/issues\n"
            "*** (Including as much information as you can about how to "
            "reproduce this error).\n"
            "*** You have apparently encountered a bug in the compiler that we'd "
            "like to fix!\n***\n");
    abort();
}

///////////////////////////////////////////////////////////////////////////

// http://en.wikipedia.org/wiki/Levenshtein_distance
int
StringEditDistance(const std::string &str1, const std::string &str2, int maxDist) {
    // Small hack: don't return 0 if the strings are the same; if we've
    // gotten here, there's been a parsing error, and suggesting the same
    // string isn't going to actually help things.
    if (str1 == str2)
        return maxDist;

    int n1 = (int)str1.size(), n2 = (int)str2.size();
    int nmax = std::max(n1, n2);

    int *current =  (int *)alloca((nmax+1) * sizeof(int));
    int *previous = (int *)alloca((nmax+1) * sizeof(int));

    for (int i = 0; i <= n2; ++i)
        previous[i] = i;

    for (int y = 1; y <= n1; ++y) {
        current[0] = y;
        int rowBest = y;

        for (int x = 1; x <= n2; ++x) {
            current[x] = std::min(previous[x-1] + (str1[y-1] == str2[x-1] ? 0 : 1),
                                  std::min(current[x-1], previous[x])+1);
            rowBest = std::min(rowBest, current[x]);
        }

        if (maxDist != 0 && rowBest > maxDist)
            return maxDist + 1;

        std::swap(current, previous);
    }

    return previous[n2];
}


std::vector<std::string> 
MatchStrings(const std::string &str, const std::vector<std::string> &options) {
    if (str.size() == 0 || (str.size() == 1 && !isalpha(str[0])))
        // don't even try...
        return std::vector<std::string>();

    const int maxDelta = 2;
    std::vector<std::string> matches[maxDelta+1];

    // For all of the options that are up to maxDelta edit distance, store
    // them in the element of matches[] that corresponds to their edit
    // distance.
    for (int i = 0; i < (int)options.size(); ++i) {
        int dist = StringEditDistance(str, options[i], maxDelta+1);
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


void
GetDirectoryAndFileName(const std::string &currentDirectory, 
                        const std::string &relativeName,
                        std::string *directory, std::string *filename) {
#ifdef ISPC_IS_WINDOWS
    char path[MAX_PATH];
    const char *combPath = PathCombine(path, currentDirectory.c_str(),
                                       relativeName.c_str());
    Assert(combPath != NULL);
    const char *filenamePtr = PathFindFileName(combPath);
    *filename = filenamePtr;
    *directory = std::string(combPath, filenamePtr - combPath);
#else
    // We need a fully qualified path.  First, see if the current file name
    // is fully qualified itself--in that case, the current working
    // directory isn't needed.  
    // @todo This probably needs to be smarter for Windows...
    std::string fullPath;
    if (relativeName[0] == '/')
        fullPath = relativeName;
    else {
        fullPath = g->currentDirectory;
        if (fullPath[fullPath.size()-1] != '/')
            fullPath.push_back('/');
        fullPath += relativeName;
    }

    // now, we need to separate it into the base name and the directory
    const char *fp = fullPath.c_str();
    const char *basenameStart = strrchr(fp, '/');
    Assert(basenameStart != NULL);
    ++basenameStart;
    Assert(basenameStart != '\0');
    *filename = basenameStart;
    *directory = std::string(fp, basenameStart - fp);
#endif // ISPC_IS_WINDOWS
}
