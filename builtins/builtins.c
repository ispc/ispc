/*
  Copyright (c) 2010-2020, Intel Corporation
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

/** @file builtins-c.c
    @brief Standard library function implementations written in C.

    This file provides C implementations of various functions that can be
    called from ispc programs; in other words, this file is *not* linked
    into the ispc compiler executable, but rather provides functions that
    can be compiled into ispc programs.

    When the ispc compiler is built, this file is compiled with clang to
    generate LLVM bitcode.  This bitcode is later linked in to the program
    being compiled by the DefineStdlib() function.  The first way to access
    definitions from this file is by asking for them name from the
    llvm::Module's' symbol table (e.g. as the PrintStmt implementation does
    with __do_print() below.  Alternatively, if a function defined in this
    file has a signature that can be mapped back to ispc types by the
    lLLVMTypeToIspcType() function, then its declaration will be made
    available to ispc programs at compile time automatically.
  */

#ifndef WASM

#ifdef _MSC_VER
// We do want old school sprintf and don't want secure Microsoft extensions.
// And we also don't want warnings about it, so the define.
#define _CRT_SECURE_NO_WARNINGS
#else
// Some versions of glibc has "fortification" feature, which expands sprintf
// to __builtin___sprintf_chk(..., __builtin_object_size(...), ...).
// We don't want this kind of expansion, as we don't support these intrinsics.
#define _FORTIFY_SOURCE 0
#endif

#ifndef _MSC_VER
// In unistd.h we need the definition of sysconf and _SC_NPROCESSORS_ONLN used as its arguments.
// We should include unistd.h, but it doesn't really work well for cross compilation, as
// requires us to carry around unistd.h, which is not available on Windows out of the box.
#include <unistd.h>

// Just for the reference: these lines are eventually included from unistd.h
// #define _SC_NPROCESSORS_ONLN 58
// long sysconf(int);
#endif // !_MSC_VER
#endif // !WASM
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int Bool;

#define PRINT_BUF_SIZE 4096

#define APPEND(str)                                                                                                    \
    do {                                                                                                               \
        int offset = bufp - &printString[0];                                                                           \
        *bufp = '\0';                                                                                                  \
        strncat(bufp, str, PRINT_BUF_SIZE - offset);                                                                   \
        bufp += strlen(str);                                                                                           \
        if (bufp >= &printString[PRINT_BUF_SIZE])                                                                      \
            goto done;                                                                                                 \
    } while (0) /* eat semicolon */

#define PRINT_SCALAR(fmt, type)                                                                                        \
    sprintf(tmpBuf, fmt, *((type *)ptr));                                                                              \
    APPEND(tmpBuf);                                                                                                    \
    break

#define PRINT_VECTOR(fmt, type)                                                                                        \
    *bufp++ = '[';                                                                                                     \
    if (bufp == &printString[PRINT_BUF_SIZE])                                                                          \
        break;                                                                                                         \
    for (int i = 0; i < width; ++i) {                                                                                  \
        /* only print the value if the current lane is executing */                                                    \
        if (mask & (1ull << i))                                                                                        \
            sprintf(tmpBuf, fmt, ((type *)ptr)[i]);                                                                    \
        else                                                                                                           \
            sprintf(tmpBuf, "((" fmt "))", ((type *)ptr)[i]);                                                          \
        APPEND(tmpBuf);                                                                                                \
        *bufp++ = (i != width - 1 ? ',' : ']');                                                                        \
    }                                                                                                                  \
    break

/** This function is called by PrintStmt to do the work of printing values
    from ispc programs.  Note that the function signature here must match
    the parameters that PrintStmt::EmitCode() generates.

    @param format  Print format string
    @param types   Encoded types of the values being printed.
                   (See lEncodeType()).
    @param width   Vector width of the compilation target
    @param mask    Current lane mask when the print statemnt is called
    @param args    Array of pointers to the values to be printed
 */
void __do_print(const char *format, const char *types, int width, uint64_t mask, void **args) {
    char printString[PRINT_BUF_SIZE + 1]; // +1 for trailing NUL
    char *bufp = &printString[0];
    char tmpBuf[256];

    int argCount = 0;
    while (*format && bufp < &printString[PRINT_BUF_SIZE]) {
        // Format strings are just single percent signs.
        if (*format != '%') {
            *bufp++ = *format;
        } else {
            if (*types) {
                void *ptr = args[argCount++];
                // Based on the encoding in the types string, cast the
                // value appropriately and print it with a reasonable
                // printf() formatting string.
                switch (*types) {
                case 'b': {
                    sprintf(tmpBuf, "%s", *((Bool *)ptr) ? "true" : "false");
                    APPEND(tmpBuf);
                    break;
                }
                case 'B': {
                    *bufp++ = '[';
                    if (bufp == &printString[PRINT_BUF_SIZE])
                        break;
                    for (int i = 0; i < width; ++i) {
                        if (mask & (1ull << i)) {
                            sprintf(tmpBuf, "%s", ((Bool *)ptr)[i] ? "true" : "false");
                            APPEND(tmpBuf);
                        } else
                            APPEND("_________");
                        *bufp++ = (i != width - 1) ? ',' : ']';
                    }
                    break;
                }
                case 'i':
                    PRINT_SCALAR("%d", int);
                case 'I':
                    PRINT_VECTOR("%d", int);
                case 'u':
                    PRINT_SCALAR("%u", unsigned int);
                case 'U':
                    PRINT_VECTOR("%u", unsigned int);
                case 'f':
                    PRINT_SCALAR("%f", float);
                case 'F':
                    PRINT_VECTOR("%f", float);
                case 'l':
                    PRINT_SCALAR("%lld", long long);
                case 'L':
                    PRINT_VECTOR("%lld", long long);
                case 'v':
                    PRINT_SCALAR("%llu", unsigned long long);
                case 'V':
                    PRINT_VECTOR("%llu", unsigned long long);
                case 'd':
                    PRINT_SCALAR("%f", double);
                case 'D':
                    PRINT_VECTOR("%f", double);
                case 'p':
                    PRINT_SCALAR("%p", void *);
                case 'P':
                    PRINT_VECTOR("%p", void *);
                default:
                    APPEND("UNKNOWN TYPE ");
                    *bufp++ = *types;
                }
                ++types;
            }
        }
        ++format;
    }

done:
    *bufp = '\0';
    fputs(printString, stdout);
    fflush(stdout);
}

#ifdef WASM
int __num_cores() { return 1; }
#else // WASM
int __num_cores() {
#if defined(_MSC_VER) || defined(__MINGW32__)
    // This is quite a hack.  Including all of windows.h to get this definition
    // pulls in a bunch of stuff that leads to undefined symbols at link time.
    // So we don't #include <windows.h> but instead have the equivalent declarations
    // here.  Presumably this struct declaration won't be changing in the future
    // anyway...
    struct SYSTEM_INFO {
        int pad0[2];
        void *pad1[2];
        int *pad2;
        int dwNumberOfProcessors;
        int pad3[3];
    };

    struct SYSTEM_INFO sysInfo;
    extern void __stdcall GetSystemInfo(struct SYSTEM_INFO *);
    GetSystemInfo(&sysInfo);
    return sysInfo.dwNumberOfProcessors;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif // !_MSC_VER
}
#endif // !WASM
