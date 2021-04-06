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

/** @file builtins-c-cpu.cpp
    @brief Standard library function implementations written in C/C++.

    This file provides C/C++ implementations of various functions that can be
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

#include "array.hpp"
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using SizeT = int;
using MaskT = uint64_t;
constexpr SizeT RES_STR_SIZE = 8196;
constexpr SizeT ARG_STR_SIZE = 1024;
template <typename T, SizeT Size> using StaticContainer = notstd::array<T, Size>;
template <typename T, SizeT Size> using StaticContainerRef = StaticContainer<T, Size> &;

template <SizeT Size> using StaticString = StaticContainer<char, Size>;
template <SizeT Size> using StaticStringRef = StaticContainerRef<char, Size>;
using WidthT = int;

#include "builtins/builtins-c-common.hpp"

class ArgWriter {
    const void *const *const args_;
    int curArgIdx_;
    WidthT width_;
    MaskT mask_;

  public:
    ArgWriter(const void *const *args, WidthT width, MaskT mask)
        : args_(args), curArgIdx_(0), width_(width), mask_(mask) {}

    template <typename T> auto uniform2Str() {
        auto fmt = PrintInfo::type2Specifier<T>();
        auto argPtr = getArg();
        StaticString<ARG_STR_SIZE> res;
        snprintf(&res[0], ARG_STR_SIZE, fmt, ValueAdapter<T>(*argCast<T>(argPtr)));
        return res;
    }

    template <typename T> auto varying2Str() {
        auto fmt = PrintInfo::type2Specifier<T>();
        StaticString<ARG_STR_SIZE> res;
        res[0] = '[';
        int haveBeenWritten = 1;
        auto argPtr = getArg();
        for (int lane = 0; lane < width_; ++lane) {
            if (mask_ & (1ull << lane)) {
                haveBeenWritten +=
                    snprintf(&res[haveBeenWritten], ARG_STR_SIZE, fmt, ValueAdapter<T>(argCast<T>(argPtr)[lane]));
            } else {
                haveBeenWritten = writeOffLane<T>(res, haveBeenWritten, argPtr, lane);
            }
            res[haveBeenWritten] = lane == width_ - 1 ? ']' : ',';
            ++haveBeenWritten;
        }
        res[haveBeenWritten] = '\0';
        return res;
    }

  private:
    const void *getArg() { return args_[curArgIdx_++]; }

    // casts void* to proper pointer
    // T is the type of argument (not pointer)
    template <typename T> auto argCast(const void *argPtr) { return reinterpret_cast<const T *>(argPtr); }

    // bools are passed as ints
    template <> auto argCast<bool>(const void *argPtr) { return reinterpret_cast<const int *>(argPtr); }

    template <typename T>
    int writeOffLane(StaticString<ARG_STR_SIZE> &res, int haveBeenWritten, const void *argPtr, int lane) {
        haveBeenWritten += snprintf(&res[haveBeenWritten], ARG_STR_SIZE, "((");
        auto fmt = PrintInfo::type2Specifier<T>();
        haveBeenWritten +=
            snprintf(&res[haveBeenWritten], ARG_STR_SIZE, fmt, ValueAdapter<T>(argCast<T>(argPtr)[lane]));
        haveBeenWritten += snprintf(&res[haveBeenWritten], ARG_STR_SIZE, "))");
        return haveBeenWritten;
    }

    template <>
    int writeOffLane<bool>(StaticString<ARG_STR_SIZE> &res, int haveBeenWritten, const void *argPtr, int lane) {
        auto fmt = PrintInfo::type2Specifier<bool>();
        haveBeenWritten += snprintf(&res[haveBeenWritten], ARG_STR_SIZE, fmt, OffLaneBoolStr);
        return haveBeenWritten;
    }
};

/** This function is called by PrintStmt to do the work of printing values
    from ispc programs.  Note that the function signature here must match
    the parameters that PrintStmt::EmitCode() generates.

    @param format  Print format string
    @param types   Encoded types of the values being printed.
                   (See lEncodeType()).
    @param width   Vector width of the compilation target
    @param mask    Current lane mask when the print statement is called
    @param args    Array of pointers to the values to be printed
 */
extern "C" void __do_print(const char *format, const char *types, WidthT width, MaskT mask, const void *const *args) {
    ArgWriter argWriter(args, width, mask);
    StaticString<RES_STR_SIZE> resultingStr = GetFormatedStr(format, types, argWriter);
    fputs(&resultingStr[0], stdout);
    fflush(stdout);
}

#ifdef WASM
extern "C" int __num_cores() { return 1; }
#else // WASM

extern "C" int __num_cores() {
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
