/*
  Copyright (c) 2019, Intel Corporation
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

/** @file builtins-c-genx.cpp
    @brief Standard library function implementations written in CM.

    This file provides CM implementations of various functions that can be
    called from ispc programs; in other words, this file is *not* linked
    into the ispc compiler executable, but rather provides functions that
    can be compiled into ispc programs.

    When the ispc compiler is built, this file is compiled with clang to
    generate LLVM bitcode.  This bitcode is later linked in to the program
    being compiled by the DefineStdlib() function.  The first way to access
    definitions from this file is by asking for them name from the
    llvm::Module's' symbol table (e.g. as the PrintStmt implementation does
    with __do_print_cm() below.  Alternatively, if a function defined in this
    file has a signature that can be mapped back to ispc types by the
    lLLVMTypeToIspcType() function, then its declaration will be made
    available to ispc programs at compile time automatically.
  */

using SizeT = unsigned;
using MaskT = unsigned long long;
using ArgT = unsigned;

constexpr SizeT RES_STR_SIZE = 128;
constexpr SizeT ARG_STR_SIZE = 100;

template <typename T, SizeT Size> using StaticContainer = vector<T, Size>;
template <typename T, SizeT Size> using StaticContainerRef = vector_ref<T, Size>;

template <SizeT Size> using StaticString = StaticContainer<char, Size>;
template <SizeT Size> using StaticStringRef = StaticContainerRef<char, Size>;
using WidthT = int;

#include "builtins/builtins-c-common.hpp"
#include <cm/cm.h>

enum LaneState : bool { OFF = false, ON = true };

// returns length of the string
// null character is not taken into account
template <typename T> inline int strLen(T str) {
    int len = 0;
    for (; str[len] != '\0'; ++len)
        ;
    return len;
}

// is meant to be used only for integral types
template <typename T> inline T max(T a, T b) {
    if (a > b)
        return a;
    return b;
}

template <typename T> inline const char *cmType2Specifier() { return type2Specifier<T>(); }

// cm printf doesn't support %p, so it is emulated with %X
template <> inline const char *cmType2Specifier<void *>() {
    if constexpr (sizeof(void *) > sizeof(unsigned)) {
        return "%016llX";
    } else {
        return "%08X";
    }
}

// returns how many characters are required to write another vector element (element of varying argument)
template <typename T, LaneState lane> inline int requiredSpace4VecElem() {
    auto specifierLen = strLen(cmType2Specifier<T>());
    if constexpr (lane) {
        // additional 1 for ',' or ']'
        return specifierLen + 1;
    } else {
        // additional 4 for "((" and "))", 1 for ',' or ']'
        return specifierLen + 5;
    }
}

template <> inline int requiredSpace4VecElem<bool, LaneState::ON>() {
    return max(strLen(ValueAdapter<bool>(false)), strLen(ValueAdapter<bool>(true)));
}

template <> inline int requiredSpace4VecElem<bool, LaneState::OFF>() { return strLen(OffLaneBoolStr); }

class ArgWriter {
    details::OffsetT offset_;
    const ArgT *const args_;
    int curArgIdx_;
    WidthT width_;
    MaskT mask_;

  public:
    inline ArgWriter(details::OffsetT offset, const ArgT *args, WidthT width, MaskT mask)
        : offset_(offset), args_(args), curArgIdx_(0), width_(width), mask_(mask) {}

    template <typename T> inline auto uniform2Str() {
        auto fmt = cmType2Specifier<T>();
        StaticString<ARG_STR_SIZE> res;
        auto wereCopied = CopyFullText(fmt, res, 0, ARG_STR_SIZE - 1);
        res[wereCopied] = '\0';

        writeArg<T>();
        return res;
    }

    template <> inline auto uniform2Str<bool>() {
        StaticString<ARG_STR_SIZE> res;
        auto wereCopied = CopyFullText(ValueAdapter<bool>(getElementaryArg()), res, 0, ARG_STR_SIZE - 1);
        res[wereCopied] = '\0';
        return res;
    }

    template <typename T> inline auto varying2Str() {
        StaticString<ARG_STR_SIZE> res;
        res[0] = '[';
        int haveBeenWritten = 1;
        for (int lane = 0; lane < width_; ++lane) {
            if (mask_ & (1ull << lane)) {
                // require 1 additional character for '\0'
                if (haveBeenWritten >= ARG_STR_SIZE - requiredSpace4VecElem<T, LaneState::ON>() - 1) {
                    // there's not enough space to write another value
                    break;
                }
                haveBeenWritten = writeFormat4VecElem<T, LaneState::ON>(res, haveBeenWritten);
            } else {
                // require 1 additional character for '\0'
                if (haveBeenWritten >= ARG_STR_SIZE - requiredSpace4VecElem<T, LaneState::OFF>() - 1) {
                    // there's not enough space to write another value
                    break;
                }
                haveBeenWritten = writeFormat4VecElem<T, LaneState::OFF>(res, haveBeenWritten);
            }
            res[haveBeenWritten] = lane == width_ - 1 ? ']' : ',';
            ++haveBeenWritten;

            writeArg<T>();
        }
        res[haveBeenWritten] = '\0';
        return res;
    }

  private:
    inline ArgT getElementaryArg() { return args_[curArgIdx_++]; }

    // accesses next scalar argument or element of vector argument from args_ (reads 2 elementary args if 64 bit value)
    // and writes it to cm printf surface
    template <typename T> inline void writeArg() {
        ArgT low, high;
        if constexpr (sizeof(T) > 4) {
            low = getElementaryArg();
            high = getElementaryArg();
        } else {
            low = getElementaryArg();
            high = 0;
        }
        SurfaceIndex BTI = details::__cm_intrinsic_impl_predefined_surface(details::PRINT_SURF_IDX);
        details::_cm_print_args_raw<T>(BTI, offset_, low, high);
        offset_ += CMPHF_VEC_BSZ;
    }

    // cm printf doesn't support %p specifier, so it is emulated with unsigned
    // look at cmType2Specifier for more details
    template <> inline void writeArg<void *>() {
        if constexpr (sizeof(void *) > sizeof(unsigned))
            writeArg<unsigned long long>();
        else
            writeArg<unsigned>();
    }

    // in case of bool we do not push the argument to surface, we directly write true or false into format string
    template <> inline void writeArg<bool>() {}

    // appends format string for vector element to res starting from position haveBeenWritten
    // behaves differently for bools, look at specializations for details
    // NOTE: this is a method because bool specialization require access to getElementaryArg
    template <typename T, LaneState lane>
    inline int writeFormat4VecElem(StaticStringRef<ARG_STR_SIZE> res, int haveBeenWritten) {
        auto fmt = cmType2Specifier<T>();
        if constexpr (!lane)
            haveBeenWritten += CopyFullText("((", res, haveBeenWritten, ARG_STR_SIZE - haveBeenWritten - 1);
        haveBeenWritten += CopyFullText(fmt, res, haveBeenWritten, ARG_STR_SIZE - haveBeenWritten - 1);
        if constexpr (!lane)
            haveBeenWritten += CopyFullText("))", res, haveBeenWritten, ARG_STR_SIZE - haveBeenWritten - 1);
        return haveBeenWritten;
    }

    // in case of bools we directly write "true" or "false" to format string, as cm printf doesn't support %s
    template <>
    inline int writeFormat4VecElem<bool, LaneState::ON>(StaticStringRef<ARG_STR_SIZE> res, int haveBeenWritten) {
        haveBeenWritten += CopyFullText(ValueAdapter<bool>(getElementaryArg()), res, haveBeenWritten,
                                        ARG_STR_SIZE - haveBeenWritten - 1);
        return haveBeenWritten;
    }

    // off lane bool unlike other types doesn't get additional parenthesis (not "((true))"), but becomes "_____"
    template <>
    inline int writeFormat4VecElem<bool, LaneState::OFF>(StaticStringRef<ARG_STR_SIZE> res, int haveBeenWritten) {
        haveBeenWritten += CopyFullText(OffLaneBoolStr, res, haveBeenWritten, ARG_STR_SIZE - haveBeenWritten - 1);
        return haveBeenWritten;
    }
};

/** This function is called by PrintStmt to do the work of printing values
    from ispc programs.  Note that the function signature here must match
    the parameters that PrintStmt::EmitCode() generates.

    @param format       Print format string
    @param types        Encoded types of the values being printed.
                        (See lEncodeType()).
    @param width        Vector width of the compilation target
    @param mask         Current lane mask when the print statement is called
    @param args         Array of split into uints values
                        (if a value bigger than uint it stored in 2 uints: low bits, then high bits,
                         vector values stored by scalar values with the previous rule being applied)
    @param numNotBoolUniArgs   number of uniform values, except bool ones (details are below)
    @param numNotBoolVarArgs   number of varying values, except bool ones (details are below)

    WARNING!!!
    CM supports maximum of 128 characters in format string. If your input doesn't fit, it will be cut.
    NOTE: 128 is the maximum size of printf format string, not ISPC print one. So consider space for brackets, commas,
    printf type specifiers, e.g. % with a vector of int being passed, will require 3*width+1 characters
    (for more details look for explanation below).

    Root: cm printf works in the following way. It takes its arguments (format string and the arguments itself),
    adds some headers and sends it to host through predefined surface. Host gets all this arguments, and calls
    the real printf.

    Idea: GetFormatedStr() transforms ISPC format string to normal printf format string:
        adds qualifiers (% -> %d,% -> %f,...),
        adds brackets for vector values (% -> [%d,((%d)),...,%d]).
    Real writing of an argument to surface happens in uniform2Str, varying2Str methods.

    FIXME: __declspec(genx_SIMT(1)) is used to make __do_print_cm function be called unpredicated in simd cf.
    It is possibly an abuse of this attribute, in this case it's better to solve the issue differently.
 */
extern "C" __declspec(cm_builtin) __declspec(genx_SIMT(1)) void __do_print_cm(const char *format, const char *types,
                                                                              WidthT width, MaskT mask,
                                                                              const ArgT *args, int numNotBoolUniArgs,
                                                                              int numNotBoolVarArgs) {
    // boolean args are not included in total_len as we won't write them into the print surface
    // they will be written directly to format string
    // FIXME: if we won't write all arguments (which can happen if we don't fit in 128 characters), behavior
    // on runtime side can be undefined. We should either fix it here (will make this code even more heavier),
    // or tighten up contract of print (require from user to not overflow 128 characters).
    details::OffsetT total_len =
        CMPHF_VEC_BSZ + CMPHF_STR_SZ + (numNotBoolUniArgs + numNotBoolVarArgs * width) * CMPHF_VEC_BSZ;
    SurfaceIndex BTI = details::__cm_intrinsic_impl_predefined_surface(details::PRINT_SURF_IDX);
    auto init_offset = details::_cm_print_init_offset(BTI, total_len);
    auto arg_offset = init_offset + CMPHF_STR_SZ + CMPHF_VEC_BSZ;

    ArgWriter argWriter(arg_offset, args, width, mask);
    StaticString<RES_STR_SIZE> resultingStr = GetFormatedStr(format, types, argWriter);
    details::_cm_print_format(BTI, init_offset, resultingStr);
}

