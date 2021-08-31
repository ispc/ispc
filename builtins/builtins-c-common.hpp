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

/** @file builtins-c-common.cpp
    @brief Basic methods for print implementations. Currenly is used by __do_print

    This library provides instruments for parsing ISPC print format strings,
    some useful adapters, and other minor helper instruments.

    FIXME: this file can be included only after
    StaticString, StaticStringRef, SizeT, RES_STR_SIZE, ARG_STR_SIZE
    are defined.
    Can be fixed with an additional template argument, but only when CMC includes are fixed.
*/

#include "src/builtins-info.h"

namespace details {
template <typename T> inline T ValueAdapterImpl(T val) { return val; }

static inline const char *ValueAdapterImpl(bool val) { return val ? "true" : "false"; }

// Copies everything from src (starting from srcIdx) to dst (starting from dstIdx), until it meets a separator.
// Separators are defined by SEPS.
// T is anything indexable, meaning you can write src[idx]
template <char... SEPS, typename T, SizeT SIZE>
inline int CopyTillSep(T src, int srcIdx, StaticStringRef<SIZE> dst, int dstIdx, int leftSpace) {
    const auto origDstIdx = dstIdx;
    while (((src[srcIdx] != SEPS) && ...) && leftSpace) {
        dst[dstIdx++] = src[srcIdx++];
        --leftSpace;
    }
    return dstIdx - origDstIdx;
}

// FIXME: ugly copy-paste to work around cmc frontend issues. Remove it when CMC includes are fixed.
template <char... SEPS, SizeT SRC_SIZE, SizeT SIZE>
inline int CopyTillSep(StaticStringRef<SRC_SIZE> src, int srcIdx, StaticStringRef<SIZE> dst, int dstIdx,
                       int leftSpace) {
    const auto origDstIdx = dstIdx;
    while (((src[srcIdx] != SEPS) && ...) && leftSpace) {
        dst[dstIdx++] = src[srcIdx++];
        --leftSpace;
    }
    return dstIdx - origDstIdx;
}

// If type encodes uniform T, function puts result of uniform2Str<T>() in res and returns true,
// otherwise it does nothing with res and returns false.
template <typename T, typename ArgWriter>
inline bool UniArg2StrIfSuitable(char type, ArgWriter &argWriter, StaticStringRef<ARG_STR_SIZE> res) {
    if (type == PrintInfo::getEncoding4Uniform<T>()) {
        res = argWriter.template uniform2Str<T>();
        return true;
    }
    return false;
}

// If type encodes varying T, function puts result of varying2Str<T>() in res and returns true,
// otherwise it does nothing with res and returns false.
template <typename T, typename ArgWriter>
inline bool VarArg2StrIfSuitable(char type, ArgWriter &argWriter, StaticStringRef<ARG_STR_SIZE> res) {
    if (type == PrintInfo::getEncoding4Varying<T>()) {
        res = argWriter.template varying2Str<T>();
        return true;
    }
    return false;
}

// If type encodes uniform T or varying T, function calls corresponding *2Str method, writes its result into res,
// returns true, otherwise function does nothing with res and returns false.
template <typename T, typename ArgWriter>
inline bool Arg2StrIfSuitable(char type, ArgWriter &argWriter, StaticStringRef<ARG_STR_SIZE> res) {
    return UniArg2StrIfSuitable<T>(type, argWriter, res) || VarArg2StrIfSuitable<T>(type, argWriter, res);
}

// Requests the next argument from argWriter. Relies on uniform2Str<T> and varying2Str<T> methods of argWriter.
// T is deduced from provided type argument. Returns the string representation of the argument returned from
// uniform2Str<T> or varying2Str<T>.
template <typename ArgWriter> inline StaticString<ARG_STR_SIZE> Arg2Str(char type, ArgWriter &argWriter) {
    StaticString<ARG_STR_SIZE> res;
    Arg2StrIfSuitable<bool>(type, argWriter, res) || Arg2StrIfSuitable<int>(type, argWriter, res) ||
        Arg2StrIfSuitable<unsigned>(type, argWriter, res) || Arg2StrIfSuitable<float>(type, argWriter, res) ||
        Arg2StrIfSuitable<long long>(type, argWriter, res) ||
        Arg2StrIfSuitable<unsigned long long>(type, argWriter, res) ||
        Arg2StrIfSuitable<double>(type, argWriter, res) || Arg2StrIfSuitable<void *>(type, argWriter, res);
    return res;
}
} // namespace details

const char UnsupportedTypeStr[] = "UNSUPPORTED TYPE";
const char OffLaneBoolStr[] = "_________";

// For every T except bool it just returns the same value.
// In this case value must have T type.
// For T==bool it returns const char* to "true" or "false".
//
// Function is presumed to be used only for integral types.
template <typename T> inline auto ValueAdapter(T val) { return details::ValueAdapterImpl(val); }

// Copies everything from src to dst (starting from dstIdx), until it meets '%' or '\0'
// Doesn't copy '%' or '\0'. Doesn't null terminate dst string.
template <SizeT SIZE>
inline int CopyPlainText(const char *const src, StaticStringRef<SIZE> dst, const int dstIdx, const int leftSpace) {
    return details::CopyTillSep<'%', '\0'>(src, 0, dst, dstIdx, leftSpace);
}

// Copies everything from src to dst (starting from dstIdx), until it meets '\0' (whole string).
// Doesn't copy '\0', so dst string is not null terminated.
template <SizeT SIZE>
inline int CopyFullText(const char *const src, StaticStringRef<SIZE> dst, const int dstIdx, const int leftSpace) {
    return details::CopyTillSep<'\0'>(src, 0, dst, dstIdx, leftSpace);
}

// Copies everything from src (starting from srcIdx) to dst (starting from dstIdx), until it meets '\0' (whole string).
// Doesn't copy '\0', so dst string is not null terminated.
template <SizeT SRC_SIZE, SizeT DST_SIZE>
inline int CopyFullText(StaticStringRef<SRC_SIZE> src, const int srcIdx, StaticStringRef<DST_SIZE> dst,
                        const int dstIdx, const int leftSpace) {
    return details::CopyTillSep<'\0'>(src, srcIdx, dst, dstIdx, leftSpace);
}

// This function parses format string and string of arg types.
// User must implement ArgWriter class, which has such template methods as uniform2Str<type> and varying2Str<type>.
// Those functions must return text representation (c-string) of the next argument, the type of argument is provided
// with the template parameter. Returned c-string is placed instead of '%' in format string.
// The function returns resulting string.
template <typename ArgWriter>
inline StaticString<RES_STR_SIZE> GetFormatedStr(const char *format, const char *types, ArgWriter &argWriter) {
    StaticString<RES_STR_SIZE> resultingStr;
    int haveBeenWritten = 0;
    int leftSpace = RES_STR_SIZE - 1; // space for null terminator

    for (;; ++format, ++types) {
        auto wereCopied = CopyPlainText(format, resultingStr, haveBeenWritten, leftSpace);
        format += wereCopied;
        haveBeenWritten += wereCopied;
        leftSpace -= wereCopied;
        if (!leftSpace || *format == '\0') {
            // reached the end of format or limits of resultingStr
            break;
        }
        StaticString<ARG_STR_SIZE> argStr = details::Arg2Str(*types, argWriter);
        wereCopied = CopyFullText(argStr, 0, resultingStr, haveBeenWritten, leftSpace);
        haveBeenWritten += wereCopied;
        leftSpace -= wereCopied;
    }
    resultingStr[haveBeenWritten] = '\0';
    return resultingStr;
}
