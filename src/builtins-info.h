/*
  Copyright (c) 2019-2021, Intel Corporation
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

/** @file builtins-info.h
    @brief This file contains information and utils shared between code,
           that uses builtins and the code that implements them.
*/

#ifndef ISPC_BUILTINS_INFO_H
#define ISPC_BUILTINS_INFO_H 1

namespace PrintInfo {

/* '\0' is excluded as encodings must form a c-string.
   L0 depends on Encoding being contiguous numbers.
 */
enum Encoding : char {
    Bool = 1,
    Int,
    UInt,
    Float,
    Long,
    ULong,
    Double,
    Ptr,
    VecBool,
    VecInt,
    VecUInt,
    VecFloat,
    VecLong,
    VecULong,
    VecDouble,
    VecPtr,
    Size = VecPtr - Bool + 1
};

// Currently the max format string length supported by L0 runtime.
// FIXME: get rid of this constant and corresponeding splitting code when
//        L0 supports arbitrary length format string.
constexpr int LZMaxFormatStrSize = 16 * 1024;

// get encoding (as a char) for uniform T type, that is used in type argument of __do_print
template <typename T> Encoding getEncoding4Uniform();

// get encoding (as a char) for uniform T type, that is used in type argument of __do_print
template <typename T> Encoding getEncoding4Varying();

template <> inline constexpr Encoding getEncoding4Uniform<bool>() { return Encoding::Bool; }
template <> inline constexpr Encoding getEncoding4Uniform<int>() { return Encoding::Int; }
template <> inline constexpr Encoding getEncoding4Uniform<unsigned>() { return Encoding::UInt; }
template <> inline constexpr Encoding getEncoding4Uniform<float>() { return Encoding::Float; }
template <> inline constexpr Encoding getEncoding4Uniform<long long>() { return Encoding::Long; }
template <> inline constexpr Encoding getEncoding4Uniform<unsigned long long>() { return Encoding::ULong; }
template <> inline constexpr Encoding getEncoding4Uniform<double>() { return Encoding::Double; }
template <> inline constexpr Encoding getEncoding4Uniform<void *>() { return Encoding::Ptr; }

template <> inline constexpr Encoding getEncoding4Varying<bool>() { return Encoding::VecBool; }
template <> inline constexpr Encoding getEncoding4Varying<int>() { return Encoding::VecInt; }
template <> inline constexpr Encoding getEncoding4Varying<unsigned>() { return Encoding::VecUInt; }
template <> inline constexpr Encoding getEncoding4Varying<float>() { return Encoding::VecFloat; }
template <> inline constexpr Encoding getEncoding4Varying<long long>() { return Encoding::VecLong; }
template <> inline constexpr Encoding getEncoding4Varying<unsigned long long>() { return Encoding::VecULong; }
template <> inline constexpr Encoding getEncoding4Varying<double>() { return Encoding::VecDouble; }
template <> inline constexpr Encoding getEncoding4Varying<void *>() { return Encoding::VecPtr; }

/* Takes encoding for varying type, returns encoding for corresponding uniform type.
   For example: Encoding::VecLong -> Encoding::Long
               (varying long long -> uniform long long)
 */
inline Encoding getCorrespondingEncoding4Uniform(Encoding type) {
    return static_cast<Encoding>(type - (Encoding::VecBool - Encoding::Bool));
}

// same as getEncoding4Uniform, but it is a functor
struct Encoding4Uniform {
    template <typename T> constexpr Encoding call() const { return getEncoding4Uniform<T>(); }
};

// same as getEncoding4Varying, but it is a functor
struct Encoding4Varying {
    template <typename T> constexpr Encoding call() const { return getEncoding4Varying<T>(); }
};

// Converts type to corresponding printf type specifier
template <typename T> inline const char *type2Specifier();

template <> inline const char *type2Specifier<bool>() {
    // %s is because we will eventually print "true" or "false"
    return "%s";
}

template <> inline const char *type2Specifier<int>() { return "%d"; }
template <> inline const char *type2Specifier<unsigned>() { return "%u"; }
template <> inline const char *type2Specifier<float>() { return "%f"; }
template <> inline const char *type2Specifier<long long>() { return "%lld"; }
template <> inline const char *type2Specifier<long long unsigned>() { return "%llu"; }
template <> inline const char *type2Specifier<double>() { return "%f"; }
template <> inline const char *type2Specifier<void *>() { return "%p"; }

// same as type2Specifier, but now it is a functor
struct Type2Specifier {
    template <typename T> const char *call() const { return type2Specifier<T>(); }
};

namespace detail {

template <typename T, typename Func, typename Encoder>
inline bool applyIfProperEncoding(Encoding type, Func f, Encoder encoder) {
    if (type == encoder.template call<T>()) {
        f.template call<T>();
        return true;
    }
    return false;
}

template <typename Func, typename Encoder> inline bool switchEncoding(Encoding type, Func f, Encoder encoder) {
    return applyIfProperEncoding<bool>(type, f, encoder) || applyIfProperEncoding<int>(type, f, encoder) ||
           applyIfProperEncoding<unsigned>(type, f, encoder) || applyIfProperEncoding<float>(type, f, encoder) ||
           applyIfProperEncoding<long long>(type, f, encoder) ||
           applyIfProperEncoding<unsigned long long>(type, f, encoder) ||
           applyIfProperEncoding<double>(type, f, encoder) || applyIfProperEncoding<void *>(type, f, encoder);
}

} // namespace detail

// Functions to avoid writing "switch(type) case getEncoding..." by hand.
// When there's a match, f.call<T>() with corresponding T is called
// and true is retruned, otherwise false is returned.
// 4Uniform means that only encodings for uniform types are checked,
// similary 4Varying is for varying types.
template <typename Func> inline bool switchEncoding4Uniform(Encoding type, Func f) {
    return detail::switchEncoding(type, f, Encoding4Uniform{});
}

template <typename Func> inline bool switchEncoding4Varying(Encoding type, Func f) {
    return detail::switchEncoding(type, f, Encoding4Varying{});
}

template <typename Func4Uniform, typename Func4Varying>
inline bool switchEncoding(Encoding type, Func4Uniform fUni, Func4Varying fVar) {
    return switchEncoding4Uniform(type, fUni) || switchEncoding4Varying(type, fVar);
}

} // namespace PrintInfo

#endif // ISPC_BUILTINS_INFO_H
