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

/** @file builtins-info.h
    @brief This file contains information shared between code, that uses
           builtins and the code that implements them.
*/

#ifndef ISPC_BUILTINS_INFO_H
#define ISPC_BUILTINS_INFO_H 1

namespace PrintInfo {

// get encoding (as a char) for uniform T type, that is used in type argument of __do_print(_cm)
template <typename T> char getEncoding4Uniform();

// get encoding (as a char) for uniform T type, that is used in type argument of __do_print(_cm)
template <typename T> char getEncoding4Varying();

template <> inline constexpr char getEncoding4Uniform<bool>() { return 'b'; }
template <> inline constexpr char getEncoding4Uniform<int>() { return 'i'; }
template <> inline constexpr char getEncoding4Uniform<unsigned>() { return 'u'; }
template <> inline constexpr char getEncoding4Uniform<float>() { return 'f'; }
template <> inline constexpr char getEncoding4Uniform<long long>() { return 'l'; }
template <> inline constexpr char getEncoding4Uniform<unsigned long long>() { return 'v'; }
template <> inline constexpr char getEncoding4Uniform<double>() { return 'd'; }
template <> inline constexpr char getEncoding4Uniform<void *>() { return 'p'; }

template <> inline constexpr char getEncoding4Varying<bool>() { return 'B'; }
template <> inline constexpr char getEncoding4Varying<int>() { return 'I'; }
template <> inline constexpr char getEncoding4Varying<unsigned>() { return 'U'; }
template <> inline constexpr char getEncoding4Varying<float>() { return 'F'; }
template <> inline constexpr char getEncoding4Varying<long long>() { return 'L'; }
template <> inline constexpr char getEncoding4Varying<unsigned long long>() { return 'V'; }
template <> inline constexpr char getEncoding4Varying<double>() { return 'D'; }
template <> inline constexpr char getEncoding4Varying<void *>() { return 'P'; }
} // namespace PrintInfo

#endif // ISPC_BUILTINS_INFO_H
