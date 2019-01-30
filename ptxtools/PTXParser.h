// -*- mode: c++ -*-
/*
   Copyright (c) 2014, Evghenii Gaburov
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
/*
   Based on GPU Ocelot PTX parser : https://code.google.com/p/gpuocelot/
   */

#pragma once

#undef yyFlexLexer
#define yyFlexLexer ptxFlexLexer
#include <FlexLexer.h>

#include "PTXLexer.h"

#include <sstream>
#include <string>
#include <vector>
namespace ptx {
extern int yyparse(parser::PTXLexer &, parser::PTXParser &);
}

namespace parser {
/*! \brief An implementation of the Parser interface for PTX */
class PTXParser {
  private:
    typedef int token_t;
    std::ostream &out;
    std::string _identifier;
    token_t _dataTypeId;
    int _alignment;

    bool isArgumentList, isReturnArgumentList;
    struct argument_t {
        token_t type;
        std::string name;
        int dim;

        argument_t(const token_t _type, const std::string &_name, const int _dim = 1)
            : type(_type), name(_name), dim(_dim) {}
    };
    std::vector<argument_t> argumentList, returnArgumentList;
    std::vector<int> arrayDimensionsList;

  public:
    PTXParser(std::ostream &_out) : out(_out) {
        isArgumentList = isReturnArgumentList = false;
        _alignment = 1;
    }

    void printHeader() {
        std::stringstream s;
#if 0
      s << "template<int N> struct __align__(N)   b8_t  { unsigned char  _v[N]; __device__ b8_t()  {}; __device__ b8_t (const int value) {}}; \n";
      s << "template<int N> struct __align__(2*N) b16_t { unsigned short _v[N]; __device__ b16_t() {}; __device__ b16_t(const int value) {}}; \n";
#else
        s << "template<int N> struct b8_t  { unsigned char  _v[N]; __device__ b8_t()  {}; __device__ b8_t (const int "
             "value) {}}; \n";
        s << "template<int N> struct b16_t { unsigned short _v[N]; __device__ b16_t() {}; __device__ b16_t(const int "
             "value) {}}; \n";
#endif
        s << "struct b8d_t  { unsigned char  _v[1]; }; \n";
        s << "struct b16d_t { unsigned short _v[1]; }; \n";

        s << "typedef unsigned int       b32_t; \n";
        s << "typedef unsigned int       u32_t; \n";
        s << "typedef int                s32_t; \n";

        s << "typedef unsigned long long b64_t; \n";
        s << "typedef unsigned long long u64_t; \n";
        s << "typedef long long          s64_t; \n";

        s << "typedef float              f32_t; \n";
        s << "typedef double             f64_t; \n";
        s << " \n";
        out << s.str();
    }

#define LOC YYLTYPE &location

    void identifier(const std::string &s) { _identifier = s; }
    void dataTypeId(const token_t token) { _dataTypeId = token; }
    void argumentListBegin(LOC) { isArgumentList = true; }
    void argumentListEnd(LOC) { isArgumentList = false; }
    void returnArgumentListBegin(LOC) { isReturnArgumentList = true; }
    void returnArgumentListEnd(LOC) { isReturnArgumentList = false; }
    void argumentDeclaration(LOC) {
        assert(arrayDimensionsList.size() <= 1);
        const int dim = arrayDimensionsList.empty() ? 1 : arrayDimensionsList[0];
        const argument_t arg(_dataTypeId, _identifier, dim);
        if (isArgumentList)
            argumentList.push_back(arg);
        else if (isReturnArgumentList)
            returnArgumentList.push_back(arg);
        else
            assert(0);
        arrayDimensionsList.clear();
    }
    void alignment(const int value) { _alignment = value; }

    void arrayDimensions(const int value) { arrayDimensionsList.push_back(value); }

    std::string printArgument(const argument_t arg, const bool printDataType = true) {
        std::stringstream s;
        if (printDataType)
            s << tokenToDataType(arg.type, arg.dim) << " ";
        s << arg.name << " ";
        return s.str();
    }

    std::string printArgumentList(const bool printDataType = true) {
        std::stringstream s;
        if (argumentList.empty())
            return s.str();
        const int n = argumentList.size();
        s << " " << printArgument(argumentList[0], printDataType);
        for (int i = 1; i < n; i++)
            s << ",\n " << printArgument(argumentList[i], printDataType);
        return s.str();
    }

    void visibleEntryDeclaration(const std::string &calleeName, LOC) {
        std::stringstream s;
        assert(returnArgumentList.empty());
        s << "extern \"C\" \n";
        s << "__global__ void " << calleeName << " (\n";
        s << printArgumentList();
        s << "\n ) { asm(\" // entry \"); }\n";

        /* check if this is an "export"  entry */
        const int entryNameLength = calleeName.length();
        const int hostNameLength = std::max(entryNameLength - 9, 0);
        const std::string ___export(&calleeName.c_str()[hostNameLength]);
        if (___export.compare("___export") == 0) {
            std::string hostCalleeName;
            hostCalleeName.append(calleeName.c_str(), hostNameLength);
            s << "/*** host interface ***/\n";
            s << "extern \"C\" \n";
            s << "__host__ void " << hostCalleeName << " (\n";
            s << printArgumentList();
            s << "\n )\n";
            s << "{\n   ";
            //        s << " cudaFuncSetCacheConfig (" << calleeName << ", ";
            s << " cudaDeviceSetCacheConfig (";
#if 1
            s << " cudaFuncCachePreferEqual ";
#elif 1
            s << " cudaFuncCachePreferL1 ";
#else
            s << " cudaFuncCachePreferShared ";
#endif
            s << ");\n";
            s << calleeName;
            s << "<<<1,32>>>(\n";
            s << printArgumentList(false);
            s << ");\n";
            s << " cudaDeviceSynchronize(); \n";
            s << "}\n";
        }
        s << "\n";
        argumentList.clear();

        out << s.str();
    }

    void visibleFunctionDeclaration(const std::string &calleeName, LOC) {
        std::stringstream s;
        assert(returnArgumentList.size() < 2);
        s << "extern \"C\" \n";
        s << "__device__ ";
        if (returnArgumentList.empty())
            s << " void ";
        else
            s << " " << tokenToDataType(returnArgumentList[0].type, returnArgumentList[0].dim);
        s << calleeName << " (\n";
        s << printArgumentList();

        if (returnArgumentList.empty())
            s << "\n ) { asm(\" // function \"); }\n\n";
        else {
            s << "\n ) { asm(\" // function \"); return 0;} /* return value to disable warnings */\n\n";
            //        s << "\n ) { asm(\" // function \"); } /* this will generate warrning */\n\n";
        }

        argumentList.clear();
        returnArgumentList.clear();

        out << s.str();
    }

    void visibleInitializableDeclaration(const std::string &name, LOC) {
        assert(arrayDimensionsList.size() == 1);
        std::stringstream s;
        s << "extern \"C\" __device__ ";
        if (_alignment > 0)
            s << "__attribute__((aligned(" << _alignment << "))) ";
        s << tokenToDataType(_dataTypeId, 0);
        if (arrayDimensionsList[0] == 0)
            s << name << ";\n\n";
        else
            s << name << "[" << arrayDimensionsList[0] << "] = {0};\n\n";
        out << s.str();
        arrayDimensionsList.clear();
    }

#undef LOC

    std::string tokenToDataType(token_t token, int dim) {
        std::stringstream s;
        switch (token) {
        case TOKEN_B8:
            if (dim > 0)
                s << "b8_t<" << dim << "> ";
            else
                s << "b8d_t ";
            break;
        case TOKEN_U8:
            assert(0);
            s << "u8_t ";
            break;
        case TOKEN_S8:
            assert(0);
            s << "s8_t ";
            break;

        case TOKEN_B16:
            if (dim > 0)
                s << "b16_t<" << dim << "> ";
            else
                s << "b16d_t ";
            break;
        case TOKEN_U16:
            assert(0);
            s << "u16_t ";
            break;
        case TOKEN_S16:
            assert(0);
            s << "s16_t ";
            break;

        case TOKEN_B32:
            assert(dim <= 1);
            s << "b32_t ";
            break;
        case TOKEN_U32:
            assert(dim <= 1);
            s << "u32_t ";
            break;
        case TOKEN_S32:
            assert(dim <= 1);
            s << "s32_t ";
            break;

        case TOKEN_B64:
            assert(dim <= 1);
            s << "b64_t ";
            break;
        case TOKEN_U64:
            assert(dim <= 1);
            s << "u64_t ";
            break;
        case TOKEN_S64:
            assert(dim <= 1);
            s << "s64_t ";
            break;

        case TOKEN_F32:
            assert(dim <= 1);
            s << "f32_t ";
            break;
        case TOKEN_F64:
            assert(dim <= 1);
            s << "f64_t ";
            break;
        default:
            std::cerr << "token= " << token << std::endl;
            assert(0);
        }

        return s.str();
    }
};
} // namespace parser
