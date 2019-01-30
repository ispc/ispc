// -*- mode: c++ -*-
/*
   Copyright (c) 2014-2015, Evghenii Gaburov
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
   Based on "ptxgen" NVVM example from CUDA Toolkit
   */
#include "GPUTargets.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nvvm.h>
#include <sys/stat.h>

template <typename T> static std::string lValueToString(const T &value) { return std::to_string(value); }

struct Exception : public std::exception {
    std::string s;
    Exception(std::string ss) : s(ss) {}
    ~Exception() throw() {} // Updated
    const char *what() const throw() { return s.c_str(); }
};

struct NVVMProg {
    nvvmProgram prog;
    NVVMProg() {
        if (nvvmCreateProgram(&prog) != NVVM_SUCCESS)
            throw Exception(std::string("Failed to create the compilation unit."));
    }
    ~NVVMProg() { nvvmDestroyProgram(&prog); }
    nvvmProgram get() const { return prog; }
};

static std::string getLibDeviceName(int computeArch) {
    const char *env = getenv("LIBNVVM_HOME");
#ifdef LIBNVVM_HOME
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
    const std::string libnvvmPath1(env ? env : TOSTRING(LIBNVVM_HOME));
#undef TOSTRING
#undef STRINGIFY
#else
    const std::string libnvvmPath1(env);
#endif

    const std::string libnvvmPath(env == nullptr ? libnvvmPath1 : std::string(env));

    if (libnvvmPath.empty())
        throw Exception("The environment variable LIBNVVM_HOME is undefined");

    /* Use libdevice for compute_20, if the target is not compute_20, compute_30,
     * or compute_35. */
    if (computeArch == 37)
        computeArch = 35;
    const std::string libdevice = std::string("/libdevice/libdevice.compute_") + lValueToString(computeArch) + "." +
                                  lValueToString(LIBDEVICE_MAJOR_VERSION) + lValueToString(LIBDEVICE_MINOR_VERSION) +
                                  ".bc";

    return libnvvmPath + libdevice;
}

static void addFileToProgram(const std::string &filename, NVVMProg &prog) {

    /* Open the input file. */
    FILE *f = fopen(filename.c_str(), "rb");
    if (f == NULL)
        throw Exception(std::string("Failed to open ") + filename);

    /* Allocate buffer for the input. */
    struct stat fileStat;
    fstat(fileno(f), &fileStat);
    std::string buffer(fileStat.st_size, 0);

    /* Read input file */
    const size_t size = fread(&buffer[0], 1, fileStat.st_size, f);
    const auto error = ferror(f);
    fclose(f);
    if (error)
        throw Exception(std::string("Failed to read ") + filename + ".");

    /* Add IR block to a program */
    if (nvvmAddModuleToProgram(prog.get(), buffer.c_str(), size, filename.c_str()) != NVVM_SUCCESS)
        throw Exception(std::string("Failed to add the module ") + filename + " to the compilation unit.");
}

static void printWarningsAndErrors(NVVMProg &prog) {
    size_t logSize;
    if (nvvmGetProgramLogSize(prog.get(), &logSize) == NVVM_SUCCESS) {
        std::string log(logSize, 0);
        if (nvvmGetProgramLog(prog.get(), &log[0]) == NVVM_SUCCESS && logSize > 1) {
            std::cerr << "--------------------------------------\n";
            std::cerr << log << std::endl;
            std::cerr << "--------------------------------------\n";
        } else
            throw Exception("Failed to get the compilation log.");
    } else
        throw Exception("Failed to get the compilation log size.");
}

static std::string generatePTX(const std::vector<std::string> &nvvmOptions, const std::vector<std::string> &nvvmFiles,
                               const int computeArch) {
    std::string ptxString;

    /* Create the compiliation unit. */
    NVVMProg prog;

    /* Add libdevice. */
    try {
        const std::string &libDeviceName = getLibDeviceName(computeArch);
        addFileToProgram(libDeviceName, prog);
    } catch (const std::exception &ex) {
        throw Exception(ex.what());
    }

    std::vector<const char *> options;
    for (const auto &f : nvvmFiles)
        addFileToProgram(f, prog);

    for (const auto &o : nvvmOptions)
        options.push_back(o.c_str());

    try {
        if (nvvmVerifyProgram(prog.get(), options.size(), &options[0]) != NVVM_SUCCESS)
            throw Exception("Failed to verify the compilation unit.");

        /* Compile the compilation unit. */
        if (nvvmCompileProgram(prog.get(), options.size(), &options[0]) == NVVM_SUCCESS) {
            size_t ptxSize;
            if (nvvmGetCompiledResultSize(prog.get(), &ptxSize) == NVVM_SUCCESS) {
                ptxString.resize(ptxSize);
                if (nvvmGetCompiledResult(prog.get(), &ptxString[0]) != NVVM_SUCCESS)
                    throw Exception("Failed to get the PTX output.");
            } else
                throw Exception("Failed to get the PTX output size.");
        } else
            throw Exception("Failed to generate PTX from the compilation unit.");
    } catch (const std::exception &ex) {
        std::cerr << "NVVM exception: " << ex.what() << std::endl;
        printWarningsAndErrors(prog);
        throw Exception("");
    }

    return ptxString;
};

static void showUsage() {
    fprintf(stderr, "Usage: ptxgen [OPTION]... [FILE]...\n"
                    "  [FILE] could be a .bc file or a .ll file\n");
}

static void lUsage(const int ret) {
    fprintf(stdout, "\nusage: ptxgen [options] file.[ll,bc] \n");
    fprintf(stdout, "    [--help]\t\t This help\n");
    fprintf(stdout, "    [--verbose]\t\t Be verbose\n");
    fprintf(stdout, "    [--arch=]\t\t GPU target architectures:\n");
    fprintf(stdout, "     \t\t\t   ");
    for (const auto &mode : GPUTargets::computeMode)
        fprintf(stdout, "%s ", mode);
    fprintf(stdout, "\n");
    fprintf(stdout, "    [-o <name>]\t\t Output file name\n");
    fprintf(stdout, "    [-g]\t\t Enable generation of debuggin information \n");
    fprintf(stdout, "    [--opt=]\t\t Optimization parameters \n");
    fprintf(stdout, "     \t\t\t    0 - disable optimizations \n");
    fprintf(stdout, "     \t\t\t    3 - defalt, enable optimizations \n");
    fprintf(stdout,
            "    [--ftz=]\t\t Flush-to-zero mode when performsing single-precision floating-point operations\n");
    fprintf(stdout, "     \t\t\t    0 - default, preserve denormal values\n");
    fprintf(stdout, "     \t\t\t    1 - flush denormal values to zero\n");
    fprintf(stdout, "    [--prec-sqrt=]\t Precision mode for single-precision floating-point square root\n");
    fprintf(stdout, "     \t\t\t    0 - use a faster approximation\n");
    fprintf(stdout, "     \t\t\t    1 - default, use IEEE round-to-nearest mode\n");
    fprintf(stdout,
            "    [--prec-div=]\t Precision mode for single-precision floating-point division and reciprocals\n");
    fprintf(stdout, "     \t\t\t    0 - use a faster approximation\n");
    fprintf(stdout, "     \t\t\t    1 - default, use IEEE round-to-nearest mode\n");
    fprintf(stdout, "    [--fma=]\t\t FMA contraction mode \n");
    fprintf(stdout, "     \t\t\t    0 - disable\n");
    fprintf(stdout, "     \t\t\t    1 - default, enable\n");
    fprintf(stdout, "    [--use_fast_math]\t Make use of fast maih. Implies --ftz=1 --prec-div=0 --prec-sqrt=0\n");
    fprintf(stdout, " \n");
    exit(ret);
}

int main(int argc, char *argv[]) {
    int _opt = 3;
    int _ftz = 0;
    int _precSqrt = 1;
    int _precDiv = 1;
    int _fma = 1;
    bool _useFastMath = false;
    bool _debug = false;
    bool _verbose = false;
    std::string _arch = *GPUTargets::computeMode.begin();
    std::string fileIR, filePTX;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help"))
            lUsage(0);
        else if (!strncmp(argv[i], "--arch=", 7))
            _arch = std::string(argv[i] + 7);
        else if (!strncmp(argv[i], "-g", 2))
            _debug = true;
        else if (!strncmp(argv[i], "--verbose", 9))
            _verbose = true;
        else if (!strncmp(argv[i], "--opt=", 6))
            _opt = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "--ftz=", 6))
            _ftz = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "--prec-sqrt=", 12))
            _precSqrt = atoi(argv[i] + 12);
        else if (!strncmp(argv[i], "--prec-div=", 11))
            _precDiv = atoi(argv[i] + 11);
        else if (!strncmp(argv[i], "--fma=", 6))
            _fma = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "--use_fast_math", 15))
            _useFastMath = true;
        else if (!strcmp(argv[i], "-o")) {
            if (++i == argc) {
                fprintf(stderr, "No output file specified after -o option.\n");
                lUsage(1);
            }
            filePTX = std::string(argv[i]);
        } else {
            const char *ext = strrchr(argv[i], '.');
            if (ext == NULL) {
                fprintf(stderr, " Unknown argument: %s \n", argv[i]);
                lUsage(1);
            } else if (strncmp(ext, ".ll", 3) && strncmp(ext, ".bc", 3)) {
                fprintf(stderr, " Unkown extension of the input file: %s \n", ext);
                lUsage(1);
            } else if (filePTX.empty()) {
                fileIR = std::string(argv[i]);
                if (filePTX.empty()) {
                    char *baseName = argv[i];
                    while (baseName != ext)
                        filePTX += std::string(baseName++, 1);
                }
                filePTX += ".ptx";
            }
        }
    }

    if (fileIR.empty()) {
        fprintf(stderr, "ptxgen fatal : No input file specified; use option --help for more information\n");
        exit(1);
    }

#if 0
  fprintf(stderr, "fileIR= %s\n", fileIR.c_str());
  fprintf(stderr, "filePTX= %s\n", filePTX.c_str());
  fprintf(stderr, "arch= %s\n", _arch.c_str());
  fprintf(stderr, "debug= %s\n", _debug ? "true" : "false");
  fprintf(stderr, "verbose= %s\n", _verbose ? "true" : "false");
  fprintf(stderr, "opt= %d\n", _opt);
  fprintf(stderr, "ftz= %d\n", _ftz);
  fprintf(stderr, "prec-sqrt= %d\n", _precSqrt);
  fprintf(stderr, "prec-div= %d\n", _precDiv);
  fprintf(stderr, "fma= %d\n", _fma);
  fprintf(stderr, "use_fast_math= %s\n", _useFastMath ? "true" : "false");
#endif

    if (std::find(GPUTargets::computeMode.begin(), GPUTargets::computeMode.end(), _arch) ==
        GPUTargets::computeMode.end()) {
        fprintf(stderr, "ptxcc fatal : --arch=%s is not supported; use option --help for more information\n",
                _arch.c_str());
        exit(1);
    }

    if (_useFastMath) {
        _ftz = 1;
        _precSqrt = _precDiv = 0;
    }

    /* replace "sm" with "compute" */
    assert(_arch[0] == 's' && _arch[1] == 'm' && _arch[2] == '_');
    const std::string _mode = std::string("compute_") + &_arch[3];
    const int computeArch = atoi(&_arch[3]);

    std::vector<std::string> nvvmOptions;
    nvvmOptions.push_back("-arch=" + _mode);
    nvvmOptions.push_back("-ftz=" + lValueToString(_ftz));
    nvvmOptions.push_back("-prec-sqrt=" + lValueToString(_precSqrt));
    nvvmOptions.push_back("-prec-div=" + lValueToString(_precDiv));
    nvvmOptions.push_back("-fma=" + lValueToString(_fma));
    if (_debug)
        nvvmOptions.push_back("-g");

    std::vector<std::string> nvvmFiles;
    nvvmFiles.push_back(fileIR);

    try {
        std::ofstream outputPTX(filePTX.c_str());
        outputPTX << generatePTX(nvvmOptions, nvvmFiles, computeArch);
    } catch (const std::exception &ex) {
        std::cerr << "Error: ptxgen failed with exception \n   " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
