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
   Based on GPU Ocelot PTX parser : https://code.google.com/p/gpuocelot/
   */

#include "GPUTargets.h"
#include "PTXParser.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>

/*
 * The C++ code below is based on the following bash-script:
      #!/bin/sh

      PTXSRC=$1__tmp_ptx.ptx
      PTXCU=$1___tmp_ptx.cu
      PTXSH=$1___tmp_ptx.sh

      NVCCPARM=${@:2}

      DEPTX=dePTX
      NVCC=nvcc

      $(cat $1 | sed 's/\.b0/\.b32/g' > $PTXSRC) &&
      $DEPTX < $PTXSRC > $PTXCU &&
      $NVCC -arch=sm_35 -dc $NVCCPARM -dryrun $PTXCU 2>&1 | \
        sed 's/\#\$//g'| \
        awk '{ if ($1 == "LIBRARIES=") print $1$2; else if ($1 == "cicc") print "cp '$PTXSRC'", $NF; else print $0 }' >
 $PTXSH && sh $PTXSH

      # rm $PTXCU $PTXSH
 *
 */

static char lRandomAlNum() {
    const char charset[] = "0123456789"
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
}

static std::string lRandomString(const size_t length) {
    timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);
    std::string str(length, 0);
    std::generate_n(str.begin(), length, lRandomAlNum);
    return str;
}

static void lGetAllArgs(int Argc, char *Argv[], int &argc, char *argv[128]) {
    // Copy over the command line arguments (passed in)
    for (int i = 0; i < Argc; ++i)
        argv[i] = Argv[i];
    argc = Argc;
}
const char *lGetExt(const char *fspec) {
    const char *e = strrchr(fspec, '.');
    return e;
}

static std::vector<std::string> lSplitString(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty())
            elems.push_back(item);
    }
    return elems;
}

static void lUsage(const int ret) {
    fprintf(stdout, "\nusage: ptxcc [options] file.ptx \n");
    fprintf(stdout, "    [--help]\t\t\t This help\n");
    fprintf(stdout, "    [--verbose]\t\t\t Be verbose\n");
    fprintf(stdout, "    [--arch=]\t\t\t GPU target architectures:\n");
    fprintf(stdout, "     \t\t\t\t   ");
    for (const auto &mode : GPUTargets::computeMode)
        fprintf(stdout, "%s ", mode);
    fprintf(stdout, "\n");
    fprintf(stdout, "    [-o <name>]\t\t\t Output file name\n");
    fprintf(stdout, "    [-Xnvcc=<arguments>]\t Arguments to pass through to \"nvcc\"\n");
    fprintf(stdout, " \n");
    exit(ret);
}

int main(int _argc, char *_argv[]) {
    int argc;
    char *argv[128];
    lGetAllArgs(_argc, _argv, argc, argv);

    std::string arch = *GPUTargets::computeMode.begin();
    std::string filePTX;
    std::string fileOBJ;
    std::string extString = ".ptx";
    bool keepTemporaries = false;
    bool verbose = false;
    std::string nvccArguments;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help"))
            lUsage(0);
        else if (!strncmp(argv[i], "--arch=", 7))
            arch = std::string(argv[i] + 7);
        else if (!strncmp(argv[i], "--keep-temporaries", 11))
            keepTemporaries = true;
        else if (!strncmp(argv[i], "--verbose", 9))
            verbose = true;
        else if (!strncmp(argv[i], "-Xnvcc=", 7))
            nvccArguments = std::string(argv[i] + 7);
        else if (!strcmp(argv[i], "-o")) {
            if (++i == argc) {
                fprintf(stderr, "No output file specified after -o option.\n");
                lUsage(1);
            }
            fileOBJ = std::string(argv[i]);
        } else {
            const char *ext = strrchr(argv[i], '.');
            if (ext == NULL) {
                fprintf(stderr, " Unknown argument: %s \n", argv[i]);
                lUsage(1);
            } else if (strncmp(ext, extString.c_str(), 4)) {
                fprintf(stderr, " Unkown extension of the input file: %s \n", ext);
                lUsage(1);
            } else if (filePTX.empty()) {
                filePTX = std::string(argv[i]);
                if (fileOBJ.empty()) {
                    char *baseName = argv[i];
                    while (baseName != ext)
                        fileOBJ += std::string(baseName++, 1);
                }
                fileOBJ += ".o";
            }
        }
    }
#if 0
  fprintf(stderr, " fileOBJ= %s\n", fileOBJ.c_str());
  fprintf(stderr, " arch= %s\n", arch.c_str());
  fprintf(stderr, " file= %s\n", filePTX.empty() ? "$stdin" : filePTX.c_str());
  fprintf(stderr, " num_args= %d\n", (int)nvccArgumentList.size());
  for (int i= 0; i < (int)nvccArgumentList.size(); i++)
    fprintf(stderr, " arg= %d : %s \n", i, nvccArgumentList[i].c_str());
#endif
    if (std::find(GPUTargets::computeMode.begin(), GPUTargets::computeMode.end(), arch) ==
        GPUTargets::computeMode.end()) {
        fprintf(stderr, "ptxcc fatal : --arch=%s is not supported; use option --help for more information\n",
                arch.c_str());
        exit(1);
    }
    if (filePTX.empty()) {
        fprintf(stderr, "ptxcc fatal : No input file specified; use option --help for more information\n");
        exit(1);
    }

    // open a file handle to a particular file:
    std::ifstream inputPTX(filePTX.c_str());
    if (!inputPTX) {
        fprintf(stderr, "ptxcc: error: %s: No such file\n", filePTX.c_str());
        exit(1);
    }

    std::string randomBaseName =
        std::string("/tmp/") + lRandomString(8) + "_" + lSplitString(lSplitString(filePTX, '/').back(), '.')[0];
    if (verbose)
        fprintf(stderr, "baseFileName= %s\n", randomBaseName.c_str());

    std::string fileCU = randomBaseName + ".cu";
    std::ofstream outputCU(fileCU.c_str());
    assert(outputCU);

    std::istream &input = inputPTX;
    std::ostream &output = outputCU;
    std::ostream &error = std::cerr;
    parser::PTXLexer lexer(&input, &error);
    parser::PTXParser state(output);

    // parse through the input until there is no more:
    //

    do {
        ptx::yyparse(lexer, state);
    } while (!input.eof());

    inputPTX.close();
    outputCU.close();

    // process output from nvcc
    //
    /* nvcc -dc -arch=$arch -dryrun -argumentlist fileCU */

    std::string fileSH = randomBaseName + ".sh";

    std::string nvccExe("nvcc");
    std::string nvccCmd;
    nvccCmd += nvccExe + std::string(" ");
    nvccCmd += "-dc ";
    nvccCmd += std::string("-arch=") + arch + std::string(" ");
    nvccCmd += "-dryrun ";
    nvccCmd += nvccArguments + std::string(" ");
    nvccCmd += std::string("-o ") + fileOBJ + std::string(" ");
    nvccCmd += fileCU + std::string(" ");
    nvccCmd += std::string("2> ") + fileSH;
    if (verbose)
        fprintf(stderr, "%s\n", nvccCmd.c_str());
    const int nvccRet = std::system(nvccCmd.c_str());
    if (nvccRet)
        fprintf(stderr, "FAIL: %s\n", nvccCmd.c_str());

    std::ifstream inputSH(fileSH.c_str());
    assert(inputSH);
    std::vector<std::string> nvccSteps;
    while (!inputSH.eof()) {
        nvccSteps.push_back(std::string());
        std::getline(inputSH, nvccSteps.back());
        if (nvccRet)
            fprintf(stderr, " %s\n", nvccSteps.back().c_str());
    }
    inputSH.close();
    if (nvccRet)
        exit(-1);

    for (int i = 0; i < (int)nvccSteps.size(); i++) {
        std::string cmd = nvccSteps[i];
        for (int j = 0; j < (int)cmd.size() - 1; j++)
            if (cmd[j] == '#' && cmd[j + 1] == '$')
                cmd[j] = cmd[j + 1] = ' ';
        std::vector<std::string> splitCmd = lSplitString(cmd, ' ');

        if (!splitCmd.empty()) {
            if (splitCmd[0] == std::string("cicc"))
                cmd = std::string("   cp ") + filePTX + std::string(" ") + splitCmd.back();
            if (splitCmd[0] == std::string("LIBRARIES="))
                cmd = "";
        }
        nvccSteps[i] = cmd;
        if (verbose)
            fprintf(stderr, "%3d: %s\n", i, cmd.c_str());
        const int ret = std::system(cmd.c_str());
        if (ret) {
            fprintf(stderr, " Something went wrong .. \n");
            for (int j = 0; j < i; j++)
                fprintf(stderr, "PASS: %s\n", nvccSteps[j].c_str());
            fprintf(stderr, "FAIL: %s\n", nvccSteps[i].c_str());
            exit(-1);
        }
    }

    if (!keepTemporaries) {
        /* remove temporaries */
    }
}
