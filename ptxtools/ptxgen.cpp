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
   Based on "ptxgen" NVVM example from CUDA Toolkit
   */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nvvm.h>
#include <sys/stat.h>


template<typename T>
static std::string lValueToString(const T& value)
{
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

typedef struct stat Stat;


#define PTXGENStatus int
enum {
  PTXGEN_SUCCESS                    = 0x0000,
  PTXGEN_FILE_IO_ERROR              = 0x0001,
  PTXGEN_BAD_ALLOC_ERROR            = 0x0002,
  PTXGEN_LIBNVVM_COMPILATION_ERROR  = 0x0004,
  PTXGEN_LIBNVVM_ERROR              = 0x0008,
  PTXGEN_INVALID_USAGE              = 0x0010,
  PTXGEN_LIBNVVM_HOME_UNDEFINED     = 0x0020,
  PTXGEN_LIBNVVM_VERIFICATION_ERROR = 0x0040
};

static PTXGENStatus getLibDeviceName(const int computeArch, std::string &libDeviceName)
{
  const char *env = getenv("LIBNVVM_HOME");
#ifdef LIBNVVM_HOME
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
  const std::string libnvvmPath(env ? env : TOSTRING(LIBNVVM_HOME));
#undef TOSTRING
#undef STRINGIFY
#else
  const std::string libnvvmPath(env);
#endif

  if (libnvvmPath.empty())
  {
    fprintf(stderr, "The environment variable LIBNVVM_HOME is undefined\n");
    return PTXGEN_LIBNVVM_HOME_UNDEFINED;
  }

  /* Use libdevice for compute_20, if the target is not compute_20, compute_30,
   * or compute_35. */
  const std::string libdevice = 
    std::string("/libdevice/libdevice.compute_") +
    lValueToString(computeArch)+ "." +
    lValueToString(LIBDEVICE_MAJOR_VERSION) + 
    lValueToString(LIBDEVICE_MINOR_VERSION) +
    ".bc";

  libDeviceName = libnvvmPath + libdevice;

  return PTXGEN_SUCCESS;
}

static PTXGENStatus addFileToProgram(const std::string &filename, nvvmProgram prog)
{
  char        *buffer;
  size_t       size;
  Stat         fileStat;

  /* Open the input file. */
  FILE *f = fopen(filename.c_str(), "rb");
  if (f == NULL) {
    fprintf(stderr, "Failed to open %s\n", filename.c_str());
    return PTXGEN_FILE_IO_ERROR;
  }

  /* Allocate buffer for the input. */
  fstat(fileno(f), &fileStat);
  buffer = (char *) malloc(fileStat.st_size);
  if (buffer == NULL) {
    fprintf(stderr, "Failed to allocate memory\n");
    return PTXGEN_BAD_ALLOC_ERROR;
  }
  size = fread(buffer, 1, fileStat.st_size, f);
  if (ferror(f)) {
    fprintf(stderr, "Failed to read %s\n", filename.c_str());
    fclose(f);
    free(buffer);
    return PTXGEN_FILE_IO_ERROR;
  }
  fclose(f);

  if (nvvmAddModuleToProgram(prog, buffer, size, filename.c_str()) != NVVM_SUCCESS) {
    fprintf(stderr,
            "Failed to add the module %s to the compilation unit\n",
            filename.c_str());
    free(buffer);
    return PTXGEN_LIBNVVM_ERROR;
  }

  free(buffer);
  return PTXGEN_SUCCESS;
}

static PTXGENStatus generatePTX(
    std::vector<std::string> nvvmOptions, 
    std::vector<std::string> nvvmFiles, 
    std::ostream &out,
    const int computeArch)
{
  nvvmProgram prog;
  PTXGENStatus status;

  /* Create the compiliation unit. */
  if (nvvmCreateProgram(&prog) != NVVM_SUCCESS) 
  {
    fprintf(stderr, "Failed to create the compilation unit\n");
    return PTXGEN_LIBNVVM_ERROR;
  }
  

  /* Add libdevice. */
  std::string libDeviceName;
  status = getLibDeviceName(computeArch, libDeviceName);
  if (status != PTXGEN_SUCCESS) 
  {
    nvvmDestroyProgram(&prog);
    return status;
  }
  status = addFileToProgram(libDeviceName, prog);
  if (status != PTXGEN_SUCCESS) 
  {
    fprintf(stderr, "Please double-check LIBNVVM_HOME environmental variable.\n");
    nvvmDestroyProgram(&prog);
    return status;
  }

  /* Add the module to the compilation unit. */
  for (int i = 0; i < (int)nvvmFiles.size(); ++i) 
  {
    status = addFileToProgram(nvvmFiles[i], prog);
    if (status != PTXGEN_SUCCESS) 
    {
      nvvmDestroyProgram(&prog);
      return status;
    }
  }

  const int numOptions = nvvmOptions.size();
  std::vector<const char*> options(numOptions);
  for (int i = 0; i < numOptions; i++)
    options[i] = nvvmOptions[i].c_str();

  /* Verify the compilation unit. */
  if (nvvmVerifyProgram(prog, numOptions, &options[0]) != NVVM_SUCCESS) 
  {
    fprintf(stderr, "Failed to verify the compilation unit\n");
    status |= PTXGEN_LIBNVVM_VERIFICATION_ERROR;
  }

  /* Print warnings and errors. */
  {
    size_t logSize;
    if (nvvmGetProgramLogSize(prog, &logSize) != NVVM_SUCCESS) 
    {
      fprintf(stderr, "Failed to get the compilation log size\n");
      status |= PTXGEN_LIBNVVM_ERROR;
    } 
    else 
    {
      std::string log(logSize,0);
      if (nvvmGetProgramLog(prog, &log[0]) != NVVM_SUCCESS) 
      {
        fprintf(stderr, "Failed to get the compilation log\n");
        status |= PTXGEN_LIBNVVM_ERROR;
      } 
      else 
      {
        fprintf(stderr, "%s\n", log.c_str());
      }
    }
  }

  if (status & PTXGEN_LIBNVVM_VERIFICATION_ERROR) 
  {
    nvvmDestroyProgram(&prog);
    return status;
  }
  
  /* Compile the compilation unit. */
  if (nvvmCompileProgram(prog, numOptions, &options[0]) != NVVM_SUCCESS) 
  {
    fprintf(stderr, "Failed to generate PTX from the compilation unit\n");
    status |= PTXGEN_LIBNVVM_COMPILATION_ERROR;
  } 
  else 
  {
    size_t ptxSize;
    if (nvvmGetCompiledResultSize(prog, &ptxSize) != NVVM_SUCCESS) 
    {
      fprintf(stderr, "Failed to get the PTX output size\n");
      status |= PTXGEN_LIBNVVM_ERROR;
    } 
    else 
    {
      std::string ptx(ptxSize,0);
      if (nvvmGetCompiledResult(prog, &ptx[0]) != NVVM_SUCCESS) 
      {
        fprintf(stderr, "Failed to get the PTX output\n");
        status |= PTXGEN_LIBNVVM_ERROR;
      } 
      else 
      {
        out << ptx;
      }
    }
  }

  /* Print warnings and errors. */
  {
    size_t logSize;
    if (nvvmGetProgramLogSize(prog, &logSize) != NVVM_SUCCESS) 
    {
      fprintf(stderr, "Failed to get the compilation log size\n");
      status |= PTXGEN_LIBNVVM_ERROR;
    } 
    else 
    {
      std::string log(logSize,0);
      if (nvvmGetProgramLog(prog, &log[0]) != NVVM_SUCCESS) 
      {
        fprintf(stderr, "Failed to get the compilation log\n");
        status |= PTXGEN_LIBNVVM_ERROR;
      } 
      else 
      {
        fprintf(stderr, "%s\n", log.c_str());
      }
    }
  }

  /* Release the resources. */
  nvvmDestroyProgram(&prog);

  return PTXGEN_SUCCESS;
}

static void showUsage()
{
  fprintf(stderr,"Usage: ptxgen [OPTION]... [FILE]...\n"
                 "  [FILE] could be a .bc file or a .ll file\n");
}

static void lUsage(const int ret)
{
  fprintf(stdout, "\nusage: ptxgen [options] file.[ll,bc] \n");
  fprintf(stdout, "    [--help]\t\t This help\n");
  fprintf(stdout, "    [--verbose]\t\t Be verbose\n");
  fprintf(stdout, "    [--arch={%s}]\t GPU target architecture\n", "sm_35");
  fprintf(stdout, "    [-o <name>]\t\t Output file name\n");
  fprintf(stdout, "    [-g]\t\t Enable generation of debuggin information \n");
  fprintf(stdout, "    [--opt=]\t\t Optimization parameters \n");
  fprintf(stdout, "     \t\t\t    0 - disable optimizations \n");
  fprintf(stdout, "     \t\t\t    3 - defalt, enable optimizations \n");
  fprintf(stdout, "    [--ftz=]\t\t Flush-to-zero mode when performsing single-precision floating-point operations\n");
  fprintf(stdout, "     \t\t\t    0 - default, preserve denormal values\n");
  fprintf(stdout, "     \t\t\t    1 - flush denormal values to zero\n");
  fprintf(stdout, "    [--prec-sqrt=]\t Precision mode for single-precision floating-point square root\n");
  fprintf(stdout, "     \t\t\t    0 - use a faster approximation\n");
  fprintf(stdout, "     \t\t\t    1 - default, use IEEE round-to-nearest mode\n");
  fprintf(stdout, "    [--prec-div=]\t Precision mode for single-precision floating-point division and reciprocals\n");
  fprintf(stdout, "     \t\t\t    0 - use a faster approximation\n");
  fprintf(stdout, "     \t\t\t    1 - default, use IEEE round-to-nearest mode\n");
  fprintf(stdout, "    [--fma=]\t\t FMA contraction mode \n");
  fprintf(stdout, "     \t\t\t    0 - disable\n");
  fprintf(stdout, "     \t\t\t    1 - default, enable\n");
  fprintf(stdout, "    [--use_fast_math]\t Make use of fast maih. Implies --ftz=1 --prec-div=0 --prec-sqrt=0\n");
  fprintf(stdout, " \n");
  exit(ret);
}

int main(int argc, char *argv[])
{
  int _opt      = 3;
  int _ftz      = 0;
  int _precSqrt = 1;
  int _precDiv  = 1;
  int _fma      = 1;
  bool _useFastMath = false;
  bool _debug       = false;
  bool _verbose     = false;
  std::string _arch = "sm_35";
  std::string fileIR, filePTX;

  for (int i = 1; i < argc; ++i) 
  {
    if (!strcmp(argv[i], "--help"))
      lUsage(0);
    else if (!strncmp(argv[i], "--arch=", 7))
      _arch = std::string(argv[i]+7);
    else if (!strncmp(argv[i], "-g", 2))
      _debug = true;
    else if (!strncmp(argv[i], "--verbose", 9))
      _verbose = true;
    else if (!strncmp(argv[i], "--opt=", 6))
      _opt = atoi(argv[i]+6);
    else if (!strncmp(argv[i], "--ftz=", 6))
      _ftz = atoi(argv[i]+6);
    else if (!strncmp(argv[i], "--prec-sqrt=", 12))
      _precSqrt = atoi(argv[i]+12);
    else if (!strncmp(argv[i], "--prec-div=", 11))
      _precDiv = atoi(argv[i]+11);
    else if (!strncmp(argv[i], "--fma=", 6))
      _fma = atoi(argv[i]+6);
    else if (!strncmp(argv[i], "--use_fast_math", 15))
      _useFastMath = true;
    else if (!strcmp(argv[i], "-o"))
    {
      if (++i == argc)
      {
        fprintf(stderr, "No output file specified after -o option.\n");
        lUsage(1);
      }
      filePTX = std::string(argv[i]);
    }
    else 
    {
      const char * ext = strrchr(argv[i], '.');
      if (ext == NULL)
      {
        fprintf(stderr, " Unknown argument: %s \n", argv[i]);
        lUsage(1);
      }
      else if (strncmp(ext, ".ll", 3) && strncmp(ext, ".bc", 3))
      {
        fprintf(stderr, " Unkown extension of the input file: %s \n", ext);
        lUsage(1);
      }
      else if (filePTX.empty())
      {
        fileIR = std::string(argv[i]);
        if (filePTX.empty())
        {
          char * baseName = argv[i];
          while (baseName != ext)
            filePTX += std::string(baseName++,1);
        }
        filePTX += ".ptx";
      }
    }
  }
  
  if (fileIR.empty())
  {
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

  int computeArch = 35;
  assert(_arch == std::string("sm_35"));

  if (_useFastMath)
  {
    _ftz = 1;
    _precSqrt = _precDiv = 0;
  }

  std::vector<std::string> nvvmOptions;
  nvvmOptions.push_back("-arch=compute_35");
  nvvmOptions.push_back("-ftz="       + lValueToString(_ftz));
  nvvmOptions.push_back("-prec-sqrt=" + lValueToString(_precSqrt));
  nvvmOptions.push_back("-prec-div="  + lValueToString(_precDiv));
  nvvmOptions.push_back("-fma="       + lValueToString(_fma));
  if (_debug)
    nvvmOptions.push_back("-g");

  std::vector<std::string> nvvmFiles;
  nvvmFiles.push_back(fileIR);

  std::ofstream outputPTX(filePTX.c_str());
  assert(outputPTX);

  const int ret = generatePTX(nvvmOptions, nvvmFiles, outputPTX, computeArch);
    outputPTX.open(filePTX.c_str());
  return ret;
}

