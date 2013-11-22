/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <nvvm.h>

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

/* Two levels of indirection to stringify LIBDEVICE_MAJOR_VERSION and
 * LIBDEVICE_MINOR_VERSION correctly. */
#define getLibDeviceNameForArch(ARCH)                 \
  _getLibDeviceNameForArch(ARCH,                      \
                           LIBDEVICE_MAJOR_VERSION,   \
                           LIBDEVICE_MINOR_VERSION)
#define _getLibDeviceNameForArch(ARCH, MAJOR, MINOR)  \
  __getLibDeviceNameForArch(ARCH, MAJOR, MINOR)
#define __getLibDeviceNameForArch(ARCH, MAJOR, MINOR) \
  ("/libdevice/libdevice.compute_" #ARCH "." #MAJOR #MINOR ".bc")

#define getLibnvvmHome _getLibnvvmHome(LIBNVVM_HOME)
#define _getLibnvvmHome(NVVM_HOME) __getLibnvvmHome(NVVM_HOME)
#define __getLibnvvmHome(NVVM_HOME) (#NVVM_HOME)

typedef struct stat Stat;

typedef enum {
  PTXGEN_SUCCESS                    = 0x0000,
  PTXGEN_FILE_IO_ERROR              = 0x0001,
  PTXGEN_BAD_ALLOC_ERROR            = 0x0002,
  PTXGEN_LIBNVVM_COMPILATION_ERROR  = 0x0004,
  PTXGEN_LIBNVVM_ERROR              = 0x0008,
  PTXGEN_INVALID_USAGE              = 0x0010,
  PTXGEN_LIBNVVM_HOME_UNDEFINED     = 0x0020,
  PTXGEN_LIBNVVM_VERIFICATION_ERROR = 0x0040
} PTXGENStatus;

static PTXGENStatus getLibDeviceName(int computeArch, char **buffer)
{
  const char *libnvvmPath = getLibnvvmHome;
  const char *libdevice   = NULL;

  if (libnvvmPath == NULL) {
    fprintf(stderr, "The environment variable LIBNVVM_HOME undefined\n");
    return PTXGEN_LIBNVVM_HOME_UNDEFINED;
  }

  /* Use libdevice for compute_20, if the target is not compute_20, compute_30,
   * or compute_35. */
  switch (computeArch) {
  default:
    libdevice = getLibDeviceNameForArch(20);
    break;
  case 30:
    libdevice = getLibDeviceNameForArch(30);
    break;
  case 35:
    libdevice = getLibDeviceNameForArch(35);
    break;
  }

  *buffer = (char *) malloc(strlen(libnvvmPath) + strlen(libdevice) + 1);
  if (*buffer == NULL) {
    fprintf(stderr, "Failed to allocate memory\n");
    return PTXGEN_BAD_ALLOC_ERROR;
  }

  /* Concatenate libnvvmPath and name. */
  *buffer = strcat(strcpy(*buffer, libnvvmPath), libdevice);

  return PTXGEN_SUCCESS;
}

static PTXGENStatus addFileToProgram(const char *filename, nvvmProgram prog)
{
  char        *buffer;
  size_t       size;
  Stat         fileStat;

  /* Open the input file. */
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    fprintf(stderr, "Failed to open %s\n", filename);
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
    fprintf(stderr, "Failed to read %s\n", filename);
    fclose(f);
    free(buffer);
    return PTXGEN_FILE_IO_ERROR;
  }
  fclose(f);

  if (nvvmAddModuleToProgram(prog, buffer, size, filename) != NVVM_SUCCESS) {
    fprintf(stderr,
            "Failed to add the module %s to the compilation unit\n",
            filename);
    free(buffer);
    return PTXGEN_LIBNVVM_ERROR;
  }

  free(buffer);
  return PTXGEN_SUCCESS;
}

static PTXGENStatus generatePTX(int numOptions,   const char **options,
                                int numFilenames, const char **filenames,
                                int computeArch)
{
  PTXGENStatus status;
  nvvmProgram  prog;
  char        *libDeviceName;
  int          i;

  /* Create the compiliation unit. */
  if (nvvmCreateProgram(&prog) != NVVM_SUCCESS) {
    fprintf(stderr, "Failed to create the compilation unit\n");
    return PTXGEN_LIBNVVM_ERROR;
  }

  /* Add libdevice. */
  status = getLibDeviceName(computeArch, &libDeviceName);
  if (status != PTXGEN_SUCCESS) {
    nvvmDestroyProgram(&prog);
    return status;
  }
  status = addFileToProgram(libDeviceName, prog);
  free(libDeviceName);
  if (status != PTXGEN_SUCCESS) {
    nvvmDestroyProgram(&prog);
    return status;
  }

  /* Add the module to the compilation unit. */
  for (i = 0; i < numFilenames; ++i) {
    status = addFileToProgram(filenames[i], prog);
    if (status != PTXGEN_SUCCESS) {
      nvvmDestroyProgram(&prog);
      return status;
    }
  }

  /* Verify the compilation unit. */
  if (nvvmVerifyProgram(prog, numOptions, options) != NVVM_SUCCESS) {
    fprintf(stderr, "Failed to verify the compilation unit\n");
    status |= PTXGEN_LIBNVVM_VERIFICATION_ERROR;
  }

  /* Print warnings and errors. */
  {
    size_t logSize;
    char  *log;
    if (nvvmGetProgramLogSize(prog, &logSize) != NVVM_SUCCESS) {
      fprintf(stderr, "Failed to get the compilation log size\n");
      status |= PTXGEN_LIBNVVM_ERROR;
    } else {
      log = (char *) malloc(logSize);
      if (log == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        status |= PTXGEN_BAD_ALLOC_ERROR;
      } else if (nvvmGetProgramLog(prog, log) != NVVM_SUCCESS) {
        fprintf(stderr, "Failed to get the compilation log\n");
        status |= PTXGEN_LIBNVVM_ERROR;
      } else {
        fprintf(stderr, "%s\n", log);
      }
      free(log);
    }
  }

  if (status & PTXGEN_LIBNVVM_VERIFICATION_ERROR) {
    nvvmDestroyProgram(&prog);
    return status;
  }
  
  /* Compile the compilation unit. */
  if (nvvmCompileProgram(prog, numOptions, options) != NVVM_SUCCESS) {
    fprintf(stderr, "Failed to generate PTX from the compilation unit\n");
    status |= PTXGEN_LIBNVVM_COMPILATION_ERROR;
  } else {
    size_t ptxSize;
    char  *ptx;
    if (nvvmGetCompiledResultSize(prog, &ptxSize) != NVVM_SUCCESS) {
      fprintf(stderr, "Failed to get the PTX output size\n");
      status |= PTXGEN_LIBNVVM_ERROR;
    } else {
      ptx = (char *) malloc(ptxSize);
      if (ptx == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        status |= PTXGEN_BAD_ALLOC_ERROR;
      } else if (nvvmGetCompiledResult(prog, ptx) != NVVM_SUCCESS) {
        fprintf(stderr, "Failed to get the PTX output\n");
        status |= PTXGEN_LIBNVVM_ERROR;
      } else {
        fprintf(stdout, "%s\n", ptx);
      }
      free(ptx);
    }
  }

  /* Print warnings and errors. */
  {
    size_t logSize;
    char  *log;
    if (nvvmGetProgramLogSize(prog, &logSize) != NVVM_SUCCESS) {
      fprintf(stderr, "Failed to get the compilation log size\n");
      status |= PTXGEN_LIBNVVM_ERROR;
    } else {
      log = (char *) malloc(logSize);
      if (log == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        status |= PTXGEN_BAD_ALLOC_ERROR;
      } else if (nvvmGetProgramLog(prog, log) != NVVM_SUCCESS) {
        fprintf(stderr, "Failed to get the compilation log\n");
        status |= PTXGEN_LIBNVVM_ERROR;
      } else {
        fprintf(stderr, "%s\n", log);
      }
      free(log);
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

int main(int argc, char *argv[])
{
  PTXGENStatus status         = PTXGEN_SUCCESS;
  int          numOptions     = 0;
  char       **options        = NULL;
  int          numFilenames   = 0;
  char       **filenames      = NULL;
  int          computeArch    = 35;
  int          i;

  /* Process the command-line arguments to extract the libnvvm options and the
   * input file names. */
  if (argc == 1) {
    showUsage();
    return PTXGEN_INVALID_USAGE;
  }

  options   = (char **) malloc((argc ) * sizeof (char *));
  filenames = (char **) malloc((argc - 1) * sizeof (char *));

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] == '-') {
      options[numOptions] = argv[i];
      ++numOptions;
    } else {
      filenames[numFilenames] = argv[i];
      ++numFilenames;
    }
  }
  options[numOptions] = "-arch=compute_35";
  numOptions++;

  if (numFilenames == 0) {
    /* If no input filename is found, then show the usage. */
    showUsage();
    status = PTXGEN_INVALID_USAGE;
  } else {
    /* Run libnvvm to generate PTX. */
    status = generatePTX(numOptions,   (const char **) options,
                         numFilenames, (const char **) filenames,
                         computeArch);
  }

  free(options);
  free(filenames);
  return status;
}
