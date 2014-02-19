#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cassert>
#include <algorithm>
#include "PTXParser.h"

static char lRandomAlNum()
{
  const char charset[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";
  const size_t max_index = (sizeof(charset) - 1);
  return charset[ rand() % max_index ];
}

static std::string lRandomString(const size_t length)
{
  std::string str(length,0);
  std::generate_n( str.begin(), length, lRandomAlNum);
  return str;
}

static void lGetAllArgs(int Argc, char *Argv[], int &argc, char *argv[128]) 
{
  // Copy over the command line arguments (passed in)
  for (int i = 0; i < Argc; ++i)
    argv[i] = Argv[i];
  argc = Argc;
}
const char *lGetExt (const char *fspec) 
{
  char *e = strrchr (fspec, '.');
  return e;
}

static std::vector<std::string> lSplitString(const std::string &s, char delim)
{
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty())
      elems.push_back(item);
  }
  return elems;
}

static void lUsage(const int ret)
{
  fprintf(stderr, "\nusage: ptxc\n");
  fprintf(stderr, "    [--help]\t\t\t\t This help\n");
  fprintf(stderr, "    [--arch={%s}]\t\t\t GPU target architecture\n", "sm_35");
  fprintf(stderr, "    [-o <name>]\t\t\t\t Output file name\n");
  fprintf(stderr, "    [-Xnvcc=<arguments>]\t\t Arguments to pass through to \"nvcc\"\n");
  fprintf(stderr, " \n");
  exit(ret);
}

int main(int _argc, char * _argv[])
{
  int argc;
  char *argv[128];
  lGetAllArgs(_argc, _argv, argc, argv);

  std::string arch="sm_35";
  std::string filePTX;
  std::string fileOBJ;
  std::string extString = ".ptx";
  bool keepTemporaries = false;
  std::string nvccArguments;

  for (int i = 1; i < argc; ++i) 
  {
    if (!strcmp(argv[i], "--help"))
      lUsage(0);
    else if (!strncmp(argv[i], "--arch=", 7))
      arch = std::string(argv[i]+7);
    else if (!strncmp(argv[i], "--keep-temporaries", 11))
      keepTemporaries = true;
    else if (!strncmp(argv[i], "-Xnvcc=", 7))
      nvccArguments = std::string(argv[i]+7);
    else if (!strcmp(argv[i], "-o"))
    {
      if (++i == argc)
      {
        fprintf(stderr, "No output file specified after -o option.\n");
        lUsage(1);
      }
      fileOBJ = std::string(argv[i]);
    }
    else 
    {
      const char * ext = strrchr(argv[i], '.');
      if (ext == NULL)
      {
        fprintf(stderr, " Unknown argument: %s \n", argv[i]);
        lUsage(1);
      }
      else if (strncmp(ext, extString.c_str(), 4))
      {
        fprintf(stderr, " Unkown extension of the input file: %s \n", ext);
        lUsage(1);
      }
      else if (filePTX.empty())
      {
        filePTX = std::string(argv[i]);
        if (fileOBJ.empty())
        {
          char * baseName = argv[i];
          while (baseName != ext)
            fileOBJ += std::string(baseName++,1);
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
  assert(arch == std::string("sm_35"));
  if (filePTX.empty())
  {
    fprintf(stderr, "ptxc fatal : No input file specified; use option --help for more information\n");
    exit(1);
  }

	// open a file handle to a particular file:
  std::ifstream inputPTX(filePTX.c_str());
  if (!inputPTX)
  {
    fprintf(stderr, "ptxc: error: %s: No such file\n", filePTX.c_str());
    exit(1);
  }

  std::string randomBaseName = std::string("/tmp/") + lRandomString(12);
  fprintf(stderr, " randomBaseName= %s\n", randomBaseName.c_str());

  std::string fileCU= randomBaseName + ".cu";
  std::ofstream outputCU(fileCU.c_str());
  assert(outputCU);

  std::istream &  input = inputPTX;
  std::ostream & output = outputCU;
  std::ostream &  error = std::cerr;
  parser::PTXLexer lexer(&input, &error);
  parser::PTXParser state(output);

	// parse through the input until there is no more:
  //

  do {
    ptx::yyparse(lexer, state);
  }
  while (!input.eof());

  inputPTX.close();
  outputCU.close();

  // process output from nvcc
  //
  /* nvcc -dc -arch=$arch -dryrun -argumentlist fileCU */

  std::string fileSH= randomBaseName + ".sh";

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
  fprintf(stderr , " nvccCmd= %s\n", nvccCmd.c_str());
  std::system(nvccCmd.c_str());


  std::ifstream inputSH(fileSH.c_str());
  assert(inputSH);
  std::vector<std::string> nvccSteps;
  while (!inputSH.eof())
  {
    nvccSteps.push_back(std::string());
    std::getline(inputSH, nvccSteps.back());
  }
  inputSH.close();

  for (int i = 0; i < (int)nvccSteps.size(); i++)
  {
    std::string cmd = nvccSteps[i];
    for (int j = 0; j < (int)cmd.size()-1; j++)
      if (cmd[j] == '#' && cmd[j+1] == '$')
        cmd[j] = cmd[j+1] = ' ';
    std::vector<std::string> splitCmd = lSplitString(cmd, ' ');

    if (!splitCmd.empty())
    {
      if (splitCmd[0] == std::string("cicc"))
        cmd = std::string("cp ") + filePTX + std::string(" ") + splitCmd.back();
      if (splitCmd[0] == std::string("LIBRARIES="))
        cmd = "";
    }
    nvccSteps[i] = cmd;
    const int ret = std::system(cmd.c_str());
    if (ret)
    {
      fprintf(stderr, " Something went wrong .. \n");
      for (int j = 0; j < i; j++)
        fprintf(stderr, "PASS: %s\n", nvccSteps[j].c_str());
      fprintf(stderr, "FAIL: %s\n", nvccSteps[i].c_str());
      exit(-1);
    }
  }

  if (!keepTemporaries)
  {
    /* remove temporaries */
  }
  

	
}
