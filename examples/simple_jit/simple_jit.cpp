#include "module.h"
#include "options.h"
#include <llvm-c/Target.h>
#include <stdio.h>

static const char* src = R"(
export void vadd(
    uniform float a[],
    uniform float b[],
    uniform float r[],
    uniform int n) {
    foreach (i = 0 ... n) {
        r[i] = a[i] + b[i];
    }
}
)";

typedef void (*VADDF)(float*, float*, float*, int);

int main(int argc, char* argv[]) {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();
    LLVMInitializeX86TargetMC();

    OptionParseResult opr = ParseOptions(argc,argv);
    int ret = (opr & OPTION_PARSE_RESULT_ERROR) ? 1 : 0;
    switch (opr) {
    case OPTION_PARSE_RESULT_REQUEST_VERSION:
        PrintVersion();
        exit(0);
    case OPTION_PARSE_RESULT_REQUEST_USAGE:
        PrintUsage(ret);
    case OPTION_PARSE_RESULT_REQUEST_DEV_USAGE:
        PrintDevUsage(ret);
    default:
        break;
    }
    if (ret == 1) {
        printf("Failed to parse command line options.\n");
        return 1;
    }

    int ec = Module::CompileAndJIT(src);
    if (ec != 0) {
        printf("Failed to compile JIT.\n");
        return 1;
    }

    uint64_t addr = Module::GetFunctionAddress("vadd");
    printf("Function address: %lx\n", addr);

    VADDF f = (VADDF)addr;

    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    float c[5];

    f(a, b, c, 5);

    for (int i = 0; i < 5; ++i) {
        printf("%f ", c[i]);
    }
    printf("\n");

    return 0;
}

