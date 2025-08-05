/*
  Copyright (c) 2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#if defined(__WASM__)
#define ISPC_IS_WASM
#elif defined(_WIN32) || defined(_WIN64)
#define ISPC_IS_WINDOWS
#elif defined(__linux__) || defined(__FreeBSD__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#else
#error "Host OS was not detected"
#endif

#if defined(_WIN64)
#define ISPC_IS_WINDOWS64
#endif

#ifdef ISPC_IS_WINDOWS
#include <windows.h>
#endif // ISPC_IS_WINDOWS

#include <assert.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#if defined ISPC_IS_LINUX || defined ISPC_IS_WASM
#include <malloc.h>
#endif

#include "ispc/ispc.h"

// For current tests we need max width multiplied by 4, i.e. 64*4
#define ARRAY_SIZE 256

#if defined(ISPC_IS_WINDOWS64)
#ifdef VECTORCALL_CONV
#define CALLINGCONV __vectorcall
#else
// __vectorcall calling convention is off by default.
#define CALLINGCONV //__vectorcall
#endif              // VECTORCALL_CONV
#else
#define CALLINGCONV
#endif // ISPC_IS_WINDOWS64

// Function pointer types for different test signatures
typedef void(CALLINGCONV *f_v_func_t)(float *result);
typedef void(CALLINGCONV *f_f_func_t)(float *result, float *a);
typedef void(CALLINGCONV *f_fu_func_t)(float *result, float *a, float b);
typedef void(CALLINGCONV *f_fi_func_t)(float *result, float *a, int *b);
typedef void(CALLINGCONV *f_du_func_t)(float *result, double *a, double b);
typedef void(CALLINGCONV *f_duf_func_t)(float *result, double *a, float b);
typedef void(CALLINGCONV *f_di_func_t)(float *result, double *a, int *b);
typedef void(CALLINGCONV *print_uf_func_t)(float a);
typedef void(CALLINGCONV *print_f_func_t)(float *a);
typedef void(CALLINGCONV *print_fuf_func_t)(float *a, float b);
typedef void(CALLINGCONV *result_func_t)(float *val);
typedef void(CALLINGCONV *print_result_func_t)();

void ISPCLaunch(void **handlePtr, void *f, void *d, int, int, int);
void ISPCSync(void *handle);
void *ISPCAlloc(void **handlePtr, int64_t size, int32_t alignment);

int width() {
#if defined(TEST_WIDTH)
    return TEST_WIDTH;
#else
#error "Unknown or unset TEST_WIDTH value"
#endif
}

void ISPCLaunch(void **handle, void *f, void *d, int count0, int count1, int count2) {
    *handle = (void *)(uintptr_t)0xdeadbeef;
    typedef void (*TaskFuncType)(void *, int, int, int, int, int, int, int, int, int, int);
    TaskFuncType func = (TaskFuncType)f;
    int count = count0 * count1 * count2, idx = 0;
    for (int k = 0; k < count2; ++k)
        for (int j = 0; j < count1; ++j)
            for (int i = 0; i < count0; ++i)
                func(d, 0, 1, idx++, count, i, j, k, count0, count1, count2);
}

void ISPCSync(void *) {}

void *ISPCAlloc(void **handle, int64_t size, int32_t alignment) {
    *handle = (void *)(uintptr_t)0xdeadbeef;
    // and now, we leak...
#ifdef ISPC_IS_WINDOWS
    return _aligned_malloc(size, alignment);
#elif defined ISPC_IS_LINUX
    return memalign(alignment, size);
#elif defined ISPC_IS_APPLE || defined ISPC_IS_WASM
    void *mem = malloc(size + (alignment - 1) + sizeof(void *));
    char *amem = ((char *)mem) + sizeof(void *);
    amem = amem + uint32_t(alignment - (reinterpret_cast<uint64_t>(amem) & (alignment - 1)));
    ((void **)amem)[-1] = mem;
    return amem;
#else
#error "Host OS was not detected"
#endif
}

#if defined(_WIN32) || defined(_WIN64)
#define ALIGN __declspec(align(64))
#else
#define ALIGN __attribute__((aligned(64)))
#endif

// Create temporary ISPC file from test content
bool createTemporaryISPCFile(const std::string &filename, const std::string &content) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        printf("Failed to create temporary ISPC file: %s\n", filename.c_str());
        return false;
    }

    file << content;
    file.close();
    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <ispc_source_file> [target]\n", argv[0]);
        return 1;
    }

    std::string ispc_file = argv[1];
    std::string target = (argc > 2) ? argv[2] : "host";

    // Initialize ISPC JIT
    if (!ispc::Initialize()) {
        printf("Failed to initialize ISPC JIT\n");
        return 1;
    }

    int w = width();
    assert(w <= 64);

    ALIGN float returned_result[ARRAY_SIZE];
    ALIGN float vfloat[ARRAY_SIZE];
    ALIGN double vdouble[ARRAY_SIZE];
    ALIGN int vint[ARRAY_SIZE];
    ALIGN int vint2[ARRAY_SIZE];

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        returned_result[i] = -1e20;
        vfloat[i] = i + 1;
        vdouble[i] = i + 1;
        vint[i] = 2 * (i + 1);
        vint2[i] = i + 5;
    }

    float b = 5.;

    // Create ISPC engine with include path for test_static.isph and TEST_SIG definition
    std::string test_sig_def = "-DTEST_SIG=" + std::to_string(TEST_SIG);
    std::vector<std::string> args = {"--target=" + target, "-I", "tests", test_sig_def, "--woff"};
    auto engine = ispc::ISPCEngine::CreateFromArgs(args);

    if (!engine) {
        printf("Failed to create ISPC engine\n");
        ispc::Shutdown();
        return 1;
    }

    // Set the runtime functions that JIT will use
    if (!engine->SetJitRuntimeFunction("ISPCLaunch", (void*)ISPCLaunch) ||
        !engine->SetJitRuntimeFunction("ISPCSync", (void*)ISPCSync) ||
        !engine->SetJitRuntimeFunction("ISPCAlloc", (void*)ISPCAlloc)) {
        printf("Failed to set JIT runtime functions\n");
        ispc::Shutdown();
        return 1;
    }

    // Compile ISPC file using JIT
    int compile_result = engine->CompileFromFileToJit(ispc_file);
    if (compile_result != 0) {
        printf("Failed to compile ISPC file: %s (error code: %d)\n", ispc_file.c_str(), compile_result);
        ispc::Shutdown();
        return 1;
    }

    // Get function pointers based on TEST_SIG - use CPU entry point names
    void *main_func = nullptr;
    void *result_func = nullptr;

#if (TEST_SIG == 0)
    main_func = engine->GetJitFunction("f_v_cpu_entry_point");
    result_func = engine->GetJitFunction("result_cpu_entry_point");
#elif (TEST_SIG == 1)
    main_func = engine->GetJitFunction("f_f_cpu_entry_point");
    result_func = engine->GetJitFunction("result_cpu_entry_point");
#elif (TEST_SIG == 2)
    main_func = engine->GetJitFunction("f_fu_cpu_entry_point");
    result_func = engine->GetJitFunction("result_cpu_entry_point");
#elif (TEST_SIG == 3)
    main_func = engine->GetJitFunction("f_fi_cpu_entry_point");
    result_func = engine->GetJitFunction("result_cpu_entry_point");
#elif (TEST_SIG == 4)
    main_func = engine->GetJitFunction("f_du_cpu_entry_point");
    result_func = engine->GetJitFunction("result_cpu_entry_point");
#elif (TEST_SIG == 5)
    main_func = engine->GetJitFunction("f_duf_cpu_entry_point");
    result_func = engine->GetJitFunction("result_cpu_entry_point");
#elif (TEST_SIG == 6)
    main_func = engine->GetJitFunction("f_di_cpu_entry_point");
    result_func = engine->GetJitFunction("result_cpu_entry_point");
#elif (TEST_SIG == 7)
    // Size test - we'll need to handle this differently in JIT mode
    printf("Size test (TEST_SIG=7) not yet supported in JIT mode\n");
    ispc::Shutdown();
    return 0;
#elif (TEST_SIG == 32)
    main_func = engine->GetJitFunction("print_uf_cpu_entry_point");
    result_func = engine->GetJitFunction("print_result_cpu_entry_point");
#elif (TEST_SIG == 33)
    main_func = engine->GetJitFunction("print_f_cpu_entry_point");
    result_func = engine->GetJitFunction("print_result_cpu_entry_point");
#elif (TEST_SIG == 34)
    main_func = engine->GetJitFunction("print_fuf_cpu_entry_point");
    result_func = engine->GetJitFunction("print_result_cpu_entry_point");
#else
#error "Unknown or unset TEST_SIG value"
#endif

    if (!main_func) {
        printf("Failed to get main function from JIT\n");
        ispc::Shutdown();
        return 1;
    }

    // Execute the function based on TEST_SIG
#if (TEST_SIG == 0)
    ((f_v_func_t)main_func)(returned_result);
#elif (TEST_SIG == 1)
    ((f_f_func_t)main_func)(returned_result, vfloat);
#elif (TEST_SIG == 2)
    ((f_fu_func_t)main_func)(returned_result, vfloat, b);
#elif (TEST_SIG == 3)
    ((f_fi_func_t)main_func)(returned_result, vfloat, vint);
#elif (TEST_SIG == 4)
    ((f_du_func_t)main_func)(returned_result, vdouble, b);
#elif (TEST_SIG == 5)
    ((f_duf_func_t)main_func)(returned_result, vdouble, static_cast<float>(b));
#elif (TEST_SIG == 6)
    ((f_di_func_t)main_func)(returned_result, vdouble, vint2);
#elif (TEST_SIG == 32)
    ((print_uf_func_t)main_func)(static_cast<float>(b));
#elif (TEST_SIG == 33)
    ((print_f_func_t)main_func)(vfloat);
#elif (TEST_SIG == 34)
    ((print_fuf_func_t)main_func)(vfloat, static_cast<float>(b));
#endif

    // Get expected results for comparison (except for print functions)
    float expected_result[ARRAY_SIZE];
    memset(expected_result, 0, ARRAY_SIZE * sizeof(float));

#if (TEST_SIG < 32)
    if (!result_func) {
        printf("Failed to get result function from JIT\n");
        ispc::Shutdown();
        return 1;
    }
    ((result_func_t)result_func)(expected_result);
#else
    if (result_func) {
        ((print_result_func_t)result_func)();
    }
    ispc::Shutdown();
    return 0;
#endif

    // Compare results
    int errors = 0;
    for (int i = 0; i < w; ++i) {
        if (returned_result[i] != expected_result[i]) {
#ifdef EXPECT_FAILURE
            // bingo, failed
            ispc::Shutdown();
            return 1;
#else
            printf("%s: value %d disagrees: returned %f [%a], expected %f [%a]\n", argv[0], i, returned_result[i],
                   returned_result[i], expected_result[i], expected_result[i]);
            ++errors;
#endif // EXPECT_FAILURE
        }
    }

    ispc::Shutdown();

#ifdef EXPECT_FAILURE
    // Don't expect to get here
    return 0;
#else
    return errors > 0;
#endif
}