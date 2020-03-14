#include "L0_helpers.h"
#include "Matrix.h"
#include <cmath>
#include <iostream>
#include <level_zero/ze_api.h>
#include <limits>

using namespace hostutil;

static int run(int m, int niter, int gx, int gy) {

    std::cout.setf(std::ios::unitbuf);
    ze_device_handle_t hDevice = nullptr;
    ze_module_handle_t hModule = nullptr;
    ze_driver_handle_t hDriver = nullptr;
    ze_command_queue_handle_t hCommandQueue = nullptr;

    storage_type_t st = RowMajor;
    float alpha = +1.0, beta = +0.0;

    int n = m, k = m;

    // Initialization
    m = (m / TILE_m) * TILE_m;
    n = k = m;

    int lda = ((k + 15) & ~15);
    int ldb = ((n + 15) & ~15);
    int ldc = ldb;

    const int lda_tmp = (((GEMM_BLOCK / TILE_m) * TILE_m) + 15) & ~15;
    const int ldb_tmp = lda_tmp;
    const int ldc_tmp = lda_tmp;
    printf("SGEMM: C(%d, %d) = %.2f * C(%d, %d) + %.2f A(%d, %d) * B(%d, %d)\n", m, n, beta, m, n, alpha, m, k, k, n);
    printf("Thread-group setting: %d x %d \n", gx, gy);

#ifdef CMKERNEL
    L0InitContext(hDriver, hDevice, hModule, hCommandQueue, "naive_sgemm_cm_mt.spv");
#else
    L0InitContext(hDriver, hDevice, hModule, hCommandQueue, "naive_sgemm_mt.spv");
#endif

    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;

#ifdef CMKERNEL
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "sgemm_kernel");
#else
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "SGEMM_naive_task");
#endif

    // Allocate matrices
    Matrix A(m, k, lda, NULL, true, "A", st);
    Matrix B(k, n, ldb, NULL, true, "B", st);
    Matrix C_gold(m, n, ldc, NULL, false, "C_gold", st);
    Matrix C(C_gold, "C");
    Matrix A_result(A, "A_result");
    Matrix B_result(B, "B_result");
    Matrix C_result(C_gold, "C_result");

    if (niter == 1) {
        printf("** validation run, only one iteration **\n");
        printf("** For performance run, add cmd-args: Sgemm 2048 1000 ** \n");
        // Compute gold result
        printf("Compute gold result\n");
        sgemmNxN(m, n, k, alpha, &A(0, 0), A.l_dim(), &B(0, 0), B.l_dim(), beta, &C_gold(0, 0), C_gold.l_dim(), st);
    } else {
        printf("CPU result not computed: Make #iterations=1 to compute CPU result\n");
    }

    void *a_ref = &A(0, 0);
    void *b_ref = &B(0, 0);
    void *c_ref = &C(0, 0);
    void *a_res_ref = &A_result(0, 0);
    void *b_res_ref = &B_result(0, 0);
    void *c_res_ref = &C_result(0, 0);

    int mtA_size = A.l_dim() * m;
    int mtB_size = B.l_dim() * B.n_row();
    int mtC_size = C.l_dim() * m;

    ze_device_mem_alloc_desc_t alloc_desc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                             0};
    ze_host_mem_alloc_desc_t host_alloc_desc = {ZE_HOST_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_HOST_MEM_ALLOC_FLAG_DEFAULT};

    L0_SAFE_CALL(
        zeDriverAllocSharedMem(hDriver, &alloc_desc, &host_alloc_desc, mtA_size * sizeof(float), 0, hDevice, &a_ref));
    L0_SAFE_CALL(
        zeDriverAllocSharedMem(hDriver, &alloc_desc, &host_alloc_desc, mtB_size * sizeof(float), 0, hDevice, &b_ref));
    L0_SAFE_CALL(
        zeDriverAllocSharedMem(hDriver, &alloc_desc, &host_alloc_desc, mtC_size * sizeof(float), 0, hDevice, &c_ref));

    // TODO: remove
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, a_ref, a_res_ref, mtA_size * sizeof(float), nullptr));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, b_ref, b_res_ref, mtB_size * sizeof(float), nullptr));

    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(a_ref), &a_ref));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(b_ref), &b_ref));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(c_ref), &c_ref));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 3, sizeof(int), &m));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 4, sizeof(int), &n));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 5, sizeof(int), &k));

    // EXECUTION
    // set group size
    // FIXME
    uint32_t groupSpaceWidth = 1;
    uint32_t groupSpaceHeight = 1;

    uint32_t group_size = groupSpaceWidth * groupSpaceHeight;
    L0_SAFE_CALL(zeKernelSetGroupSize(hKernel, /*x*/ groupSpaceWidth, /*y*/ groupSpaceHeight, /*z*/ 1));

    // set grid size
    ze_group_count_t dispatchTraits = {(uint32_t)gx, (uint32_t)gy, 1};

    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            C(i, j) = C_result(i, j) = -1;

    double total = 0;
    auto tot_wct = std::chrono::system_clock::now();

    for (int i = 0; i < niter; ++i) {
        auto wct = std::chrono::system_clock::now();
        // launch
        L0_SAFE_CALL(zeCommandListAppendLaunchKernel(hCommandList, hKernel, &dispatchTraits, nullptr, 0, nullptr));
        L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

        // TODO: enable
        // L0_SAFE_CALL(zeCommandListClose(hCommandList));
        // L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
        // L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()));
        auto dur = (std::chrono::system_clock::now() - wct);
        auto secs = std::chrono::duration_cast<std::chrono::nanoseconds>(dur);
        total += (secs.count() / 1e+6);
        // copy result to host
        // L0_SAFE_CALL(zeCommandListReset(hCommandList));
        // L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));
        L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, c_res_ref, c_ref, mtC_size * sizeof(float), nullptr));
        // dispatch & wait
        L0_SAFE_CALL(zeCommandListClose(hCommandList));
        L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
        L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()));
    }
    auto tot_dur = (std::chrono::system_clock::now() - tot_wct);
    auto tot_secs = std::chrono::duration_cast<std::chrono::nanoseconds>(tot_dur);
    std::cout << "Time is: " << tot_secs.count() / 1e+6 / niter << " milliseconds" << std::endl;
    std::cout << "No memory time is: " << total / niter << " ms" << std::endl;

    // RESULT CHECK
    bool pass = false;
    if (niter == 1) {
        if (C_result == C_gold) {
            printf("PASSED\n");
            pass = true;
        } else
            printf("FAILED\n");
    } else
        printf("Result not checked - make #iterations=1 to check result!\n");

    printf("----------------------------\n");

    L0_SAFE_CALL(zeDriverFreeMem(hDriver, a_ref));
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, b_ref));
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, c_ref));

    return (pass) ? 0 : 1;
}

int main(int argc, char *argv[]) {
    int m = GEMM_BLOCK;
    int niterations = 1;
    int gx = 2, gy = 1;

    if (argc >= 3) {
        m = atoi(argv[1]);
        niterations = atoi(argv[2]);
        if (argc == 5) {
            gx = atoi(argv[3]);
            gy = atoi(argv[4]);
        }
    }

    int success = 0;

    std::cout << "Running test with " << niterations << " iterations on " << gx << " * " << gy << " threads."
              << std::endl;
    success = run(m, niterations, gx, gy);

    return success;
}
