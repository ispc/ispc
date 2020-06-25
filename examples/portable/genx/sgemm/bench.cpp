// ======================================================================== //
// Copyright 2019-2020 Intel Corporation                                         //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// Google Benchmark
#include <benchmark/benchmark.h>
#include <chrono>
#include <iostream>

#include "sgemm.hpp"

static void run_sgemm(benchmark::State &state) {
    SGEMMApp app(false);

    app.initialize();

    for (auto _ : state) {
        constexpr int subiterations = 10;
        SGEMMApp::RunResult result;
        app.run(result, state.range(0), subiterations, 2, 1, false);
        auto gpu_nsec = std::chrono::nanoseconds(result.gpuTime);
        using double_sec = std::chrono::duration<double, std::chrono::seconds::period>;
        state.SetIterationTime(double_sec(gpu_nsec).count() / subiterations);
    }

    app.cleanup();

    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(run_sgemm)->Threads(1)->RangeMultiplier(2)->Range(32, 256)->MinTime(3)->UseManualTime()->Unit(
    benchmark::kMillisecond);
BENCHMARK(run_sgemm)->Threads(1)->RangeMultiplier(2)->Range(512, 512)->MinTime(4)->UseManualTime()->Unit(
    benchmark::kMillisecond);

BENCHMARK_MAIN();