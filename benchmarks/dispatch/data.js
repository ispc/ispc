window.BENCHMARK_DATA = {
  "lastUpdate": 1793988265419,
  "repoUrl": "https://github.com/ispc/ispc",
  "entries": {
    "dispatch (sse4-i32x4,avx2-i32x8,avx512gnr-x8)": [
      {
        "commit": {
          "author": {
            "name": "Arina Neshlyaeva",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "committer": {
            "name": "aneshlya",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "id": "9d7bbd11079a39a22bce083a700f85c8dddc5eec",
          "message": "Show the output of failed CTest tests in CI builds\n\nCTest only printed that a benchmark failed, without the cases that\nreported incorrect results.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-22T20:44:30Z",
          "url": "https://github.com/ispc/ispc/commit/9d7bbd11079a39a22bce083a700f85c8dddc5eec"
        },
        "date": 1792777796105,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test/4096",
            "value": 225630807.02093428,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 223106449.99999994 ns\nthreads: 1"
          },
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test_multithread/10000",
            "value": 1422903418.0170856,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 210653213.00000006 ns\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Arina Neshlyaeva",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "committer": {
            "name": "aneshlya",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "id": "9d7bbd11079a39a22bce083a700f85c8dddc5eec",
          "message": "Show the output of failed CTest tests in CI builds\n\nCTest only printed that a benchmark failed, without the cases that\nreported incorrect results.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-22T20:44:30Z",
          "url": "https://github.com/ispc/ispc/commit/9d7bbd11079a39a22bce083a700f85c8dddc5eec"
        },
        "date": 1792781798513,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test/4096",
            "value": 224715963.99554983,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 222620252 ns\nthreads: 1"
          },
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test_multithread/10000",
            "value": 1423401746.986201,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 209813555.0000002 ns\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Arina Neshlyaeva",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "committer": {
            "name": "aneshlya",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "id": "29c07f332af1d56d0ce7eac9b2675679ed52edc8",
          "message": "Declare structs named by typedef names in their own scope\n\nThe \"struct B;\" declaration and struct definitions handled a typedef name\nB of a struct A differently: \"struct B;\" declared nothing, while a\ndefinition of struct B in a function body declared a local struct A, which\nhid the global A, and B still referred to the global A.\n\nBoth now go through lDeclareStructInCurrentScope().  If the typedef is in\nan outer scope, they declare a new struct B in the current scope, as C does\nfor the struct tag, and leave A unchanged.  If the typedef is in the current\nscope, \"struct B;\" still refers to A, and a definition still defines A if A\nis declared in the current scope; otherwise it is now reported as a\nredefinition of B instead of declaring a local A.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-22T18:45:56Z",
          "url": "https://github.com/ispc/ispc/commit/29c07f332af1d56d0ce7eac9b2675679ed52edc8"
        },
        "date": 1792894432296,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test/4096",
            "value": 225719359.99440032,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 222903289 ns\nthreads: 1"
          },
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test_multithread/10000",
            "value": 1422835856.96714,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 210017656.0000002 ns\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Arina Neshlyaeva",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "committer": {
            "name": "aneshlya",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "id": "aa0734e76853fba56e037b77032e653cc5074a49",
          "message": "ci: don't source ~/.cargo/env for ispc-downsampler\n\nOn macos-15 rustup is preinstalled by the system package manager, so\ndtolnay/rust-toolchain doesn't run rustup-init and ~/.cargo/env doesn't\nexist; sourcing it failed the job. cargo is already on PATH.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-30T19:30:39Z",
          "url": "https://github.com/ispc/ispc/commit/aa0734e76853fba56e037b77032e653cc5074a49"
        },
        "date": 1793498290447,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test/4096",
            "value": 225359939.0387535,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 223114770 ns\nthreads: 1"
          },
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test_multithread/10000",
            "value": 1429526071.9954967,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 209496833.0000002 ns\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Arina Neshlyaeva",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "committer": {
            "name": "aneshlya",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "id": "ded7463f4926d2fdca9562f712e68292349f5ed2",
          "message": "Fix #3983 (3/3): mark x86 gather and scatter calls nocapture\n\nLLVM's AVX2 and AVX-512 gather intrinsics don't mark their base pointer\ncaptures(none), and the AVX-512 scatters have no memory attributes at all.\nAdd nocapture to the base pointer at the calls and memory(write) to the\nscatters. They aren't marked argmem-only: ISPC can pick a base from\ninteger arithmetic whose offsets reach another object.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-11-02T22:11:26Z",
          "url": "https://github.com/ispc/ispc/commit/ded7463f4926d2fdca9562f712e68292349f5ed2"
        },
        "date": 1793729739102,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test/4096",
            "value": 224344895.00301424,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 221992164.99999985 ns\nthreads: 1"
          },
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test_multithread/10000",
            "value": 1426737628.0040479,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 213357236.99999985 ns\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Arina Neshlyaeva",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "committer": {
            "name": "aneshlya",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "id": "e1d896f50606570ab5b75114136f471f8ebd3b36",
          "message": "Add XFAIL reproducers for the known address gaps\n\nGuard-page runtime tests and IR tests for the gather, scatter and vector\nload/store address gaps that remain after #3985: lane 0 off with a signed\noffset that overflows, offsets that wrap at 2^32 between lanes, a signed\nconstant split without nsw, a struct member at 2 GB through a varying\npointer, new[] sign extending an unsigned count, an unfolded +2^31\ndisplacement, nneg copied to split or rescaled operands, a nested-index\nsum of two signed-i32 byte offsets that reaches 2^31, an explicit int64\nsum converted to a varying pointer, and the integer-to-pointer provenance\nloss of #3983. Each XFAIL test runs only the configurations that fail\ntoday; address-gaps-ok.ispc, address-nested-sum.ispc and\naddress-provenance.ispc run the passing ones as regression gates, with\nthe kernels and drivers shared in address-gaps.inc and\naddress-provenance.inc.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-11-03T21:55:02Z",
          "url": "https://github.com/ispc/ispc/commit/e1d896f50606570ab5b75114136f471f8ebd3b36"
        },
        "date": 1793754771174,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test/4096",
            "value": 224530723.99366647,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 222499169 ns\nthreads: 1"
          },
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test_multithread/10000",
            "value": 1428095562.994713,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 207280917.0000003 ns\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "claude[bot]",
            "username": "claude[bot]",
            "email": "209825114+claude[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "aneshlya",
            "username": "aneshlya",
            "email": "arina.neshlyaeva@intel.com"
          },
          "id": "ae2ae73ba80eccc3b5b989188a8d8124c51663ff",
          "message": "Fix #4008: Renumber metadata before printing final IR on LLVM 24\n\nLLVM trunk now gives each metadata node a persistent print ID\n(\"[IR] Use persistent metadata IDs for printing\"), so printed IR only\ngets the canonical !0, !1, ... numbering when the caller asks for it.\nispc-opt did not ask, which broke the keeps_metadata check in\nmemory-access-materialize.ll, and --emit-llvm-text output had the same\nnon-canonical numbering.\n\nAsk ispc-opt's PrintModulePass to renumber, as upstream opt does, and\ncall Module::renumberMetadataForAssembly() before printing for\n--emit-llvm-text. Both are guarded by ISPC_LLVM_24_0. Intermediate\ndebug dumps (--debug-phase, debug module prints) are not renumbered,\nbecause LLVM says renumbering is only for final output.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-11-06T11:59:41Z",
          "url": "https://github.com/ispc/ispc/commit/ae2ae73ba80eccc3b5b989188a8d8124c51663ff"
        },
        "date": 1793988264012,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test/4096",
            "value": 224261433.99206918,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 222007134.00000006 ns\nthreads: 1"
          },
          {
            "name": "16_dispatcher_overhead/dispatcher_overhead_test_multithread/10000",
            "value": 1425856543.9959056,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 211639769.99999967 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}