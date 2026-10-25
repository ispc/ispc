window.BENCHMARK_DATA = {
  "lastUpdate": 1792894433805,
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
      }
    ]
  }
}