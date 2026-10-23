window.BENCHMARK_DATA = {
  "lastUpdate": 1792777796873,
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
      }
    ]
  }
}