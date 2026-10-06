// Verify that dispatch-module dependency output preserves -MT/-MQ semantics.
// RUN: %{python} "%S/Inputs/deps-escaping.py" %{ispc} "%t.dir" sse4-i32x4,avx2-i32x8
// REQUIRES: X86_ENABLED
