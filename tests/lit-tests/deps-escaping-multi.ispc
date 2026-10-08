// Check the same quoting cases through dispatch-module dependency output.
// REQUIRES: X86_ENABLED
// RUN: mkdir -p "%t.dir"
// RUN: cp "%s" "%t.dir/source.ispc"
// RUN: cd "%t.dir"
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MT '$(OBJDIR)/one.o two.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=RAW --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ '$(OBJDIR)/one.o two.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=QUOTED --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MT 'C:\some path\out#$.o	two.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=RAW-SPECIAL --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ unused -MT '$(OBJDIR)/one.o two.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=RAW --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MT unused -MQ '$(OBJDIR)/one.o two.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=QUOTED --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ 'space name.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=SPACE --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ 'tab	name.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=TAB --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ 'back\slash.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=BACKSLASH --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ 'two\\slashes.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=TWO-BACKSLASHES --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ 'back\ space.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=BACKSLASH-SPACE --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ 'two\\ space.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=TWO-BACKSLASHES-SPACE --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ 'back\	name.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=BACKSLASH-TAB --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ 'hash#dollar$.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=HASH-DOLLAR --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ 'C:/some path/out.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=DRIVE-FORWARD --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MQ 'C:\some path\out.o' | FileCheck "%S/deps-escaping.ispc" --check-prefix=DRIVE-BACKSLASH --strict-whitespace --match-full-lines
// RUN: cp source.ispc 'space #$.ispc'
// RUN: %{ispc} 'space #$.ispc' --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M | FileCheck "%S/deps-escaping.ispc" --check-prefix=IMPLICIT --strict-whitespace --match-full-lines
// RUN: %{ispc} 'space #$.ispc' --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MT out.o | FileCheck "%S/deps-escaping.ispc" --check-prefix=SOURCE --strict-whitespace --match-full-lines
// RUN: %{ispc} 'space #$.ispc' --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MF deps.d -MQ 'out file.o'
// RUN: FileCheck "%S/deps-escaping.ispc" --check-prefix=DEPFILE --strict-whitespace --match-full-lines --input-file=deps.d
// RUN: echo '#include "space #$.ispc"' > include.ispc
// RUN: %{ispc} include.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MT out.o | FileCheck "%S/deps-escaping.ispc" --check-prefix=INCLUDE --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -M -MF deps.d -o 'out #$.o'
// RUN: FileCheck "%S/deps-escaping.ispc" --check-prefix=OUTPUT --strict-whitespace --match-full-lines --input-file=deps.d
// RUN: %{ispc} 'space #$.ispc' --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -MMM flat.d
// RUN: FileCheck "%S/deps-escaping.ispc" --check-prefix=FLAT --strict-whitespace --match-full-lines --input-file=flat.d
// RUN: not %{ispc} source.ispc --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -MQ 2>&1 | FileCheck "%S/deps-escaping.ispc" --check-prefix=MISSING

uniform int value;
