// Verify Make dependency quoting with raw -MT, quoted -MQ, and implicit targets.
// Literal tabs in RUN and CHECK lines are intentional. Strict whitespace checks
// distinguish a tab from spaces and preserve backslashes before whitespace.
// RUN: mkdir -p "%t.dir"
// RUN: cp "%s" "%t.dir/source.ispc"
// RUN: cd "%t.dir"
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MT '$(OBJDIR)/one.o two.o' | FileCheck "%s" --check-prefix=RAW --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ '$(OBJDIR)/one.o two.o' | FileCheck "%s" --check-prefix=QUOTED --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MT 'C:\some path\out#$.o	two.o' | FileCheck "%s" --check-prefix=RAW-SPECIAL --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ unused -MT '$(OBJDIR)/one.o two.o' | FileCheck "%s" --check-prefix=RAW --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MT unused -MQ '$(OBJDIR)/one.o two.o' | FileCheck "%s" --check-prefix=QUOTED --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ 'space name.o' | FileCheck "%s" --check-prefix=SPACE --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ 'tab	name.o' | FileCheck "%s" --check-prefix=TAB --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ 'back\slash.o' | FileCheck "%s" --check-prefix=BACKSLASH --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ 'two\\slashes.o' | FileCheck "%s" --check-prefix=TWO-BACKSLASHES --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ 'back\ space.o' | FileCheck "%s" --check-prefix=BACKSLASH-SPACE --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ 'two\\ space.o' | FileCheck "%s" --check-prefix=TWO-BACKSLASHES-SPACE --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ 'back\	name.o' | FileCheck "%s" --check-prefix=BACKSLASH-TAB --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ 'hash#dollar$.o' | FileCheck "%s" --check-prefix=HASH-DOLLAR --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ 'C:/some path/out.o' | FileCheck "%s" --check-prefix=DRIVE-FORWARD --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MQ 'C:\some path\out.o' | FileCheck "%s" --check-prefix=DRIVE-BACKSLASH --strict-whitespace --match-full-lines
// RUN: cp source.ispc 'space #$.ispc'
// RUN: %{ispc} 'space #$.ispc' --target=host --nostdlib --nowrap -M | FileCheck "%s" --check-prefix=IMPLICIT --strict-whitespace --match-full-lines
// RUN: %{ispc} 'space #$.ispc' --target=host --nostdlib --nowrap -M -MT out.o | FileCheck "%s" --check-prefix=SOURCE --strict-whitespace --match-full-lines
// RUN: %{ispc} 'space #$.ispc' --target=host --nostdlib --nowrap -M -MF deps.d -MQ 'out file.o'
// RUN: FileCheck "%s" --check-prefix=DEPFILE --strict-whitespace --match-full-lines --input-file=deps.d
// RUN: echo '#include "space #$.ispc"' > include.ispc
// RUN: %{ispc} include.ispc --target=host --nostdlib --nowrap -M -MT out.o | FileCheck "%s" --check-prefix=INCLUDE --strict-whitespace --match-full-lines
// RUN: %{ispc} source.ispc --target=host --nostdlib --nowrap -M -MF deps.d -o 'out #$.o'
// RUN: FileCheck "%s" --check-prefix=OUTPUT --strict-whitespace --match-full-lines --input-file=deps.d
// RUN: %{ispc} 'space #$.ispc' --target=host --nostdlib --nowrap -MMM flat.d
// RUN: FileCheck "%s" --check-prefix=FLAT --strict-whitespace --match-full-lines --input-file=flat.d
// RUN: not %{ispc} source.ispc --target=host --nostdlib --nowrap -MQ 2>&1 | FileCheck "%s" --check-prefix=MISSING

// RAW:$(OBJDIR)/one.o two.o: source.ispc
// QUOTED:$$(OBJDIR)/one.o\ two.o: source.ispc
// RAW-SPECIAL:C:\some path\out#$.o	two.o: source.ispc
// SPACE:space\ name.o: source.ispc
// TAB:tab\	name.o: source.ispc
// BACKSLASH:back\slash.o: source.ispc
// TWO-BACKSLASHES:two\\slashes.o: source.ispc
// BACKSLASH-SPACE:back\\\ space.o: source.ispc
// TWO-BACKSLASHES-SPACE:two\\\\\ space.o: source.ispc
// BACKSLASH-TAB:back\\\	name.o: source.ispc
// HASH-DOLLAR:hash\#dollar$$.o: source.ispc
// DRIVE-FORWARD:C\:/some\ path/out.o: source.ispc
// DRIVE-BACKSLASH:C\:\some\ path\out.o: source.ispc
// IMPLICIT:space\ \#$$.o: space\ \#$$.ispc
// SOURCE:out.o: space\ \#$$.ispc
// DEPFILE:out\ file.o: space\ \#$$.ispc
// INCLUDE:out.o: include.ispc \
// INCLUDE-NEXT: .{{[\\/]}}space\ \#$$.ispc
// OUTPUT:out\ \#$$.o: source.ispc
// FLAT:space #$.ispc
// MISSING:No target name specified after -MQ option.

uniform int value;
