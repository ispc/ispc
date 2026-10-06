// Actual filenames containing tabs, backslashes, and colons are POSIX-only.
// Literal tabs in RUN and CHECK lines are intentional.
// UNSUPPORTED: WINDOWS_HOST
// RUN: mkdir -p "%t.dir"
// RUN: cp "%s" "%t.dir/source.ispc"
// RUN: cd "%t.dir"
//
// RUN: cp source.ispc 'tab	name.ispc'
// RUN: %{ispc} 'tab	name.ispc' --target=host --nostdlib --nowrap -M | FileCheck "%s" --check-prefix=TAB --strict-whitespace --match-full-lines
// RUN: %{ispc} 'tab	name.ispc' --target=host --nostdlib --nowrap -M -MT out.o | FileCheck "%s" --check-prefix=TAB-SOURCE --strict-whitespace --match-full-lines
// RUN: %{ispc} 'tab	name.ispc' --target=host --nostdlib --nowrap -M -MF deps.d -MQ 'out file.o'
// RUN: FileCheck "%s" --check-prefix=TAB-DEPFILE --strict-whitespace --match-full-lines --input-file=deps.d
// RUN: echo '#include "tab	name.ispc"' > include.ispc
// RUN: %{ispc} include.ispc --target=host --nostdlib --nowrap -M -MT out.o | FileCheck "%s" --check-prefix=TAB-INCLUDE --strict-whitespace --match-full-lines
//
// RUN: cp source.ispc 'back\slash.ispc'
// RUN: %{ispc} 'back\slash.ispc' --target=host --nostdlib --nowrap -M | FileCheck "%s" --check-prefix=BACKSLASH --strict-whitespace --match-full-lines
// RUN: %{ispc} 'back\slash.ispc' --target=host --nostdlib --nowrap -M -MT out.o | FileCheck "%s" --check-prefix=BACKSLASH-SOURCE --strict-whitespace --match-full-lines
// RUN: %{ispc} 'back\slash.ispc' --target=host --nostdlib --nowrap -M -MF deps.d -MQ 'out file.o'
// RUN: FileCheck "%s" --check-prefix=BACKSLASH-DEPFILE --strict-whitespace --match-full-lines --input-file=deps.d
// RUN: echo '#include "back\slash.ispc"' > include.ispc
// RUN: %{ispc} include.ispc --target=host --nostdlib --nowrap -M -MT out.o | FileCheck "%s" --check-prefix=BACKSLASH-INCLUDE --strict-whitespace --match-full-lines
//
// RUN: cp source.ispc 'back\ space.ispc'
// RUN: %{ispc} 'back\ space.ispc' --target=host --nostdlib --nowrap -M | FileCheck "%s" --check-prefix=BACKSLASH-SPACE --strict-whitespace --match-full-lines
// RUN: %{ispc} 'back\ space.ispc' --target=host --nostdlib --nowrap -M -MT out.o | FileCheck "%s" --check-prefix=BACKSLASH-SPACE-SOURCE --strict-whitespace --match-full-lines
// RUN: %{ispc} 'back\ space.ispc' --target=host --nostdlib --nowrap -M -MF deps.d -MQ 'out file.o'
// RUN: FileCheck "%s" --check-prefix=BACKSLASH-SPACE-DEPFILE --strict-whitespace --match-full-lines --input-file=deps.d
// RUN: echo '#include "back\ space.ispc"' > include.ispc
// RUN: %{ispc} include.ispc --target=host --nostdlib --nowrap -M -MT out.o | FileCheck "%s" --check-prefix=BACKSLASH-SPACE-INCLUDE --strict-whitespace --match-full-lines
//
// RUN: cp source.ispc 'colon:name.ispc'
// RUN: %{ispc} 'colon:name.ispc' --target=host --nostdlib --nowrap -M | FileCheck "%s" --check-prefix=COLON --strict-whitespace --match-full-lines
// RUN: %{ispc} 'colon:name.ispc' --target=host --nostdlib --nowrap -M -MT out.o | FileCheck "%s" --check-prefix=COLON-SOURCE --strict-whitespace --match-full-lines
// RUN: %{ispc} 'colon:name.ispc' --target=host --nostdlib --nowrap -M -MF deps.d -MQ 'out file.o'
// RUN: FileCheck "%s" --check-prefix=COLON-DEPFILE --strict-whitespace --match-full-lines --input-file=deps.d
// RUN: echo '#include "colon:name.ispc"' > include.ispc
// RUN: %{ispc} include.ispc --target=host --nostdlib --nowrap -M -MT out.o | FileCheck "%s" --check-prefix=COLON-INCLUDE --strict-whitespace --match-full-lines

// TAB:tab\	name.o: tab\	name.ispc
// TAB-SOURCE:out.o: tab\	name.ispc
// TAB-DEPFILE:out\ file.o: tab\	name.ispc
// TAB-INCLUDE:out.o: include.ispc \
// TAB-INCLUDE-NEXT: ./tab\	name.ispc

// BACKSLASH:back\slash.o: back\slash.ispc
// BACKSLASH-SOURCE:out.o: back\slash.ispc
// BACKSLASH-DEPFILE:out\ file.o: back\slash.ispc
// BACKSLASH-INCLUDE:out.o: include.ispc \
// BACKSLASH-INCLUDE-NEXT: ./back\slash.ispc

// BACKSLASH-SPACE:back\\\ space.o: back\\\ space.ispc
// BACKSLASH-SPACE-SOURCE:out.o: back\\\ space.ispc
// BACKSLASH-SPACE-DEPFILE:out\ file.o: back\\\ space.ispc
// BACKSLASH-SPACE-INCLUDE:out.o: include.ispc \
// BACKSLASH-SPACE-INCLUDE-NEXT: ./back\\\ space.ispc

// COLON:colon\:name.o: colon\:name.ispc
// COLON-SOURCE:out.o: colon\:name.ispc
// COLON-DEPFILE:out\ file.o: colon\:name.ispc
// COLON-INCLUDE:out.o: include.ispc \
// COLON-INCLUDE-NEXT: ./colon\:name.ispc

uniform int value;
