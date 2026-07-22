; Verifies the RestoreInlineAttrPass behavior in isolation:
;   - The custom `ispc-defer-alwaysinline` string attribute is removed.
;   - `alwaysinline` is added in its place — unless `noinline` is also present,
;     in which case `noinline` wins and `alwaysinline` is not added.
;   - Functions without `ispc-defer-alwaysinline` are left untouched.

; RUN: %{ispc-opt} --passes=restore-inline-attr %s -o - | FileCheck %s

; CHECK-LABEL: define void @forced()
; CHECK-SAME: #[[FORCED:[0-9]+]]
define void @forced() #0 {
  ret void
}

; CHECK-LABEL: define void @forced_but_noinline()
; CHECK-SAME: #[[NOINLINE:[0-9]+]]
define void @forced_but_noinline() #1 {
  ret void
}

; CHECK-LABEL: define void @plain()
; CHECK-SAME: #[[PLAIN:[0-9]+]]
define void @plain() #2 {
  ret void
}

attributes #0 = { "ispc-defer-alwaysinline" }
attributes #1 = { noinline "ispc-defer-alwaysinline" }
attributes #2 = { nounwind }

; "ispc-defer-alwaysinline" must not survive on any function. alwaysinline must be
; added on @forced but NOT on @forced_but_noinline (which keeps noinline).
; CHECK-NOT: ispc-defer-alwaysinline
; CHECK-DAG: attributes #[[FORCED]] = { alwaysinline }
; CHECK-DAG: attributes #[[NOINLINE]] = { noinline }
; CHECK-DAG: attributes #[[PLAIN]] = { nounwind }
