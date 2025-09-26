define i64 @__clock() nounwind {
  %r = call i64 asm sideeffect "rdtime $0", "=r"() nounwind
  ret i64 %r
}
