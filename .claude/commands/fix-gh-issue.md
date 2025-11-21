Analyze and fix GitHub issue #$ARGUMENTS from https://github.com/ispc/ispc

1. Use `gh issue view` to get the issue details
2. Explain the problem and reproduce the issue using the reproducer from the GitHub issue, if available
3. Search codebase for relevant files
4. **Create fix plan and get user approval**
6. Implement the fix
7. Write regression tests in `tests/lit-tests/` and verify the fix. Use `ispc-lit-tests` skill.
8. Run `agent-code-review` to review the changes and address its feedback
9. Run `clang-format -i` on modified C/C++/header files
10. Commit with message: `Fix #$ARGUMENTS: <summary>`
11. Show results and next steps

Use the GitHub CLI (`gh`) for all GitHub-related tasks.
Stop and ask if anything is unclear or fails.
