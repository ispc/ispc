[![License](https://img.shields.io/github/license/ispc/ispc)](https://github.com/ispc/ispc)

Welcome to Intel® ISPC contributing guide
=========================================

Thank you for your interest in contributing to our project!

In this guide you will get an overview of the contribution workflow from opening an issue to merging the PR.

Here are some important resources:
- To get an overview of the project, read the [README](README.md).
- Review our plans for the upcoming releases in [ISPC project](https://github.com/orgs/ispc/projects/)
- Join our [developer list](https://groups.google.com/g/ispc-dev)

How to contribute
-----------------
### Create an issue

If you spot an issue, ensure the bug was not already reported by searching on [ISPC Issues](https://github.com/ispc/ispc/issues)

If you're unable to find an open issue addressing the problem, open a new one. Be sure to include:
- Steps to reproduce the issue including your environment (operating system, ISPC version, LLVM version).
- Code snippet or an executable test case demonstrating the expected behavior that is not occurring.

If it is possible, create a reproducer using [Compiler Explorer](https://godbolt.org/) and just paste the link to the issue.

### Solve an issue

Scan through our [existing issues](https://github.com/ispc/ispc/issues) to find one that interests you. You can narrow down the search
using `labels` as filters. If you don't know where to start use issues with [good first issue](https://github.com/ispc/ispc/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
label.

### Make changes

Review our [ISPC Development Guide](https://github.com/ispc/ispc/wiki/ISPC-Development-Guide) for build and test instructions.

Note, that every compiler change should be covered by [lit test](https://github.com/ispc/ispc/wiki/ISPC-Development-Guide#ISPC_lit_tests)
Also look into [existing lit tests](https://github.com/ispc/ispc/tree/main/tests/lit-tests) to learn how to write them.

If you made a language change new [functional tests](https://github.com/ispc/ispc/tree/main/tests) checking compiler behaviour in runtime are needed.

### Submitting PR

Follow [Working with forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks) approach to submit a PR to ISPC.
Describe the change you've made in the PR description, put an issue number if exists.

Before submitting a PR, format the changed .cpp/.c/.h files with `clang-format` (included into LLVM distribution). For example:
`clang-format -i src/ispc.cpp`.

Squash git history to meaningful commits. One commit is responsible for one change.
