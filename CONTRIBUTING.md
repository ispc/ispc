[![License](https://img.shields.io/github/license/ispc/ispc)](https://github.com/ispc/ispc)

Welcome to Intel® ISPC contributing guide
=========================================

Thank you for your interest in contributing to our project!

We are looking for all kinds of contributions ranging from [submitting a bug](https://github.com/ispc/ispc/issues) or
participating in the [discussions](https://github.com/ispc/ispc/discussions) to submitting a bug fix, a feature or
maintaining part of the compiler.

In this guide you will get an overview of the contribution workflow from opening an issue to merging the PR.

Here are some important resources:
- To get an overview of the project, read the [README](README.md).
- Review our plans for the upcoming releases in [ISPC project](https://github.com/orgs/ispc/projects/1).
- Submit an [issue](https://github.com/ispc/ispc/issues) or participate in
  [discussions](https://github.com/ispc/ispc/discussions) on GitHub.
- Check [Wiki pages](https://github.com/ispc/ispc/wiki) for additional information on development process, setting up
  environment instructions, etc.
- Use [Compiler Explorer](https://godbolt.org/) to interactively compile ISPC program.

How to contribute
-----------------
### Create an issue

[File an issue](https://github.com/ispc/ispc/issues/new) if you:
- Spotted a stability or performance bug.
- Would like to request a feature or suggest an improvement.

Before filing a new issue search through existing ones at [ISPC Issues](https://github.com/ispc/ispc/issues). If the
issue exists, give it a like, subscribe for updates, contribute to investigation or add details about your specific
case.

When creating an issue, try to be specific and give all the relevant details. If this is a bug, provide steps to
reproduce, ISPC version and information about your environment that is relevant to the problem. Try to have a minimal
reproducer, so it is clear where the problem is. An ideal reproducer is a code snippet pasted to the issue with the
command line to compile and execute it. If it's duplicated by the link to [Compiler Explorer](https://godbolt.org/) - it
is even better. If it's a feature request, describe what is needed and why you need it or think it is useful.

If the bug or the feature is important to you, for example it blocks something in your project or affects how/where you
use ISPC - don't hesitate to mention it, this will help prioritizing the bug higher.

### Start a discussion

If you have a question about ISPC, need help or advice, then starting a
[Discussion](https://github.com/ispc/ispc/discussions) is the best way proceed.

If you have a success story applying ISPC in your project, get expected (or unexpected!) performance gain due to using
ISPC - please post it in Discussions as well, [Show and tell](https://github.com/ispc/ispc/discussions/categories/show-and-tell)
category is there for you. If you evaluated ISPC and it didn't work for you for some reason and you think it worth
sharing your story, you are also very welcome. We love feedback and willing to improve!

### Find an issue to fix

Scan through our [existing issues](https://github.com/ispc/ispc/issues) to find one that interests you. You can narrow
down the search using `labels` as filters. If you don't know where to start use issues with
[good first issue](https://github.com/ispc/ispc/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) label.

Feel free to comment on the issues asking for guidance or help with the fix.

### Make changes

Review our [ISPC Development Guide](https://github.com/ispc/ispc/wiki/ISPC-Development-Guide) for build and test instructions.

Note, that every compiler change should be covered by [lit test](https://github.com/ispc/ispc/wiki/ISPC-Development-Guide#ISPC_lit_tests)
Also look into [existing lit tests](https://github.com/ispc/ispc/tree/main/tests/lit-tests) to learn how to write them.

If you made a language change new [functional tests](https://github.com/ispc/ispc/tree/main/tests) checking compiler
behavior in runtime are needed.

### Submitting PR

Follow [Working with forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks)
approach to submit a PR to ISPC.  Describe the change you've made in the PR description, put an issue number if exists.

Before submitting a PR, format the changed .cpp/.c/.h files with `clang-format` (included into LLVM distribution). For
example: `clang-format -i src/ispc.cpp`.

Squash git history to meaningful commits. One commit is responsible for one change for easier review.


### Contribute a feature or maintain a target

We are looking for individual, academic and institutional contributors who are interested in more significant
contributions than a single fix as well. The possible list of topic include:
- Fine-tuning existing targets, x86 and non-x86. Contributions to ARM targets are especially appreciated, as the core
  team doesn't have enough bandwidth to fully cover this.
- Adding and maintaining new hardware targets - Webassembly (WASM), RISC-V, ARM v9, etc.
- Adding and/or maintaining a new OS/platform - PS4/PS5 (hey Sony, I'm calling for you!), Windows on ARM, FreeBSD, etc.
- Adding new optimizations or doing an experimental implementation for a novel research idea - for example, adding a
  [superoptimization](https://en.wikipedia.org/wiki/Superoptimization) pass.

If you are a professor teaching a compiler class or advising undergraduate or graduate student, who is looking for a
topic for their thesis, feel free to use our project as a playground or a potential target for applying your research.
