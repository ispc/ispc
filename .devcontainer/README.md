Codespaces - ISPC developer's environment (CPU only)
====================================================

This is the minimal environment that enables CPU-only ISPC build (i.e. GPU backend is not enabled).

To build ISPC and run the tests do the following:
```bash
# `build` folder already contains result of `cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
cd build
# build ISPC and examples
make -j4
# run LIT tests
make check-all
```

Note that `build` folder already contains `compile_commands.json`, which is needed for VSCode to be able correctly parse and browse C++ files.

The container also contains pre-installed [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack) for VSCode.

Enjoy and please let us know if you use Codespaces to hack ISPC and have any suggestions or feedback: [Github Discussions](https://github.com/ispc/ispc/discussions)
