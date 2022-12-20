/*
  Copyright (c) 2022-2023, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "DebugPass.h"

namespace ispc {

char DebugPass::ID = 0;

bool DebugPass::runOnModule(llvm::Module &module) {
    fprintf(stderr, "%s", str_output);
    fflush(stderr);
    module.print(llvm::errs(), nullptr);
    return true;
}

llvm::Pass *CreateDebugPass(char *output) { return new DebugPass(output); }

char DebugPassFile::ID = 0;

/**
 * Strips all non-alphanumeric characters from given string.
 */
static std::string lSanitize(std::string in) {
    llvm::Regex r("[^[:alnum:]]");
    while (r.match(in))
        in = r.sub("", in);
    return in;
}

void DebugPassFile::run(llvm::Module &module, bool init) {
    std::ostringstream oss;
    oss << (init ? "init_" : "ir_") << pnum << "_" << lSanitize(std::string{pname}) << ".ll";

    const std::string pathFile{oss.str()};

#ifdef ISPC_HOST_IS_WINDOWS
    const std::string pathSep{"\\"};
#else
    const std::string pathSep{"/"};
#endif // ISPC_HOST_IS_WINDOWS

    std::string pathDirFile;

    if (!pdir.empty()) {
        llvm::sys::fs::create_directories(pdir);
        pathDirFile = pdir + pathSep + pathFile;
    } else {
        pathDirFile = pathFile;
    }

    std::error_code EC;
    llvm::raw_fd_ostream OS(pathDirFile, EC, llvm::sys::fs::OF_None);
    Assert(!EC && "IR dump file creation failed!");
    module.print(OS, 0);
}

bool DebugPassFile::runOnModule(llvm::Module &module) {
    run(module, false);
    return true;
}

bool DebugPassFile::doInitialization(llvm::Module &module) {
    run(module, true);
    return true;
}

llvm::Pass *CreateDebugPassFile(int number, llvm::StringRef name, std::string dir) {
    return new DebugPassFile(number, name, dir);
}

} // namespace ispc
