@echo off

REM If LLVM_INSTALL_DIR isn't set globally in your environment,
REM it can be set here_
set LLVM_INSTALL_DIR=c:\users\mmp\llvm-dev
set LLVM_VERSION=3.1svn

REM Both the LLVM binaries and python need to be in the path
set path=%LLVM_INSTALL_DIR%\bin;%PATH%;c:\cygwin\bin

msbuild ispc.vcxproj /V:m /p:Platform=Win32 /p:Configuration=Release
