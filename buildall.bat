@echo off

REM If LLVM_INSTALL_DIR isn't set globally in your environment,
REM it can be set here_
set LLVM_INSTALL_DIR=c:\users\mmp\llvm-dev

REM Both the LLVM binaries and python need to be in the path
set path=%LLVM_INSTALL_DIR%\bin;%PATH%;c:\cygwin\bin

msbuild ispc.vcxproj /m:4 /V:m /p:Platform=Win32 /p:Configuration=Release

msbuild examples\examples.sln /m:4 /V:m /p:Platform=x64 /p:Configuration=Release
msbuild examples\examples.sln /m:4 /V:m /p:Platform=x64 /p:Configuration=Debug
msbuild examples\examples.sln /m:4 /V:m /p:Platform=Win32 /p:Configuration=Release
msbuild examples\examples.sln /m:4 /V:m /p:Platform=Win32 /p:Configuration=Debug


