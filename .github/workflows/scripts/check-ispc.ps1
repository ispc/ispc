$env:Path += ";$pwd\build\bin\Release"
check_isa.exe
ispc.exe --support-matrix
cmake --build build --target check-all --config ${env:BUILD_TYPE}
