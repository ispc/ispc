set WD=%cd%
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
git pull
git checkout 3.1.31
call emsdk.bat install 3.1.31
call emsdk.bat activate 3.1.31
cd %WD%
