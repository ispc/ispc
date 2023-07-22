set WD=%cd%
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
git pull
call emsdk.bat install 3.1.17
call emsdk.bat activate 3.1.17
cd %WD%
