set WD=%cd%
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
git pull
git checkout 2.0.4
call emsdk.bat install sdk-1.39.11-64bit
call emsdk.bat activate sdk-1.39.11-64bit
cd %WD%
