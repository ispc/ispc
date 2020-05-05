set WD=%cd%
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
git pull
call emsdk.bat install latest
call emsdk.bat activate latest
cd %WD%
