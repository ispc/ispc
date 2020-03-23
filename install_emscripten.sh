WD=`pwd`
git clone https://github.com/emscripten-core/emsdk.git && \
cd emsdk
git pull && \
./emsdk install latest && \
./emsdk activate latest && \
source ./emsdk_env.sh
cd $WD