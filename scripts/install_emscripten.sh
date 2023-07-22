WD=`pwd`
git clone https://github.com/emscripten-core/emsdk.git && \
cd emsdk
git pull && \
./emsdk install 3.1.17 && \
./emsdk activate 3.1.17  && \
source ./emsdk_env.sh
cd $WD
