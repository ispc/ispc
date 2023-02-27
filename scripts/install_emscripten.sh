WD=`pwd`
git clone https://github.com/emscripten-core/emsdk.git && \
cd emsdk
git pull && \
git checkout 3.1.31 && \
./emsdk install 3.1.31 && \
./emsdk activate 3.1.31 && \
source ./emsdk_env.sh
cd $WD
