WD=`pwd`
git clone https://github.com/emscripten-core/emsdk.git && \
cd emsdk
git pull && \
./emsdk install 4.0.7 && \
./emsdk activate 4.0.7  && \
source ./emsdk_env.sh
cd $WD
