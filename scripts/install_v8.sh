WD=`pwd`
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
export PATH=$WD/depot_tools:"$PATH"
fetch v8 &&  \
cd v8 &&  \
gclient sync &&  \
./build/install-build-deps.sh &&  \
./tools/dev/v8gen.py x64.release &&  \
ninja -j12 -C out.gn/x64.release &&  \
export PATH=$WD/v8/out.gn/x64.release:"$PATH"

cd $WD
