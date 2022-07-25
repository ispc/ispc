#!/bin/bash -x

cd $GITHUB_WORKSPACE
# Download embree if we're on the HW with Ray Tracing capabilities
if [ "$RAY_TRACING_HW" != "OFF" ]; then
  curl --connect-timeout 5 --max-time 3600 --retry 5 --retry-delay 0 --retry-max-time 40 --fail -H "X-JFrog-Art-Api:$ARTIFACTORY_ISPC_API_KEY" $EMBREE_URL --output embree.tar.gz
  tar -xvf embree.tar.gz
  export EMBREE_INSTALL_DIR=$GITHUB_WORKSPACE/install
fi
git clone https://$ACCESS_TOKEN@github.com/intel-sandbox/aneshlya.ispc-dpcpp-interop ispc-dpcpp-interop
cd ispc-dpcpp-interop
git checkout $INTEROP_BRANCH
./run_tests.sh

