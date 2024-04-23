#!/bin/bash -e

# Fetch xwin to get Windows VcTools and SDK (windows cross-compile on linux or macos)
XWIN_URL="https://github.com/Jake-Shadle/xwin/releases/download/0.6.7/xwin-0.6.7-x86_64-unknown-linux-musl.tar.gz"
if [[ $OSTYPE == 'darwin'* ]]; then
    XWIN_URL="https://github.com/Jake-Shadle/xwin/releases/download/0.6.6-rc.2/xwin-0.6.6-rc.2-x86_64-apple-darwin.tar.gz"
fi
mkdir xwin-install
cd xwin-install
wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 -O xwin.tar.gz $XWIN_URL
tar --strip-components=1 -xf xwin.tar.gz
./xwin --accept-license splat --output ../winsdk
cd ..
rm -rf xwin-install
