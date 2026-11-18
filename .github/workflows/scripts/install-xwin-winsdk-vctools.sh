#!/bin/bash -e

# Fetch xwin to get Windows VcTools and SDK (windows cross-compile on linux or macos).
# xwin older than 0.10.0 fails on the current Microsoft manifest with
# "unable to find x86_64 headers for Win11SDK_*".
XWIN_VERSION=0.10.0
mkdir xwin-install
cd xwin-install
if [[ $OSTYPE == 'darwin'* ]]; then
    # xwin does not publish macOS binaries after 0.6.6-rc.2, so build it from crates.io.
    cargo install --locked --root . --version "$XWIN_VERSION" xwin
    XWIN=./bin/xwin
else
    XWIN_URL="https://github.com/Jake-Shadle/xwin/releases/download/${XWIN_VERSION}/xwin-${XWIN_VERSION}-x86_64-unknown-linux-musl.tar.gz"
    wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 -O xwin.tar.gz "$XWIN_URL"
    tar --strip-components=1 -xf xwin.tar.gz
    XWIN=./xwin
fi
$XWIN --accept-license splat --output ../winsdk
cd ..
rm -rf xwin-install
