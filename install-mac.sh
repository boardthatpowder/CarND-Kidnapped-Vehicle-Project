#!/bin/bash

brew install openssl libuv cmake
git clone https://github.com/uWebSockets/uWebSockets 
cd uWebSockets
git checkout e94b6e1
patch CMakeLists.txt < ../cmakepatch.txt
mkdir build
export PKG_CONFIG_PATH=/usr/local/opt/openssl/lib/pkgconfig 
cd build

OPENSSL_ROOT_DIR=/usr/local/Cellar/openssl/1.0.2r

cmake -DOPENSSL_ROOT_DIR=$OPENSSL_ROOT_DIR -DOPENSSL_LIBRARIES=$OPENSSL_ROOT_DIR/lib ..
make 
sudo make install
cd ..
cd ..
sudo rm -r uWebSockets
