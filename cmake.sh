#!/bin/sh
mkdir -p build
cp cmake/config.cmake build/
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j4

