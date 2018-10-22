#!/bin/sh
# install tvm package for the current user
# NOTE: if you installed python via homebrew, --user is not needed during installaiton
#       it will be automatically installed to your user directory.
#       providing --user flag may trigger error during installation in such case.

export MACOSX_DEPLOYMENT_TARGET=10.9  # This is required for mac to avoid symbol conflicts with libstdc++
#cd python; python setup.py install --user; cd ..
cd python; python setup.py install; cd ..
#cd topi/python; python setup.py install --user; cd ../..
cd topi/python; python setup.py install; cd ../..
#cd nnvm/python; python setup.py install --user; cd ../..
cd nnvm/python; python setup.py install; cd ../..
