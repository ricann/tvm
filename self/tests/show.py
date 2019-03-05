"""
Introduction to TOPI
====================
**Author**: `Ehsan M. Kermani <https://github.com/ehsanmok>`_

This is an introductory tutorial to TVM Operator Inventory (TOPI).
TOPI provides numpy-style generic operations and schedules with higher abstractions than TVM.
In this tutorial, we will see how TOPI can save us from writing boilerplates code in TVM.
"""
from __future__ import absolute_import, print_function

import tvm
import topi
import numpy as np

x, y = 100, 10
a = tvm.placeholder((x, y, y), name="a")
g = topi.sum(a)
with tvm.target.cuda():
#with tvm.target.create("llvm"):
    sg = topi.generic.schedule_reduce(g)
    print(tvm.lower(sg, [a], simple_mode=True))

######################################################################
# As you can see, scheduled stages of computation have been accumulated and we can examine them by
#
print(sg.stages)

