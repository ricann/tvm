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

######################################################################
# Basic example
# -------------
# Let's revisit the sum of rows operation (equivalent to :code:`B = numpy.sum(A, axis=1)`') \
# To compute the sum of rows of a two dimensional TVM tensor A, we should
# specify the symbolic operation as well as schedule as follows
#
#dim = tvm.var("dim")
dim = tvm.var("dim", "int")
A = tvm.placeholder((dim), dtype="int32", name='A')
B = tvm.placeholder((dim), dtype="int32", name='B')
x = tvm.placeholder((dim), dtype="int32", name='x')

#y = tvm.compute((dim,), lambda i: A[i]*x[i]+B[i], name="B")
y = tvm.compute((dim,), lambda i: A[i]*x[i] + B[i], name="y")
s = tvm.create_schedule(y.op)
print(tvm.lower(s, [A, x, B], simple_mode=True))
module = tvm.build(s, [A, x, B, y], "llvm")

ctx = tvm.context("llvm", 0)
ctx2 = tvm.cpu(0)
ctx3 = tvm.gpu(0)

# tvm.nd.array
# a_np = tvm.nd.array(np.random.uniform(size=(4)).astype(A.dtype), ctx)
# b_np = tvm.nd.array(np.random.uniform(size=(4)).astype(B.dtype), ctx)
# x_np = tvm.nd.array(np.random.uniform(size=(4)).astype(x.dtype), ctx)
a_np = tvm.nd.array(np.array([2,2,2,2], dtype="int32"), ctx)
x_np = tvm.nd.array(np.array([3,3,3,3], dtype="int32"), ctx)
b_np = tvm.nd.array(np.array([4,4,4,4], dtype="int32"), ctx)
y_np = tvm.nd.array(np.zeros(4, dtype=y.dtype), ctx)

t_np = np.random.uniform(size=(4)).astype(A.dtype)

module(b_np, a_np, x_np, y_np)

print(y_np)

######################################################################
# and to examine the IR code in human readable format, we can do
#
#with tvm.target.create("llvm"):
#    sg = topi.generic.schedule_reduce(B)
#    print(tvm.lower(sg, [A], simple_mode=True))

