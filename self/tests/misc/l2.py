import numpy as np
import nnvm.compiler
import nnvm.testing

net, params = nnvm.testing.resnet.get_workload(layers=50, batch_size=1, image_shape=(3, 224, 224))
g = nnvm.graph.create(net)

