#! /usr/bin/env python

import sys

surgery_dir = '../'
sys.path.insert(0, surgery_dir)

import caffe
import surgery, score

import numpy as np
import os

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

#weights = '../voc-fcn16s/voc-fcn16s.caffemodel'

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_device(1)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)
#val = np.loadtxt('../data/pascal/segvalid11.txt', dtype=str)
val = np.loadtxt('../data/pascal/seg11valid.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
