
import sys

#caffe_root = '/home/alexey/repository/caffe'
#sys.path.insert(0, caffe_root)

surgery_dir = '../'
sys.path.insert(0, surgery_dir)

import caffe
#from caffe import surgery, score
import surgery, score

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = '../fcn32s-heavy-pascal.caffemodel'

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_device(1)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)

print 'run solver ...'
solver.step(1000)


#for _ in range(25):
#    print '---> solver step ... '
#    solver.step(4000)
#    score.seg_tests(solver, False, val, layer='score')

