#! /usr/bin/env python

#
# Alexey Abramov <alexey.abramov@continental-corporation.com>
#
#
#

import caffe
import glob
import cv2
import numpy as np


def main():

    #mean_data = caffe.io.  .io.read_mean('../../data/lane_markings')
    #return

    #compute_image_mean()
    #return

    caffe.set_device(1)
    caffe.set_mode_gpu()

    caffe_train()
    #caffe_test()


def compute_image_mean():

    # path with all training samples
    file_list = glob.glob('/home/alexey/repository/caffe/data/lane_markings/train/*.png')

    mean = np.zeros([256,256,3],dtype=float)
    n = 0

    for fname in file_list:
        img = cv2.imread(fname,cv2.CV_LOAD_IMAGE_COLOR)
        mean[...] += img[...]
        n += 1

        if n % 1000 == 0:
            mean[...] /= n
            n = 0

    if n != 0:
        mean[...] /= n

    cv2.imwrite('mean-image.png', mean)

    mean_scaled = cv2.resize(mean,(227,227),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('mean-image-scaled.png', mean_scaled)


def caffe_train():

    # pre-trained weights
    weights = '/home/alexey/repository/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

    # load the network
    # net = caffe.Net('conv.prototxt', caffe.TRAIN)
    # print 'network loaded ...'
    # print net.blobs['conv1'].data.shape

    # load the solver
    solver = caffe.SGDSolver('solver.prototxt')
    # solver = caffe.get_solver('solver.prototxt')


    print '------------------------------'
    print '       solver loaded !        '
    print '------------------------------'

    solver.net.copy_from(weights)

    print '------------------------------'
    print '     weights are copied!      '
    print '------------------------------'

    training_net = solver.net
    test_net = solver.test_nets[0]

    print '------------------------------'
    print '     Network layers:          '
    print '------------------------------'


    for layer_name, blob in training_net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)

    #for layer_name, blob in test_net.blobs.iteritems():
    #    print layer_name + '\t' + str(blob.data.shape)

    print '------------------------------'
    print '     Forward pass...          '
    print '------------------------------'

    #outputs = solver.net.forward()

    #outputs = solver.step(20)
    #print outputs

    # forward/backward pass with weight update
    # solver.step(20)

    # run the solver until the last iteration
    solver.solve()
    print 'last solver iteration: ', solver.iter

    # solver.net.forward()
    # solver.test_nets[0].forward()


def caffe_test():

    # load a trained model
    net = caffe.Net('val.prototxt', 'models/model_iter_5000.caffemodel', caffe.TEST)

    print '------------------------------'
    print '    Network initialized !     '
    print '------------------------------'

    #net.forward()
    #fc7 = net.blobs['fc7'].data
    #print 'fc7 = ', fc7.sum()

    accuracy = 0
    loss = 0
    iterations_test = 100

    for i in range(iterations_test):

        # one iteration (load the next mini-batch as defined in the net)
        outputs = net.forward()
        accuracy += net.blobs['accuracy'].data
        loss += net.blobs['loss'].data
        #print outputs
        #print 'loss = ', net.blobs['loss'].data
        #print 'accuracy = ', net.blobs['accuracy'].data

    avg_accuracy = accuracy / iterations_test
    avg_loss = loss / iterations_test

    print '------------------------------'
    print '   accuracy = ', avg_accuracy
    print '   loss     = ', avg_loss
    print '------------------------------'


if __name__ == "__main__":
    main()
