#! /usr/bin/env python

import caffe


def main():

    caffe.set_device(0)
    caffe.set_mode_gpu()

    caffe_train()
    #caffe_test()


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

    # forward/backward pass with weight update
    # solver.step(20)

    # print 'solver.iter = ', solver.iter
    # print 'solver.iter = ', solver.

    # run the solver until the last iteration
    solver.solve()
    print 'solver.iter = ', solver.iter

    # solver.net.forward()
    # solver.test_nets[0].forward()
    #    solver.solve()
    # solver.step(2000)


def caffe_test():

    # load a trained model
    net = caffe.Net('val.prototxt', 'model_iter_100.caffemodel', caffe.TEST)

    accuracy = 0
    iterations_test = 1000

    for i in range(iterations_test):

        # one iteration
        outputs = net.forward()
        accuracy += net.blobs['accuracy'].data
        #print outputs
        #print 'loss = ', net.blobs['loss'].data
        #print 'accuracy = ', net.blobs['accuracy'].data

    avg_accuracy = accuracy / iterations_test

    print '------------------------------'
    print '   accuracy = ', avg_accuracy
    print '------------------------------'


if __name__ == "__main__":
    main()
