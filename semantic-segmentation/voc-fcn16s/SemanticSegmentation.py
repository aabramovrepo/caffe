#! /usr/bin/env python

import sys

# Make sure that caffe is on the python path:
caffe_root = '../../repository/caffe/'
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, '../')

import caffe


def main():

    caffe.set_device(1)
    caffe.set_mode_gpu()

    caffe_train()
    #caffe_test()


def caffe_train():

    # pre-trained weights
    weights = './fcn16s-heavy-pascal.caffemodel'

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

    outputs = solver.net.forward()

    #outputs = solver.step(20)
    #print outputs

    return

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

    net.forward()
    #fc7 = net.blobs['fc7'].data
    #print 'fc7 = ', fc7.sum()

    return

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
