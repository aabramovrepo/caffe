#
# Alexey Abramov <alexey.abramov@continental-corporation.com>
#
# Python layer used for the data augmentation
#
# Data augmentation is randomly applied or not to a sample. The batch is not extended by modified versions,
# the samples are being modified in place. Over the long term (doing multiple epochs) the network sees both
# augmented and non-augmented versions of the sample. That is also the way the built-in augmentation (mirroring) works.
#


import caffe
import numpy as np
import scipy as sp
import cv2
import random


class DataAugmentationLayer(caffe.Layer):


    def setup(self, bottom, top):

        #print '--- DataAugmentationLayer: setUp'

        params = eval(self.param_str)
        self.augmentation = np.array(params['augmentation'])
        self.mean_val = np.array(params['mean'])


    def reshape(self, bottom, top):

        #print
        #print '--- DataAugmentationLayer: reshape'
        #print

        #print 'bottom[0].data = ', bottom[0].data.shape
        top[0].reshape(*bottom[0].shape)
        #print 'top[0].data = ', top[0].data.shape

        top[0].data[...] = bottom[0].data[...]


    def forward(self, bottom, top):

        #print
        #print '--- DataAugmentationLayer: forward'
        #print

        if self.augmentation:

            if random.random() > 0.5:
                #print
                #print '=====> apply augmentation ---'
                #print

                # do data augmentation for the whole batch
                for ind in range(bottom[0].data.shape[0]):
                    bottom[0].data[ind][...] *= random.uniform(0.1, 5.)
                    bottom[0].data[ind][...][(bottom[0].data[ind][...] > 255.).nonzero()] = 255.

                    #a = np.zeros([227, 227, 3], dtype=float)
                    #a[:, :, 0] = bottom[0].data[ind][0, :, :]
                    #a[:, :, 1] = bottom[0].data[ind][1, :, :]
                    #a[:, :, 2] = bottom[0].data[ind][2, :, :]
                    #cv2.imwrite('/home/alexey/augmentation/aug-' + str(ind) + '.png', a)

        top[0].reshape(*bottom[0].shape)
        top[0].data[...] = bottom[0].data[...]


    def backward(self, top, propagate_down, bottom):

        print '--- DataAugmentationLayer: backward'
        pass
