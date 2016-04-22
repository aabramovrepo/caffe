
import caffe
import numpy as np
import scipy as sp
import yaml

class DataAugmentationLayer(caffe.Layer):

    def setup(self, bottom, top):
        print '--- DataAugmentationLayer: setUp'
        self.num = yaml.load(self.param_str)["num"]
        print "Parameter num : ", self.num

    def reshape(self, bottom, top):
        print '--- DataAugmentationLayer: reshape'

        print 'bottom[0].data = ', bottom[0].data.shape
        top[0].reshape(*bottom[0].shape)
        print 'top[0].data = ', top[0].data.shape

    def forward(self, bottom, top):
        print '--- DataAugmentationLayer: forward'

        print '---'
        print 'bottom[0].data.shape = ', bottom[0].data[0].shape
        print bottom[0].data[0]
        sp.misc.imsave('/home/alexey/augmentation/1.png', [bottom[0].data[0][2,:,:],bottom[0].data[0][1,:,:],bottom[0].data[0][0,:,:]])

        #img = np.zeros(bottom[0].data[0].shape,dtype=float)
        #print 'img.shape = ', img.shape

        #img[0,:,:] = bottom[0].data[0][2,:,:]
        #img[1,:,:] = bottom[0].data[0][1,:,:]
        #img[2,:,:] = bottom[0].data[0][0,:,:]
        #sp.misc.imsave('/home/alexey/augmentation/2.png', img)
        #print '---'
        top[0].reshape(*bottom[0].shape)

    def backward(self, top, propagate_down, bottom):
        print '--- DataAugmentationLayer: backward'
        pass
