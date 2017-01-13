#
# Alexey Abramov <alexey.abramov@continental-corporation.com>
#
# Python layer used for the data augmentation
#
# Data augmentation is randomly applied or not to a sample. The batch is not extended by modified versions,
# the samples are being modified in place. Over the long term (doing multiple epochs) the network sees both
# augmented and non-augmented versions of the sample. That is also the way the built-in augmentation (mirroring) works.
#

# images in the network appear in the following format: <batch_size N><channels K><height H><width W>


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
        #self.mean_img = cv2.imread('mean-image-scaled.png',cv2.CV_LOAD_IMAGE_COLOR)
        
        #print 'self.augmentation = ', self.augmentation
        #print 'self.mean_val = ', self.mean_val


    def reshape(self, bottom, top):

        #print
        #print '--- DataAugmentationLayer: reshape'
        #print

        top[0].reshape(*bottom[0].shape)
        top[0].data[...] = bottom[0].data[...]


    def forward(self, bottom, top):

        #print
        #print '--- DataAugmentationLayer: forward'
        #print
        
        #print 'bottom shape = ', bottom[0].data[...].shape        
        #cv2.imwrite('bottom-0.png', bottom[0].data[0][...].transpose((1,2,0)))
        
        if self.augmentation:
            
            #print '---> bottom data shape = ', bottom[0].data.shape[0]
            
            if random.random() > 0.2:
                # do data augmentation for the whole batch
                for ind in range(bottom[0].data.shape[0]):
                    
                    # randomly jitter contrast
                    bottom[0].data[ind][...] *= random.uniform(0.2,3.0)
                    bottom[0].data[ind][...][(bottom[0].data[ind][...] > 255.).nonzero()] = 255.
                    
                    #cv2.imwrite('augmentation/' + str(ind) + '-0.png', bottom[0].data[ind][...].transpose((1,2,0)))
                    
                    # horizontal flip
                    #bottom[0].data[ind][0,:,:] = np.fliplr(bottom[0].data[ind][0,:,:])
                    #bottom[0].data[ind][1,:,:] = np.fliplr(bottom[0].data[ind][1,:,:])
                    #bottom[0].data[ind][2,:,:] = np.fliplr(bottom[0].data[ind][2,:,:])
                    
                    #cv2.imwrite('augmentation/' + str(ind) + '-1.png', bottom[0].data[ind][...].transpose((1,2,0)))

        # subtract the image mean for each image in the batch
        #for ind in range(bottom[0].data.shape[0]):
        #    bottom[0].data[ind][0, :, :] -= self.mean_img[:, :, 0]
        #    bottom[0].data[ind][1, :, :] -= self.mean_img[:, :, 1]
        #    bottom[0].data[ind][2, :, :] -= self.mean_img[:, :, 2]

        top[0].reshape(*bottom[0].shape)
        top[0].data[...] = bottom[0].data[...]


    def backward(self, top, propagate_down, bottom):

        print '--- DataAugmentationLayer: backward'
        pass

