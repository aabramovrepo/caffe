
import caffe

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import random
import pascal_voc_dict


class VOCSegDataLayer_Alexey(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL VOC semantic segmentation.

        example

        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        
        print ' voc_dir = ', self.voc_dir
        print ' split = ', self.split
        print ' mean = ', self.mean

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.voc_dir,
                self.split)
        
        print 'split_f = ', split_f
        
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/SegmentationClass/{}.png'.format(self.voc_dir, idx))
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label


class SBDDSegDataLayer_Alexey(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.sbdd_dir = params['sbdd_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        
        print ' '
        print 'self.sbdd_dir = ', self.sbdd_dir
        print 'self.split = ', self.split
        print 'self.mean = ', self.mean
        print 'self.random = ', self.random
        print 'self.seed = ', self.seed
        print ' '

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.sbdd_dir,self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0
        
        print 'split_f = ', split_f
        print 'self.indices (number of training samples) = ', len(self.indices)
        print ' '

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        
#        print ''
#        print 'reshape: load image and label ...'
        
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        
#        print 'data shape = ', self.data.shape
#        print 'label shape = ', self.label.shape
#        print 'labels = ', np.unique(self.label)

        #fig = plt.figure()
        #plt.title('Input image', color='white')
        #plt.axis('off')
        
        #input_img = np.zeros((500,330,3), dtype=np.uint8)
        #input_img[:,:,0] = self.data[0,:,:]
        #input_img[:,:,1] = self.data[1,:,:]
        #input_img[:,:,2] = self.data[2,:,:]

        #plt.imshow(input_img)
        #plt.savefig('input.png', facecolor='black')

        #fig = plt.figure()
        #plt.title('Labels', color='white')
        #plt.axis('off')
        
        #input_labels = np.zeros((500,330), dtype=np.uint8)
        #input_labels[:,:,0] = self.label[0,:,:]

        #plt.imshow(input_labels)
        #plt.savefig('labels.png', facecolor='black')
        
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):

#        print ''
#        print 'forward: ...'

        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
#        if self.random:
#            self.idx = random.randint(0, len(self.indices)-1)
#        else:
#            self.idx += 1
#            if self.idx == len(self.indices):
#                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        print ''
        print 'backward: ...'

        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """

        print ''
        print 'load_image: ', '{}/img/{}.jpg'.format(self.sbdd_dir, idx)
        
        im = Image.open('{}/img/{}.jpg'.format(self.sbdd_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        
        # save input image
        #self.save_image(in_)
        
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        
        print ''
        print 'load_label: ', '{}/cls/{}.mat'.format(self.sbdd_dir, idx)
        
        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        
        # save input segments
        #self.save_segments(label)
        
        label = label[np.newaxis, ...]
        return label
    
    
    def save_image(self, img):
        """Only for debugging purposes: saving input image"""
    
        # plot input image
        plt.figure()
        plt.title('Input image')
        input_img = np.zeros(img.shape, dtype=np.uint8)
        input_img[...] = img[...]
        plt.imshow(input_img)
        plt.savefig('input-data.png', facecolor='black')


    def save_segments(self, label):
        """Only for debugging purposes: saving input segments"""

        segments = np.zeros((label.shape[0],label.shape[1],3), dtype=np.uint8)
        
        for obj_id in list(np.unique(label)):
            segments[:,:,0] += np.uint8(pascal_voc_dict.pascal_voc_2011[obj_id][1][0]) * (label == obj_id)
            segments[:,:,1] += np.uint8(pascal_voc_dict.pascal_voc_2011[obj_id][1][1]) * (label == obj_id)
            segments[:,:,2] += np.uint8(pascal_voc_dict.pascal_voc_2011[obj_id][1][2]) * (label == obj_id)

        # plot input segments
        plt.figure()
        plt.title('Input segments')
        plt.imshow(segments)
        plt.savefig('input-segments.png', facecolor='black')
