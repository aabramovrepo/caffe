
#
# Caffe Python layers for loading images and labels for both training and validation
#

import caffe

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import sys
import glob

import random

sys.path.insert(0, os.environ['CITYSCAPES_SCRIPTS'] + '/helpers')
import labels

downsample_factor=0.75


class SegDataLayerVal(caffe.Layer):
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


class SegDataLayerTrain(caffe.Layer):
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
        
        #print ' '
        #print 'self.sbdd_dir = ', self.sbdd_dir
        #print 'self.split = ', self.split
        #print 'self.mean = ', self.mean
        #print 'self.random = ', self.random
        #print 'self.seed = ', self.seed
        #print ' '

        # load indices for images and labels
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
        #search_images = os.path.join(cityscapesPath, 'leftImg8bit', 'train', 'aachen', '*_leftImg8bit.png')
        #search_labels = os.path.join(cityscapesPath, 'gtFine', 'train', 'aachen', '*_gtFine_labelTrainIds.png')

        print 'cityscapesPath = ', cityscapesPath

        search_images = os.path.join(cityscapesPath, 'leftImg8bit', 'train', '*', '*_leftImg8bit.png')
        #search_labels = os.path.join(cityscapesPath, 'gtFine', 'train', '*', '*_gtFine_labelTrainIds.png')
        search_labels = os.path.join(cityscapesPath, 'gtFine', 'train', '*', '*_gtFine_labelIds.png')
        search_disparity = os.path.join(cityscapesPath, 'disparity', 'train', '*', '*_disparity.png')

        self.idx = 0
        self.files_images = glob.glob(search_images)
        self.files_labels = glob.glob(search_labels)
        self.files_disparity = glob.glob(search_disparity)

        self.files_images.sort()
        self.files_labels.sort()
        self.files_disparity.sort()
        self.length = max(len(self.files_images),len(self.files_labels))

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, self.length-1)
            #self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        
        # load image + label image pair
        self.data = self.load_image(self.idx)
        self.label = self.load_label(self.idx)

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
        if self.random:
            self.idx = random.randint(0, self.length-1)
        else:
            self.idx += 1
            if self.idx == self.length:
                self.idx = 0


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

        #print 'image: ', self.files_images[idx]

        _im = cv2.imread('/media/ssd_drive/Cityscapes_dataset/leftImg8bit/train/monchengladbach/monchengladbach_000000_009930_leftImg8bit.png',1)
        #_im = cv2.imread(self.files_images[idx],1)
        _im_small = cv2.resize(_im, (0,0), fx=downsample_factor, fy=downsample_factor)

        im = np.asarray(_im_small)
        #im = np.asarray(_im)
        in_ = np.array(im, dtype=np.float32)

        #print 'in_.shape = ', in_.shape
        #self.save_image(in_, idx)
        
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """

        #print 'labels: ', self.files_labels[idx]
        
        _label = cv2.imread('/media/ssd_drive/Cityscapes_dataset/gtFine/train/monchengladbach/monchengladbach_000000_009930_gtFine_labelIds.png',0)
        #_label = cv2.imread(self.files_labels[idx],0)
        _label_small = cv2.resize(_label, (0,0), fx=downsample_factor, fy=downsample_factor)
        label = np.asarray(_label_small)
        #label = np.asarray(_label)

        #print 'label.shape = ', label.shape
        #print 'label unique = ', np.unique(label)
        #self.save_segments(label, idx)
        
        label = label[np.newaxis, ...]
        return label
    
    
    def save_image(self, img, idx):
        """Only for debugging purposes: saving input image"""
    
        # plot input image
        plt.figure()
        plt.title('Input image')
        input_img = np.zeros(img.shape, dtype=np.uint8)
        input_img[:,:,0] = img[:,:,2]
        input_img[:,:,1] = img[:,:,1]
        input_img[:,:,2] = img[:,:,0]
        plt.imshow(input_img)
        plt.savefig('input-data-' + str(idx) + '.png', facecolor='black')


    def save_segments(self, label, idx):
        """Only for debugging purposes: saving input segments"""

        segments = np.zeros((label.shape[0],label.shape[1],3), dtype=np.uint8)
        
        for object_id in list(np.unique(label)):
            #print 'id: ', object_id, ', name: ', labels.id2label[object_id].name, ', category: ', labels.id2label[object_id].category, ', color: ', labels.id2label[object_id].color
            segments[:,:,0] += np.uint8(labels.id2label[object_id].color[0]) * (label == object_id)
            segments[:,:,1] += np.uint8(labels.id2label[object_id].color[1]) * (label == object_id)
            segments[:,:,2] += np.uint8(labels.id2label[object_id].color[2]) * (label == object_id)

        # plot input segments
        plt.figure()
        plt.title('Input segments')
        plt.imshow(segments)
        plt.savefig('input-segments-' + str(idx) + '.png', facecolor='black')

