import caffe

import numpy as np
from PIL import Image

import random
import os
import sys
import glob
import cv2

height_scaled = 800  # 500

sys.path.insert(0, os.environ['CITYSCAPES_SCRIPTS'] + '/helpers')

import labels


class CityscapesDataLayerVal(caffe.Layer):
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
        self.random = False #params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        cityscapesPath = os.environ['CITYSCAPES_DATASET']
        #search_images = os.path.join(cityscapesPath, 'leftImg8bit', 'val', 'frankfurt', '*_leftImg8bit.png')
        #search_labels = os.path.join(cityscapesPath, 'gtFine', 'val', 'frankfurt', '*_gtFine_labelTrainIds.png')

        search_images = os.path.join(cityscapesPath, 'leftImg8bit', 'val', '*', '*_leftImg8bit.png')
        search_labels = os.path.join(cityscapesPath, 'gtFine', 'val', '*', '*_gtFine_labelTrainIds.png')

        self.idx = 0
        self.files_images = glob.glob(search_images)
        self.files_labels = glob.glob(search_labels)

        self.files_images.sort()
        self.files_labels.sort()
        self.length = max(len(self.files_images),len(self.files_labels))

        print 'files_images = ', len(self.files_images)
        print 'files_labels = ', len(self.files_labels)

    def reshape(self, bottom, top):

        # load image + label image pair
        print '   val reshape'
        print 'file image: ', self.files_images[self.idx]
        print 'file label: ', self.files_labels[self.idx]

        self.data = self.load_image(self.files_images[self.idx])
        self.label = self.load_label(self.files_labels[self.idx])

        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
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
        pass

    def load_image(self, fname):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        print 'fname image = ', fname

        img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_COLOR)
        print img.shape
        height, width = img.shape[:2]

        height_new = height_scaled
        width_new = int((width / float(height)) * height_new)

        dst = cv2.resize(img, (width_new, height_new), interpolation=cv2.INTER_NEAREST)

        in_ = np.array(dst, dtype=np.float32)
        print in_.shape
        in_ = in_[:, :, ::-1]
        # in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))

        return in_

    def load_label(self, fname):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        print 'fname label = ', fname

        img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        print img.shape
        height, width = img.shape[:2]

        height_new = height_scaled
        width_new = int((width / float(height)) * height_new)

        dst = cv2.resize(img, (width_new, height_new), interpolation=cv2.INTER_NEAREST)
        label = np.uint8(dst)
        print label.shape
        print 'label IDS: ', list(np.unique(label))
        return label


class CityscapesDataLayer(caffe.Layer):
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
        self.random = False #params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
        #search_images = os.path.join(cityscapesPath, 'leftImg8bit', 'train', 'aachen', '*_leftImg8bit.png')
        #search_labels = os.path.join(cityscapesPath, 'gtFine', 'train', 'aachen', '*_gtFine_labelTrainIds.png')

        search_images = os.path.join(cityscapesPath, 'leftImg8bit', 'train', '*', '*_leftImg8bit.png')
        #search_labels = os.path.join(cityscapesPath, 'gtFine', 'train', '*', '*_gtFine_labelTrainIds.png')
        search_labels = os.path.join(cityscapesPath, 'gtFine', 'train', '*', '*_gtFine_labelIds.png')

        self.idx = 0
        self.files_images = glob.glob(search_images)
        self.files_labels = glob.glob(search_labels)

        self.files_images.sort()
        self.files_labels.sort()
        self.length = max(len(self.files_images),len(self.files_labels))

        print 'files_images = ', len(self.files_images)
        print 'files_labels = ', len(self.files_labels)

        #print '   ', self.files_images[1114]
        #print '   ', self.files_labels[1114]

    def reshape(self, bottom, top):

        print 'TRAIN reshape ...'
        print 'idx = ', self.idx

        self.data = self.load_image(self.files_images[self.idx])
        self.label = self.load_label(self.files_labels[self.idx])

        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):

        print '   TRAIN forward ...'

        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        #img = np.zeros(self.data.shape, dtype=np.float32)
        #img[...] = self.data
        #img = img.transpose((1, 2, 0))
        #label = self.label.reshape((256,256))
        #cv2.imwrite('input-image.png', img)
        #cv2.imwrite('input-label.png', label)

        print '   data shape = ', top[0].data[...].shape
        print '   label shape = ', top[1].data[...].shape

        # pick next input
        if self.random:
            self.idx = random.randint(0, self.length-1)
        else:
            self.idx += 1
            if self.idx == self.length:
                self.idx = 0

    def backward(self, top, propagate_down, bottom):

        print ' '
        print '   TRAIN backward ...'
        print ' '

        pass

    def load_image(self, fname):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        print 'fname image = ', fname

        img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_COLOR)
        #print img.shape
        height, width = img.shape[:2]

        height_new = height_scaled
        width_new = int((width / float(height)) * height_new)

        dst = cv2.resize(img, (width_new, height_new), interpolation=cv2.INTER_NEAREST)

        #cv2.imwrite('/home/alexey/semantic-segmentation/1.png', dst)
        #cv2.imwrite('/home/alexey/augmentation/0.png', dst)

        middle_u = width_new / 2.
        middle_v = height_new / 2.
        cropped = dst[middle_v - 400:middle_v + 400, middle_u - 400:middle_u + 400]
        #cv2.imwrite('/home/alexey/augmentation/1.png', cropped)

        #cv2.imwrite('/home/alexey/semantic-segmentation/cropped-image.png', cropped)

        #in_ = np.array(dst, dtype=np.float32)
        in_ = np.array(cropped, dtype=np.float32)

        #cv2.imwrite('/home/alexey/augmentation/0-1.png', in_)

        print 'in_ shape = ', in_.shape
        #in_ = in_[:,:,::-1]
        #print 'in_ shape = ', in_.shape
        #cv2.imwrite('/home/alexey/augmentation/0-2.png', in_)

        #in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        #print 'in_ shape = ', in_.shape
        #cv2.imwrite('/home/alexey/augmentation/0-2.png', in_)

        return in_

    def load_label(self, fname):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        print 'fname label = ', fname

        img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        print img.shape
        height, width = img.shape[:2]

        height_new = height_scaled
        width_new = int((width / float(height)) * height_new)
        dst = cv2.resize(img, (width_new, height_new), interpolation=cv2.INTER_NEAREST)

        middle_u = width_new / 2.
        middle_v = height_new / 2.
        cropped = dst[middle_v - 400:middle_v + 400, middle_u - 400:middle_u + 400]

        #cv2.imwrite('/home/alexey/semantic-segmentation/2.png', dst)
        #label = np.uint8(dst)
        #label = np.float32(dst)
        label = np.float32(cropped)
        #cv2.imwrite('/home/alexey/semantic-segmentation/label.png', label)

        print 'label shape = ', label.shape
        print 'label IDS: ', list(np.unique(label))

        label = label[np.newaxis, ...]
        #label = label[..., np.newaxis]
        print 'new label shape = ', label.shape

        return label


class CityscapesDataLayerDeploy(caffe.Layer):

    def setup(self, bottom, top):

        print 'DEPLOYMENT setup ...'

        # config
        params = eval(self.param_str)

        cityscapesPath = os.environ['CITYSCAPES_DATASET']
        fname = cityscapesPath + '/' + 'leftImg8bit/test/munich/munich_000254_000019_leftImg8bit.png'
        print 'fname = ', fname

        img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_COLOR)
        height, width = img.shape[:2]

        height_new = height_scaled
        width_new = int((width / float(height)) * height_new)
        dst = cv2.resize(img, (width_new, height_new), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite('infer-input-image.png', dst)

        middle_u = width_new / 2.
        middle_v = height_new / 2.
        cropped = dst[middle_v - 400:middle_v + 400, middle_u - 400:middle_u + 400]
        #cv2.imwrite('deploy-input-image.png', cropped)

        # in_ = np.array(cropped, dtype=np.float32)
        self.in_ = np.array(cropped, dtype=np.uint8)
        #self.in_ = np.array(dst, dtype=np.uint8)
        # print 'in_.shape = ', in_.shape
        #self.in_ = self.in_[:, :, ::-1]
        # in_ -= self.mean

        print 'in_.shape = ',  self.in_.shape

        self.in_ = self.in_.transpose((2, 0, 1))

        print 'in_.shape = ', self.in_.shape

    def reshape(self, bottom, top):

        print 'DEPLOYMENT reshape ...'

        top[0].reshape(1, *self.in_.shape)

    def forward(self, bottom, top):

        print 'DEPLOYMENT forward ...'

        # assign output
        top[0].data[...] = self.in_
        print 'top[0].data[...].shape = ', top[0].data[...].shape

        #data_plot = top[0].data[...].transpose(2, 0, 1)
        #print 'data shape = ',  data_plot.shape
        #cv2.imwrite('deploy-forward.png', data_plot)
