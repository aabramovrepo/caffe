#! /usr/bin/env python

#
# The current script runs the pre-trained Caffe network on PASCAL VOC data
#

import numpy as np
from PIL import Image

import caffe
import cv2
import os
import glob
import sys

import matplotlib.pyplot as plt
from matplotlib import gridspec

# PASCAL VOC classes
# 0 - background
pascal_voc_2011 = {0: ('background', [0, 0, 0]), 1: ('aeroplane', [46, 131, 193]), 2: ('bicycle', [84, 139, 108]),
                   3: ('bird', [190, 242, 73]), 4: ('boat', [158, 29, 0]), 5: ('bottle', [112, 67, 115]),
                   6: ('bus', [164, 192, 191]), 7: ('car', [65, 127, 36]), 8: ('cat', [234, 118, 26]),
                   9: ('chair', [198, 121, 73]), 10: ('cow', [149, 159, 62]), 11: ('diningtable', [246, 222, 141]),
                   12: ('dog', [229, 17, 93]), 13: ('horse', [243, 219, 110]), 14: ('motorbike', [210, 250, 216]),
                   15: ('person', [2, 222, 131]), 16: ('potted plant', [251, 213, 226]), 17: ('sheep', [193, 0, 1]),
                   18: ('sofa', [172, 79, 96]), 19: ('train', [190, 7, 76]), 20: ('tv/monitor', [48, 129, 217])}


#(1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car , 8=cat, 9=chair, 10=cow, 11=diningtable, 12=dog, 13=horse,
# 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor)

# path to PASCAL VOC images
pascal_path = '/media/ssd_drive/PASCAL_VOC/TrainVal/VOCdevkit/VOC2011/'


def main():
    
    '''nrow = 10
    ncol = 3

    fig = plt.figure(figsize=(4, 10)) 

    gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1, 1, 1],
                           wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845) 

    for i in range(10):
        for j in range(3):
            im = np.random.rand(28,28)
            ax= plt.subplot(gs[i,j])
            ax.imshow(im)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    plt.savefig('output/test-plot.png')
    return'''

    images = os.path.join(pascal_path, 'JPEGImages', '*.jpg')
    list_images = glob.glob(images)

    caffe.set_device(0)
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()

    # load net
    #net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
    net = caffe.Net('voc-fcn8s/deploy.prototxt', 'models/fcn8s-heavy-pascal.caffemodel', caffe.TEST)

    #for index in range(len(list_images)):
    for index in range(5):
        run_inference(index, list_images[index], net)


def run_inference(index, fname, net):
    
    print '\n fname = ', fname

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    #im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
    im = Image.open(fname)
    in_ = np.array(im, dtype=np.float32)
    
    input_img = np.zeros(in_.shape, dtype=np.uint8)
    input_img[...] = in_[...]
    
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    
    print 'net.blobs = ', net.blobs['score'].data[0].shape
    print 'net.blobs = ', net.blobs['score'].data[0].argmax(axis=0).shape
    print 'out shape = ', out.shape
    
    # get predicted labels
    labels = list(np.unique(out))    
    segments = np.zeros(input_img.shape, dtype=np.uint8)
    
    print 'labels = ', labels
    
    for obj_id in labels:        
        segments[:,:,0] += np.uint8(pascal_voc_2011[obj_id][1][0]) * (out == obj_id)
        segments[:,:,1] += np.uint8(pascal_voc_2011[obj_id][1][1]) * (out == obj_id)
        segments[:,:,2] += np.uint8(pascal_voc_2011[obj_id][1][2]) * (out == obj_id)

    # create image overlay with predicted segments
    overlay = np.zeros(input_img.shape, dtype=np.uint8)
    overlay[...] = input_img[...]

    alpha = 0.6
    cv2.addWeighted(segments, alpha, overlay, 1.0-alpha, 0., overlay)
    
    plot_output_signals(index, input_img, segments, overlay, net.blobs['score'].data[0])


def plot_output_signals(index, img, segments, overlay, output):
    
    # subplot: rows x cols x plot number
    s_rows = 4
    s_cols = 6

    #fig = plt.figure(figsize=(50,40), facecolor='black')
    fig = plt.figure(figsize=(50,40))
    fig.patch.set_facecolor('black')

    fig.add_subplot(s_rows,s_cols,1)
    plt.title('Input image', color='white')
    plt.axis('off')
    plt.imshow(img)

    fig.add_subplot(s_rows,s_cols,2)
    plt.title('Predicted segments', color='white')
    plt.axis('off')
    plt.imshow(segments)

    fig.add_subplot(s_rows,s_cols,3)
    plt.title('Overlay',color='white')
    plt.axis('off')
    plt.imshow(overlay)

    nmb = 4
    n = 0
    min_scale_v = 0.
    max_scale_v = 0.
    
    for key in pascal_voc_2011:
        
        min_v = -15.
        max_v = 15.
        
        fig.add_subplot(s_rows,s_cols,nmb)
        plt.axis('off')
        plt.title(pascal_voc_2011[key][0], color='white')
        plt.imshow(output[key])
        plt.clim(min_v, max_v)
        
        nmb += 1

    plt.savefig('output/infer-final-' + str(index) + '.png', facecolor='black')

    
def plot_output_heat_map(row, col, number, data, title_str, scale_min, scale_max):

    plt.subplot(row, col, number)
    plt.axis('off')
    plt.title(title_str)
    plt.imshow(data)
    plt.clim(scale_min, scale_max)
    #plt.clim(data.min(), data.max())
    
    
if __name__ == "__main__":
    main()
