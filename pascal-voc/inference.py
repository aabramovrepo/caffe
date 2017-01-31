#! /usr/bin/env python

#
# The current script runs the pre-trained Caffe network on PASCAL VOC data
#

#from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from PIL import Image

import caffe
import cv2
import os
import glob
import sys
import six

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors as mcolors

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


def plot_matplotlib_colors(main_fig,index):

    n = 21
    ncols = 1
    nrows = 21

    #fig.add_subplot(s_rows,s_cols,nmb)
    fig, ax = main_fig.add_subplot(5,6,index)

    #fig, ax = plt.subplots(figsize=(2,5))
    X, Y = fig.get_dpi() * fig.get_size_inches()

    # row height
    h = Y / (nrows + 1)
    # col width
    w = X / ncols
    
    for key in pascal_voc_2011:
        col = 0
        row = key
        y = Y - (row * h) - h

        xi_line = w * (col+0.05)
        xf_line = w * (col+0.25)
        xi_text = w * (col+0.3)

        ax.text(xi_text, y, pascal_voc_2011[key][0], fontsize=(h*0.8), horizontalalignment='left', verticalalignment='center', color='white')
        
        color = pascal_voc_2011[key][1]
        ax.hlines(y + h*0.1, xi_line, xf_line, color=(color[0]/255.,color[1]/255.,color[2]/255.), linewidth=(h*0.6))

    ax.set_xlim(0,X)
    ax.set_ylim(0,Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    #plt.savefig('output/classes-colors.png', facecolor='black')


def main():
    
    images = os.path.join(pascal_path, 'JPEGImages', '*.jpg')
    list_images = glob.glob(images)

    caffe.set_device(1)
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()

    # load net
    #net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
    #net = caffe.Net('voc-fcn8s/deploy.prototxt', 'models/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
    net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s-atonce/snapshot/train_iter_100000.caffemodel', caffe.TEST)

    #for index in range(len(list_images)):
    for index in range(30):
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
    s_rows = 5
    s_cols = 6

    #fig = plt.figure(figsize=(50,40), facecolor='black')
    #fig = plt.figure(figsize=(50,40))
    fig = plt.figure(figsize=(20,10))
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
        
        fig.add_subplot(s_rows,s_cols,nmb)
        plt.axis('off')
        plt.title(pascal_voc_2011[key][0], color='white')
        plt.imshow(output[key])
        plt.clim(output[key].min(), output[key].max())
        cbar=plt.colorbar()
        
        # set white color for bar ticks
        for t in cbar.ax.get_yticklabels(): 
            t.set_color("w") 
        
        nmb += 1

    #plot_matplotlib_colors(fig,nmb)
        
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
