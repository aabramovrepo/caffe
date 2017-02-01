#! /usr/bin/env python

#
# The current script runs the pre-trained FCN Caffe network on PASCAL VOC data
#

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

import pascal_voc_dict


def plot_matplotlib_colors(main_fig,index):
    '''Create a legend for all classes and their colors'''

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
    
    for key in pascal_voc_dict.pascal_voc_2011:
        col = 0
        row = key
        y = Y - (row * h) - h

        xi_line = w * (col+0.05)
        xf_line = w * (col+0.25)
        xi_text = w * (col+0.3)

        ax.text(xi_text, y, pascal_voc_dict.pascal_voc_2011[key][0], fontsize=(h*0.8), horizontalalignment='left', verticalalignment='center', color='white')
        
        color = pascal_voc_dict.pascal_voc_2011[key][1]
        ax.hlines(y + h*0.1, xi_line, xf_line, color=(color[0]/255.,color[1]/255.,color[2]/255.), linewidth=(h*0.6))

    ax.set_xlim(0,X)
    ax.set_ylim(0,Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    #plt.savefig('output/classes-colors.png', facecolor='black')


def main():
    '''Load the pre-trained FCN network and do inference'''
    
    images = os.path.join(pascal_voc_dict.pascal_voc_path, 'JPEGImages', '*.jpg')
    list_images = glob.glob(images)

    caffe.set_device(1)
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()

    # load pre-trained FCN caffe net
    #net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
    #net = caffe.Net('voc-fcn8s/deploy.prototxt', 'models/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
    net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s-atonce/snapshot/train_iter_300.caffemodel', caffe.TEST)

    run_inference(0, '/media/ssd_drive/SBD_dataset/dataset/img/2010_000132.jpg', net)
    
    #for index in range(len(list_images)):
    #for index in range(3):
        #run_inference(index, list_images[index], net)


def run_inference(index, fname, net):
    '''Do inference for a given input image'''
    
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
    print 'labels = ', labels

    # create image with predicted segments
    segments = np.zeros(input_img.shape, dtype=np.uint8)

    for obj_id in labels:        
        segments[:,:,0] += np.uint8(pascal_voc_dict.pascal_voc_2011[obj_id][1][0]) * (out == obj_id)
        segments[:,:,1] += np.uint8(pascal_voc_dict.pascal_voc_2011[obj_id][1][1]) * (out == obj_id)
        segments[:,:,2] += np.uint8(pascal_voc_dict.pascal_voc_2011[obj_id][1][2]) * (out == obj_id)

    # create image overlay with predicted segments
    overlay = np.zeros(input_img.shape, dtype=np.uint8)
    overlay[...] = input_img[...]

    alpha = 0.6
    cv2.addWeighted(segments, alpha, overlay, 1.0-alpha, 0., overlay)
    
    # plot all output signals
    plot_output_signals(index, input_img, segments, overlay, net.blobs['score'].data[0])


def plot_output_signals(index, img, segments, overlay, output):
    '''Plot heat maps for all classes the network was trained on'''
    
    # subplot: rows x cols x plot number
    s_rows = 4
    s_cols = 6

    #fig = plt.figure(figsize=(50,40), facecolor='black')
    #fig = plt.figure(figsize=(50,40))
    fig = plt.figure(figsize=(20,10))
    fig.patch.set_facecolor('black')

    # plot input image
    fig.add_subplot(s_rows,s_cols,1)
    plt.title('Input image', color='white')
    plt.axis('off')
    plt.imshow(img)

    # plot predicted segments
    fig.add_subplot(s_rows,s_cols,2)
    plt.title('Predicted segments', color='white')
    plt.axis('off')
    plt.imshow(segments)

    # plot image overlay with predicted segments
    fig.add_subplot(s_rows,s_cols,3)
    plt.title('Overlay',color='white')
    plt.axis('off')
    plt.imshow(overlay)

    nmb = 4
    n = 0
    min_scale_v = 0.
    max_scale_v = 0.
    
    # plot heat maps for all classes the network was trained on
    for key in pascal_voc_dict.pascal_voc_2011:
        
        fig.add_subplot(s_rows,s_cols,nmb)
        plt.axis('off')
        plt.title(pascal_voc_dict.pascal_voc_2011[key][0], color='white')
        plt.imshow(output[key])
        plt.clim(output[key].min(), output[key].max())
        cbar=plt.colorbar()
        
        # set white color for bar ticks
        for t in cbar.ax.get_yticklabels(): 
            t.set_color("w") 
        
        nmb += 1

    #plot_matplotlib_colors(fig,nmb)
        
    plt.savefig('output/infer-final-' + str(index) + '.png', facecolor='black')
    

if __name__ == "__main__":
    main()
