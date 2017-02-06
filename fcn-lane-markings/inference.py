#! /usr/bin/env python

#
# The current script runs the pre-trained FCN Caffe network on lane marking data
#

import numpy as np
from PIL import Image

import caffe
import cv2
import os
import glob
import sys
import six
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors as mcolors

import classes


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

        xi_line = w*(col+0.05)
        xf_line = w*(col+0.25)
        xi_text = w*(col+0.3)

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
    
    caffe.set_device(1)
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()

    # load pre-trained FCN caffe net
    #net = caffe.Net('fcn8s-atonce/deploy.prototxt', 'fcn8s-atonce/snapshot/lane_markings/train_iter_1000.caffemodel', caffe.TEST)
    net = caffe.Net('fcn8s-atonce/deploy.prototxt', 'fcn8s-atonce/snapshot/train_iter_4000.caffemodel', caffe.TEST)

    # do inference
#    indices_train = [0,100,104,109,11,12,15,19,1,23,24,27,29,2,31,34,3,52,5,70,7,82,88,92,95,98,9]

#    for idx in indices_train:
#        f_img = classes.lanes_path + 'train/images/color_rect_' + str(idx) + '.png'
#        f_label = classes.lanes_path + 'train/labels/color_rect_' + str(idx) + '_roi.png'
#        run_inference(idx, f_img, f_label, net)
 
    indices_val = [0,10]

    for idx in indices_val:
        f_img = classes.lanes_path + 'val/images/color_rect_' + str(idx) + '.png'
        f_label = classes.lanes_path + 'val/labels/color_rect_' + str(idx) + '_roi.png'
        run_inference(idx, f_img, f_label, net)    

        
def run_inference(index, f_image, f_gt, net):
    '''Do inference for a given input image'''
    
    print '\n f_image = ', f_image
    print '\n f_gt = ', f_gt

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(f_image)
    in_ = np.array(im, dtype=np.float32)
    
    in_ = cv2.resize(in_, (0,0), fx=0.1, fy=0.1)
    
    input_img = np.zeros(in_.shape, dtype=np.uint8)
    input_img[...] = in_[...]
    
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take argmax for prediction
    start_t = time.time()
    net.forward()
    print 'Runtime: ', time.time() - start_t
    
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
        segments[:,:,0] += np.uint8(classes.lanes_classes[obj_id][1][0]) * (out==obj_id)
        segments[:,:,1] += np.uint8(classes.lanes_classes[obj_id][1][1]) * (out==obj_id)
        segments[:,:,2] += np.uint8(classes.lanes_classes[obj_id][1][2]) * (out==obj_id)

    # create image overlay with predicted segments
    overlay = np.zeros(input_img.shape, dtype=np.uint8)
    overlay[...] = input_img[...]

    alpha = 0.6
    cv2.addWeighted(segments, alpha, overlay, 1.0-alpha, 0., overlay)

    # create image with ground truth labels
    _gt = cv2.imread(f_gt,0)
    gt = np.asarray(_gt)
    gt = cv2.resize(gt, (0,0), fx=0.1, fy=0.1)
    
    #gt[(gt == 255).nonzero()] = 1
    gt[(gt == 50).nonzero()] = 1
    gt[(gt == 100).nonzero()] = 2
    gt[(gt == 150).nonzero()] = 3
    gt[(gt > 3).nonzero()] = 0
    
    print 'gt labels = ', np.unique(gt)

    # create image with predicted segments
    gt_segments = np.zeros(input_img.shape, dtype=np.uint8)

    for obj_id in np.unique(gt):
        gt_segments[:,:,0] += np.uint8(classes.lanes_classes[obj_id][1][0]) * (gt==obj_id)
        gt_segments[:,:,1] += np.uint8(classes.lanes_classes[obj_id][1][1]) * (gt==obj_id)
        gt_segments[:,:,2] += np.uint8(classes.lanes_classes[obj_id][1][2]) * (gt==obj_id)

    # plot all output signals
    plot_output_signals(index, input_img, segments, overlay, net.blobs['score'].data[0], gt_segments)


def plot_output_signals(index, img, segments, overlay, output, ground_truth):
    '''Plot heat maps for all classes the network was trained on'''
    
    # subplot: rows x cols x plot number
    s_rows = 2
    s_cols = 4

    fig = plt.figure(figsize=(30,10))
    fig.patch.set_facecolor('black')
#    fig.tight_layout()

    # plot input image
    fig.add_subplot(s_rows,s_cols,1)
    fig.tight_layout()
    plt.title('Input image', color='white')
    plt.axis('off')
    plt.imshow(img)

    # plot ground truth
    fig.add_subplot(s_rows,s_cols,2)
    fig.tight_layout()
    plt.title('Ground truth', color='white')
    plt.axis('off')
    plt.imshow(ground_truth)

    # plot predicted segments
    fig.add_subplot(s_rows,s_cols,3)
    fig.tight_layout()
    plt.title('Predicted segments', color='white')
    plt.axis('off')
    plt.imshow(segments)

    # plot image overlay with predicted segments
    fig.add_subplot(s_rows,s_cols,4)
    fig.tight_layout()
    plt.title('Overlay',color='white')
    plt.axis('off')
    plt.imshow(overlay)

    nmb = 5
    min_v = min([output[key].min() for key in classes.lanes_classes])
    max_v = max([output[key].max() for key in classes.lanes_classes])

    # plot heat maps for all classes the network was trained on
    for key in classes.lanes_classes:
        fig.add_subplot(s_rows,s_cols,nmb)
        fig.tight_layout()
        plt.axis('off')
        plt.title(classes.lanes_classes[key][0], color='white')
        plt.imshow(output[key])        
        plt.clim(min_v, max_v)
        #cbar=plt.colorbar()
        
        # set white color for bar ticks
#        for t in cbar.ax.get_yticklabels(): 
#            t.set_color("w") 
        nmb += 1

    plt.savefig('output/infer-final-' + str(index) + '.png', facecolor='black')
    

if __name__ == "__main__":
    main()
