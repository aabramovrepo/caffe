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

downsample_factor=0.75

sys.path.insert(0, os.environ['CITYSCAPES_SCRIPTS'] + '/helpers')
import labels


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
    net = caffe.Net('fcn8s-atonce/deploy.prototxt', 'fcn8s-atonce/snapshot/train_iter_2000.caffemodel', caffe.TEST)


    # load indices for images and labels
    cityscapesPath = os.environ['CITYSCAPES_DATASET']

    val_images = os.path.join(cityscapesPath, 'leftImg8bit', 'val', '*', '*_leftImg8bit.png')
    val_labels = os.path.join(cityscapesPath, 'gtFine', 'val', '*', '*_gtFine_labelIds.png')
    val_disparity = os.path.join(cityscapesPath, 'disparity', 'val', '*', '*_disparity.png')

    files_images = glob.glob(val_images)
    files_labels = glob.glob(val_labels)
    files_disparity = glob.glob(val_disparity)

    files_images.sort()
    files_labels.sort()
    files_disparity.sort()
    val_length = max(len(files_images),len(files_labels))

#    for idx in range(10):
#        run_inference(idx, files_images[idx], files_labels[idx], net)

    f_img = '/media/ssd_drive/Cityscapes_dataset/leftImg8bit/train/monchengladbach/monchengladbach_000000_009930_leftImg8bit.png'
    f_label = '/media/ssd_drive/Cityscapes_dataset/gtFine/train/monchengladbach/monchengladbach_000000_009930_gtFine_labelIds.png'
    run_inference(0, f_img, f_label, net)

        
def run_inference(index, f_image, f_gt, net):
    '''Do inference for a given input image'''
    
    print '\n f_image = ', f_image
    print '\n f_gt = ', f_gt

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(f_image)
    in_ = np.array(im, dtype=np.float32)
    
    in_ = cv2.resize(in_, (0,0), fx=downsample_factor, fy=downsample_factor)
    
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
    found_labels = list(np.unique(out))
    print 'labels = ', labels

    # create image with predicted segments
    segments = np.zeros(input_img.shape, dtype=np.uint8)

    for object_id in found_labels:
        #print 'id: ', object_id, ', name: ', labels.id2label[object_id].name, ', category: ', labels.id2label[object_id].category, ', color: ', labels.id2label[object_id].color
        segments[:,:,0] += np.uint8(labels.id2label[object_id].color[0]) * (out==object_id)
        segments[:,:,1] += np.uint8(labels.id2label[object_id].color[1]) * (out==object_id)
        segments[:,:,2] += np.uint8(labels.id2label[object_id].color[2]) * (out==object_id)

    # create image overlay with predicted segments
    overlay = np.zeros(input_img.shape, dtype=np.uint8)
    overlay[...] = input_img[...]

    alpha = 0.6
    cv2.addWeighted(segments, alpha, overlay, 1.0-alpha, 0., overlay)

    # create image with ground truth labels
    _gt = cv2.imread(f_gt,0)
    gt = np.asarray(_gt)
    gt = cv2.resize(gt, (0,0), fx=downsample_factor, fy=downsample_factor)

    print 'gt labels = ', np.unique(gt)

    # create image with predicted segments
    gt_segments = np.zeros(input_img.shape, dtype=np.uint8)

    for object_id in np.unique(gt):
        #print 'id: ', object_id, ', name: ', labels.id2label[object_id].name, ', category: ', labels.id2label[object_id].category, ', color: ', labels.id2label[object_id].color
        gt_segments[:,:,0] += np.uint8(labels.id2label[object_id].color[0]) * (gt==object_id)
        gt_segments[:,:,1] += np.uint8(labels.id2label[object_id].color[1]) * (gt==object_id)
        gt_segments[:,:,2] += np.uint8(labels.id2label[object_id].color[2]) * (gt==object_id)

    # plot all output signals
    plot_output_signals(index, input_img, segments, overlay, net.blobs['score'].data[0], gt_segments)


def plot_output_signals(index, img, segments, overlay, output, ground_truth):
    '''Plot heat maps for all classes the network was trained on'''
    
    # subplot: rows x cols x plot number
    s_rows = 6
    s_cols = 4

    fig = plt.figure(figsize=(30,20))
    fig.patch.set_facecolor('black')

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

    # show heat maps for the most important classes
    nmb = 5
    main_classes = [1, 6, 7, 11, 13, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    min_v = min([output[object_id].min() for object_id in main_classes])
    max_v = max([output[object_id].max() for object_id in main_classes])

    print 'min_v = ', [output[object_id].min() for object_id in main_classes]
    print 'max_v = ', [output[object_id].max() for object_id in main_classes]
    

    for object_id in main_classes:
        fig.add_subplot(s_rows,s_cols,nmb)
        fig.tight_layout()
        plt.axis('off')
        plt.title(labels.id2label[object_id].name, color='white')
        plt.imshow(output[object_id])
        plt.clim(-1e3, 1e3)
        #plt.clim(output[object_id].min(), output[object_id].max())
        nmb += 1

# plot heat maps for all classes the network was trained on
#    for key in classes.lanes_classes:
#        fig.add_subplot(s_rows,s_cols,nmb)
#        fig.tight_layout()
#        plt.axis('off')
#        plt.title(classes.lanes_classes[key][0], color='white')
#        plt.imshow(output[key])        
#        plt.clim(min_v, max_v)
        #cbar=plt.colorbar()
        
        # set white color for bar ticks
#        for t in cbar.ax.get_yticklabels(): 
#            t.set_color("w") 
#        nmb += 1

    plt.savefig('output/infer-final-' + str(index) + '.png', facecolor='black')
    

if __name__ == "__main__":
    main()
