# Fully Convolutional Networks for Semantic Segmentation on PASCAL VOC

This project has originated from the "Fully Convolutional Networks for Semantic Segmentation" by Berkeley.

* inference.py -- inference for a pre-trained FCN Caffe model
* pascal_voc_dict.py -- python dictionary for PASCAL VOC 2011 classes
* voc_layers_alexey.py -- python layers for reading input images and ground truth segments

## voc-fcn8s-atonce

all-at-once, three stream, 8 pixel prediction stride net, scoring 65.4 mIU on seg11valid

## voc-fcn8s

three stream, 8 pixel prediction stride net, scoring 65.5 mIU on seg11valid and 67.2 mIU on seg12test

## voc-fcn16s

two stream, 16 pixel prediction stride net, scoring 65.0 mIU on seg11valid

## voc-fcn32s

single stream, 32 pixel prediction stride net, scoring 63.6 mIU on seg11valid
