
import numpy as np
from PIL import Image


def run_inference2(fname, net):
    print '111'
    print '222'
    
    
def run_inference(fname, net):
    
    print '\n fname = ', fname
    print '\n fname = ', fname
    return
    
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    #im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
    im = Image.open(fname)
    print '111'
    in_ = np.array(im, dtype=np.float32)
    print '222'
    
    # plot input image
    plt.figure()
    plt.title('Input image')
    input_img = np.zeros(in_.shape, dtype=np.uint8)
    input_img[...] = in_[...]
    plt.imshow(input_img)
    
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
    
    for obj_id in labels:
        
        print 'class: ', pascal_voc_2011[obj_id]
        
        segments[:,:,0] += np.uint8(pascal_voc_2011[obj_id][1][0]) * (out == obj_id)
        segments[:,:,1] += np.uint8(pascal_voc_2011[obj_id][1][1]) * (out == obj_id)
        segments[:,:,2] += np.uint8(pascal_voc_2011[obj_id][1][2]) * (out == obj_id)

    # create image overlay with predicted segments
    overlay = np.zeros(input_img.shape, dtype=np.uint8)
    overlay[...] = input_img[...]

    alpha = 0.6
    cv2.addWeighted(segments, alpha, overlay, 1.0-alpha, 0., overlay)
    
    plt.figure()
    plt.title('Predicted segments')
    plt.imshow(segments)

    plt.figure()
    plt.title('Overlay')
    plt.imshow(overlay)
    
    print 'out shape = ', out.shape
    
    class_map = net.blobs['score'].data[0][3]
    print class_map.shape
    
    #plot_output_heat_map(s_rows, s_cols, s_number, output[list_classes[class_ind]],
    #                     labels.id2label[list_classes[class_ind]].name, scale_min, scale_max)

    #plt.figure()
    #plt.title('')
    #plt.imshow(class_map)
    #plt.clim(class_map.min(), class_map.max())
    
    ind = 0
    plot_output_signals(ind, input_img, segments, overlay, net.blobs['score'].data[0])
