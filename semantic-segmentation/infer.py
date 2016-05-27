
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import caffe
import os
import sys
import cv2
import glob

sys.path.insert(0, os.environ['CITYSCAPES_SCRIPTS'] + '/helpers')
import labels


def main():

    caffe.set_device(1)
    caffe.set_mode_gpu()

    # load net
    net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/snapshot/train_iter_2000.caffemodel', caffe.TEST)

    cityscapesPath = os.environ['CITYSCAPES_DATASET']
    search_test_images = os.path.join(cityscapesPath, 'leftImg8bit', 'test', '*', '*_leftImg8bit.png')
    test_images = glob.glob(search_test_images)

    #for ind in range(len(test_images)):
    for ind in range(1):

        fname = '/media/ssd_drive/Cityscapes_dataset/leftImg8bit/test/munich/munich_000279_000019_leftImg8bit.png'
        #print test_images[ind]
        #img = prepare_input_image(test_images[ind])

        print fname
        img = prepare_input_image(fname)

        in_ = np.array(img, dtype=np.uint8)
        in_ = in_[:, :, ::-1]
        # in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_

        # run net and take argmax for prediction
        net.forward()
        out = net.blobs['score'].data[0].argmax(axis=0)
        out = np.uint8(out)

        #print 'net.blobs = ', net.blobs['score'].data[0].shape
        #print 'net.blobs = ', net.blobs['score'].data[0].argmax(axis=0).shape
        #print 'out shape = ', out.shape

        label_ids = list(np.unique(out))
        print 'label_ids = ', label_ids

        # heat maps for each class
        for i in range(net.blobs['score'].data[0].shape[0]-1):

            map = net.blobs['score'].data[0][i]
            class_name = labels.id2label[i].name
            plot_heat_map(class_name, map, -10., 10., 'output/heat-map-' + str(i))


        overlay = np.zeros(img.shape, dtype=np.uint8)

        for trainID in label_ids:
            # print
            # print 'trainID = ', trainID
            # print

            #  map from trainID to label
            # name = labels.trainId2label[trainID].name
            name = labels.id2label[trainID].name
            category = labels.id2label[trainID].category
            color = labels.id2label[trainID].color

            # print
            # print "Name of label with trainID '{id}': {name}".format(id=trainID, name=name)
            # print "   Category of label with ID '{id}': {category}".format(id=trainID, category=category)
            # print 'color = ', color

            object = np.zeros(img.shape, dtype=np.uint8)
            # indices = (out == trainID).nonzero()
            # print 'indices = ', indices

            object[:, :, 0] = color[2]
            object[:, :, 1] = color[1]
            object[:, :, 2] = color[0]

            overlay[:, :, 0] += object[:, :, 0] * (out == trainID)
            overlay[:, :, 1] += object[:, :, 1] * (out == trainID)
            overlay[:, :, 2] += object[:, :, 2] * (out == trainID)

        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1.0-alpha, 0., img)
        cv2.imwrite('output/infer-final-' + str(ind) + '.png', img)


def plot_heat_map(title_str, data, scale_min, scale_max, fname):

    plt.figure(figsize=(20, 10))
    plt.title(title_str)
    plt.imshow(data)
    plt.clim(scale_min, scale_max)
    plt.colorbar()
    plt.savefig(fname + '.png')
    plt.close()


def prepare_input_image(fname):

    img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_COLOR)
    height, width = img.shape[:2]

    height_new = 800
    width_new = int((width / float(height)) * height_new)

    dst = cv2.resize(img, (width_new, height_new), interpolation=cv2.INTER_NEAREST)
    #cv2.imwrite('infer-input-image.png', dst)

    middle_u = width_new / 2.
    middle_v = height_new / 2.
    cropped = dst[middle_v - 400:middle_v + 400, middle_u - 400:middle_u + 400]
    #cv2.imwrite('infer-input-image-cropped.png', cropped)

    return cropped


if __name__ == "__main__":
    main()
