#! /usr/bin/env python
# -*- coding: utf-8 -*-

# how to use
# (using CPU) python classify.py
# (using GPU) python classify.py --gpu

import sys
import os
import numpy as np
import caffe
import argparse
import time

from caffe.io import blobproto_to_array
from caffe.proto import caffe_pb2

# MEAN_FILE = 'places_mean.npy'
MODEL_FILE = 'deploy_alexnet_places365.prototxt'
PRETRAINED = 'alexnet_places365.caffemodel'
MEAN_FILE = "places365CNN_mean.npy"
# LAYER = 'fc6wi'
# INDEX = 4


def init_net():
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='227,227',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    if not os.path.exists(MEAN_FILE):
        print("creating mean file..")
        blob = caffe_pb2.BlobProto()
        with open(MEAN_FILE.rstrip("npy")+"binaryproto", "rb") as fp:
            blob.ParseFromString(fp.read())
        np.save("places365CNN_mean.npy", blobproto_to_array(blob))

    channel_swap = [int(s) for s in args.channel_swap.split(',')]

    # if args.gpu:
    #     caffe.set_mode_gpu()
    #     caffe.set_device(2)
    #     print("GPU mode")
    # else:
    #     caffe.set_mode_cpu()
    #     print("CPU mode")

    # Make classifier.
    classifier = caffe.Classifier(MODEL_FILE, PRETRAINED,
        image_dims=image_dims,
        mean=np.load(MEAN_FILE).reshape((3, 256, 256)).mean(1).mean(1),
        input_scale=args.input_scale,
        raw_scale=args.raw_scale,
        channel_swap=channel_swap
    )
    return classifier

def make365caffe(path):
    cdirs = os.listdir(path)
    print 'Method called'
    for item in cdirs:
        if os.path.isfile(path+item) and item.endswith('.jpg'):
            f, e = os.path.splitext(path+item)
            print 'Loading item: ', f

            caffe.set_mode_gpu()
            caffe.set_device(2)
            net = init_net()

            # loading images. (loading too many images at once may not work because of consuming large GPU memory)
            # if you want to classify large number of images, repeat the code below per small batchsize(32, 64, 128, ..)


            images = [caffe.io.load_image(path+item)]   # this is just an example. loading the same images for 50 times
            predicted = net.predict(images)
            print predicted  # predicted[i][j] indicates probability of belonging to j-th class (classes are in categories_places365.txt) in an i-th image
            print predicted[0]
            print predicted.shape  # =(batch_size, 365)
            np.save(f, predicted)
            print 'Item Saved: ', item

path = raw_input("Please enter full directory to execute: ")
make365caffe(path)

