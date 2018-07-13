import sys
import os
caffe_root='---caffe path---'
sys.path.insert(0,caffe_root+'python')
import caffe
import numpy as np
import cv2

input = np.array([[1,2,3,3,4],[4,5,6,6,4],[7,8,9,9,4],[6,5,4,4,4]])
input = input[np.newaxis,np.newaxis,:,:]

caffe.set_mode_cpu()
net = caffe.Net('./extra/test.prototxt', caffe.TEST)

net.blobs['data'].reshape(*input.shape)
net.blobs['data'].data[...] = input

net.forward()
