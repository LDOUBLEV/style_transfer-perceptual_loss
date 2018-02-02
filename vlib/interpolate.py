#coding:utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque
import os
import math
import json
import logging
from PIL import Image
from datetime import datetime
# from models import *
# from utils import save_image

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2, normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)


def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
    batch_size = len(real_batch)
    half_batch_size = int(batch_size/2)

    self.sess.run(self.z_r_update)
    tf_real_batch = to_nchw_numpy(real_batch)
    for i in trange(train_epoch):
        z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
    z = self.sess.run(self.z_r)

    z1, z2 = z[:half_batch_size], z[half_batch_size:]
    real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

    generated = []
    for idx, ratio in enumerate(np.linspace(0, 1, 10)):
        z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
        z_decode = self.generate(z, save=False)
        generated.append(z_decode)

    generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
    for idx, img in enumerate(generated):
        save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

    all_img_num = np.prod(generated.shape[:2])
    batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
    save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)

def interpolate(batch_size, noise):
    half_batch = batch_size//2
    z1 = noise[:half_batch]
    z2 = noise[half_batch:]

    generate = []
    for idx, ratio in enumerate(np.linspace(0, 1, 10)):
        z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
        generate.append(G(z))
    generate = np.stack(generate).transpose([1,0,2,3,4])
    for idx, img in enumerate(generate):
        save_image(img, os.path.join(os.getcwd()+'/interpolate', 'interp_G_{}.png'.format(idx)), nrow=10)

    all_img_num = np.prod(generate.shape[:2])
    new_shape = [all_img_num] + list(generated.shape[2:])
    batch_generate = np.reshape(generate, new_shape)
    save_image(batch_generate, os.path.join(os.getcwd()+'/interpolate', 'interp_G.png'), nrow=10)





