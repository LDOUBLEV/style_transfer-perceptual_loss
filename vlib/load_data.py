# coding=utf-8
import tensorflow as tf
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.misc as scm

'''
Function introduction:
dir:          the data direction, for example:'/home/liuvv/Desktop/picture'
batch_size:   the batch size
scle:         defult is False, if True, the result data is crop from raw data
scale_size:   the data size of result data of we need
is_gratscale: defult is False, if True, the result data is gratscale pic
'''

def get_loader(dir, batch_size, scale_size, scale=False,is_gratscale=False):

    # dataser_dir = os.path.basename(dir)
    for ext in ["png", "jpg"]:
        path = glob("{}/*.{}".format(dir, ext))
        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png

    with Image.open(path[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    filename_quene = tf.train.string_input_producer(list(path), shuffle=False, seed=None)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_quene)
    image_ = tf_decode(data, channels=3)
    image = tf.image.resize_images(image_, size=[300, 300])
    if is_gratscale:
        image = tf.image.rgb_to_grayscale(image)

    # image.set_shape(shape)
    image = img_process(image, True)  # for image process

    quene = tf.train.shuffle_batch([image], batch_size=batch_size, num_threads=4, capacity=5000+3*batch_size,
                                   min_after_dequeue=5000, name='synthetic_inputs')
    if scale:
        quene = tf.image.crop_to_bounding_box(quene, 0, 0, 64, 64)
        quene = tf.image.resize_nearest_neighbor(quene, [scale_size, scale_size])
    else:
        quene = tf.image.resize_nearest_neighbor(quene, [scale_size, scale_size])

    image = tf.to_float(quene)
    image = tf.cast(image, tf.float32)

    # image = image/127.5 - 1.

    return image


'''
Function introduction:
dir:        the data direction, for example:'/home/liuvv/Desktop/picture'
batch_size: the batch size
idx:        the index from [0, iters]
'''
def load_data(data_dir, batch_size, idx):
    def get_image(img_path):
        img = scm.imread(img_path) / 255. - 0.5

        # img = img[..., ::-1]  # rgb to bgr
        return img

    data = sorted(glob(os.path.join(data_dir, "*.*")))
    # random_order = np.random.permutation(len(data))
    # data = [data[i] for i in random_order[:]]
    batch_files = data[idx * batch_size: (idx + 1) * batch_size]
    batch_data = [get_image(batch_file) for batch_file in batch_files]
    return batch_data


""" image process """
means = [123.68, 116.779, 103.939]

def _mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, 2)


def _mean_image_add(image, means):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(channels, 2)


def img_process(img, process):
    if process:
        out = _mean_image_subtraction(img, means)
    else:
        out = _mean_image_add(img, means)

    return out
