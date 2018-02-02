import os
from PIL import Image
import numpy as np
import tensorflow as tf
import imageio
imageio.plugins.ffmpeg.download()
import moviepy.editor as mpy
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS

# Original image dimensions
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512
COLOR_CHAN = 3
BATCH_SIZE = 3
IMG_HEIGHT = 64
IMG_WIDTH = 64
datadir = '/home/liu/Downloads/datasets/push/push_train'
# datadir = '/home/liu/Downloads/datasets/push/test'

def extract_img():
    # filenames = '/home/liuvv/tensorflow/push_data_extract/push_train.tfrecord-00001-of-00264'
    # filenames = [os.path.join(datadir, 'push_train.tfrecord-00%d-of-00264'%i) for i in range(264)]
    # filenames = os.path.join(datadir, '*')
    filenames = gfile.Glob(os.path.join(datadir, '*'))

    if not filenames:
        raise RuntimeError('No data files found.')
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq = []
    #(23+1)/2=12
    for i in range(24):
        image_name = 'move/' + str(i) + '/image/encoded'
        features = {image_name: tf.FixedLenFeature([1], tf.string)}
        features = tf.parse_single_example(serialized_example, features=features)

        image_buffer = tf.reshape(features[image_name], shape=[])
        image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
        image.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])

        image = tf.reshape(image, [1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
        image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
        if i % 2:
            image_seq.append(image)#extract one frame from 2 frames,
                                   # this can deduce the caculation

    image_seq = tf.concat(image_seq, 0)

    image_batch = tf.train.batch(
        [image_seq],
        BATCH_SIZE,
        num_threads=1,
        capacity=2500)
    try:
        shape = image_batch.get_shape().as_list()
    except:
        shape = image_batch.shape()
    image_batch = tf.reshape(image_batch, [shape[0]*shape[1], shape[2], shape[3], shape[4]])
    image_batch = image_batch/255. - 0.5  #[0, 255]-[-0.5, 0.5]
    return image_batch

'''
def npy_to_gif(npy, filename):
    clip = mpy.ImageSequenceClip(list(npy), fps=10)# 10fps per second
    clip.write_gif(filename)


train_image_tensor = extract_img()
sess = tf.InteractiveSession()
tf.train.start_queue_runners(sess)
sess.run(tf.global_variables_initializer())
train_videos = sess.run(train_image_tensor)

for i in range(BATCH_SIZE):
    video = train_videos[i]
    npy_to_gif(video, '~/train_' + str(i) + '.gif')
shape = train_videos.shape()
'''
'''
#the max num of frames is 28,so the img_batch=[BATCH_SIZE, time, , , ] max time is 28
for i in xrange(2):
    img_batch = train_videos[i]
    for j in xrange(28):
        img = img_batch[j]
        # img = tf.cast(img, tf.uint8)*255
        img = tf.cast(img, tf.uint8)
        img = sess.run(img)
        img = Image.fromarray(img, 'RGB')
        print j
        img.save(os.getcwd() + '/data2/' + str(i)+str(j) + '.jpg')        
'''
