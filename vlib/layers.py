#coding:utf-8
import tensorflow as tf
import numpy as np
regular_num = 0.0001 # regularizer number

def conv2d(name, tensor, ksize, out_dim, stride=2, sted=True, padding='SAME'):
    with tf.variable_scope(name):
        def uniform(stdev, size):
            return np.random.uniform(low=-stdev*np.sqrt(3),
                                     high=stdev*np.sqrt(3),
                                     size=size).astype('float32')

        tensr_dim = tensor.get_shape().as_list()
        fan_in =tensr_dim[3]*ksize**2
        fan_out = out_dim*ksize**2/(stride**2)
        if sted:
            filter_stdev = np.sqrt(4./(fan_in+fan_out))
        else:
            filter_stdev = 2/(out_dim+tensr_dim[3])
        filter_value = uniform(filter_stdev,(ksize,ksize,tensr_dim[3],out_dim))

        w = tf.get_variable('weight', validate_shape=True, dtype=tf.float32,initializer=filter_value)
        tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(regular_num)(w))
        var = tf.nn.conv2d(tensor, w, [1, stride, stride, 1], padding=padding)
        b = tf.get_variable('bias', [out_dim], 'float32', initializer=tf.constant_initializer(0.))
        # tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(regular_num)(b))
        return tf.nn.bias_add(var, b)


def deconv2d(name, tensor, ksize, outshape, stride=2, sted=True, padding='SAME'):
    with tf.variable_scope(name):
        def uniform(stdev, size):
            return np.random.uniform(low=-stdev*np.sqrt(3),
                                     high=stdev*np.sqrt(3),
                                     size=size).astype('float32')
        tensr_dim = tensor.get_shape().as_list()
        in_dim = tensr_dim[3]
        out_dim = outshape[3]
        fan_in = in_dim*ksize**2/(stride**2)
        fan_out = out_dim*ksize**2
        if sted:
            filter_stdev = np.sqrt(4./(fan_out+fan_in))
        else:
            filter_stdev = 2/(in_dim+out_dim)
        filter_value = uniform(filter_stdev,(ksize,ksize,out_dim,in_dim))

        w = tf.get_variable('weight', validate_shape=True, dtype=tf.float32,initializer=filter_value)
        tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(regular_num)(w))
        var = tf.nn.conv2d_transpose(tensor, w, outshape, strides=[1, stride, stride, 1], padding=padding)
        b = tf.get_variable('bias', [outshape[-1]], 'float32', initializer=tf.constant_initializer(0.))
        # tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(regular_num)(b))
        return tf.nn.bias_add(var, b)

def fc(name,value, output_shape):
    with tf.variable_scope(name, reuse=None) as scope:
        shape = value.get_shape().as_list()
        def uniform(stdev, size):
            return np.random.uniform(low=-stdev*np.sqrt(3),
                                     high=stdev*np.sqrt(3),
                                     size=size).astype('float32')
        weight_val = uniform(np.sqrt(2./(shape[1]+output_shape)),
                             (shape[1], output_shape))

        w = tf.get_variable('weight', validate_shape=True, dtype=tf.float32,initializer=weight_val)
        tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(regular_num)(w))
        b = tf.get_variable('bias', [output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        # tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(regular_num)(b))
        return tf.matmul(value, w) + b


def fused_norm (name, inputs, norm='fused_norm'):
    with tf.variable_scope(name):
        if norm == 'fused_norm':
            offset = tf.get_variable(name+'offset', dtype='float32',
                                     shape=[inputs.get_shape()[1]],
                                     initializer=tf.constant_initializer(0))
            scale = tf.get_variable(name+'scale',  dtype='float32',
                                    shape=[inputs.get_shape()[1]],
                                    initializer=tf.constant_initializer(1))
            output, _, _ = tf.nn.fused_batch_norm(inputs,scale,offset,epsilon=1e-5)

        elif norm =='layer_norm':
            output = tf.contrib.layers.layer_norm(name, inputs, center=True, scale=True)

        return output

def relu(name, tensor):
    return tf.nn.relu(tensor, name)

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def resize_nn(x, size):
    return tf.image.resize_nearest_neighbor(x, size=(int(size), int(size)))

def avg_pool(x):
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x-y))

def input_img(data_dir):
    print ('loading data....')
    filename_queue = tf.train.string_input_producer([data_dir])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#读取文件名和文件内容
    features = tf.parse_single_example(serialized_example,
                                       features={'img_raw': tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    img_batch = tf.train.shuffle_batch([img], batch_size=16, capacity=50000+16,
                                                    min_after_dequeue=50000)
    print ('loading data done')
    return img_batch

#train_phase: True or False
#True :when training, or False when testing
def batch_norm_layer(x, train_phase, scope_bn):
    with tf.variable_scope(scope_bn):
        train_phase = tf.convert_to_tensor(train_phase, dtype='bool')
        beta = tf.Variable(tf.constant(0., shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        # axises = np.arange(len(x.shape)-1)
        axises = list(range(len(x.shape)-1))
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

def batch_norm_layer2(x, is_test=False, scope_bn='bn'):
    with tf.variable_scope(scope_bn):
        is_test = tf.convert_to_tensor(is_test, dtype='bool')
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = np.arange(len(x.shape) - 1)
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_test, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
