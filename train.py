# coding:utf-8

import tensorflow as tf
import numpy as np
import os
import model
import time
import vlib.plot as plot
import vlib.save_images as save_img
import vlib.load_data as load_data
import vgg_simple as vgg
import scipy.misc as scm

import model

slim = tf.contrib.slim

def load_style_img(styleImgPath):
    img = tf.read_file(styleImgPath)
    style_img = tf.image.decode_jpeg(img, 3)

    style_img = tf.image.resize_images(style_img, [256, 256])

    style_img = load_data.img_process(style_img, True)  # True for substract means

    images = tf.expand_dims(style_img, 0)
    style_imgs = tf.concat([images, images, images, images], 0)  # batch is 4
    # style_imgs = tf.image.resize_images(style_imgs, [256, 256])

    return style_imgs

def load_test_img(img_path):
    style_img = tf.read_file(img_path)

    style_img = tf.image.decode_jpeg(style_img, 3)
    shape = tf.shape(style_img)

    style_img = tf.image.resize_images(style_img, [shape[0], shape[1]])
    style_img = load_data.img_process(style_img, True)

    images = tf.expand_dims(style_img, 0)
    return images


class Train(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = 4
        self.img_size = 256

        self.img_dim = 3
        self.gamma = 0.7
        self.lamda = 0.001
        self.load_model = False
        self.max_step = 500000
        self.save_step = 10000
        self.lr_update_step = 100000
        self.img_save = 500

        self.args = args

    def build_model(self):
        data_path = self.args.train_data_path

        imgs = load_data.get_loader(data_path, self.batch_size, self.img_size)

        style_imgs = load_style_img(self.args.style_data_path)

        with slim.arg_scope(model.arg_scope()):
            gen_img, variables = model.gen_net(imgs, reuse=False, name='transform')

            with slim.arg_scope(vgg.vgg_arg_scope()):
                gen_img_processed = [load_data.img_process(image, True)
                                     for image in tf.unstack(gen_img, axis=0, num=self.batch_size)]

                f1, f2, f3, f4, exclude = vgg.vgg_16(tf.concat([gen_img_processed, imgs, style_imgs], axis=0))

                gen_f, img_f, _ = tf.split(f3, 3, 0)
                content_loss = tf.nn.l2_loss(gen_f - img_f) / tf.to_float(tf.size(gen_f))

                style_loss = model.styleloss(f1, f2, f3, f4)

                # load vgg model
                vgg_model_path = self.args.vgg_model
                vgg_vars = slim.get_variables_to_restore(include=['vgg_16'], exclude=exclude)
                # vgg_init_var = slim.get_variables_to_restore(include=['vgg_16/fc6'])
                init_fn = slim.assign_from_checkpoint_fn(vgg_model_path, vgg_vars)
                init_fn(self.sess)
                # tf.initialize_variables(var_list=vgg_init_var)
                print 'vgg s weights load done'

            self.gen_img = gen_img

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            self.content_loss = content_loss
            self.style_loss = style_loss*self.args.style_w
            self.loss = self.content_loss + self.style_loss
            self.opt = tf.train.AdamOptimizer(0.0001).minimize(self.loss, global_step=self.global_step, var_list=variables)

        all_var = tf.global_variables()
        # init_var = [v for v in all_var if 'beta' in v.name or 'global_step' in v.name or 'Adam' in v.name]
        init_var = [v for v in all_var if 'vgg_16' not in v.name]
        init = tf.variables_initializer(var_list=init_var)
        self.sess.run(init)

        self.save = tf.train.Saver(var_list=variables)
        # self.save.restore(self.sess, os.getcwd()+'/model_saved/wave25.ckpt')
        # print 'load the mave ckpt'
        # var_load = slim.get_variables_to_restore(exclude=['vgg_16'])
        # init_fn2 = slim.assign_from_checkpoint_fn(os.getcwd()+'/model_saved_wave/wave3.ckpt', variables)
        # init_fn2(self.sess)

    def train(self):
        print ('start to training')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                # start_time = time.time()
                _, loss, step, cl, sl = self.sess.run([self.opt, self.loss, self.global_step, self.content_loss, self.style_loss])

                if step%100 == 0:
                    gen_img = self.sess.run(self.gen_img)
                    if not os.path.exists('gen_img'):
                        os.mkdir('gen_img')
                    save_img.save_images(gen_img, './gen_img/{0}.jpg'.format(step/100))

                print ('[{}/40000],loss:{}, content:{},style:{}'.format(step, loss, cl, sl))

                if step % 2000 == 0:
                    if not os.path.exists('model_saved_s'):
                        os.mkdir('model_saved_s')
                    self.save.save(self.sess, './model_saved_s/wave{}.ckpt'.format(step/2000))
                if step >= 40000:
                    break

        except tf.errors.OutOfRangeError:
                self.save.save(sess, os.path.join(os.getcwd(), 'fast-style-model.ckpt-done'))
        finally:
            coord.request_stop()
        coord.join(threads)

    def test(self):
        print ('test model')
        test_img_path = self.args.test_data_path
        test_img = load_test_img(test_img_path)
        # test_img = tf.random_uniform(shape=(1, 500, 800, 3), minval=0, maxval=1.)
        test_img = self.sess.run(test_img)
        with slim.arg_scope(model.arg_scope()):

            gen_img, _ = model.gen_net(test_img, reuse=False, name='transform')

            # load model
            model_path = self.args.transfer_model

            vars = slim.get_variables_to_restore(include=['transform'])
            # vgg_init_var = slim.get_variables_to_restore(include=['vgg_16/fc6'])
            init_fn = slim.assign_from_checkpoint_fn(model_path, vars)
            init_fn(self.sess)
            # tf.initialize_variables(var_list=vgg_init_var)
            print 'vgg s weights load done'

            gen_img = self.sess.run(gen_img)
            save_img.save_images(gen_img, self.args.new_img_name)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-is_training', help='train or test', type=bool, default=False)
parser.add_argument('-vgg_model', help='the path of pretrained vgg model', type=str,
                    default='model_saved/vgg_16.ckpt')
parser.add_argument('-transfer_model', help='the path of transfer net model', type=str,
                    default='model_save/wave8.ckpt')
parser.add_argument('-train_data_path', help='the path of train data', type=str,
                    default='img')
parser.add_argument('-style_data_path', help='the path of style image', type=str, default=os.getcwd() + '/img/star.jpg')
parser.add_argument('-test_data_path', help='the path of style image', type=str, default='test.jpg')
parser.add_argument('-new_img_name', help='the path of style image', type=str, default='transfer.jpg')
parser.add_argument('-style_w', help='the weight of style loss', type=float, default=100)

args = parser.parse_args()

if __name__ == '__main__':

    with tf.Session() as sess:
        Model = Train(sess, args)
        is_training = args.is_training

        if is_training:
            Model.build_model()
            Model.train()
        else:
            Model.test()
            print 'the test'
