"""
Hack code on mnist dataset
The function of the code is to restore images from feature. 
In other words, given feature F and a feature extractor model W,
we want to get images I, where M(I) is similar to F.
We first set W as encoder and train a decoder according to it. 
Then we use FGSM technique to finetune images.

Params:
-- feature_path          : The path containing feature we want to restore.
-- mnist_checkpoint_path : The path containing feature extractor model W

Returns:
-- save_feature_path     : Feature M(I)
-- save_image            : Hacked images I
-- checkpoint_path       : Decoder and Encoder
""" 
from __future__ import division
import time
import math
from glob import glob
from six.moves import xrange
import pathlib
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from utils import *
import scipy.misc


batch_size = 100
epoch_num = 25
epoch_att_num = 3000
feature_path = './test_all.csv'
mnist_checkpoint_path = './model/'
checkpoint_path = './checkpoint'
save_image_path = './images/'
save_feature_path = './csv/'


def save(sess,saver,checkpoint_dir, step):
    model_name = "hack.model"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

def load(sess,saver,checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
    else:
        print(" [*] Failed to find a checkpoint")

def discriminator(x, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        W = tf.get_variable('d_w0', [784,10], tf.float32)
        b = tf.get_variable('d_b0', [10], tf.float32)

        W_conv1 = weight_variable([5, 5, 1, 32],'d_w1')
        b_conv1 = bias_variable([32],'d_b1')

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64],'d_w2')
        b_conv2 = bias_variable([64],'d_b2')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024],'d_w3')
        b_fc1 = bias_variable([1024],'d_b3')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        return h_fc1

def generator( z, batch_size):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = 28, 28
        gf_dim = 64
        c_dim = 1
        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')
        g_bn3 = batch_norm(name='g_bn3')
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        z2, h02_w, h02_b = linear(
        z, gf_dim*8*s_h16*s_w16, 'g_h_lin', with_w=True)

        
        h02 = tf.reshape(
            z2, [-1, s_h16, s_w16, gf_dim * 8])
        h02 = tf.nn.relu(g_bn0(h02))

        h1, h1_w, h1_b = deconv2d(
            h02, [batch_size, s_h8, s_w8, gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(g_bn1(h1))

        h2, h2_w, h2_b = deconv2d(
            h1, [batch_size, s_h4, s_w4, gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(g_bn2(h2))

        h3, h3_w, h3_b = deconv2d(
            h2, [batch_size, s_h2, s_w2, gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(g_bn3(h3))

        h4, h4_w, h4_b = deconv2d(
            h3, [batch_size, s_h, s_w, c_dim], name='g_h4', with_w=True)

        return tf.nn.sigmoid(h4)

def FGSM(inputs, y, eps=1, clip_min=0.,clip_max=1.):
    x = tf.identity(inputs)
    loss = tf.reduce_mean(tf.square(y-discriminator(x, reuse=True)))
    dy_dx, = tf.gradients(loss, x)
    x = tf.stop_gradient(x - eps*dy_dx)
    x = tf.clip_by_value(x, clip_min, clip_max)
    return loss, tf.squeeze(x)

def main():
    for path in [checkpoint_path, save_image_path, save_feature_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    sess = tf.Session()
    z = tf.placeholder( tf.float32, [None, 1024], name='z')
    G  = generator(z, batch_size)
    G_img = tf.squeeze(G)
    feature_gen =  discriminator(G, reuse=False)
    g_loss = tf.reduce_mean(tf.square(feature_gen - z))
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'g_' in var.name]
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_optim = tf.train.AdamOptimizer(1e-3).minimize(g_loss, var_list = g_vars)
    x_att = tf.placeholder(tf.float32, [None, 28, 28 ,1], name='x_att')
    y_att = tf.placeholder(tf.float32, [None, 1024], name='y_att')
    loss_att, x_FGSM = FGSM(x_att,y_att)
    feature_att = discriminator(x_att, reuse=True)
    sess.run(tf.global_variables_initializer())
    features = read_csv(feature_path)
    saver = tf.train.Saver(d_vars)
    load(sess,saver, mnist_checkpoint_path)
 
    saver = tf.train.Saver()
    
    for epoch in xrange(epoch_num):
        batch_idxs = features.shape[0] // batch_size

        for idx in xrange(0, batch_idxs):
            batch_z = features[idx*batch_size:(idx+1)*batch_size,:]
            temp = sess.run(g_optim, feed_dict = { z:batch_z})
            err_G = g_loss.eval(session=sess, feed_dict={z:batch_z})
            print("Epoch: [%2d] [%4d/%4d]  loss:%.8f" % (epoch, idx, batch_idxs, err_G))
        save(sess,saver,checkpoint_path,epoch)
    ''' 
    Extract features and images
    '''
    batch_idxs = features.shape[0] // batch_size
    images = np.empty((features.shape[0],28,28))
    features_1 = np.empty((features.shape[0],1024))
    for idx in xrange(0, batch_idxs):
        batch_z = features[idx*batch_size:(idx+1)*batch_size,:]
        feature_temp, image_temp = sess.run([feature_gen,G_img], feed_dict = { z:batch_z})
        features_1[idx*batch_size:(idx+1)*batch_size,:] = feature_temp
        images[idx*batch_size:(idx+1)*batch_size,:,:] = image_temp
    save_csv(features_1,'./csv/test_temp_gen1.csv')

    '''
    FGSM attack
    '''
    features_FGSM = np.empty((features.shape[0], 1024))

    for i in range(features.shape[0]):
        for epoch in range(epoch_att_num):
            images_att = images[i,:,:].copy()
            images_att = images_att[np.newaxis,:,:]
            images_att = images_att[:,:,:,np.newaxis]
            features_att = features[i,:].copy()
            features_att = features_att[np.newaxis,:]
        
            loss_temp,images[i,:,:] = sess.run([loss_att, x_FGSM], feed_dict={x_att: images_att, y_att: features_att})
        images_att = images[i,:,:].copy()
        images_att = images_att[np.newaxis,:,:]
        images_att = images_att[:,:,:,np.newaxis]

        print("Image: [%4d] loss:%.8f" % (i, loss_temp))
        features_FGSM[i,:] = sess.run(feature_att , feed_dict={x_att: images_att})
        scipy.misc.imsave(save_image_path + str(i) + '.png', images[i,:,:])
    save_csv(features_FGSM, save_feature_path + 'test_FGSM.csv')


if __name__ == "__main__":
    main()
