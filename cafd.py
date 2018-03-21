#!/usr/bin/env python3
'''
Class-Aware Frechet Distance (CAFD) 

The CAFD  metric evaluates the similarity of two image sets' distribution. 
For datasets with multiple classes, CAFD employs a Gussian mixture model on 
the feature space to better fit the multi-manifold feature distribution.
Different from FID, we use domain-specific model as feature extractor.

The inputs of the he function are two image sets where images are stored as 
PNG/JPEG and domain-specific model or two feature sets need to be compared.

'''
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp
import os
import gzip, pickle
import tensorflow as tf
from scipy.misc import imread
import pathlib
import urllib
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import sys

GPU_ID = '0'
MODEL_PATH =  './fashion-expert-graph.pb'
IMAGE_PATH = ['./test0', './test1']

def create_graph(pth):
    """
    Creates a graph from saved GraphDef file.
    Params:
    --  pth   :model path
    """
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='CAFD_Net')


def _get_layer(sess, is_feature=0):
    """
    Return certain layers in the graph
    Params:
    --  sess       : currenet session
    -- is_feature  : the type of inputs. True for feature and false for images.
    
    Returns:
    -- feature layer : the layer whose output is activations used for CAFD
    -- softmax       : the probability of each classes
    """
    feature_dim = 1024
    if is_feature == 0:
        layername = 'CAFD_Net/fc1:0'
        feature_layer = sess.graph.get_tensor_by_name(layername)
        ops = feature_layer.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                if shape._dims is not None:
                    shape = [s.value for s in shape]
                    new_shape = []
                    for j, s in enumerate(shape):
                        if s == 1 and j == 0:
                            new_shape.append(None)
                        else:
                            new_shape.append(s)
                    o._shape = tf.TensorShape(new_shape)
    else: 
        feature_layer = tf.placeholder("float", shape=[None, feature_dim], name='feature')
    w = sess.graph.get_tensor_by_name('CAFD_Net/constant_W_fc2:0')
    b = sess.graph.get_tensor_by_name('CAFD_Net/constant_b_fc2:0')
    softmax = tf.nn.softmax(tf.matmul(tf.nn.relu(feature_layer),w) + b)
    return feature_layer, softmax
#-------------------------------------------------------------------------------


def get_activations(data, sess, batch_size=50, is_feature=0):
    """
    Calculates the activations and probability of images or feature.

    Params:
    -- data        : Images (sample_number, hi, wi, channel_dim) or feature (sample_number, feature_dim). 
    -- sess        : current session
    -- batch_size  : the data numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- is_feature  : the type of inputs. True for feature and false for images.

    Returns:
    -- pred_arr    : A numpy array of dimension (sample_number, activations_dim) that contains the
                     activations of a batch used for CAFD.
    -- pro_arr     : A numpy array of dimension (sample_number, class_number) that contains the
                     probability for each class of a batch used for CAFD.

    """
    class_number = 10
    activations_dim = 1024
    data = data.astype(np.float32)
    layer, softmax = _get_layer(sess,is_feature)
    d0 = data.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pro_arr = np.empty((n_used_imgs, class_number))
    if is_feature == 0:
        pred_arr = np.empty((n_used_imgs, activations_dim))
    else:
        pred_arr = data[0:n_used_imgs,:]
    for i in range(n_batches):
        start = i*batch_size
        end = start + batch_size
        batch = data[start:end]
        if is_feature == 0:
            pred, pro = sess.run([layer,softmax], {'CAFD_Net/input:0': batch})
            pro_arr[start:end] = pro.reshape(batch_size, -1)
            pred_arr[start:end] = pred.reshape(batch_size,-1)
        else:
            pro = sess.run(softmax, feed_dict={'feature:0': batch})
            pro_arr[start:end] = pro.reshape(batch_size, -1)   
    return pred_arr, pro_arr
#-------------------------------------------------------------------------------


def calculate_CAFD(mu1, sigma1, mu2, sigma2):
    """
    Numpy implementation of CAFD.
    We assume the distribution of activations for class i is Gaussian distribution:
    X_i^r ~ N(mu_i^r, C_i^r), X_i^g ~ N(mu_i^g, C_i^g). K is the number of classes
    The CAFD between two distributions X^r and X^g is:
            CAFD = (\sigma_k ||mu_k^r - mu_k^g||^2 + Tr(C_k^r + C_2 - 2*sqrt(C_1*C_2)))/K.


    Params:
    -- mu1   : List containing the mean values of X^r for each class 
    -- mu2   : List containing the mean values of X^g for each class 
    -- sigma1: List containing covariance matrixs of x^r for each class
    -- sigma2: List containing covariance matrixs of x^g for each class

    Returns:
    -- dist  : CAFD.

    """
    dist_sum = 0
    for i in range(len(mu1)):
        m = np.square(mu1[i] - mu2[i]).sum()
        s = sp.linalg.sqrtm(np.dot(sigma1[i], sigma2[i]))
        dist = m + np.trace(sigma1[i]+sigma2[i] - 2*s)
        if np.isnan(dist):
            raise InvalidCAFDException("nan occured in distance calculation.")
        dist_sum += dist
    return dist_sum/len(mu1)


def calculate_activation_statistics(data, sess, batch_size=50, is_feature=0):
    """
    Calculation of activations statistics.
    Params:
    -- data        : The input of domain-specific model. It can be images or feature. 
    -- sess        : current session
    -- batch_size  : the data numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- is_feature  : the type of inputs. True for feature and false for images.

    Returns:
    -- mu    : The mean value of the activations for each class
    -- sigma : The covariance matrix of the activations for each class
    -- softmax : The sum of probability for each class
    """
    class_number = 10
    act, pro = get_activations(data, sess, batch_size, is_feature)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    pro = np.array(pro)
    softmax = list(np.sum(pro, 0))

    mu = [ np.zeros(act.shape[1]) for i in range(class_number)]
    sigma =[ np.zeros((act.shape[1],act.shape[1])) for i in range(class_number) ]
    for i in range(class_number):
        pro_temp = pro[:,i]/softmax[i]
        pro_temp = np.tile(pro_temp, (act.shape[1],1)).T
        act_temp = np.multiply(pro_temp, act)
        mu[i] = np.sum(act_temp, axis=0)
        act_temp1 = act - np.tile(mu[i],(act.shape[0],1))
        sigma[i] =  np.array(np.mat(np.multiply(pro_temp,act_temp1)).T*np.mat(act_temp1))
    softmax = softmax/np.sum(softmax)
    return mu, sigma, softmax

def read_csv(path):
    """
    Get feature from certain paths
    Params:
    -- path  : The path of feature
    Returns:
    -- data  : Numpy array (sample_number, feature_dim)
    """
    df = pd.read_csv(path, sep=',')
    df = df[df.columns[1:]]
    data = df.as_matrix()
    return data

def _handle_path(path, sess):
    if path.endswith('.csv'):
        x = read_csv(path)
        m, s, softmax = calculate_activation_statistics(x, sess, is_feature=1)
        print("\nmode distribution for", path, ":\n", softmax)
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        x = np.array([Image.open(str(fn)).getdata() for fn in files])
        x = x.astype(np.float32)/255.0
        x = x[0:10000, :]
        m, s, softmax = calculate_activation_statistics(x, sess)
        print("\nmode distribution for", path, ":\n", softmax)
  
    return m, s, softmax

def calculate_kl(softmax1, softmax2):
    """ 
    Calculate KL divergence of two distribution
    Params:
    -- softmax1 : The probability for each class of x^r
    -- softmax2 : The probability for each class of x^g

    Returns:
    -- kl_value : The KL divergence of these two probability
    """
    kl_value = 0
    for i in range(len(softmax1)):
        kl_value += softmax1[i]*(np.log(softmax1[i]/softmax2[i]))
    return kl_value

def calculate_given_paths(paths, model_path):
    """
    Calculates CAFD and KL divergence  of two paths.
    Params: 
    -- paths     : Paths of two data sets x^r and x^g
    -- model_path: Path of domain-specific model
 
    Returns:
    -- cafd_value: CAFD
    -- kl_value  : KL divergence
    """

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    create_graph(str(model_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1, softmax1 = _handle_path(paths[0], sess) 
        m2, s2, softmax2 = _handle_path(paths[1], sess)
        cafd_value = calculate_CAFD(m1, s1, m2, s2)
        kl_value = calculate_kl(softmax1, softmax2)
        return cafd_value, kl_value


if __name__ == "__main__":
    if len(sys.argv) == 3:
        IMAGE_PATH[0] = sys.argv[1]
        IMAGE_PATH[1] = sys.argv[2]
    elif len(sys.argv) != 1:
        print("\nUsage: \"python cafd.py\" or \"python cafd.py folder1 folder2\"\n")
        raise Exception("usage error")

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    cafd_value, kl_value = calculate_given_paths(IMAGE_PATH, MODEL_PATH)
    print("\nCAFD: \t ", cafd_value)
    print("KLD: \t ", kl_value)  
    
