#! /usr/bin/env python3

'''
Classify sounds using database
Author: Scott H. Hawley

This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

Trained using Fraunhofer IDMT's database of monophonic guitar effects,
   clips were 2 seconds long, sampled at 44100 Hz
'''
from __future__ import print_function
import numpy as np
import librosa
from models import ELM 
from panotti.datautils import *
import tensorflow as tf
#from keras.callbacks import ModelCheckpoint #,EarlyStopping
import os
from os.path import isfile
from timeit import default_timer as timer
# from panotti.multi_gpu import MultiGPUModelCheckpoint


def train_network(weights_file="weights.hdf5", classpath="Preproc/Train/", epochs=50, batch_size=50, val_split=0.25,tile=False):
    np.random.seed(1)
    from keras import backend as K
    # prevent TF from consuming whole memory in GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    K.set_session(sess)
    K.set_image_data_format('channels_last') #make sure we use current 

    print("GPU available ", K.tensorflow_backend._get_available_gpus())
    # Get the data
    X_train, Y_train, paths_train, class_names = build_dataset(path=classpath, batch_size=batch_size, tile=tile)


    # Score the model against Test dataset
    X_test, Y_test, paths_test, class_names_test  = build_dataset(path=classpath+"../Test/", tile=tile)
    shape=get_sample_dimensions(class_names,path=classpath)

    assert( class_names == class_names_test )
    # Construct ELM
    hidden_num = 150
    print("batch_size : {}".format(batch_size))
    print("hidden_num : {}".format(hidden_num))
    elm = ELM(sess, batch_size, 784, hidden_num, 10)
    elm.feed(X_train, Y_train)
    # testing
    elm.test(X_test, Y_test)

    



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="trains network using training dataset")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-c', '--classpath', #type=argparse.string,
        help='Train dataset directory with list of classes', default="Preproc/Train/")
    parser.add_argument('--epochs', default=20, type=int, help="Number of iterations to train for")
    parser.add_argument('--batch_size', default=40, type=int, help="Number of clips to send to GPU at once")
    parser.add_argument('--val', default=0.25, type=float, help="Fraction of train to split off for validation")
    parser.add_argument("--tile", help="tile mono spectrograms 3 times for use with imagenet models",action="store_true")
    args = parser.parse_args()
    train_network(weights_file=args.weights, classpath=args.classpath, epochs=args.epochs, batch_size=args.batch_size,
        val_split=args.val, tile=args.tile)
