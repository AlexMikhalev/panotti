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
from RMDL import RMDL_Image
from panotti.datautils import *
import tensorflow as tf
#from keras.callbacks import ModelCheckpoint #,EarlyStopping
import os
from os.path import isfile
from timeit import default_timer as timer
# from panotti.multi_gpu import MultiGPUModelCheckpoint


def train_network(weights_file="weights.hdf5", classpath="Preproc/Train/", epochs=50, batch_size=20, val_split=0.25,tile=False):
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
    n_epochs = [100, 100, 100]  ## DNN-RNN-CNN
    Random_Deep = [3, 3, 3]  ## DNN-RNN-CNN
    assert( class_names == class_names_test )
    RMDL_Image.Image_Classification(X_train, Y_train, X_test, Y_test, shape, batch_size=batch_size,
                         sparse_categorical=False, random_deep=Random_Deep, epochs=n_epochs, plot=True,
                         min_hidden_layer_dnn=1, max_hidden_layer_dnn=8, min_nodes_dnn=128, max_nodes_dnn=1024,
                         min_hidden_layer_rnn=1, max_hidden_layer_rnn=5, min_nodes_rnn=32, max_nodes_rnn=128,
                         min_hidden_layer_cnn=3, max_hidden_layer_cnn=10, min_nodes_cnn=128, max_nodes_cnn=512,
                         random_state=np.random.seed(1), random_optimizor=True, dropout=0.6)

    



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
