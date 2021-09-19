# ========================================================== #
# File name: SNN_Simulation_cifar100.py
# Python Version: 3.6
# Tensorflow-gpu Version: 2.1.0
# Keras Version: 2.3.1
# dataset: cifar100
#
# This file is based on snntoolbox (converting ANN to SNN and emulating it):
# https://github.com/NeuromorphicProcessorProject/snn_toolbox
#
# You can either run it directly or set the parameters of the config file in your code according to your particular circumstances.
# You can refer to snntoolbox's documentation for how to set the parameters:
# https://github.com/NeuromorphicProcessorProject/snn_Toolbox/blob/master/docs/source/index.rst
# ========================================================== #
import os
import sys
import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser
# save the output results to an external file
def savefile(name):
    deskpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '.'))
    fullpath = deskpath + name + '.txt'
    file = open(fullpath, 'w')
filename = 'logfile_SNN_cifar100'
savefile(filename)
output = sys.stdout
outputfile = open(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '.')) + filename + '.txt',
                  'w')
sys.stdout = outputfile
time_start= datetime.datetime.now()
print("start     "+time_start.strftime('%Y.%m.%d-%H:%M:%S'))
# WORKING DIRECTORY #
#####################

# Define path where model and output files will be stored.
path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '.'))

# GET DATASET #
###############
num_classes = 100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

# Save dataset so SNN toolbox can find it.
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
# SNN toolbox will not do any training, but we save a subset of the training
# set so the toolbox can use it when normalizing the network parameters.
np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train[::100])

model_name = 'cifar100'

# SNN TOOLBOX CONFIGURATION #
#############################

# Create a config file with experimental setup for SNN Toolbox.
# For parameter setting, you can refer to: https://snntoolbox.readthedocs.io/en/latest/guide/configuration.html#

configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
        'path_wd ': path_wd,  # Path to model.
        'dataset_path': path_wd,  # Path to dataset.
        'filename_ann': model_name  # Name of input model.
    }

config['tools'] = {
        'evaluate_ann': True,  # Test ANN on dataset before conversion.
        'normalize': True, # Normalize weights for full dynamic range.
        'parse': True
    }

config['simulation'] = {
        'simulator': 'INI',  # Chooses execution backend of SNN toolbox.
        'duration': 1100,  # Number of time steps to run each sample.
        'num_to_test': 10000,  # How many test samples to run.
        'batch_size': 20,  # Batch size for simulation.
        'keras_backend': 'tensorflow'  # Which keras backend to use.
    }

# config['output']={
#      'plot_vars':{
#         'error_t'
#     }
# }

# Store config file.
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# RUN SNN TOOLBOX #
###################
main(config_filepath)
time_end= datetime.datetime.now()
print("end     "+time_end.strftime('%Y.%m.%d-%H:%M:%S'))
outputfile.close()