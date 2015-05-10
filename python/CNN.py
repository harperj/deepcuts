
# coding: utf-8

## Deepcuts CNN

# This is a fully self-contained CNN Architecture for shot boundary detection. The code performs the following tasks:
# 
# - Create a FrameDataSet class inheriting 'DenseDesignMatrix' to hold the data. The Pylearn2 CNN architecture expects a DenseDesignMatrix container.
# 
# - Read all data files
# 
# - Create an ndarray, frame_pair, of dimension Channel x Height x Width for the data of each frame-pair in both the train & test set. The format of the array is as follows:
#     - Channel = The signal type (e.g. horizontal flow)
#     - Height = The height of the image 
#     - Width = The Width of the images
# 
# - Put all of the frame_pairs into a FrameDataSet
# 
# - Define the network architecture using the YAML markup language
# 
# - Run the CNN

### Dependencies

# - <a href=http://deeplearning.net/software/pylearn2/> Pylearn2</a>
# 
# - <a href=http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz3ZD4QUCPt> CUDA (optional)</a>
#     - <a href=http://www.quantstart.com/articles/Installing-Nvidia-CUDA-on-Ubuntu-14-04-for-Linux-GPU-Computing> A helpful CUDA install guide </a>

### Preparation

# - <a href=http://deeplearning.net/software/theano/tutorial/using_gpu.html> Theano GPU Settings </a> (if using cuda)

### References & Helpful Links

# - <a href=http://deeplearning.net/software/pylearn2/tutorial/index.html#tutorial> Python Quick Start Tutorial </a>
# - <a href=http://deeplearning.net/software/pylearn2/yaml_tutorial/index.html#yaml-tutorial> YAML for Pylearn2</a>
# - <a href=http://nbviewer.ipython.org/github/lisa-lab/pylearn2/blob/master/pylearn2/scripts/tutorials/convolutional_network/convolutional_network.ipynb>iPython Notebook CNN Tutorial</a>
# - <a href=http://nbviewer.ipython.org/gist/cancan101/ea563e394ea968127e0e > Fully Contained iPython Notebook CNN </a>
# - <a href=https://github.com/kastnerkyle/kaggle-cifar10/blob/master/kaggle_dataset.py> Sample Dataset Class </a>

# In[227]:

#Put all of the imports here
#NOTE: Importing all of pylearn2 is VERY slow. 
# - Import only the needed components.

get_ipython().magic(u'matplotlib inline')

import numpy as np
from scipy import misc

import theano

from theano.compat.six.moves import xrange
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.datasets import cache
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng

from os import listdir
from os.path import isfile, isdir, join, basename, splitext

import matplotlib
import matplotlib.pyplot as plt

import sys
import Image

import csv





# In[228]:

#Check which device theano uses (gpu or cpu
print theano.config.device


# In[246]:

#Define Global Parameters
TRAIN_DATA = '/home/alex/Desktop/deepcuts_data/signals/'
TEST_DATA = '/home/alex/Desktop/deepcuts_data/signals/'
TRAIN_LABELS = "/home/alex/Desktop/deepcuts_data/shot_annot/csv/"
TEST_LABELS = "/home/alex/Desktop/deepcuts_data/shot_annot/csv/"


FORCED_HEIGHT = 512
FORCED_WIDTH = 512

SIGNALS = ['sample_signal']
SIG_COUNT = 3

#Signals that should be converted to single channel
NEEDS_COLOR_CONVERSION = ['h_flow', 'v_flow', 'luminance'] 



# In[247]:

#Each pixel value is the average from rgb
#This is for mathematical accuracy
#This is NOT for producing visually accurate rgb2gray conversion
def convert(signal_type, im):
    
    im = im.resize((FORCED_HEIGHT, FORCED_WIDTH))
    image = np.asarray(im)
    
    h,w,ch = image.shape
    assert (ch < h and ch < w), 'Invalid Input Size: Expected data input is of size H x W x Ch'
    
    if signal_type in NEEDS_COLOR_CONVERSION:
        output = np.zeros([h,w])
        for i in range(0, ch):
            output = output + image[:,:,i]
        
        image = output / ch        
    #After color conversion, we reshape for CNN input. 
    #CNN Expects ndarry to be N x Channel x Height x Width
    
    #move channels to front axis
    image = image.swapaxes(0,2) #Now image is Ch x W x H
    
    #swap W and H
    image = image.swapaxes(1,2) #Now image is Ch x H x W
    
    return image
    
            
    


# In[248]:

#Get all files in directory
def get_all_files(path):
    files = [f for f in listdir(path) if isfile(join(path, f)) and              f[0] != '.' ] #ignore hidden files
    return sorted(files)


# In[249]:

#Get all subdirectories in a directory
def get_all_dirs(path):
    folders = [f for f in listdir(path) if isdir(join(path, f)) and                f[0] != '.'] #ignore hiddin directories
    return sorted(folders)


# In[250]:

#get all signal files for movie in directory
def get_images(movie, signal, data_path):
    path = data_path + movie + '/' + signal + '/'
    files = get_all_files(path)
    image_dict = {}
    
    for f in files:
        im = Image.open(path + f)
        im.load()
        image_dict[int(f[:-4])] = convert(signal, im)
        
    return image_dict

#get_images('anni005', 'sample_signal')
    


# In[251]:

def get_movie_signals(movie, data_path):
    mov_sigs = {}
    for s in SIGNALS:
        mov_sigs[s] = get_images(movie, s, data_path)
        
    #now vstack them. 
    mov_compiled = mov_sigs[SIGNALS[0]]
    
    for f in mov_compiled:
        for s in SIGNALS[1:]:
            mov_compiled[f] =  np.vstack(mov_compiled[f], mov_sigs[s])
            
    return mov_compiled
    
    
        


# In[252]:

def get_all_sigs(data_path):
    dirs = get_all_dirs(data_path)
    print('Found ' + str(len(dirs)) + ' movies: ')
    print(dirs)
    all_sigs = {}
    for d in dirs:
        all_sigs[d] = get_movie_signals(d, data_path)
    
    return all_sigs


# In[253]:

#Read shot_annot csv files

# Dictionary of all movies
# Each movie will have a list of ShotAnnotations
# read the files into dictionary




def get_labels(label_path):
    
    files = [f for f in listdir(label_path) if isfile(join(label_path, f)) and f[0] != '.']
    annotations = {}
    annot_arrays = {}
    colNames = ['style', 'pre_frame', 'post_frame']
    
    for f in files:
        full_path = label_path + f
        name = (basename(f)).split('.')[0] #get only movie title
        annotations[name] = []
        annot_arrays[name] = {}
        with open(label_path + f, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)    
            idx = 0
            for row in reader:
                #new_annot = ShotAnnotation(f, row[0], row[1], row[2])
                annotations[name].append(new_annot)
                if(row[0] == 'CUT'):
                    annot_arrays[name][row[1]] = 1

    #Debug check: did the dictionary populate as expected?        
    print('Gathered labels for ',len(annot_arrays), 'videos')
    return annot_arrays
    
    
#get_labels()


# In[254]:

#Make the final image and label arrays

def prep_data(data_path, label_path):
    signals = get_all_sigs(data_path)
    labels = get_labels(label_path)
    
    final_labels = []
    signal_ndArrays = np.empty((1, SIG_COUNT, FORCED_HEIGHT, FORCED_WIDTH))
    for m in sorted(signals):
        max_val = max([int(k) for k in signals[m]])
        
        these_labels = [0] * (max_val - 1)
        for cut in labels[m]:
            #-1 because frames are indexed starting at 1
            #these_labels[int(cut) - 1] = labels[cut]
            x = 1
        final_labels.append(these_labels)
        itermov = iter(sorted(signals[m]))
        
        if(len(signal_ndArrays) == 1):
            signal_ndArrays[0,:,:,:] = signals[m][1]
            next(itermov)
            
        for sig in itermov:
            new_array = np.empty((1, SIG_COUNT, FORCED_HEIGHT, FORCED_WIDTH))
            new_array[0,:,:,:] = signals[m][sig]
            signal_ndArrays = np.vstack((signal_ndArrays, new_array))
            
    print final_labels
    print(signal_ndArrays.shape)
    
    return{'labels':final_labels, 'signals':signal_ndArrays}
            


# In[254]:




# In[255]:

#Create the FrameDataSet Class
class FrameDataSet(dense_design_matrix.DenseDesignMatrix):
    
    def __init__(self, data_path, label_path):
        data = prep_data(data_path, label_path)
        labels = data['labels']
        ims = data['signals']
        
        self.num_ims, self.ch, self.h, self.w = ims.shape
        self.img_shape = (self.ch, self.h, self.w)
        start_idx = 0
        self.max_count = sys.maxsize
        self.label_names = [0, 1]
        self.n_classes = len(self.label_names)    
            
            
        #The following parts might not be necessary.
        self.one_hot = False
        self.gcn = 55. #Don't know what this does.
        
    
        def get_labels():
            return labels
        
        self.label_map = {k: v for k, v in zip(self.label_names,                                               range(self.n_classes))}
        self.label_unmap = {v: k for k, v in zip(self.label_names,                                               range(self.n_classes))}
        
        
        
                                         
            
    
            
        
    


# In[260]:

#all_data = FrameDataSet(TRAIN_DATA, TRAIN_LABELS)


# In[259]:

#!/usr/bin/env python
from pylearn2.models import mlp, maxout
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.preprocessing import Pipeline, ZCA
from pylearn2.datasets.preprocessing import GlobalContrastNormalization
from pylearn2.space import Conv2DSpace
from pylearn2.train import Train
from pylearn2.train_extensions import best_params, window_flip
from pylearn2.utils import serial

#Define Datasets here
trn = FrameDataSet(TRAIN_DATA, TRAIN_LABELS)

tst = FrameDataSet(TEST_DATA, TEST_LABELS)


#Define the network here
in_space = Conv2DSpace(shape=(32, 32),
                       num_channels=3,
                       axes=('c', 0, 1, 'b'))

l1 = maxout.MaxoutConvC01B(layer_name='l1',
                           pad=4,
                           tied_b=1,
                           W_lr_scale=.05,
                           b_lr_scale=.05,
                           num_channels=96,
                           num_pieces=2,
                           kernel_shape=(8, 8),
                           pool_shape=(4, 4),
                           pool_stride=(2, 2),
                           irange=.005,
                           max_kernel_norm=.9,
                           partial_sum=33)

l2 = maxout.MaxoutConvC01B(layer_name='l2',
                           pad=3,
                           tied_b=1,
                           W_lr_scale=.05,
                           b_lr_scale=.05,
                           num_channels=192,
                           num_pieces=2,
                           kernel_shape=(8, 8),
                           pool_shape=(4, 4),
                           pool_stride=(2, 2),
                           irange=.005,
                           max_kernel_norm=1.9365,
                           partial_sum=15)

l3 = maxout.MaxoutConvC01B(layer_name='l3',
                           pad=3,
                           tied_b=1,
                           W_lr_scale=.05,
                           b_lr_scale=.05,
                           num_channels=192,
                           num_pieces=2,
                           kernel_shape=(5, 5),
                           pool_shape=(2, 2),
                           pool_stride=(2, 2),
                           irange=.005,
                           max_kernel_norm=1.9365)

l4 = maxout.Maxout(layer_name='l4',
                   irange=.005,
                   num_units=500,
                   num_pieces=5,
                   max_col_norm=1.9)

output = mlp.Softmax(layer_name='y',
                     n_classes=10,
                     irange=.005,
                     max_col_norm=1.9365)

layers = [l1, l2, l3, l4, output]

#Define the model
#Not sure if we need to modify this.
mdl = mlp.MLP(layers,
              input_space=in_space)


#Not sure what this does from here to next comment.
trainer = sgd.SGD(learning_rate=.17,
                  batch_size=128,
                  learning_rule=learning_rule.Momentum(.5),
                  # Remember, default dropout is .5
                  cost=Dropout(input_include_probs={'l1': .8},
                               input_scales={'l1': 1.}),
                  termination_criterion=EpochCounter(max_epochs=475),
                  monitoring_dataset={'valid': tst,
                                      'train': trn})

preprocessor = Pipeline([GlobalContrastNormalization(scale=55.), ZCA()])
trn.apply_preprocessor(preprocessor=preprocessor, can_fit=True)
tst.apply_preprocessor(preprocessor=preprocessor, can_fit=False)
serial.save('deepcuts_preprocessor.pkl', preprocessor)

watcher = best_params.MonitorBasedSaveBest(
    channel_name='valid_y_misclass',
    save_path='deepcuts_maxout_zca.pkl')

velocity = learning_rule.MomentumAdjustor(final_momentum=.65,
                                          start=1,
                                          saturate=250)

decay = sgd.LinearDecayOverEpoch(start=1,
                                 saturate=500,
                                 decay_factor=.01)

win = window_flip.WindowAndFlipC01B(pad_randomized=8,
                                    window_shape=(32, 32),
                                    randomize=[trn],
                                    center=[tst])

#Define experiment
experiment = Train(dataset=trn,
                   model=mdl,
                   algorithm=trainer,
                   extensions=[watcher, velocity, decay, win])

#Run experiment
experiment.main_loop()



# In[ ]:



