import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

def lums_diff(im1, im2):
    assert(im1.shape[2] == 3)
    assert(im2.shape[2] == 3)
    
    def lum(im):
        return 0.2126 * im[:, :, 0] + 0.7152 * im[:, :, 1] + 0.0722 * im[:, :, 2]
    return lum(im1) - lum(im2)

def rgb_diff(im1, im2):
    assert(im1.shape[2] == 3)
    assert(im2.shape[2] == 3)
    
    r_diff = im1[:, :, 0] - im2[:, :, 0]
    g_diff = im1[:, :, 1] - im2[:, :, 1]
    b_diff = im1[:, :, 2] - im2[:, :, 2]
    return r_diff, g_diff, b_diff

folders = glob('.././signals/*')
for folder in folders:  
    for s in ['bw', 'red', 'green', 'blue']:
        if not os.path.exists(os.path.join(folder, s)):
            os.mkdir(os.path.join(folder, s)) 

    inpath = os.path.join(folder, 'img')
    #inpath = os.path.join(folder, 'sample_signal')
    
    images = glob(inpath + '/*')
    for i in range(len(images)):
        im1 = plt.imread(images[i - 1])
        im2 = plt.imread(images[i])
        
        bw_diff = lums_diff(im1, im2)
        r_diff, g_diff, b_diff = rgb_diff(im1, im2)
        
        outfname = os.path.basename(images[i - 1])

        plt.imsave(os.path.join(folder, 'bw', outfname),   bw_diff)
        plt.imsave(os.path.join(folder, 'red', outfname),   r_diff)
        plt.imsave(os.path.join(folder, 'green', outfname), g_diff)
        plt.imsave(os.path.join(folder, 'blue', outfname),  b_diff)
        