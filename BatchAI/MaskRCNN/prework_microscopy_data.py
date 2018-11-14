# to install Conx package go to https://github.com/Calysto/conx
# https://github.com/Calysto/conx-notebooks/tree/master/HowToRun#how-to-run-conx
# exercise for all deep learning : https://github.com/Calysto/conx-notebooks
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
tile_size = (64, 64)
# dataset from :https://www.kaggle.com/kmader/electron-microscopy-3d-segmentation

train_em_image_vol = imread('./training.tif')[:40, ::2, ::2]
train_em_seg_vol = imread('./training_groundtruth.tif')[:40, ::2, ::2]>0
test_em_image_vol = imread('./training.tif')[:40, ::2, ::2]
test_em_seg_vol = imread('./training_groundtruth.tif')[:40, ::2, ::2]>0
print("Data Loaded, Dimensions", train_em_image_vol.shape,'->',train_em_seg_vol.shape)
def g_random_tile(em_image_vol, em_seg_vol):
    z_dim, x_dim, y_dim = em_image_vol.shape
    z_pos = np.random.choice(range(z_dim))
    x_pos = np.random.choice(range(x_dim-tile_size[0]))
    y_pos = np.random.choice(range(y_dim-tile_size[1]))
    return np.expand_dims(em_image_vol[z_pos, x_pos:(x_pos+tile_size[0]), y_pos:(y_pos+tile_size[1])],-1), \
            np.expand_dims(em_seg_vol[z_pos, x_pos:(x_pos+tile_size[0]), y_pos:(y_pos+tile_size[1])],-1).astype(float)
np.random.seed(2018)
t_x, t_y = g_random_tile(test_em_image_vol, test_em_seg_vol)
print('x:', t_x.shape, 'Range:', t_x.min(), '-', t_x.max())
print('y:', t_y.shape, 'Range:', t_y.min(), '-', t_y.max())
test_em_image_vol.shape, test_em_seg_vol.shape
num_imgs=50
track=0
for i in range(num_imgs):
    im_x,im_y=g_random_tile(test_em_image_vol, test_em_seg_vol)
    if len(np.unique(im_y))>1 and track<=num_imgs:
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(im_x.squeeze())
        plt.title("img"+str(track))
        plt.imsave('./data/images/{}.jpg'.format(str(track)),im_x.squeeze())
        f.add_subplot(1,2, 2)
        plt.imshow(im_y.squeeze())
        plt.title("mask"+str(track))
        plt.imsave('./data/masks/{}.jpg'.format(str(track)),im_y.squeeze())
        print(np.max(im_y), np.min(im_y))
        print(np.unique(im_y, return_counts=True), len(np.unique(im_y)))
        plt.show(block=True)
        track+=1
