from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
import argparse
image_rows = int(420)
image_cols = int(580)
def fetch_train_idx(train_data_path):
    names=os.listdir(train_data_path)
    d=dict([(x.split('.')[0],x.split('.')[0].endswith('mask')) for x in names])
    d1=sorted(d)
    l=len(d1)
    newimg=[]
    newmask=[]
    for i in range(l-1):
        if d1[i+1].endswith('mask'):
            newimg.append(d1[i]+'.tif')
            newmask.append(d1[i+1]+'.tif')
    return newimg, newmask

def create_train_data(data_path):
    train_data_path = os.path.join(data_path, 'train')
    images, mask4images = fetch_train_idx(train_data_path)
    
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name , mask_name in zip(images, mask4images) :

        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(train_data_path, mask_name), as_gray=True)

        img = np.array([img])
        img_mask = np.array([img_mask])
        
        imgs[i] = img
        imgs_mask[i] = img_mask
        #print('Loading done.')

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    

    np.save(data_path+'imgs_train.npy', imgs)
    np.save(data_path+'imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data(data_path):
    imgs_train = np.load(data_path+'imgs_train.npy')
    imgs_mask_train = np.load(data_path+'imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data(data_path):
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = int(len(images))

    imgs = np.ndarray((total, image_rows, image_cols),dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
            print("shape so far", imgs.shape)
        i += 1
    print('Loading done.')

    np.save(data_path+'imgs_test.npy', imgs)
    np.save(data_path+'imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data(data_path):
    imgs_test = np.load(data_path+'imgs_test.npy')
    imgs_id = np.load(data_path+'imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    image_rows = int(420)
    image_cols = int(580)
    parser = argparse.ArgumentParser(description='keras Unet data preprocess')
    parser.add_argument('--parent', required=True,   help='location of the root directory')
    args = parser.parse_args()
    data_path = args.parent
    create_test_data(data_path)
    create_train_data(data_path)
