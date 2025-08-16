"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import numpy as np
import copy
from skimage import transform
import imageio
import nibabel as nib

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


# -----------------------------
# new added functions for cyclegan
# not used in our hyper-GAN
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


def load_test_data(image_path, domain_id):

    imgAll = nib.load(image_path)
    img = imgAll.get_data().astype('single')
    # print('img',img.shape) # (240, 240, 160)
    if domain_id == 0:
        img = img / 3000. * 2. - 1.
    elif domain_id == 1:
        img = img / 5000. * 2. - 1.
    elif domain_id == 2:
        img = img / 6000. * 2. - 1.
    else:
        img = img / 7000. * 2. - 1.
    
    img[img > 1.] = 1.
    
    return img


def load_train_data(image_path, load_size0=256, load_size1=256, fine_size0=240, fine_size1=240, is_testing=False):
    
    img_A = np.load(image_path[0])
    img_B = np.load(image_path[1])
    
    
    if not is_testing:

        padA_size0 = load_size0 - img_A.shape[0]
        padA_size1 = load_size1 - img_A.shape[1]
        padB_size0 = load_size0 - img_B.shape[0]
        padB_size1 = load_size1 - img_B.shape[1]

        img_A = np.pad(img_A, ((int(padA_size0 // 2), int(padA_size0) - int(padA_size0 // 2)),
                               (int(padA_size1 // 2), int(padA_size1) - int(padA_size1 // 2))), mode='constant',
                       constant_values=-1)
        img_B = np.pad(img_B, ((int(padB_size0 // 2), int(padB_size0) - int(padB_size0 // 2)),
                               (int(padB_size1 // 2), int(padB_size1) - int(padB_size1 // 2))), mode='constant',
                       constant_values=-1)
        
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size0 - fine_size0)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size1 - fine_size1)))
        img_A = img_A[h1:h1 + fine_size0, w1: w1 + fine_size1]
        img_B = img_B[h1:h1 + fine_size0, w1: w1 + fine_size1]

    else:
        
        padA_size = fine_size0 - img_A.shape[0]
        padB_size = fine_size0 - img_B.shape[0]

        img_A = np.pad(img_A, ((int(padA_size // 2), int(padA_size) - int(padA_size // 2)), (0, 0)), mode='constant',
                       constant_values=-1)
        img_B = np.pad(img_B, ((int(padB_size // 2), int(padB_size) - int(padB_size // 2)), (0, 0)), mode='constant',
                       constant_values=-1)
    # print('imageA',img_A.shape) # (240, 160)
    # print('imageB',img_B.shape) # (240, 160)

    img_AB = np.dstack((img_A, img_B))
    # print('imageAB',img_AB.shape) # (240, 160, 2)

    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def get_nii_info(input_domain):
    valNum = 5
    valVec = np.arange(valNum) + 1 # [1 2 3 4 5]
    shapeList = []
    headList = []
    affineList = []
    for val_id in valVec:
        input_image_path = r'../BraTS_new/VolumeData/Valid/Valid{}/Dom{}{:03d}.nii.gz'.format(input_domain,input_domain,val_id)
        input_imageAll = nib.load(input_image_path)
        img_shape = input_imageAll.shape
        shapeList.append(img_shape)
        head_output = input_imageAll.get_header() # 获取头文件
        headList.append(head_output)
        affine_output = input_imageAll.affine # 获取仿射矩阵（物理坐标相关）
        affineList.append(affine_output)
    return shapeList, headList, affineList
