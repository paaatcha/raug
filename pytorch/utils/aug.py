#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the methods and functions to proceed the data augmentation in pytorch

If you find any bug or have some suggestion, please, email me.
"""

import torch
import numpy as np
from torchvision import transforms
import glob
import os
import time
from PIL import Image
import shutil

def get_augmentation (params=None, seed_number=None, verbose=False):
    """
    This function returns a bunch of transformation in order to use for augmentation in pytorch dataset pipeline. The
    transformation should be informed in params. All transformations are performed randomly

    :param params (optional, dictionary): a dictionary containing the following params:
    - prob (float): the probability to perform any augmentation. It should be a value between 0.0 and 1.0. If the key
    is not informed, the Default value is 0.5. Note: Even with prob = 0, resize and convert to tensor are performed.

    - size (int or tuple): a tuple of ints representing width and height to resize the image. If it's just a int, so
    width = height = int. Set it None to keep the original size. If the key is not informed, the default value is
    (128, 128).

    - brightness (float or tuple): how much to jitter brightness. This is chosen uniformly from [max(0, 1 - brightness),
    1 + brightness] or the given [min, max]. Should be non negative numbers. If the key is not informed, the Default
    value is 0.3

    - contrast (float or tuple): how much to jitter contrast. It's chosen uniformly from [max(0, 1 - contrast),
    1 + contrast] or the given [min, max]. Should be non negative numbers. If the key is not informed, the Default value
     is 0.4

    - saturation (float or tuple): how much to jitter saturation. It's chosen uniformly from [max(0, 1 - saturation),
    1 + saturation] or the given [min, max]. Should be non negative numbers. If the key is not informed, the Default
    value is 0.5

    - hue (float or tuple): how much to jitter hue. It's chosen uniformly from [-hue, hue] or the given [min, max].
    Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5. If the key is not informed, the Default value is 0.05.

    - horizontal_flip (float): the probability to perform the flip. It should be 0 < p < 1. If 0 it means the flip will
    not be performed. If the key is not informed, the default value is 0.5

    - vertical_flip (float): the probability to perform the flip. It should be 0 < p < 1. If 0 it means the flip will
    not be performed. If the key is not informed, the default value is 0.5

    - rotation_degrees (float or int): range of degrees to perform a rotation. If degrees is a number instead of tuple,
     like (min, max), the range of degrees will be (-degrees, +degrees). Set to 0 to deactivate rotations. If the key
     is not informed, the default value is (-10, 10)

    - translate (tuple): tuple of maximum absolute fraction for horizontal and vertical translations. For example,
     translate=(a, b), then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a
     and vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b.  Set it None to not
     translate. If the key is not informed, the default value is (0, 0.1)

    - scale (tuple): scaling factor interval, e.g (a, b), then scale is randomly sampled from the range a <= scale <= b.
     Set it None to keep the original scale. If the key is not informed, the default value is (1,2)

    - shear (float or tuple): range of degrees to select from. If degrees is a number instead of sequence, like
    (min, max), the range of degrees will be (-degrees, +degrees). Set it None to not apply shear. If the key is not
    informed, the default value is (1,5)

    - noise (float): how much you'd like to apply random noise on the image. Set it None if you'd like to not include
    noise. The value should be <= 1.0. If the key is not informed, the default value is None.

    - blur (tuple): if you'd like to perform a blur in the image set a tuple containing the (kernel_size, std). If you
     don't wanna perform blur set it as None. If the key is not informed, the default value is None.

    If params is None, the default values of each param is used

    :param seed_number (int, optional): the seed number to keep the shuffle for multiples executions. Default is None.

    :param verbose(bool, optional): if you'd like to print the augmentation params, set it as True. Default is False.

    :return (torchvision.transforms.compose): a compose of transformations
    """

    if (seed_number is not None):
        np.random.seed(seed_number)
        torch.manual_seed(seed_number)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed_number)

    # if params is None, these are the default values:
    prob = 0.5
    to_tensor = True
    size = (128, 128)
    brightness = 0.3
    contrast = 0.4
    saturation = 0.5
    hue = 0.05
    horizontal_flip = 0.5
    vertical_flip = 0.5
    rotation_degrees = (-10, 10)
    translate = (0, 0.1)
    scale = (1,2)
    shear = (1,5)
    noise = None
    blur = None

    # However, if the params is defined, we used the values described on it:
    if (params is not None):
        if ('prob' in params.keys()):
            prob = params['prob']

        if ('to_tensor' in params.keys()):
            to_tensor = params['to_tensor']

        if ('size' in params.keys()):
            size = params['size']

        if ('brightness' in params.keys()):
            brightness = params['brightness']

        if ('contrast' in params.keys()):
            contrast = params['contrast']

        if ('saturation' in params.keys()):
            saturation = params['saturation']

        if ('hue' in params.keys()):
            hue = params['hue']

        if ('horizontal_flip' in params.keys()):
            horizontal_flip = params['horizontal_flip']

        if ('vertical_flip' in params.keys()):
            vertical_flip = params['vertical_flip']

        if ('rotation_degrees' in params.keys()):
            rotation_degrees = params['rotation_degrees']

        if ('translate' in params.keys()):
            translate = params['translate']

        if ('scale' in params.keys()):
            scale = params['scale']

        if ('shear' in params.keys()):
            shear = params['shear']

        if ('noise' in params.keys()):
            noise = params['noise']

        if ('blur' in params.keys()):
            blur = params['blur']

    def get_noise(img):
        """ Just a function to generate noise. It's used in transforms.Lambda"""
        return img + (torch.rand_like(img) * noise)

    # TODO: implement the blur
    def get_blur (img):
        return img

    if (verbose):
        print ("Augmentation parameters: ")
        print (params)

    if (np.random.rand() < prob):

        operations = list();

        if (size is not None):
            operations.append(transforms.Resize(size))

        operations += [
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            transforms.RandomHorizontalFlip(p=horizontal_flip),
            transforms.RandomVerticalFlip(p=vertical_flip),
            transforms.RandomAffine(rotation_degrees, translate=translate, scale=scale, shear=shear) # rotaciona, translada, faz zoom e distorce
        ]

        if (noise is not None):
            operations.append(transforms.Lambda(get_noise))
        if (blur is not None):
            operations.append(transforms.Lambda(get_blur))
        if (to_tensor):
            operations.append(transforms.ToTensor())

        trans = transforms.Compose(operations)
    else:
        trans = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

    return trans


def save_augmentation (img_folder_path, extra_info_suf=None, img_exts=['png'], n_img_aug=None,
                       params=None, seed_number=None, verbose=True):
    """
    This function is used to perform a list of transformations using the `get_transformation` and save all images
    transformed on the same folder informed.

    :param img_folder_path (string): the folder where the images take place and will be saved
    :param extra_info_suf (string, optional): if there are extra information along with the images, you need to inform the suffix. All
    extra files will be replicated with the augmentation. If None, there is no extra information. Default is None
    :param img_exts (list, optional): a list of images extension to load. Default is only ['png']
    :param n_img_aug (int, optional): the number of images to generate. If None, it will be generated 1 augmented image
    for each image on the folder
    :param params (dict, optional): the augmentation parameters. Please, refers to `get_augmentation` to learn about all
    parameters. If None, all default operations described in `get_augmentation` will be performed. Default is None.
    :param seed_number (int, optional): the seed number to keep the shuffle for multiples executions. Default is None.
    """

    # Getting all images path in the folder
    for ext in img_exts:
        img_paths = glob.glob(os.path.join(img_folder_path, '*.' + ext))

    # Getting the amount of images
    n_imgs = len(img_paths)

    # If the number of augmentation is not defined, it will be generate only 1 aug for each image
    if (n_img_aug is None):
        n_img_aug = n_imgs

    # Otherwise, n_img_aug will transformed
    if (n_img_aug <= n_imgs):
        aug_paths = img_paths[0:n_img_aug]
    else:
        n_fac = n_img_aug // n_imgs
        n_rest = n_img_aug % n_imgs

        aug_paths = img_paths * n_fac
        if (n_rest > 0):
            aug_paths = aug_paths + img_paths[0:n_rest]


    if (verbose):
        print("- There are {} images in the folder".format(n_imgs))
        print("- {} images will be transformed and saved in the folder".format(n_img_aug))
        print ("Starting the process... ")
        time.sleep(2)

    params_aug = dict()
    if (params is None):
        params_aug['to_tensor'] = False # it must be false, because we're gonna save the image using PIL
        params_aug['prob'] = 1.0  # it must be 1 because we always wanna an augmentation
        params_aug['size'] = None
    else:
        params_aug = params
        params_aug['to_tensor'] = False  # it must be false, because we're gonna save the image using PIL
        params_aug['prob'] = 1.0  # it must be 1 because we always wanna an augmentation


    aug_ops = get_augmentation(params_aug, seed_number)
    for k, path in enumerate(aug_paths):

        if (verbose):
            print ('Augmenting image {} of {}'.format(k, n_img_aug))

        img = Image.open(path)
        img_aug = aug_ops(img)

        p = path.split('.')
        img_name = p[0] + '_' + str(k) + '.' + p[-1]
        img_aug.save(img_name)

        if (extra_info_suf is not None):
            extra_info_path = p[0] + extra_info_suf
            extra_info_new_path = p[0] + '_' + str(k) + extra_info_suf
            shutil.copy(extra_info_path, extra_info_new_path)
