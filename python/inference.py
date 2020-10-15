"""
This file is part of midas-cpp

MIT License

Copyright (c) 2020 Shing Yan Loo (lsyan@ualberta.ca)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import midas

import glob
import os

import cv2
import numpy as np

def visualize_minmax(arr, cmap=None, min_val=None, max_val=None):
    """
    Given a 2D NumPy array (h x w), visualize the value (pixel)
    based on the min-max range

    :param arr: 2D NumPy array
    :param cmap: (Optional) One of the OpenCV colormaps (https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html)
    :param min_val: (Optional) clip the min range
    :param max_val: (Optional) clip the max range
    :return: heatmap
    """
    # Clip min-max range
    if min_val is not None:
        arr[arr < min_val] = min_val
    if max_val is not None:
        arr[arr > max_val] = max_val
    
    # Normalize the range
    arr_norm = cv2.normalize(arr, None, norm_type=cv2.NORM_MINMAX)

    # Apply OpenCV colormap
    if cmap is not None:
        arr_norm = np.expand_dims(arr_norm, axis=-1) * 255
        arr_norm = arr_norm.astype(np.uint8)
        arr_norm = cv2.applyColorMap(arr_norm, cmap)

    return arr_norm


if __name__ == '__main__':
    # Load an image
    # image = cv2.imread('../sample_image/bicycle.jpg')
    image = cv2.imread('../sample_image/horses.jpg')

    # Load MiDaS
    height, width = image.shape[:2]
    midas = midas.MiDas(width, height, '../traced_model.pt')

    # Predict depth
    depth = midas.inference(image)

    # Visualize depth
    depth_visual = visualize_minmax(depth, cmap=cv2.COLORMAP_JET)
    cv2.imshow('depth prediction', np.vstack((image, depth_visual)))
    cv2.waitKey(0)
