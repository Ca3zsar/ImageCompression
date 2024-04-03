# Description: PSNR-HVS metric.
import os
# os.environ['PSNR_HVSM_BACKEND'] = 'torch'

from psnr_hvsm import psnr_hvs_hvsm
import numpy as np

def bt601ycbcr(a):
    """Convert an RGB image into normalized YCbCr."""
    # a is of shape (N, height, width, 3)
    # r, g, b are of shape (N, height, width)
    r = a[..., 0].astype(np.float64)
    g = a[..., 1].astype(np.float64)
    b = a[..., 2].astype(np.float64)
    
    y = np.round(16 + 65.481 * r / 255 + 128.553 * g / 255 + 24.966 * b / 255) / 255
    cb = np.round(128 - 37.797 * r / 255 - 74.203 * g / 255 + 112.0 * b / 255) / 255
    cr = np.round(128 + 112.0 * r / 255 - 93.786 * g / 255 - 18.214 * b / 255) / 255
    return y, cb, cr


def compute_psnr_hvs(image_1, image_2):
    """Compute PSNR-HVS-M between two images."""
    
    # Check if the images are in the range [0, 1]
    if image_1.min() < 0 or image_1.max() > 1:
        raise ValueError('Image_1 must be in the range [0, 1]')
    if image_2.min() < 0 or image_2.max() > 1:
        raise ValueError('Image_2 must be in the range [0, 1]')
    
    image_1_y, *_ = bt601ycbcr(image_1)
    image_2_y, *_ = bt601ycbcr(image_2)

    print(image_1_y.shape)

    # Compute PSNR-HVS-M
    psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(image_1_y, image_2_y, batch=True)

    return psnr_hvs