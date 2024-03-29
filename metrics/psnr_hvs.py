# Description: PSNR-HVS metric.
import os
# os.environ['PSNR_HVSM_BACKEND'] = 'torch'

from psnr_hvsm import psnr_hvs_hvsm, bt601ycbcr

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