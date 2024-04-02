import numpy as np

def patchify(image, patch_shape):
    # image: (H, W, C)
    # patch_shape: (ph, pw)

    H, W, C = image.shape
    ph, pw = patch_shape

    # pad image to be divisible by patch_shape
    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw

    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    # split image into patches
    patches = []
    for i in range(0, H+pad_h, ph):
        for j in range(0, W+pad_w, pw):
            patch = image[i:i+ph, j:j+pw]
            patches.append(patch)

    return np.array(patches)