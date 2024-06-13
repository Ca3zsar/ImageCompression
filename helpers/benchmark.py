from typing import Callable
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from helpers.file_operations import write_image


def pad_image(image: np.ndarray, patch_size: int):
    h, w, c = image.shape
    h_pad = (patch_size - (h % patch_size)) % patch_size
    w_pad = (patch_size - (w % patch_size)) % patch_size

    return np.pad(image, ((0, h_pad), (0, w_pad), (0, 0)), mode="edge"), h_pad, w_pad


def reconstruct(
    image: np.ndarray, compress_function, decompress_function
) -> tuple[np.ndarray, np.ndarray]:
    patch_size = 32

    # pad the image
    image *= 255.0
    image, h_pad, w_pad = pad_image(image, patch_size)

    height, width, channels = image.shape

    strings, x_shape, y_shape = compress_function(image)

    reconstructed_image = decompress_function(strings, x_shape, y_shape)
    reconstructed_image = tf.squeeze(reconstructed_image)

    height = height - h_pad
    width = width - w_pad

    reconstructed_image = reconstructed_image[:height, :width]

    # convert reconstructed image to tensor
    reconstructed_image = tf.convert_to_tensor(reconstructed_image)
    reconstructed_image = tf.saturate_cast(reconstructed_image, tf.uint8)

    return reconstructed_image, strings


def compute_bitrate(image, strings):
    # calculate the size of the compressed image
    pixels = tf.cast(tf.reduce_prod(tf.shape(image)[-3:-1]), tf.float32)
    bits = len(strings.numpy()[0]) * 8

    compression_ratio = tf.cast(bits, tf.float32) / pixels

    return compression_ratio


def plot_image(image, result, stats):
    image_yuv = tf.image.yuv_to_rgb(image)
    result_yuv = tf.image.yuv_to_rgb(result)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(tf.squeeze(image))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(tf.squeeze(result))
    plt.axis("off")

    # show ssim, ms-ssim and psnr on the plot
    plt.subplot(1, 3, 3)
    plt.text(0, 0.2, f"SSIM: {stats['ssim'].numpy():.4f}", fontsize=12, color="red")
    plt.text(
        0, 0.4, f"MS-SSIM: {stats['ms_ssim'].numpy():.4f}", fontsize=12, color="red"
    )
    plt.text(0, 0.6, f"PSNR: {stats['psnr'].numpy():.4f}", fontsize=12, color="red")
    plt.text(
        0,
        0.8,
        f"Compression Ratio: {stats['compression_ratio'].numpy():.4f}",
        fontsize=12,
        color="red",
    )
    plt.axis("off")

    # center the text vertically
    plt.tight_layout()

    plt.show()


def compute_benchmark(
    data_directory: str,
    reconstruct_function: Callable,
    network: tf.keras.Model,
    plot: bool = True,
    save: bool = False,
    save_path: str = None,
):
    benchmark_images = tf.io.gfile.glob(data_directory + "*.png")

    psnrs = []
    ssims = []
    ms_ssims = []
    rates = []
    
    if save:
        os.makedirs(f"results/{save_path}",exist_ok=True)

    for image_path in benchmark_images:
        image = tf.image.decode_image(
            tf.io.read_file(image_path), channels=3, dtype="float32"
        )
        # image = tf.image.resize(image, (256, 256))
        image_copy = image

        result, strings = reconstruct_function(
            image_copy, network.compress, network.decompress
        )

        compression_ratio = compute_bitrate(image, strings)

        image = tf.saturate_cast(tf.round(image * 255), tf.uint8)

        image = tf.cast(image, tf.float64) / 255.0
        result = tf.cast(result, tf.float64) / 255.0

        image_copy = image
        result_copy = result

        image_yuv = tf.image.rgb_to_yuv(image)
        result_yuv = tf.image.rgb_to_yuv(result)

        image = image_yuv[:, :, 0]
        result = result_yuv[:, :, 0]

        image = tf.expand_dims(image, axis=-1)
        result = tf.expand_dims(result, axis=-1)

        ssim = tf.image.ssim(image, result, max_val=1)
        ms_ssim = tf.image.ssim_multiscale(image, result, max_val=1)
        psnr = tf.image.psnr(image_yuv, result_yuv, max_val=1)

        ssims.append(ssim)
        ms_ssims.append(ms_ssim)
        psnrs.append(psnr)
        rates.append(compression_ratio.numpy())

        stats = {
            "ssim": ssim,
            "ms_ssim": ms_ssim,
            "psnr": psnr,
            "compression_ratio": compression_ratio,
        }

        if plot:
            plot_image(image_copy, result_copy, stats)

        if save:
            write_image(tf.saturate_cast(tf.round(result_copy * 255), tf.uint8), f'results/{save_path}/{image_path.split("/")[-1]}')
    
    if plot:
        print(f"Average SSIM: {np.mean(ssims):.4f}")
        print(f"Average MS-SSIM: {np.mean(ms_ssims):.4f}")
        print(f"Average PSNR: {np.mean(psnrs):.4f}")
        print(f"Average bpp: {np.mean(rates):.4f}")
    
    return np.mean(ssims), np.mean(ms_ssims), np.mean(psnrs), np.mean(rates)
