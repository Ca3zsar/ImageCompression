import tensorflow as tf

def write_image(image, path):
    image = tf.image.encode_png(image)
    tf.io.write_file(path, image)