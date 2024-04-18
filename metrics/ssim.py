import tensorflow as tf


def ssim(y_true, y_pred, max_val=1.0):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.expand_dims(y_true, 0)
    y_pred = tf.expand_dims(y_pred, 0)
    return tf.image.ssim(y_true, y_pred, max_val=max_val)
