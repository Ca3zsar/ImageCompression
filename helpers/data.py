import keras.datasets
import tensorflow_datasets as tfds
import tensorflow as tf


def normalize(images):
    return images.astype("float32") / 255.0


def load_keras_dataset(dataset_name):
    try:
        dataset = getattr(keras.datasets, dataset_name)
    except AttributeError:
        raise AttributeError(f"Dataset {dataset_name} not found in keras.datasets")

    (x_train, _), (x_test, _) = dataset.load_data()

    x_train = normalize(x_train)
    x_test = normalize(x_test)

    return x_train, x_test


def load_tensorflow_dataset(dataset_name):
    if dataset_name == "oxford_flowers102":
        dataset, _ = tfds.load("oxford_flowers102", with_info=True)
        train_ds = dataset["train"]
        test_ds = dataset["test"]
        return train_ds, test_ds


def check_image_size(image, patchsize):
    shape = tf.shape(image)
    return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def crop_image(image, patchsize):
    image = tf.image.random_crop(image, (patchsize, patchsize, 3))
    return tf.cast(image, tf.keras.mixed_precision.global_policy().compute_dtype)


def get_dataset(name, split, args):
    """Creates input data pipeline from a TF Datasets dataset."""
    with tf.device("/cpu:0"):
        dataset = tfds.load(name, split=split, shuffle_files=True)
        if split.startswith("train"):
            dataset = dataset.repeat()
        dataset = dataset.filter(
            lambda x: check_image_size(x[args["field"]], args["patch_size"])
        )
        dataset = dataset.map(
            lambda x: crop_image(x[args["field"]], args["patch_size"])
        )
    return dataset
