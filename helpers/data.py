import keras.datasets
import tensorflow_datasets as tfds


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
