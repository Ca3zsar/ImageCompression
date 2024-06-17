

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import numpy as np

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_datasets as tfds

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from helpers.data import get_dataset

tf.config.run_functions_eagerly(False)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.autograph.set_verbosity(
    level=3, alsologtostdout=False
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


args = {
    "patch_size" : 256,
    "batch_size" : 8,
    "num_filters" : 512,
    "latent_dims" : 64,
    "lambda_value" : 1,
    "num_epochs" : 50,
    "steps_per_epoch" : 1000,
    "checkpoint_dir" : "models/checkpoints/",
    "model_dir" : "models/",
    "model_name" : "end_to_end_v14",
    "field" : "image"
}


class AnalysisTransform(tf.keras.Sequential):
    """
    The analysis transform.
    """

    def __init__(self, num_filters, latent_dims):
        super().__init__(name="analysis_transform")
        self.add(tf.keras.layers.Lambda(lambda x: x / 255.))
        self.add(tfc.SignalConv2D(
            num_filters, (9, 9), name='layer0', corr=True, strides_down=4,
            padding='same_zeros', use_bias=True, activation=tfc.GDN(name='gdn0')
        ))
        self.add(tfc.SignalConv2D(
            num_filters // 2, (5, 5), name='layer1', corr=True, strides_down=2,
            padding='same_zeros', use_bias=True, activation=tfc.GDN(name='gdn1')
        ))
        self.add(tfc.SignalConv2D(
            num_filters // 4, (5, 5), name='layer2', corr=True, strides_down=2,
            padding='same_zeros', use_bias=True, activation=tfc.GDN(name='gdn2')
        ))
        self.add(tfc.SignalConv2D(
            latent_dims, (5, 5), name='layer3', corr=True, strides_down=2,
            padding='same_zeros', use_bias=False, activation=None
        ))


class SynthesisTransform(tf.keras.Sequential):
    """
    The synthesis transform.
    """

    def __init__(self, num_filters):
        super().__init__(name="synthesis_transform")
        self.add(tfc.SignalConv2D(
            num_filters // 4, (5, 5), name='layer0', corr=False, strides_up=2,
            padding='same_zeros', use_bias=True, activation=tfc.GDN(name='igdn0', inverse=True)
        ))
        self.add(tfc.SignalConv2D(
            num_filters // 2, (5, 5), name='layer1', corr=False, strides_up=2,
            padding='same_zeros', use_bias=True, activation=tfc.GDN(name='igdn1', inverse=True)
        ))
        self.add(tfc.SignalConv2D(
            num_filters, (5, 5), name='layer2', corr=False, strides_up=2,
            padding='same_zeros', use_bias=True, activation=tfc.GDN(name='igdn2', inverse=True)
        ))
        self.add(tfc.SignalConv2D(
            3, (9, 9), name='layer3', corr=False, strides_up=4,
            padding='same_zeros', use_bias=True, activation=None
        ))
        self.add(tf.keras.layers.Lambda(lambda x: x * 255.))


class EndToEndModel(tf.keras.Model):
    """
    End-to-end learned image compression model.
    """

    def __init__(self, lmbda, num_filters, latent_dims, prior_values = None):
        super().__init__()
        
        self.lmbda = lmbda
        self.analysis_transform = AnalysisTransform(num_filters, latent_dims)
        self.synthesis_transform = SynthesisTransform(num_filters)

        self.prior = tfc.NoisyDeepFactorized(batch_shape=(latent_dims,))
        self.entropy_model = tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=3, compression=True)

    def call(self, x, training):
        """Computes rate and distortion losses"""
        entropy_model = tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=3, compression=False)

        x = tf.cast(x, self.compute_dtype)
        y = self.analysis_transform(x, training=training)

        y_tilde, bits = entropy_model(y, training=training)
        x_tilde = self.synthesis_transform(y_tilde, training=training)

        # Compute the number of bits divided by the number of pixels
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), tf.float32)
        bpp = tf.reduce_sum(bits) / num_pixels

        # Compute the distortion
        # distortion = tf.reduce_mean(tf.math.squared_difference(x, x_tilde))
        distortion = tf.compat.v1.losses.absolute_difference(x,
                                              x_tilde,
                                              reduction=tf.compat.v1.losses.Reduction.MEAN)
        # distortion = 1 - tf.image.ssim(x, x_tilde, max_val=255)
        distortion = tf.cast(distortion, tf.float32)

        # Compute the rate-distortion loss
        loss = bpp + self.lmbda * distortion

        return loss, bpp, distortion
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss, bpp, distortion = self(x, training=True)

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.distortion.update_state(distortion)

        return {m.name: m.result() for m in [self.loss, self.bpp, self.distortion]}
    
    def test_step(self, x):
        loss, bpp, distortion = self(x, training=False)
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.distortion.update_state(distortion)

        return {m.name: m.result() for m in [self.loss, self.bpp, self.distortion]}
    
    def compile(self, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs
        )

        self.loss = tf.keras.metrics.Mean(name="loss")
        self.bpp = tf.keras.metrics.Mean(name="bpp")
        self.distortion = tf.keras.metrics.Mean(name="distortion")

    def fit(self, *args, **kwargs):
        retval = super().fit(*args, **kwargs)

        self.entropy_model = tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=3, compression=True)
        return retval
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8)])
    def compress(self, x):
        x = tf.expand_dims(x, 0)
        x = tf.cast(x, self.compute_dtype)

        y = self.analysis_transform(x, training=False)
        
        x_shape = tf.shape(x)[1:-1]
        y_shape = tf.shape(y)[1:-1]
        
        return self.entropy_model.compress(y), x_shape, y_shape
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, ), dtype=tf.string),
        tf.TensorSpec(shape=(2, ), dtype=tf.int32),
        tf.TensorSpec(shape=(2, ), dtype=tf.int32)
    ])
    def decompress(self, string, x_shape, y_shape):
        
        y_tilde = self.entropy_model.decompress(string, y_shape)
        x_tilde = self.synthesis_transform(y_tilde, training=False)

        x_tilde = x_tilde[0, :x_shape[0], :x_shape[1], :]

        return tf.saturate_cast(tf.round(x_tilde), tf.uint8)


def train(model=None):
    """Instantiates and trains the model."""
    # reduceLRonPlateau = tf.keras.callbacks.ReduceLROnPlateau(
    reduceLRonPlateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        verbose=1,
        mode="min",
        min_delta=0.01,
        min_lr=1e-8,
    )

    if model == None:
        model = EndToEndModel(
                    args["lambda_value"], args["num_filters"], args["latent_dims"]
                )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=0.5)
    )

    train_dataset = get_dataset("clic", "train", args)
    validation_dataset = get_dataset("clic", "validation", args)

    train_dataset = train_dataset.batch(args["batch_size"]).prefetch(8)
    validation_dataset = validation_dataset.batch(args["batch_size"]).cache()
    model.fit(
        train_dataset, 
        epochs=args["num_epochs"],
        steps_per_epoch=args["steps_per_epoch"],
        validation_data=validation_dataset, 
        validation_freq=1,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'{args["model_dir"]}/{args["model_name"]}/logs', histogram_freq=1, update_freq="epoch"
            ),
            tf.keras.callbacks.BackupAndRestore(f'{args["model_dir"]}/{args["model_name"]}'),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=args["checkpoint_dir"] + args["model_name"] + ".h5",
                save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1,
                # initial_value_threshold=0.77
            ),
            reduceLRonPlateau,
        ],
        verbose=1,
    )
    model.save(args["model_dir"] + args["model_name"])


if __name__ == "__main__":
    train()
