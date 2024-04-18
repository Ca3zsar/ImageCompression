import tensorflow as tf


@tf.custom_gradient
def binary_out(inputs):
    binary = tf.sign(inputs)

    def grad(dy):
        return dy

    return binary, grad


class Encoder(tf.keras.Model):
    def __init__(self, binary_size, patch_size):
        super().__init__()
        self.binary_size = binary_size
        self.patch_size = patch_size

        sizes = [512, 256, 128]
        activations = ["tanh", "tanh", "tanh"]

        self.custom_layers = []
        for i in range(3):
            self.custom_layers.append(
                tf.keras.layers.Dense(
                    sizes[i], activation=activations[i], trainable=True
                )
            )
        self.custom_layers.append(
            tf.keras.layers.Dense(
                binary_size, activation=binary_out, use_bias=False, trainable=True
            )
        )

    def call(self, inputs, training=True):
        # reshape inputs to 1D
        inputs = tf.reshape(inputs, [-1, self.patch_size * self.patch_size * 3])
        bits = inputs
        for layer in self.custom_layers:
            bits = layer(bits)

        return bits

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(self.patch_size, self.patch_size, 3))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))


class Decoder(tf.keras.Model):
    def __init__(self, binary_size, patch_size):
        super().__init__()
        self.binary_size = binary_size
        self.patch_size = patch_size

        sizes = [128, 256, 512]
        activations = ["tanh", "tanh", "tanh"]

        self.custom_layers = []
        for i in range(3):
            self.custom_layers.append(
                tf.keras.layers.Dense(sizes[i], activations[i], trainable=True)
            )

        self.custom_layers.append(
            tf.keras.layers.Dense(patch_size * patch_size * 3, activation="tanh")
        )

    def call(self, inputs):
        x = inputs
        for layer in self.custom_layers:
            x = layer(x)
        x = tf.reshape(x, [-1, self.patch_size, self.patch_size, 3])
        return x

    def build_graph(self):
        x = tf.keras.layers.Input(shape=self.binary_size)
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))


class AutoEncoder(tf.keras.Model):
    def __init__(self, binary_size, patch_size):
        super().__init__()
        self.binary_size = binary_size
        self.patch_size = patch_size

        self.encoder = Encoder(self.binary_size, self.patch_size)
        self.decoder = Decoder(self.binary_size, self.patch_size)

        self.saved_bits = []

    def call(self, inputs):
        bits = self.encoder(inputs)
        out = self.decoder(bits)
        return out


class ResidualAutoEncoder(tf.keras.Model):
    def __init__(self, binary_size, patch_size, steps):
        super().__init__()
        self.binary_size = binary_size
        self.patch_size = patch_size
        self.steps = steps

        self.encoders = [
            Encoder(self.binary_size, self.patch_size) for _ in range(self.steps)
        ]
        self.decoders = [
            Decoder(self.binary_size, self.patch_size) for _ in range(self.steps)
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x = data
        y_pred = self(x, training=False)
        return y_pred

    def call(self, inputs, training=True):
        x = inputs
        patch = tf.zeros_like(x)
        losses = []

        for i in range(self.steps):
            cur_x = x
            bits = self.encoders[i](x)
            x = self.decoders[i](bits)

            output_patch = x

            patch = patch + output_patch

            x = cur_x - x
            losses.append(tf.reduce_mean(tf.square(x)))

        self.add_loss(
            tf.reduce_mean(losses) / self.steps / self.patch_size / self.patch_size
        )

        return patch

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(self.patch_size, self.patch_size, 3))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))
