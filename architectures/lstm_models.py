import tensorflow as tf
from tensorflow.keras import layers


@tf.custom_gradient
def binary_out(inputs):
    binary = tf.sign(inputs)

    def grad(dy):
        return dy

    return binary, grad


class Encoder(tf.keras.models.Model):
    def __init__(self, batch_size, binary_size, patch_size):
        super().__init__()

        self.batch_size = batch_size
        self.binary_size = binary_size
        self.patch_size = patch_size

        self.hidden_size = 512

        self.custom_layers = [
            layers.Dense(self.hidden_size, activation="tanh", trainable=True),
            layers.LSTM(self.hidden_size, return_state=True, trainable=True),
            layers.LSTM(self.hidden_size, return_state=True, trainable=True),
        ]
        self.custom_layers.append(
            tf.keras.layers.Dense(
                binary_size, activation=binary_out, use_bias=False, trainable=True
            )
        )

    def init_state(self):
        h_1_0 = tf.zeros([self.batch_size, self.hidden_size])
        c_1_0 = tf.zeros([self.batch_size, self.hidden_size])
        h_2_0 = tf.zeros([self.batch_size, self.hidden_size])
        c_2_0 = tf.zeros([self.batch_size, self.hidden_size])
        return (h_1_0, c_1_0), (h_2_0, c_2_0)

    def call(self, inputs, training=True):
        x = inputs

        x = tf.reshape(x, [-1, self.patch_size * self.patch_size * 3])
        x = self.custom_layers[0](x)
        h_out1, c_out1 = self.custom_layers[1](x)
        h_out2, c_out2 = self.custom_layers[2](h_out1)
        bits = self.custom_layers[3](h_out2)
        new_state = ((h_out1, c_out1[1]), (h_out2, c_out2[1]))
        return bits, new_state

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(self.patch_size, self.patch_size, 3))
        state = self.init_state()

        return tf.keras.models.Model(inputs=[x], outputs=self.call((x, state)))


class Decoder(tf.keras.models.Model):
    def __init__(self, batch_size, binary_size, patch_size):
        super().__init__()

        self.batch_size = batch_size
        self.binary_size = binary_size
        self.patch_size = patch_size

        self.hidden_size = 512

        self.custom_layers = [
            layers.LSTMCell(self.hidden_size, trainable=True),
            layers.LSTMCell(self.hidden_size, trainable=True),
            layers.Dense(self.hidden_size, activation="tanh", trainable=True),
        ]
        self.custom_layers.append(
            layers.Dense(patch_size * patch_size * 3, activation="tanh", trainable=True)
        )

    def init_state(self):
        h_1_0 = tf.zeros([self.batch_size, self.hidden_size])
        c_1_0 = tf.zeros([self.batch_size, self.hidden_size])
        h_2_0 = tf.zeros([self.batch_size, self.hidden_size])
        c_2_0 = tf.zeros([self.batch_size, self.hidden_size])
        return ((h_1_0, c_1_0), (h_2_0, c_2_0))

    def call(self, inputs):
        x, state = inputs
        h_out1, c_out1 = self.custom_layers[0](x, state[0])
        h_out2, c_out2 = self.custom_layers[1](h_out1, state[1])
        x = self.custom_layers[2](h_out2)
        x = self.custom_layers[3](x)
        x = tf.reshape(x, [-1, self.patch_size, self.patch_size, 3])
        new_state = ((h_out1, c_out1[1]), (h_out2, c_out2[1]))
        return x, new_state

    def build_graph(self):
        x = tf.keras.layers.Input(shape=self.binary_size)
        state = self.init_state()

        return tf.keras.models.Model(inputs=[x], outputs=self.call((x, state)))


class LSTM_Autoencoder(tf.keras.models.Model):
    def __init__(self, batch_size, binary_size, patch_size, steps):
        super().__init__()

        self.steps = steps
        self.batch_size = batch_size
        self.binary_size = binary_size
        self.patch_size = patch_size

        self.encoder = Encoder(batch_size, binary_size, patch_size)
        self.decoder = Decoder(batch_size, binary_size, patch_size)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        x = inputs
        encoder_state = self.encoder.init_state()
        decoder_state = self.decoder.init_state()

        for _ in range(self.steps):
            bits, encoder_state = self.encoder((x, encoder_state))
            x, decoder_state = self.decoder((bits, decoder_state))

            residual = x - inputs
            self.add_loss(tf.reduce_mean(tf.square(residual)))
            x = residual

        return residual

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(self.patch_size, self.patch_size, 3))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))
