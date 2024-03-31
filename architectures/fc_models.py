import tensorflow as tf
from keras.layers import Add
from keras.layers import Activation

@tf.custom_gradient
def binary_out(inputs):
    binary = tf.sign(inputs)

    def grad(dy):
        return dy

    return binary, grad

class BinaryLayer(tf.keras.layers.Layer):
    def __init__(self, binary_size):
        super(BinaryLayer, self).__init__()
        self.binary_size = binary_size
        self.binary_code = None

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[input_shape[-1], self.binary_size],
                                      initializer='random_normal',
                                      trainable=True)
        self.bias = self.add_weight("bias",
                                    shape=[self.binary_size],
                                    initializer='random_normal',
                                    trainable=True)
        
    @tf.custom_gradient
    def call(self, inputs):
        outputs = binary_out(inputs)

        self.binary_code = outputs[0]
        outputs[0] = tf.matmul(outputs[0], self.kernel) + self.bias

        return  outputs



class Encoder(tf.keras.Model):
    def __init__(self, binary_size, patch_size):
        super(Encoder, self).__init__()
        self.binary_size = binary_size
        self.patch_size = patch_size

        self.dense_1 = tf.keras.layers.Dense(512, activation='tanh', trainable=True)
        self.dense_2 = tf.keras.layers.Dense(512, activation='tanh', trainable=True)
        self.dense_3 = tf.keras.layers.Dense(512, activation='tanh', trainable=True)
        self.dense_4 = tf.keras.layers.Dense(binary_size, activation=binary_out, use_bias=False, trainable=True)

    def call(self, inputs, training=True):
        # reshape inputs to 1D
        inputs = tf.reshape(inputs, [-1, self.patch_size*self.patch_size*3])
        bits = inputs
        bits = self.dense_1(bits)
        bits = self.dense_2(bits)
        bits = self.dense_3(bits)
        bits = self.dense_4(bits)

        return bits
    

class Decoder(tf.keras.Model):
    def __init__(self, binary_size, patch_size):
        super(Decoder, self).__init__()
        self.binary_size = binary_size
        self.patch_size = patch_size
    
        self.custom_layers = [
            tf.keras.layers.Dense(512, activation='tanh'),
            tf.keras.layers.Dense(512, activation='tanh'),
            tf.keras.layers.Dense(512, activation='tanh'),
            tf.keras.layers.Dense(patch_size*patch_size*3, activation='tanh')
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.custom_layers:
            x = layer(x)
        x = tf.reshape(x, [-1, self.patch_size, self.patch_size, 3])
        return x
    

class AutoEncoder(tf.keras.Model):
    def __init__(self, binary_size, patch_size):
        super(AutoEncoder, self).__init__()
        self.binary_size = binary_size
        self.patch_size = patch_size

        self.encoder = Encoder(self.binary_size, self.patch_size)
        self.decoder = Decoder(self.binary_size, self.patch_size)

        self.saved_bits = []

    def call(self, inputs):
        bits = self.encoder(inputs)
        out = self.decoder(bits)
        return out


class ResidualAutoEncoderShareWeight(tf.keras.Model):
    def __init__(self, binary_size, patch_size, steps):
        super(ResidualAutoEncoderShareWeight, self).__init__()
        self.binary_size = binary_size
        self.patch_size = patch_size
        self.steps = steps

        self.encoder = Encoder(self.binary_size, self.patch_size)
        self.decoder = Decoder(self.binary_size, self.patch_size)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            for i in range(self.steps):
                # append i to the input
                x = self((x, i), training=True)
            
            loss = self.compiled_loss(y, x)

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, x)
        
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        out_bits = self.encoder(inputs)
        out_patch = self.decoder(out_bits)
                                 
        out_patch = inputs - out_patch

        return out_patch

class ResidualAutoEncoder(tf.keras.Model):
    def __init__(self, binary_size, patch_size, steps):
        super(ResidualAutoEncoder, self).__init__()
        self.binary_size = binary_size
        self.patch_size = patch_size
        self.steps = steps

        self.encoders = [Encoder(self.binary_size, self.patch_size) for _ in range(self.steps)]
        self.decoders = [Decoder(self.binary_size, self.patch_size) for _ in range(self.steps)]

    def train_step(self, data):
        x, y = data
        losses = []
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            loss = tf.reduce_sum(losses)

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        x = inputs

        for i in range(self.steps):
            bits = self.encoders[i](x)
            x = self.decoders[i](bits)
            x = inputs - x

            self.add_loss(tf.reduce_sum(tf.square(x)))
        
        return x
