import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization, GlobalAveragePooling2D, Dense, Reshape
from tensorflow.nn import relu
from sionna.phy.utils import log10, insert_dims

class SEBlock(Layer):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

    def build(self, input_shape):
        self.global_pool = GlobalAveragePooling2D()
        self.fc1 = Dense(self.channels // self.reduction, activation="relu")
        self.fc2 = Dense(self.channels, activation="sigmoid")

    def call(self, inputs):
        # Squeeze
        z = self.global_pool(inputs)  # [batch, C]
        # Excitation
        z = self.fc1(z)
        z = self.fc2(z)
        # Reshape to [batch, 1, 1, C]
        z = Reshape([1,1,self.channels])(z)
        # Scale
        return inputs * z

class ResidualSEBlock(Layer):
    def __init__(self, num_conv_channels, reduction=16):
        super().__init__()
        self._num_conv_channels = num_conv_channels
        self._reduction = reduction

    def build(self, input_shape):
        self._conv_1 = Conv2D(self._num_conv_channels, (3,3), padding="same", activation=None)
        self._layer_norm_1 = LayerNormalization()
        self._conv_2 = Conv2D(self._num_conv_channels, (3,3), padding="same", activation=None)
        self._layer_norm_2 = LayerNormalization()
        self._se = SEBlock(self._num_conv_channels, self._reduction)

    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = relu(z)
        z = self._conv_2(z)
        z = z + inputs   # residual connection
        z = self._se(z)  # squeeze and excitation
        return z

class NeuralReceiverSE(Layer):
    def __init__(self, num_bits_per_symbol, num_conv_channels=128, num_residual_blocks=5):
        super().__init__()
        self._num_conv_channels = num_conv_channels
        self._num_bits_per_symbol = num_bits_per_symbol
        self._num_residual_blocks = num_residual_blocks

    def build(self, input_shape):
        self._input_conv = Conv2D(self._num_conv_channels, (3,3), padding="same", activation=None)
        self._residual_blocks = [
            ResidualSEBlock(self._num_conv_channels) for _ in range(self._num_residual_blocks)
        ]
        self._output_conv = Conv2D(self._num_bits_per_symbol, (3,3), padding="same", activation=None)

    def call(self, y, no):
        no = log10(no)
        y = tf.transpose(y, [0,2,3,1])  # antena último eje
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])
        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       no], axis=-1)

        z = self._input_conv(z)
        for block in self._residual_blocks:
            z = block(z)
        z = self._output_conv(z)
        return z
