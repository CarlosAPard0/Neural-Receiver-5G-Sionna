import tensorflow as tf  
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization  
from tensorflow.nn import relu  
from sionna.phy.utils import log10, insert_dims  
  
class ResidualBlock(Layer):  
    r"""  
    This Keras layer implements a convolutional residual block made of two convolutional layers with ReLU activation, layer normalization, and a skip connection.  
    The number of convolutional channels of the input must match the number of kernel of the convolutional layers ``num_conv_channel`` for the skip connection to work.  
    """  
  
    def __init__(self, num_conv_channels):  
        super().__init__()  
        self._num_conv_channels = num_conv_channels  
  
    def build(self, input_shape):  
        # First conv layer  
        self._conv_1 = Conv2D(filters=self._num_conv_channels,  
                              kernel_size=[3,3],  
                              padding='same',  
                              activation=None)  
        # First layer norm  
        self._layer_norm_1 = LayerNormalization()  
        # Second conv layer  
        self._conv_2 = Conv2D(filters=self._num_conv_channels,  
                              kernel_size=[3,3],  
                              padding='same',  
                              activation=None)  
        # Second layer norm  
        self._layer_norm_2 = LayerNormalization()  
  
    def call(self, inputs):  
        z = self._layer_norm_1(inputs)  
        z = relu(z)  
        z = self._conv_1(z)  
        z = self._layer_norm_2(z)  
        z = relu(z)  
        z = self._conv_2(z)  
        # Skip connection  
        z = z + inputs  
        return z  
class NeuralReceiver(Layer):  
    def __init__(self, num_bits_per_symbol, num_conv_channels=128, num_residual_blocks=5):  
        super().__init__()  
        self._num_conv_channels = num_conv_channels  
        self._num_bits_per_symbol = num_bits_per_symbol  
        self._num_residual_blocks = num_residual_blocks

  
    def build(self, input_shape):  
        # Input convolution  
        self._input_conv = Conv2D(filters=self._num_conv_channels,  
                                  kernel_size=[3,3],  
                                  padding='same',  
                                  activation=None)  
          
        # Create residual blocks dynamically based on configuration  
        self._residual_blocks = []  
        for i in range(self._num_residual_blocks):  
            self._residual_blocks.append(ResidualBlock(self._num_conv_channels))  
          
        # Output conv  
        self._output_conv = Conv2D(filters=self._num_bits_per_symbol,  
                                   kernel_size=[3,3],  
                                   padding='same',  
                                   activation=None)  
  
    def call(self, y, no):  
        # Feeding the noise power in log10 scale helps with the performance  
        no = log10(no)  
  
        # Stacking the real and imaginary components of the different antennas along the 'channel' dimension  
        y = tf.transpose(y, [0, 2, 3, 1]) # Putting antenna dimension last  
        no = insert_dims(no, 3, 1)  
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])  
        # z : [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 1]  
        z = tf.concat([tf.math.real(y),  
                       tf.math.imag(y),  
                       no], axis=-1)  
          
        # Input conv  
        z = self._input_conv(z)  
          
        # Apply residual blocks dynamically  
        for residual_block in self._residual_blocks:  
            z = residual_block(z)  
          
        # Output conv  
        z = self._output_conv(z)  
  
        return z