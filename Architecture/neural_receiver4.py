import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization, LSTM, Bidirectional
from tensorflow.nn import relu
from sionna.phy.utils import log10, insert_dims

class ResidualBlock(Layer):
    def __init__(self, num_conv_channels):
        super().__init__()
        self._num_conv_channels = num_conv_channels

    def build(self, input_shape):
        self._conv_1 = Conv2D(filters=self._num_conv_channels,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)
        self._layer_norm_1 = LayerNormalization()
        self._conv_2 = Conv2D(filters=self._num_conv_channels,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)
        self._layer_norm_2 = LayerNormalization()

    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = relu(z)
        z = self._conv_2(z)
        return z + inputs

class NeuralReceiverCNNLSTM(Layer):
    """
    CNN + LSTM receiver that preserves the output shape:
      [batch, num_ofdm_symbols, num_subcarriers, num_bits_per_symbol]
    LSTM is applied along the OFDM-symbol (time) axis for each subcarrier independently.
    """
    def __init__(self, num_bits_per_symbol, num_conv_channels=128,
                 num_residual_blocks=3, lstm_units=128):
        super().__init__()
        self._num_conv_channels = num_conv_channels
        self._num_bits_per_symbol = num_bits_per_symbol
        self._num_residual_blocks = num_residual_blocks
        self._lstm_units = lstm_units

    def build(self, input_shape):
        # input_shape corresponds to y (before transpose) when Keras constructs the layer.
        # We don't rely on static spatial dims except for building conv layers which are shape-agnostic.
        self._input_conv = Conv2D(filters=self._num_conv_channels,
                                  kernel_size=[3,3],
                                  padding='same',
                                  activation=None)
        # Residual blocks (optional)
        self._residual_blocks = []
        for _ in range(self._num_residual_blocks):
            self._residual_blocks.append(ResidualBlock(self._num_conv_channels))

        # LSTM (bidirectional) that will be applied on flattened (batch * num_subcarriers)
        self._bilstm = Bidirectional(LSTM(self._lstm_units, return_sequences=True))

        # Final 1x1 conv to map features -> num_bits_per_symbol
        self._output_conv = Conv2D(filters=self._num_bits_per_symbol,
                                   kernel_size=[1,1],
                                   padding='same',
                                   activation=None)

    def call(self, y, no):  
        # same preprocessing as your original network  
        no = log10(no)  
      
        # Put antenna/channel last: y original assumed [batch, ant, time, subcarriers]  
        y = tf.transpose(y, [0, 2, 3, 1])  # -> [batch, time, subcarriers, antennas]  
        no = insert_dims(no, 3, 1)  
        no = tf.tile(no, [1, tf.shape(y)[1], tf.shape(y)[2], 1])  
      
        z = tf.concat([tf.math.real(y),  
                       tf.math.imag(y),  
                       no], axis=-1)   # [batch, time, subcarriers, channels]  
      
        # Initial conv + residual blocks (spatial feature extraction)  
        z = self._input_conv(z)                # [batch, time, subcarriers, num_conv_channels]  
          
        # AGREGAR: Información de shape explícita  
        z.set_shape([None, None, None, self._num_conv_channels])  
          
        for rb in self._residual_blocks:  
            z = rb(z)  
      
        # Prepare to apply LSTM per-subcarrier:  
        # current z: [B, T, F, C] where T=num_ofdm_symbols, F=num_subcarriers  
        shape = tf.shape(z)  
        B = shape[0]  
        T = shape[1]  
        F = shape[2]  
        C = shape[3]  
      
        # MODIFICAR: Usar tf.ensure_shape para ayudar al optimizador  
        z = tf.ensure_shape(z, [None, None, None, self._num_conv_channels])  
      
        # Permute to [B, F, T, C] and then merge B*F as batch for LSTM:  
        z = tf.transpose(z, [0, 2, 1, 3])     # [B, F, T, C]  
        z = tf.reshape(z, [B * F, T, C])      # [B*F, T, C]  
      
        # Apply (Bi)LSTM along time dimension for each (original) subcarrier independently  
        z = self._bilstm(z)                   # [B*F, T, 2*lstm_units]  
      
        # Restore shape to [B, F, T, channels]  
        out_channels = tf.shape(z)[-1]  
        z = tf.reshape(z, [B, F, T, out_channels])  # [B, F, T, out_ch]  
        z = tf.transpose(z, [0, 2, 1, 3])           # [B, T, F, out_ch]  
      
        # Final 1x1 conv to map to num_bits_per_symbol per (T,F)  
        z = self._output_conv(z)  # [B, T, F, num_bits_per_symbol]  
      
        return z