import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Concatenate, LayerNormalization, Add
from tensorflow.nn import relu
from sionna.phy.utils import log10, insert_dims

class InceptionResidualBlock(Layer):  
    def __init__(self, num_filters):  
        super().__init__()  
        self.num_filters = num_filters  
  
    def build(self, input_shape):  
        # Normalización inicial global  
        self.norm_input = LayerNormalization(axis=-1)  
          
        # Ramas Inception con normalización independiente por rama  
        self.conv2 = Conv2D(self.num_filters//3, (2,2), padding='same', activation=None)  
        self.norm2 = LayerNormalization(axis=-1)  # Solo normalizar canales  
          
        self.conv3 = Conv2D(self.num_filters//3, (3,3), padding='same', activation=None)  
        self.norm3 = LayerNormalization(axis=-1)  # Solo normalizar canales  
          
        self.conv5 = Conv2D(self.num_filters//3, (5,5), padding='same', activation=None)  
        self.norm5 = LayerNormalization(axis=-1)  # Solo normalizar canales  
          
        # Proyección final  
        self.proj = Conv2D(self.num_filters, (1,1), padding='same', activation=None)  
        self.norm_final = LayerNormalization(axis=-1)  
  
    def call(self, x):  
        # Guardar entrada para conexión residual  
        residual = x  
          
        # Normalización inicial y activación  
        z = self.norm_input(x)  
        z = relu(z)  
          
        # Ramas Inception con normalización independiente  
        b2 = self.conv2(z)  
        b2 = self.norm2(b2)  
        b2 = relu(b2)  
          
        b3 = self.conv3(z)  
        b3 = self.norm3(b3)  
        b3 = relu(b3)  
          
        b5 = self.conv5(z)  
        b5 = self.norm5(b5)  
        b5 = relu(b5)  
          
        # Concatenar ramas (aquí se preservan las características multi-escala)  
        z = Concatenate(axis=-1)([b2, b3, b5])  
          
        # Proyección final  
        z = self.proj(z)  
        z = self.norm_final(z)  
        z = relu(z)  
          
        # Conexión residual  
        z = Add()([z, residual])  
          
        return z
        
class NeuralReceiverInception(Layer):
    def __init__(self, num_bits_per_symbol, num_filters=128, num_blocks=4):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_filters = num_filters
        self.num_blocks = num_blocks

    def build(self, input_shape):
        # Conv inicial
        self.input_conv = Conv2D(self.num_filters, (3,3), padding='same', activation=relu)
        # Bloques Inception
        self.inception_blocks = [InceptionResidualBlock(self.num_filters) for _ in range(self.num_blocks)]
        # Conv final para bits
        self.output_conv = Conv2D(self.num_bits_per_symbol, (3,3), padding='same', activation=None)

    def call(self, y, no):
        # Procesar ruido
        no = log10(no)
        y = tf.transpose(y, [0, 2, 3, 1])  # antena al último eje
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])

        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       no], axis=-1)

        # Conv inicial
        z = self.input_conv(z)
        # Pasar por Inception blocks
        for block in self.inception_blocks:
            z = block(z)
        # Conv final
        z = self.output_conv(z)

        return z
