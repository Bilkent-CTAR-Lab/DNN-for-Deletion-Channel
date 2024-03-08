import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, GRU, Dense, LayerNormalization, Bidirectional
from keras.models import Model


class BI_Estimator(Model):
  def __init__(self, d_rnn = 512, d_mlp = [128,32], num_bi_layers = 3, rnn_type = 'gru'):
    """
    Encoder model for insertion deletion channels using Bidirectional LSTM or GRU layers.

    This model is designed for encoding sequences in deletion channels.

    Parameters:
    - d_rnn (int): Hidden size number of Bidirectional RNNs.
    - d_mlp (list of int): Hidden size numbers for the multi-layer perceptron.
    - num_bi_layers (int): Number of Bidirectional RNN layers.
    - rnn_type (str): Type of RNN to be used, either 'lstm' or 'gru'.

    Attributes:
    - num_bi_layers (int): Number of Bidirectional RNN layers.
    - d_rnn (int): Hidden size number of Bidirectional RNNs.
    - rnn_type (str): Type of RNN used, either 'lstm' or 'gru'.
    - d_mlp (list of int): Hidden size numbers for the multi-layer perceptron.
    - bir_layers (list of tf.keras.layers.Bidirectional): Bidirectional RNN layers.
    - nor_layers (list of tf.keras.layers.LayerNormalization): Batch normalization layers.
    - mlp_layers (list of tf.keras.layers.Dense): Multi-layer perceptron layers.
    - output_layer (tf.keras.layers.Dense): Output layer with sigmoid activation.

    Note:
    - The model processes sequences through Bidirectional RNN layers followed by batch normalization,
      then through a multi-layer perceptron, and finally through an output layer with sigmoid activation.
    """
    super().__init__()

    # Parameters init.
    self.num_bi_layers = num_bi_layers
    self.d_rnn = d_rnn
    self.rnn_type = rnn_type
    self.d_mlp = d_mlp

    # BI-RNN layers
    if self.rnn_type == 'lstm':
      self.bir_layers = [Bidirectional(LSTM(self.d_rnn, return_sequences = True))
                        for _ in range(self.num_bi_layers)]
    elif self.rnn_type == 'gru':
      self.bir_layers = [Bidirectional(GRU(self.d_rnn, return_sequences = True))
                        for _ in range(self.num_bi_layers)]
    else:
      assert self.rnn_type == 'gru' or self.rnn_type == 'lstm', 'RNN type must be either gru or lstm!'

    # Batch normalization layers
    self.nor_layers = [LayerNormalization() for _ in range(self.num_bi_layers)]

    # MLP layers
    self.mlp_layers = [Dense(size, activation = 'relu') for size in self.d_mlp]

    # Output layer
    self.output_layer = Dense(1, activation = 'sigmoid')

  def call(self, x):
    """
    Forward Pass of the specified model!
    """
    for bir_layer,nor_layer in zip(self.bir_layers, self.nor_layers):
      x = bir_layer(x)
      x = nor_layer(x)

    for layer in self.mlp_layers:
      x = layer(x)
    
    x = self.output_layer(x)

    return x

class NeuralDecoderConvCodes(tf.keras.Model):
  def __init__(self, d_bilstm = 400, d_ffn = 32, num_bi_layers = 2, rnn_type = 'gru'):
    """
    Initializes the encoder for insertion deletion channels
    --------------------------------------------------------------
    args
    * d_bilstm: hidden size number of bidirectional rnns
    * d_ffn: hidden size number of multi layer perceptron
    * num_bi_layers: number of bi-directional layers
    * rnn_type: type of rnn - LSTM or GRU
    """
    super().__init__()

    # parameters init.
    self.num_bi_layers = num_bi_layers
    self.d_bilstm = d_bilstm
    self.rnn_type = rnn_type
    self.d_ffn = d_ffn;

    # layers
    if self.rnn_type == 'lstm':
      self.bir_layers = [Bidirectional(LSTM(self.d_bilstm, return_sequences = True))
                        for _ in range(self.num_bi_layers)]
    elif self.rnn_type == 'gru':
      self.bir_layers = [Bidirectional(GRU(self.d_bilstm, return_sequences = True))
                        for _ in range(self.num_bi_layers)]

    # batch normalization layers
    self.nor_layers = [LayerNormalization() for _ in range(self.num_bi_layers)]

    # concat layers
    self.mlp1 = Dense(self.d_ffn, activation = 'relu')
    self.mlp2 = Dense(1, activation = 'sigmoid')

  def call(self, x):

    for bir_layer,nor_layer in zip(self.bir_layers, self.nor_layers):
      x = bir_layer(x)
      x = nor_layer(x)

    x = self.mlp1(x)
    x = self.mlp2(x)

    return x


class BI_LSTM_Insertion_Deletion(Model):
  def __init__(self, d_bilstm = 512, d_ffn = 256, num_bi_layers = 3, rnn_type = 'gru'):
    """

    Initializes the encoder for insertion deletion channels
    --------------------------------------------------------------
    args
    * d_bilstm: hidden size number of bidirectional rnns
    * d_ffn: hidden size number of multi layer perceptron
    * num_bi_layers: number of bi-directional layers
    * rnn_type: type of rnn - LSTM or GRU
    """
    super().__init__()

    # parameters init.
    self.num_bi_layers = num_bi_layers
    self.d_bilstm = d_bilstm
    self.rnn_type = rnn_type
    self.d_ffn = d_ffn;

    # layers
    if self.rnn_type == 'lstm':
      self.bir_layers = [Bidirectional(LSTM(self.d_bilstm, return_sequences = True))
                        for _ in range(self.num_bi_layers)]
    elif self.rnn_type == 'gru':
      self.bir_layers = [Bidirectional(GRU(self.d_bilstm, return_sequences = True))
                        for _ in range(self.num_bi_layers)]

    # batch normalization layers
    self.nor_layers = [LayerNormalization() for _ in range(self.num_bi_layers)]

    # concat layers
    self.mlp1 = Dense(self.d_ffn, activation = 'relu')
    self.mlp2 = Dense(128, activation = 'relu')
    self.mlp3 = Dense(32, activation = 'relu')
    self.mlp4 = Dense(1, activation = 'sigmoid')

  def call(self, x):

    for bir_layer,nor_layer in zip(self.bir_layers, self.nor_layers):
      x = bir_layer(x)
      x = nor_layer(x)

    x = self.mlp1(x)
    x = self.mlp2(x)
    x = self.mlp3(x)
    x = self.mlp4(x)

    return x