import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import csv

from scipy.io import loadmat
import time
import argparse

from scipy.io import savemat
from keras.models import Model

from training_utils import create_batch1, create_batch2
from models import BI_Estimator

parser = argparse.ArgumentParser(description='Deletion/Substition Channel Training')

parser.add_argument('--path', default = 'BI_GRU_ESTIMATOR', type=str, 
                    help='Saving directory of the trained model.')
parser.add_argument('--outer_code', choices=['LDPC', 'Conv'], default = 'LDPC', type = str, 
                    help = 'The type of the outer code (default LDPC)')
parser.add_argument('--training_approach', choices = [1,2], default = 2,
                      type = int, help = '1 or 2 - We advise you to train with the approach 2 (much faster)')
parser.add_argument('--training_Pd', default=[0.01, 0.02, 0.03, 0.04, 0.05], type=list,
                    help='Specify the training deletion probabilities (list)')       
parser.add_argument('--training_Ps', default=[0], type=list,
                    help='Specify the training substition probabilities (list)')    
parser.add_argument('--marker_sequence', default=np.array([0,1]).reshape(1,-1), type = np.array,
                      help = 'Specify the marker sequence.')                
parser.add_argument('--loss', choices=['mse', 'bce'], default = 'bce', type=str, help='Specify the loss function')              
parser.add_argument('--matrix', choices = ['small', 'medium'], type = str,
                    help = 'The size of the parity check matrix (default small). Experiments show results only for small', default = 'small')

parser.add_argument('--epochs', default=300, type=int, help='Number of total epochs to run')
parser.add_argument('--step', default= 100, type=int, help='Number of steps per epoch')             
parser.add_argument('--bs', default = 16, type=int, help='Mini-batch size')
parser.add_argument('--lr', default = 9e-4, type=float, help='Initial learning rate')

# Architecture Arguments                   
parser.add_argument('--d_rnn', default=128, type = int, help='Hidden size dimension of bi-rnn')
parser.add_argument('--mlp', default=[128, 32], type = list, help='Dimensions of MLP added on top of bi-rnn.')
parser.add_argument('--rnn_type', default='gru', choices = ['gru', 'lstm'], type=str, help='Type of rnn.')
parser.add_argument('--n_rnn', default=3, type=int, help='Number of bi-rnn layers')


parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=1000, type=int, help='seed for initializing training.')


def main():

  
  args = parser.parse_args()

  # Seeds

  # Load Parity Check Matrix & Generator Matrix for LDPC Code
  H = loadmat('Matrices/H_matrix_small')['H'] # H matrix

  """# Model Training"""

  @tf.function
  def train_step(input, labels):
      """
      One training step for training
      ------------------------------------------
      args
      input : training batch
      labels : labels batch
      """

      # TRAINING STEP FOR E2E COMM SYSTEM
      with tf.GradientTape() as tape:

        # FORWARD PART
        logits = model(input)

        # Cross-Entropy Loss
        loss = loss_fn(labels, logits)

      grads = tape.gradient(loss, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))
      train_acc_metric.update_state(labels, logits)

      return loss

  # Define the estimator model
  model = BI_Estimator(d_rnn= args.d_rnn, d_ffn = args.mlp, num_bi_layers = args.n_rnn, rnn_type=args.rnn_type)

  
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.lr,
      decay_steps=1000,
      decay_rate=0.95,
  staircase = True)

  # Set the loss function!
  if args.loss == 'mse':
    loss_fn = tf.keras.losses.MeanSquaredError()
  elif args.loss == 'bce':
    loss_fn = tf.keras.losses.BinaryCrossentropy()

  # Set optimizer with the lr specified!
  optimizer = keras.optimizers.legacy.Adam(learning_rate = args.lr)

  # Set trainining accuracy metric!
  train_acc_metric = keras.metrics.BinaryAccuracy()

  # Other training details specified!
  epochs = args.epochs
  batch_size = args.bs
  codeword_length = 204
  steps = args.step
  run_length = 2
  Pd = args.training_Pd
  Ps = args.training_Ps
  Pi = [0.]
  approach = args.training_approach
  marker_sequence = args.marker_sequence

  model.compile(loss = loss_fn,
              optimizer = optimizer,
              metrics = ['accuracy'])


  for epoch in range(epochs):
      print("Epoch %d/%d" % (epoch+1, epochs)) # print current epoch
      start_time = time.time() # start time
      train_loss_total = 0


      # Iterate over batches
      for step in range(steps):

        # create training batch
        if approach == 1:
          trainX_batch, labels, _ = create_batch1(run_length = run_length,m_total = codeword_length,
                                        num_code = batch_size, Pd = Pd, Pi = Pi, Ps = Ps, Nc = 10, 
                                        marker_sequence=marker_sequence)
        else:
           trainX_batch, labels, _ = create_batch2(m_total = H.shape[-1], num_code = batch_size,
                                          Pd = Pd, Pi = Pi, Ps = Ps, Nc = 10, marker_sequence=marker_sequence)


        # record total los
        train_loss = train_step(trainX_batch, labels)
        train_loss_total += train_loss # add overall loss

        train_acc = train_acc_metric.result()
        train_loss = train_loss_total/(step + 1)

        print("Batch: %d - %.2fs - train loss: %.4f - train acc: %.4f" % (step, time.time() - start_time,  train_loss, train_acc))

      # Here add some. validation test ! 
      train_acc = train_acc_metric.result()
      train_loss = train_loss_total/(steps)
      train_acc_metric.reset_states()

      print("%.2fs - train loss: %.4f - train acc: %.4f" % (time.time() - start_time,  train_loss, train_acc))


if __name__ == '__main__':
  main()