import numpy as np
import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
import os
import csv
import sys
import scipy.io
import time

from tensorflow import keras
from keras.layers import Bidirectional
from keras.layers import LSTM, GRU, Dense, LayerNormalization
import sys

import tensorflow as tf
from models import BI_Estimator, BI_LSTM_Insertion_Deletion, NeuralDecoderConvCodes
from test_functions import test_LDPC_BI_GRU, test_conv_BI_GRU

parser = argparse.ArgumentParser(description='Testing')

parser.add_argument('--path', default='Results of NN method', type = str,
                    help = 'The source where results will be saved!')
parser.add_argument('--outer_code', choices = ['LDPC', 'Conv'], default='LDPC', help='Choose the outer code')
parser.add_argument('--marker_sequence', default=np.array([0,1]).reshape(1,-1), type = np.array,
                      help = 'Specify the marker sequence.')   
parser.add_argument('--test_points_Pd', default=[0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                    type=list, help = 'Deletion test points')
parser.add_argument('--test_points_Ps', default=[sys.float_info.min], type=list, 
                    help = 'Substition probability test points (type: list)')
parser.add_argument('--test_points_Pi', default=[sys.float_info.min], type=list, 
                    help = 'Insertion probability test points (type: list)')
parser.add_argument('--fer_errors', default=100, type=int, 
                    help = 'Number of fer errors until simulation runs for each test point')     

parser.add_argument('--Nc', default=10, type = int, help = 'After how many bits, markers are inserted')
parser.add_argument('--iter_num', default=100, type = int, help='Iteration number of the LDPC Sum product Decoder')
parser.add_argument('--matrix', choices = ['small', 'medium', 'large', 'xlarge'], type = str,
                    help = 'The size of the parity check matrix (default small). Experiments show results only for small', 
                    default = 'small')
parser.add_argument('--print_every',default=10, type = int)
parser.add_argument('--seed', default=100, type = int, help = 'Seed for reproducibility')


args = parser.parse_args()

# Fix the seed for reproducibility!
np.random.seed(args.seed)

# Get Parity Check Matrix.
matrix_path = 'H_matrix_' + args.matrix + '.mat'
filepath = os.path.join('Matrices', matrix_path)
H = scipy.io.loadmat(filepath)['H'] 

# Get args
iter_num = args.iter_num
marker_seq = args.marker_sequence
print_every = args.print_every
Nc = args.Nc
run_length = 2



# Get arguments
Pi = [sys.float_info.min]
Ps = [sys.float_info.min]
test_points = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]

# Instantiate the model
model = BI_LSTM_Insertion_Deletion(d_bilstm = 1024, d_ffn = 256, num_bi_layers = 8, rnn_type = 'gru')
# filepath # USE run_length = 2 !!!1
filepath = 'Models/model_Nc10_1'
model.load_weights(filepath)
# feed it for built of the model
input = np.zeros((1, 210 + 42, 2))
_ = model(input)

# Instantiate the model
model_conv = NeuralDecoderConvCodes()

filepath = 'Models/CONV_MODEL2'
model_conv.load_weights(filepath)
input = np.zeros((1, 210, 2))
_ = model_conv(input)

# parameters of the marker code
Nr = marker_seq.shape[-1]
N = Nc + Nr; # total marker and meesage bit block
rm = Nc/N; # rate of marker code

# Simulation logs
print('Simulation starts for the baseline decoder with BCJR based estimator and the LDPC decoder')
print('The Partiy-Check matrix size is ', H.shape)
print('Nc (In how many codeword bits, marker sequence is added) =', Nc)
print('Iteration Number of the LDPC decoder = ', iter_num)
print('Marker sequence is ', marker_seq)

# Run the simulation
test_LDPC_BI_GRU(model, run_length, H, iter_num, marker_seq, Nc, Pi, Ps, test_points, print_every)
#test_conv_BI_GRU(model, model_conv, run_length, marker_seq, Nc, Pi, Ps, test_points, print_every)

