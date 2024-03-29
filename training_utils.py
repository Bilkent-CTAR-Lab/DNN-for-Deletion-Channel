import numpy as np
from insertion_deletion import insert_regular_markers, ins_del_channel
from utils import create_conv_code

def create_batch1(run_length, m_total, num_code, Pd, Pi, Ps, Nc, marker_sequence):
    """
    Create minibatches for training of BI-RNNs for decoding concatenated codes over
    the insertion/deletion channel.

    This function generates training batches for BI-RNN (Bidirectional Recurrent Neural Network) 
    training to decode concatenated codes transmitted over a channel with insertion and deletion errors. 
    It prepares input data, target labels, and marker masks for training.

    Parameters:
    - run_length (int): Length of the runs.
    - m_total (int): Total number of bits in a message.
    - num_code (int): Number of codes to generate.
    - Pd (list): List of deletion probabilities.
    - Pi (list): List of Insertion probabilities.
    - Ps (list): List of substitution probabilities.
    - Nc (int): Number of bits per code block.
    - marker_sequence (numpy.ndarray): Sequence of marker bits.

    Returns:
    - trainX (numpy.ndarray): Input data of shape (num_code, int(m_total/r), run_length).
    - trainY (numpy.ndarray): Target labels of shape (num_code, int(m_total/r), 1).
    - mask (numpy.ndarray): Mask indicating marker positions in the input data.

    Note:
    - Marker Code and its parameters are predefined.
    - If m_total does not divide N_c, it is adjusted to ensure divisibility.
    - Random samples are drawn from the specified Pd and Ps for each code.
    - Random messages are generated for each code.
    - Marked inserted coded bits are obtained.
    - Channel simulation is performed.
    - Labels (trainY) are set to the original coded bits.
    - Inputs (trainX) are generated by shifting and scaling the channel output.
    """
    # Function implementations for 'insert_regular_markers' and 'ins_del_channel' are assumed.
    # Ensure to import necessary libraries like numpy and implement missing functions.
    
    # Length of the marker sequence
    Nr = marker_sequence.shape[-1]
    r = Nc/(Nc+Nr)

    # If m shape does not divide N_c
    if m_total % Nc != 0:
        m_total = m_total + Nc - (m_total % Nc);

    trainX = np.zeros((num_code, int(m_total/r), run_length))
    trainY = np.zeros((num_code, int(m_total/r), 1))

    for i in range(num_code):

      # Get random samples from the specified training Ps and Pd.
      # If one specify the list with only one element, then it is deterministic.
      Pd_sample = np.random.choice(Pd)
      Ps_sample = np.random.choice(Ps)
      Pi_sample = np.random.choice(Pi)

      # Create a random message.
      m = np.random.randint(0,2, size = (1, m_total))

      # Create marked (regukar) inserted coded bits.
      c, mask = insert_regular_markers(m, Nc, marker_sequence)

      # Channel
      y,trans = ins_del_channel(c, Pd_sample, Pi_sample, Ps_sample)
      y = np.array(y).T

      # Get labels.
      trainY[i,:,:] = c.T;

      # Get inputs.
      for j in range(run_length):
        trainX[i, 0:y.shape[-1]-j,j] = -2*y[0,j:] + 1;

    mask = np.array(mask).T
    return trainX, trainY, mask


def create_batch2(m_total, num_code, Pd, Pi, Ps, Nc, marker_sequence):
  """
    Create minibatches for training of BI-RNNs for decoding concatenated codes over
    the insertion/deletion channel.

    This function generates training batches for BI-RNN (Bidirectional Recurrent Neural Network) 
    training to decode concatenated codes transmitted over a channel with insertion and deletion errors. 
    It prepares input data, target labels, and marker masks for training.

    This function generates batches with a different approach compared to 'create_batch1'. 
    (See below for more detail)

    Parameters:
    - m_total (int): Total number of bits in a message.
    - num_code (int): Number of codes to generate.
    - Pd (list): List of deletion probabilities.
    - Pi (list): List of insertion probabilities.
    - Ps (list): List of substitution probabilities.
    - Nc (int): Number of bits per code block.
    - marker_sequence (numpy.ndarray): Sequence of marker bits.

    Returns:
    - trainX (numpy.ndarray): Input data of shape (num_code, int(m_total/r), int(m_total/r)).
    - trainY (numpy.ndarray): Target labels of shape (num_code, int(m_total/r), 1).
    - mask (numpy.ndarray): Mask indicating marker positions in the input data.

    Note:
    - This function generates batches with a different approach compared to 'create_batch1'.
    - Marker Code and its parameters are predefined.
    - If m_total does not divide N_c, it is adjusted to ensure divisibility.
    - Random samples are drawn from the specified Pi,Pd, and Ps for each code.
    - Random messages are generated for each code.
    - Marked coded bits are obtained using regular markers.
    - Channel simulation is performed.
    - Labels (trainY) are set to the original coded bits.
    - Inputs (trainX) are generated based on the channel output with specific considerations.

    Differences from 'create_batch1':
    - The shape of trainX and the way input data is generated differs.
    - In this function, trainX is of shape (num_code, int(m_total/r), int(m_total/r)).
    - The input data generation process in this function involves scaling the channel output based on specific considerations.
    - The comment block within the function provides more details on the differences in trainX generation.
    """
    # Function implementations for 'insert_regular_markers' and 'ins_del_channel' are assumed.
    # Ensure to import necessary libraries like numpy and implement missing functions.
    
  # Length of the marker sequence
  Nr = marker_sequence.shape[-1]
  r = Nc/(Nc+Nr)

  # If m shape does not divide N_c
  if m_total % Nc != 0:
      m_total = m_total + Nc - (m_total % Nc);

  trainX = np.zeros((num_code, int(m_total/r), int(m_total/r)))
  trainY = np.zeros((num_code, int(m_total/r), 1))

  for i in range(num_code):

    # Get random samples from the specified training Ps and Pd.
    Pd_sample = np.random.choice(Pd)
    Ps_sample = np.random.choice(Ps)
    Pi_sample = np.random.choice(Pi)

    # create message
    m = np.random.randint(0,2, size = (1, m_total))

    # create marked code bits
    c, mask = insert_regular_markers(m, Nc, marker_sequence)

    # channel
    y,trans = ins_del_channel(c, Pd_sample, Pi_sample, Ps_sample)
    y = np.array(y).T
    numR = y.shape[-1]
    T = c.shape[-1]

    # train Y
    trainY[i,:,:] = c.T;

    # train X
    for j in range(numR):

      trainX[i, j, 0:j] = -2*y[0,0:j] + 1;
    #for j in range(T):
      #if (j + 2) % 12 == 0 or (j + 1) % 12 == 0:
        #if j <= numR - 1:
          #trainX[i, j, 0:j+1] = -2*y[0,0:j+1] + 1;
        #else:
          #ind = j - numR + 1
          #trainX[i, j, ind:numR] = -2*y[0,ind:] + 1;
    
  mask = np.array(mask).T
  return trainX, trainY, mask
  
def create_batch_for_convcode(run_length,m_total,num_code,Pd,Pi,Ps,Nc,imp,perm, marker_sequence):
    """
    Create minibatches for training of BI-RNNs for decoding concatenated codes with conv codes over
    the insertion/deletion channel.

    Args:
    - run_length (int): The length of the run for the convolutional code.
    - m_total (int): The total number of message bits per code.
    - num_code (int): The number of codes to be generated in the batch.
    - Pd (list): List of deletion probabilities for the channel.
    - Pi (list): List of insertion probabilities for the channel.
    - Ps (list): List of substitution probabilities for the channel.
    - Nc (int): The number of marker bits inserted per code.
    - imp (list): List of impulse response coefficients for the convolutional encoder.
    - perm (list): List representing the permutation applied to the encoded bits for interleaving.
    - marker_sequence (numpy.ndarray): Sequence of marker bits used for coding.

    Returns:
    - inputs (numpy.ndarray): Array of shape (num_code, int(m_total*2/r), run_length) representing the inputs to the system.
    - labels (numpy.ndarray): Array of shape (num_code, m_total, 1) containing the ground truth message bits.
    - mask_array (numpy.ndarray): Array of shape (num_code, int(m_total*2/r), 1) containing the mask information.

    Explanation:
    This function generates a batch of input-output pairs for a convolutional encoder-decoder system.
    It creates random messages of length m_total, encodes them using a convolutional code, inserts markers, 
    simulates a communication channel with specified probabilities of deletion, insertion, and substitution,
    and generates input-output pairs for the decoder. The function returns the inputs, labels, 
    and mask array for the generated batch.
    """

    # Marker code length and code rate
    Nr = marker_sequence.shape[-1]
    r = Nc/(Nc+Nr)

    # Init. the labels and the inout array.
    inputs = np.zeros((num_code, int(m_total*2/r), run_length))
    labels = np.zeros((num_code, m_total, 1))
    mask_array = np.zeros((num_code, int(m_total*2/r), 1))
    
    for i in range(num_code):

      # Get samples !
      Pd_sample = np.random.choice(Pd)
      Ps_sample = np.random.choice(Ps)
      Pi_sample = np.random.choice(Pi)

      # create message
      m = np.random.randint(0, 2, size = (m_total,))

      # conv code
      code = create_conv_code(m, imp).reshape(1,m_total*2)

      # apply interleaver (use perm array)
      code = code[:,perm]

      # create marked code bits
      c, mask = insert_regular_markers(code, Nc, marker_sequence)

      # channel
      y,_ = ins_del_channel(c, Pd_sample, Pi_sample, Ps_sample)
      y = np.array(y).T

      # Get labels
      labels[i,:,:] = m.reshape(m_total,1)

      # Get inputs
      for j in range(run_length):
        inputs[i, 0:y.shape[-1]-j,j] = -2*y[0,j:] + 1

    mask_array = np.array(mask.T)
    return inputs, labels, mask_array
