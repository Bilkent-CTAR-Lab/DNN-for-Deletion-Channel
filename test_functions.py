import numpy as np
from decoders import LDPC_decoder
from insertion_deletion import ins_del_channel, insert_regular_markers
from training_utils import create_batch_for_convcode
import itertools

def test_LDPC_BI_GRU(model, run_length, H, iter_num, marker_seq, Nc, Pi, Ps, Pd, print_every = 100,
                    max_fer = 100, max_codes = 10000, filepath = None):

  """
   Simulates the performance of a given model (tensorflow model) as an estimator
   on an deletion/substitution channel for the concatenated coding with LDPC outer codes.

   Args:
       model (tf.keras.Model): The neural network model to be evaluated as an estimator.
       run_length (int): The length of the input sequence to the model.
       H (numpy.ndarray): The parity check matrix of the LDPC code.
       iter_num (int): The number of iterations for the LDPC decoder.
       marker_code (numpy.ndarray): The marker code for identifying the positions of the inserted markers.
       Nc (int): The number of codeword bits for which markers are to be inserted.
       Pi (list): The insertion probability.
       Ps (list): The substitution probability.
       Pd (list): The deletion probability.
       print_every (int): The frequency at which the intermediate results are printed.

   Returns:
       list: A list of tuples containing the bit error rate (BER) and frame error rate (FER) for each value of Pd.

   This function simulates the transmission of codewords over an insertion/deletion/substitution channel 
   and evaluates the performance of the given model in recovering the transmitted codewords. 
   It performs the following steps:

   1. Initialize the total BER and FER to 0.
   2. For each value of Pd:
       a. Iterate over a range of codewords.
       b. Generate a random sequence and add it to an all-zero codeword to bypass the codeword generation via the generator matrix.
       c. Insert markers into the codeword.
       d. Send the codeword through the deletion/substitution channel.
       e. Feed the received vector to the neural network model.
       f. Obtain the estimated codeword from the model's output.
       g. Calculate the log-likelihood ratios (LLRs) from the estimated codeword.
       h. Reverse the effect of the random sequence added in step 2b.
       i. Decode the codeword by using LLRs in the LDPC decoder.
       j. Calculate the BER and FER for the current codeword.
       k. Update the total BER and FER.
       l. Print the intermediate results if the current iteration is a multiple of print_every.
   3. Append the final BER and FER for the current value of Pd to the result list.
   4. Print the final BER and FER for the current value of Pd.
   5. Return the result list containing the BER and FER for all values of Pd.
   """

  # Codeword length
  n = H.shape[-1]
  
  # Total codeword block
  Nr = marker_seq.shape[-1]
  N = Nc + Nr

  # Create testing points.
  test_points = list(itertools.product(Pd,Pi,Ps))

  
  print(f"------------SIMULATION STARTS---------------")
  print(f"--------------------------------------------")
  res = []
  
  for i, test_point in enumerate(test_points):

    # Init BER/FER error metrics
    ber_total = 0
    fer_total = 0

    # Get current Pd, Pi, Ps
    Pd_point, Pi_point, Ps_point = test_point
    code_simulated = 1
    print(f"------------Test point (Pd, Pi, Ps) = {test_point}---------------")

    while fer_total <= max_fer and code_simulated <= max_codes:

        # Create a sequence and add it to all zero codeword.
        # This step is done in order to bypass the creation of the codeword via
        # generator matrix of the LDPC code which is harder to obtain for some
        # cases.
        seq = np.random.randint(0,2,(1,n))
        mes = np.zeros((1,n)) + seq 

        # Insert markers
        c, mask = insert_regular_markers(mes, Nc, marker_seq)
        y, _ = ins_del_channel(c, Pd_point, Pi_point, Ps_point)
        y = np.array(y).T

        # Lengths of recieved vecors etc
        R = y.shape[-1]
        T = c.shape[-1]

        # Prepare NN inputs !
        input = np.zeros((1, c.shape[-1], run_length))
        for ind in range(run_length):
          input[0, 0:y.shape[-1]-ind, ind] = -2*y[0,ind:] + 1

        # Estimate probabilities via BI-GRU
        prob = model(input)
        c_est = np.array(prob).reshape(1,c.shape[-1])
        c_est = np.expand_dims(c_est[mask == 0], axis = 0)
        c_est = c_est[:,0:n]

        # LLR calculation
        llr = np.log((1-c_est)/c_est)
        llr[seq == 1] = -llr[seq == 1]

        # LDPC decoder
        m_est, _ = LDPC_decoder(llr, H, iter_num)

        # calculate total ber
        ber = np.sum(m_est.astype('int') != np.zeros((1,n)))
        ber_total += + ber

        # If bit error happens, increase frame error!
        if ber > 0:
          fer_total += 1

        if code_simulated % print_every == 0:
          ber = ber_total/(n*code_simulated)
          fer = fer_total/code_simulated
          print(f"{code_simulated}) (Pd, Pi, Ps) = {test_point}, BER: {ber: .7f}, FER: {fer: .7f}")

        code_simulated += 1
    
    # Calculate BER/FER
    a = ber_total/(code_simulated*n)
    b = fer_total/(code_simulated)
    res.append((a,b))
    print(f"{i+1}) Pd = {Pd} testing finished, BER: {ber_total/(n*code_simulated): .7f},  FER: {fer_total/code_simulated: .7f}")
        
  return res

def test_conv_BI_GRU(model_est, model_conv, run_length, marker_seq, Nc, Pi, Ps, Pd, print_every):
  
  # Define Pd array
  Pd_array = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]

  # interleaver
  m_total = 105

  # 
  np.random.seed(100)
  perm = np.random.permutation(m_total*2)

  # parameters of the marker code
  Nc = 10; Nr = 2; # params of marker and code
  N = Nc + Nr; # total marker and meesage bit block
  rm = Nc/(Nc+Nr); # rate of marker code

  batch_size = 32

  # simulation
  num_codes = [10000]*len(Pd_array)
  res = []

  for k in range(len(Pd_array)):

      ber_total = 0
      fer_total = 0

      # Choose random deletion prob for training
      Pd = [Pd_array[k]]
      num_code = num_codes[k]

      # Iterate over batches
      for l in range(1, num_code + 1):

        # Create training batch
        inp1, trainY, mask = create_batch_for_convcode(run_length = 2, m_total = m_total, num_code = batch_size,
                                    Pd = Pd,Pi = [0],Ps = [0],Nc = Nc,imp = [[1,0,1],[1,1,1]],perm = perm,
                                    marker_sequence=marker_seq)


        # Give to model estimator and estimate
        probs = np.array(model_est(inp1))
        probs = probs[:,mask == 0]

        # LLR
        llr = np.log((1-probs)/probs)
        llr[llr>10] = 10
        llr[llr<-10] = -10
        llr2 = np.zeros(llr.shape)
        llr2[:,perm] = llr

        # RESHAPE ! IMPORTANT !
        llr2 = llr2.reshape(-1,int(llr2.shape[1]/2), 2)

        # train
        y_hat = model_conv(llr2)
        y_hat = y_hat > 0.5

        r = y_hat != trainY
        ber =  np.sum(r)
        fer = np.sum(np.sum(y_hat != trainY, axis=1) > 0)

        ber_total += ber
        fer_total += fer

        if l % print_every == 0:
           print(f"{l}) Pd = {Pd}, BER: {ber_total/(m_total*l*batch_size)},  FER: {fer_total/(l*batch_size)}")

      a = ber_total/(num_code*m_total*batch_size);
      b = fer_total/(num_code*batch_size);
      res.append((a,b));
      print("For Deletion Probability : %5.3f, BER : %5.5f, FER: %5.5f" % (Pd[0], a, b))

  return