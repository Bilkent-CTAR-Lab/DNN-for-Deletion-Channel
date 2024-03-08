import numpy as np

def LDPC_decoder(c_llr, H, iter_num):
  """
    Implements the classical sum-product algorithm (SPA) for LDPC codes.

    Parameters:
    - c_llr (numpy.ndarray): Channel LLR (Log-Likelihood Ratio) values.
    - H (numpy.ndarray): Parity check matrix.
    - iter_num (int): Total number of iterations for decoding.

    Returns:
    - m_est (numpy.ndarray): Estimated message bits after decoding.

    This function performs LDPC decoding using the Sum-Product Algorithm (SPA). It takes the channel LLR values,
    parity check matrix, and the number of iterations as inputs, and returns the estimated message bits after decoding.

    Example:
    ```
    # Define inputs
    c_llr = np.array([0.1, -0.3, 0.5, -0.2])
    H = np.array([[1, 0, 1, 0],
                  [0, 1, 1, 1]])
    iter_num = 10

    # Perform LDPC decoding
    m_est = LDPC_decoder(c_llr, H, iter_num)
    ```
    """

  size_H = H.shape
  m,n = size_H
  
  # Initiliaze the messages - 
  # For the first initial message passing,
  # realize with the channel values.
  VN_to_CN = H*c_llr 

  for iter in range(1,iter_num+1):
      CN_to_VN = np.zeros(size_H)
    
      # Calculate CN to VN messages
      for i in range(m):

          # Get current row vector of H
          curr_row = H[i,:]
          # Find where this equals to non-zero
          ind = np.squeeze(np.argwhere(curr_row))
          # get messages coming to CN
          mes = VN_to_CN[i,ind]

          for j in range(len(ind)):

              # Delete the corresponding CN_to_VN meesage where we send info
              mes_upd = np.copy(mes)
              mes_upd = np.delete(mes_upd, j)

              # Find current CN to VN message
              CN_to_VN[i, ind[j]] = 2*np.arctanh(np.clip(np.prod(np.tanh(mes_upd/2)), -0.9999999999, 0.9999999999))

      # Estimate the messages
      llr = np.sum(CN_to_VN, 0) + c_llr
      m_est = llr < 0
      if np.sum(np.mod(m_est @ H.T, 2)) == 0:
          #print('Process is stopped at iteration: ', iter)
          break;

      VN_to_CN = np.zeros(size_H)
      # Calculate VN to CN messages
      for j in range(n):

          # Get current column vector of H
          curr_col = H[:,j]
          # Find where this equals to non-zero
          ind = np.squeeze(np.argwhere(curr_col))
          # get messages coming to CN
          mes = CN_to_VN[ind, j]

          for i in range(len(ind)):

              # Delete the corresponding CN_to_VN meesage where we send info
              mes_upd = np.copy(mes)
              mes_upd = np.delete(mes_upd, i)

              # Find current VN to CN message
              VN_to_CN[ind[i],j] = np.sum(mes_upd)+c_llr[0,j]

  # Estimate the messages
  llr = np.sum(CN_to_VN, 0) + c_llr
  m_est = llr < 0

  return m_est, llr


def Viterbi_decoder(rev_state_trans, state_out, k, r):


    # Path metric.
    PM = np.zeros((4, k))
    # State hist metric shows previous hist for states
    state_hist = np.zeros((4, k+1))
    # Initialize message array
    m_hat = np.array([], dtype=int)
    
    # INITIALIZATION
    # STEP 1
    curr_r = r[:2]
    state_temp_00 = np.array([[+1, +1], [-1, -1]])
    temp1 = curr_r * state_temp_00
    temp_sum1 = np.sum(temp1, axis=1)
    PM[0, 0] = temp_sum1[0]
    PM[2, 0] = temp_sum1[1]
    state_hist[0, 0] = 1
    state_hist[2, 0] = 1
    
    # STEP 2
    curr_r = r[2:4]
    state_temp_00 = np.array([[+1, +1], [-1, -1]])
    temp2 = curr_r * state_temp_00
    temp_sum2 = np.sum(temp2, axis=1)
    state_temp_10 = np.array([[+1, -1], [-1, +1]])
    temp3 = curr_r * state_temp_10
    temp_sum3 = np.sum(temp3, axis=1)
    
    PM[0, 1] = PM[0, 0] + temp_sum2[0]
    PM[1, 1] = PM[2, 0] + temp_sum3[0]
    PM[2, 1] = PM[0, 0] + temp_sum2[1]
    PM[3, 1] = PM[2, 0] + temp_sum3[1]
    
    state_hist[0, 1] = 1
    state_hist[1, 1] = 3
    state_hist[2, 1] = 1
    state_hist[3, 1] = 3
    
    for i in range(2, k):
        curr_r = r[2*i-2:2*i]
        temp = curr_r * state_out
        temp2 = np.sum(temp, axis=1)
        PM_temp = np.concatenate((PM[:, i-1], PM[:, i-1]))
        t = PM_temp + temp2
        for j in range(4):
            curr_met = [t[2*j], t[2*j+1]]
            PM[j, i] = max(curr_met)
            state_hist[j, i] = rev_state_trans[j, np.argmax(curr_met)]
    
    final_state = np.zeros(k+1, dtype=int)
    temp = 0
    for i in range(k, 0, -1):
        if i == k:
            final_state[i] = np.argmax(PM[:, k])
            temp = final_state[i]
        else:
            temp = state_hist[temp, i]
            final_state[i] = temp
    
    for i in range(1, k+1):
        curr = final_state[i]
        prev = final_state[i-1]
        if curr == 1 and prev == 1:
            m_hat = np.append(m_hat, 0)
        elif curr == 1 and prev == 2:
            m_hat = np.append(m_hat, 0)
        elif curr == 2 and prev == 3:
            m_hat = np.append(m_hat, 0)
        elif curr == 2 and prev == 4:
            m_hat = np.append(m_hat, 0)
        elif curr == 3 and prev == 1:
            m_hat = np.append(m_hat, 1)
        elif curr == 3 and prev == 2:
            m_hat = np.append(m_hat, 1)
        elif curr == 4 and prev == 3:
            m_hat = np.append(m_hat, 1)
        elif curr == 4 and prev == 4:
            m_hat = np.append(m_hat, 1)
            
    return m_hat, PM, state_hist, final_state
