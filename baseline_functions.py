import numpy as np
from utils import max_star
from decoders import LDPC_decoder
from insertion_deletion import ins_del_channel, insert_regular_markers
import itertools
import time

def forward_recursion_log_domain(y, T, Pd, Pi, Ps, mask, marker_seq, N, Nc):
    """
    Calculates forward probabilities in BCJR algorithm deployed for 
    insertion/deletion/substitution channels.
    
    Args:
        y (list): Channel realization (Output of the channel).
        T (int): Total transmitted bits.
        Pd (float): Deletion probability of the Channel.
        Pi (float): Insertion probability of the Channel.
        Ps (float): Substitution probability of the Channel.
        mask (array_like): Indicates whether the current bit is a marker bit (1 if the current bit is a marker bit.
        marker_seq (array_like): Code of the marker.
        N (int): Total block length (N = Nc + Nm).
        Nc (int): Total codeword block.

    Returns:
        alpha(np.darray): Forward probabilities.
    """

    R = len(y)                              # Received bit numbers 
    alpha = -np.inf * np.ones((R+1, T+1))   # Forward recursion inits
    alpha[0, 0] = 0                         # Init. the starting point
    Pt = 1 - Pd - Pi                        # Transmission Prob.

    for k in range(1, T+1): # From stage 1 to T
        
        for n in range(max(0, R-2*(T-k)), min(2*k+1, R+1)):
            
            if n == 0: # For state n = 0, only deletion event occurs
                alpha[n, k] = np.log(Pd) + alpha[n, k-1]
                
            elif n == 1: # For state n = 1, deletion or transmission events can occur
                a = np.log(Pd) + alpha[n, k-1]
                # If this is a marker bit position, we need the marker bit @ that position
                if mask[k-1] == 1:
                    curr_marker_bit = marker_seq[(k-1)%N - Nc]
                    b = np.log(Pt) + np.log((1-2*Ps)*(curr_marker_bit == y[n-1]) + Ps) + alpha[n-1, k-1]
                else:
                    b = np.log(Pt/2) + alpha[n-1, k-1]
                alpha[n, k] = max_star(a, b)

            elif n > 1: # For states bigger than 1, deletion, transmission, or insertion can occur
                a = np.log(Pd) + alpha[n, k-1]
                b = np.log(Pi/4) + alpha[n-2, k-1]
                if mask[k-1] == 1:
                    curr_marker_bit = marker_seq[(k-1)%N - Nc]
                    c = np.log(Pt) + np.log((1-2*Ps)*(curr_marker_bit == y[n-1]) + Ps) + alpha[n-1, k-1]
                else:
                    c = np.log(Pt/2) + alpha[n-1, k-1]
                alpha[n, k] = max_star(a, b, c)
    return alpha


def backward_recursion_log_domain(y, T, Pd, Pi, Ps, mask, marker_seq, N, Nc):
    """
    Calculates backward probabilities in BCJR algorithm deployed for 
    insertion/deletion/substitution channels.
    
    Args:
        y (list): Channel realization (Output of the channel).
        T (int): Total transmitted bits.
        Pd (float): Deletion probability of the Channel.
        Pi (float): Insertion probability of the Channel.
        Ps (float): Substitution probability of the Channel.
        mask (array_like): Indicates whether the current bit is a marker bit (1 if the current bit is a marker bit.
        marker_seq (array_like): Code of the marker.
        N (int): Total block length (N = Nc + Nm).
        Nc (int): Total codeword block.

    Returns:
       beta(): Backward probabilities.
    """
    
    R = len(y)                            # Received bit number
    beta = -np.inf * np.ones((R+1, T+1))  # backward recursion inits
    beta[R, T] = 0                        # Init. the starting point
    Pt = 1 - Pd - Pi

    for k in range(T, 0, -1):
        k_star = T + 1 - k
        
        # For states between R-2 to 1
        for n in range(R, max(R-1-2*k_star, -1), -1):
            
            # for states less than R - 1
            if n < R - 1:
                a = np.log(Pd)   + beta[n, k]
                b = np.log(Pi/4) + beta[n+2, k]

                if mask[k-1] == 1:
                    curr_marker_bit = marker_seq[(k-1) % N - Nc]
                    c = np.log(Pt) + np.log((1-2*Ps)*(curr_marker_bit == y[n]) + Ps) + beta[n+1, k]
                else:
                    c = np.log(Pt/2) + beta[n+1, k]

                beta[n, k-1] = max_star(a, b, c)
            
            # For state n = R - 1
            elif n == R-1:
                a = np.log(Pd) + beta[R - 1, k]
                if mask[k-1] == 1:
                    curr_marker_bit = marker_seq[(k-1) % N - Nc]
                    b = np.log(Pt) + np.log((1-2*Ps)*(curr_marker_bit == y[n]) + Ps) + beta[R, k]
                else:
                    b = np.log(Pt/2) + beta[R, k]
                beta[R-1, k-1] = max_star(a, b)

            # For state n = R
            elif n == R:
                beta[n, k-1] = np.log(Pd) + beta[n, k]

    return beta


def estimate_message(y, alpha, beta, Pd, Pi, Ps, T):
    """
    Estimate the transmitted message based on the BCJR algorithm by using
    forward and backward probabilities calculated based on the recevied sequence
    and the marker bit locations.

    Args:
        y (list): Channel realization (output of the channel).
        alpha (numpy.ndarray): Forward probabilities.
        beta (numpy.ndarray): Backward probabilities.
        Pd (float): Deletion probability of the channel.
        Pi (float): Insertion probability of the channel.
        Ps (float): Substitution probability of the channel.
        T (int): Total transmitted bits.

    Returns:
        tuple: A tuple containing:
            - m_est (numpy.ndarray): Estimated transmitted message.
            - llr (numpy.ndarray): Log-likelihood ratio of the estimated message.

    Note:
        This function estimates the transmitted message based on the forward and backward probabilities
        calculated using the BCJR algorithm. It computes the log-likelihood ratio (LLR) for each bit and
        returns the estimated message and LLR.

    """

    # Total number of received bits
    R = len(y) 

    # Transmission prob.
    Pt = 1-Pd-Pi
    
    # Likelihood arrays for individual bits
    l0 = np.zeros((T,1))
    l1 = np.zeros((T,1))

    for k in range(T):
        # Get alpha vector corresponding to alpha vector @ stage k
        alpha_k = alpha[:, k]
        # Get beta vector corresponding to beta vector @ stage (k+1)
        beta_k = beta[:, k+1]
        # Init the ins, del, trans0 and trans1 vectors to minus inf
        # Instead of minus inf, start it with a very small negative integer
        ins =     -1e100
        del_val = -1e100
        trans0 =  -1e100 
        trans1 =  -1e100 
        
        for n in range(1, min(2*k+2, R+1)):
            
            if n == R + 1:
                del_val = max_star(del_val, np.log(Pd) + beta_k[n-1] + alpha_k[n-1])
                
            elif n == R:
                del_val = max_star(del_val, np.log(Pd) + beta_k[n-1] + alpha_k[n-1])
                trans0 = max_star(trans0, np.log(Pt) + beta_k[n] + alpha_k[n-1] +
                             np.log((1-Ps)*(y[n-1] == 0) + Ps))
                trans1 = max_star(trans1, np.log(Pt) + beta_k[n] + alpha_k[n-1] +
                             np.log((1-Ps)*(y[n-1] == 1) + Ps))
            else:
                ins = max_star(ins, np.log(Pi/4) + beta_k[n+1] + alpha_k[n-1])
                del_val = max_star(del_val, np.log(Pd) + beta_k[n-1] + alpha_k[n-1])
                trans0 = max_star(trans0, np.log(Pt) + beta_k[n] + alpha_k[n-1] +
                             np.log((1-Ps)*(y[n-1] == 0) + Ps))
                trans1 = max_star(trans1, np.log(Pt) + beta_k[n] + alpha_k[n-1] +
                             np.log((1-Ps)*(y[n-1] == 1) + Ps))
                
        l0[k,0] = max_star(trans0, del_val, ins)
        l1[k,0] = max_star(trans1, del_val, ins)

    # Calculate LLR for 1,2,...,T
    llr = l0 - l1
    
    # Message estimates (It is provided in addition to LLR.)
    m_est = llr < 0

    return m_est, llr

def baseline_simulation_LDPC(H, iter_num, marker_seq, Nc, Pd, Pi, Ps, 
                            print_every = 10, max_fer = 100, max_codes = 10000,
                            CSI='known', P_est=None, filepath = None):

    """
    Perform baseline decoder testing for LDPC decoding using the BCJR algorithm 
    with insertion/deletion/substitution channels.

    Args:
        H (numpy.ndarray): Parity check matrix of the LDPC code.
        iter_num (int): Number of iterations for LDPC decoding (sum-product algorithm).
        marker_seq (array_like): Marker sequence.
        Nc (int): Total codeword block size.
        Pi (list): Insertion test probability of the channel(s).
        Ps (list): Substitution test probability of the channel(s).
        Pd (list): Deletion test probability of the channel(s).
        print_every (int): Frequency of printing progress during testing.
        CSI(str, choices = {'known', 'unk}): Channel state information (channel probabilities) is known or not
        P_est(list): Channel probability estimates if CSI is set to 'unk'. [Pi_est, Ps_est]

    Returns:
        res

    Note:
    This function performs baseline testing for LDPC decoding by simulating transmission over
    insertion/deletion/substitution channels. It generates random messages, encodes them using
    regular markers, applies the specified channel model (with insertion, deletion and substition probs.), 
    performs LDPC decoding after estimating LLRs with the BCJR algorithm, 
    and calculates the error rates (BER, FER).
    """

    # Check if correct inputs are taken !
    if  CSI != 'unk' and P_est == None:
        assert CSI != 'unk' and P_est == None, 'If CSI is unknown (unk), then P_est should not be empty!'
    elif CSI != 'unk' and CSI != 'known':
        assert CSI != 'known' and CSI != 'unk', "CSI should be set to 'known' or 'unk'!"

    # Codeword Lenght
    n = H.shape[-1]
    
    # Total codeword block
    Nr = marker_seq.shape[-1]
    N = Nc + Nr
    
    # Create testing points
    test_points = list(itertools.product(Pd,Pi,Ps))

    #num_codes_array = [1000, 1000, 1000, 1000, 1000, 1000]
    
    print(f"------------SIMULATION STARTS---------------")
    print(f"--------------------------------------------")
    # Hold results array.
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
            rand_seq = np.random.randint(low = 0, high=2, size=(1,n), dtype=int)
            m = np.zeros((1,n)) + rand_seq

            # Insert markers
            c,mask = insert_regular_markers(m, Nc, marker_seq)
            T = c.shape[-1]

            # Send codeword through the general insertion/deletion/substition channel.
            y, _ = ins_del_channel(c, Pd_point, Pi_point, Ps_point)
            R = len(y)

            if CSI == 'known':
                # Calculate forward probabilities.
                alpha = forward_recursion_log_domain(np.squeeze(y), T, Pd_point, Pi_point, Ps_point, np.squeeze(mask), 
                                                    np.squeeze(marker_seq), N, Nc)
                # Calculate backward probabilities.                                 
                beta = backward_recursion_log_domain(np.squeeze(y), T, Pd_point, Pi_point, Ps_point, np.squeeze(mask), 
                                                    np.squeeze(marker_seq), N, Nc)
                # Estimate LLRs from forward and backward probabilities.
                _, llr2 = estimate_message(np.squeeze(y), alpha, beta, Pd_point, Pi_point, Ps_point, T)
            elif CSI == 'unk':
                # Estimate the Pd
                Pd_est = (T-R)/T
                # Calculate forward probabilities.
                alpha = forward_recursion_log_domain(np.squeeze(y),T,Pd_point,P_est[0],P_est[1], np.squeeze(mask), 
                                                    np.squeeze(marker_seq), N, Nc)
                # Calculate backward probabilities.                                 
                beta = backward_recursion_log_domain(np.squeeze(y),T,Pd_point,P_est[0],P_est[1], np.squeeze(mask), 
                                                    np.squeeze(marker_seq), N, Nc)
                # Estimate LLRs from forward and backward probabilities.
                _, llr2 = estimate_message(np.squeeze(y), alpha, beta, Pd_point,P_est[0],P_est[1], T)
                

            # Get rid of the llrs corresponding to marker bits
            llr = llr2[np.squeeze(mask) == 0, :]

            # Get rid of the extra bits padded at the end of seqeunce while creating codewords
            llr = llr[0:n,:]
            llr = llr.T

            # This step is done in order to reverse the effect of the random sequence added 
            # at the beginning.
            llr[:,np.squeeze(rand_seq) == 1] = -llr[:,np.squeeze(rand_seq) == 1]

            # LDPC Decoder (clip values in order to avoid numerical errors) 
            llr = np.clip(llr, -30, 30)
            m_est, _ = LDPC_decoder(llr,H,iter_num)

            # Add errors
            ber = np.sum(m_est != np.zeros((1,n)))
            ber_total += ber

            if ber > 0:
                fer_total += 1

            if code_simulated % print_every == 0:
                ber = ber_total/(n*code_simulated)
                fer = fer_total/code_simulated
                print(f"{code_simulated}) (Pd, Pi, Ps) = {test_point}, BER: {ber: .7f}, FER: {fer: .7f}")

            code_simulated += 1

        print(f"{i+1}) (Pd, Pi, Ps) = {test_point} testing finished, BER: {ber_total/(n*code_simulated) : .7f}, FER: {fer_total/code_simulated : .7f}")
        
        # Append results.
        res.append((ber_total/(code_simulated*n),fer_total/(code_simulated)))
    
    # Save the results.
    return res
       