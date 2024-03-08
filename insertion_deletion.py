import numpy as np

def insert_regular_markers(m, Nc, marker_sequence):
    """
    Insert regular markers into the message bits. I.e., a pre-selected marker sequence with Nm
    bits is inserted after every Nc coded bits in message bits m . 

    Parameters:
    - m (numpy.ndarray): The message bits to which markers will be inserted.
    - Nc (int): The number of message bits per marker.
    - marker_code (numpy.ndarray): The marker sequence to be inserted.

    Returns:
    - c (numpy.ndarray): The message bits with markers inserted.
    - mask (numpy.ndarray): The mask indicating the positions of inserted markers.

    Example:
    >>> m = np.array([[1, 0, 0, 1]])
    >>> Nc = 2
    >>> marker_code = np.array([[1, 1]])
    >>> insert_regular_markers(m, Nc, marker_code)
    (array([[1., 0., 1., 1., 0., 1., 1., 1.]]), array([[0., 0., 1., 1., 0., 0., 1., 1.]]))
    """

    # Get parameters of the code from the specified marker and Nc.
    Nm = marker_sequence.shape[-1]  # Length of the marker sequence.
    N = Nm + Nc                     # Total codeword block with markers.
    rm = Nc/N                       # Rate of the marker code.

    # If message length does not divide Nc, then pad minimum number of zeros to message bits
    # until message length divides Nc.
    if m.shape[-1] % Nc != 0:
        m = np.concatenate((m, np.zeros((1, Nc-(m.shape[-1] % Nc)))), axis = 1) 

    # Then get the total number of meesage bits!
    mtotal = m.shape[-1]

    # Init the codeword.
    c = np.zeros((1, int(mtotal/rm)))

    # marker bit
    mask = np.zeros((1, int(mtotal/rm)))
    for i in range(int(mtotal/Nc)):

        # Get neccessarty indexes!
        low_ind = N*i # low ind
        high_ind = N*(i+1)# high ind
        low_ind_m = (N-Nm)*i
        high_ind_m = (N-Nm)*(i+1)

        # Insert markers!
        c[0, low_ind : high_ind - Nm] = m[0, low_ind_m: high_ind_m]
        c[0, high_ind - Nm: high_ind] = marker_sequence

        # Specfy the locations where markers are inserted !
        mask[0, high_ind - Nm: high_ind] = np.ones((1, Nm))

    return c, mask


def ins_del_channel(c, Pd, Pi, Ps):
    """
    Simulate an insertion/deletion/substition channel and generate received signal 
    based on the transmitted codeword.

    Parameters:
    - c (numpy.ndarray): The transmitted codeword through the channel.
    - Pd (float): Probability of deletion event in the channel.
    - Pi (float): Probability of insertion event in the channel.
    - Ps (float): Probability of substitution event in the channel.

    Returns:
    - y (list): The received signal after passing through the insertion-deletion channel.
    - trans (numpy.ndarray): An array indicating the type of operation (correct transmission, deletion, or insertion) 
    for each symbol in the codeword.

    Channel Model:
    The insertion-deletion channel operates as follows:
    - Each symbol in the transmitted codeword undergoes one of three operations: 
        1) correct transmission, 2) deletion, or 3) insertion.
    - The probabilities of these operations are defined by Pd (deletion), Pi (insertion),
         and the remaining probability Pt (correct transmission) = 1- Pd - Pi.
    - For correct transmission ('c'), the transmitted bit may be flipped with probability Ps (substitution).
    - For insertion ('i'), two random bits (with i.i.d uniform probability) are inserted into the received signal.
    - For deletion ('d'), the symbol is simply deleted from the received signal.

    Example:
    >>> c = np.array([[0, 1, 0, 1]])
    >>> Pd = 0.1
    >>> Pi = 0.2
    >>> Ps = 0.0
    >>> ins_del_channel(c, Pd, Pi, Ps)
    ([[0], [1], [1], [1]], array(['c', 'c', 'i', 'd'], dtype='<U1'))
    """

    # Codeword length.
    len_c = c.shape[-1] 
    # Prob. of a correct transmission.
    Pt = 1 - Pd - Pi 
    # Transition array!
    trans = np.random.choice(np.array(['c','d','i']), size = len_c, replace = True, p = [Pt, Pd, Pi])

    # Init the array!
    y = []

    for i in range(len_c):
        # Correct transmission case !
        if trans[i] == 'c':
            rand_m = (c[0,i] + np.random.choice([0,1], size=1, replace=True, p=[1-Ps, Ps])) % 2
            y.append(rand_m.tolist())

        # Insertion case !
        elif trans[i] == 'i':
            i_bit1 = np.random.randint(low = 0, high=2, size=(1), dtype=int)
            y.append(i_bit1.tolist())
            i_bit2 = np.random.randint(low = 0, high=2, size=(1), dtype=int)
            y.append(i_bit2.tolist())

    return y, trans