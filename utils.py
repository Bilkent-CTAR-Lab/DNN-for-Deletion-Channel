import numpy as np


def max_star(x, y, z=None, approx = None):
    """
    Calculate the max star value.

    Parameters:
    - x (float): The first input value.
    - y (float): The second input value.
    - z (float, optional): The third input value. If provided, the function calculates the max star value
      considering all three inputs. Defaults to None.
    - approx (None or bool, optional): If None (default), performs the exact calculation. If True, approximates
      the calculation with the max function. Defaults to None.

    Returns:
    - numeric: The max star value calculated based on the inputs.

    Note:
    If approx is None:
    - If z is None, calculates the max star value using the formula max(x, y) + ln(1 + exp(-abs(x - y))).
    - If z is provided, recursively calls max_star for x and y, then compares the result with z, and calculates
      the max star value considering the three values.

    If approx is True, calculates the maximum of x, y, and z directly.
    """
    if approx is None:
        if z is None:
            return max(x, y) + np.log(1 + np.exp(-np.abs(x - y)))
        else:
            inter = max_star(x, y)
            return max(inter, z) + np.log(1 + np.exp(-np.abs(inter - z)))
    else:
        return max(x,y,z)

def create_conv_code(mes, impulse_responses):
  """
   Create a convolutional code from a message and a list of impulse responses.

   Args:
       mes (numpy.ndarray): The input message represented as a binary numpy array.
       impulse_responses (list): A list of lists (of binary scalars) representing the impulse responses 
          of the convolutional encoders.

   Returns:
       numpy.ndarray: The convolutional code obtained by convolving the message 
       with each impulse response and stacking the results.

   This function generates a convolutional code from a given input message and a set of impulse responses. 
   It performs the following steps:

   1. Determine the length of the input message (`k`).
   2. Initialize an empty list (`conv`) to store the convolved sequences.
   3. Iterate over each impulse response in `impulse_responses`:
       a. Convolve the input message (`mes`) with the current impulse response (`imp`).
       b. Take the modulo 2 of the convolved sequence to obtain a binary sequence.
       c. Truncate the convolved sequence to length `k` and append it to the `conv` list.
   4. Stack the sequences in the `conv` list column-wise using `np.column_stack`.
   5. Ravel (flatten) the stacked array to obtain a 1-D convolutional code.
   6. Return the convolutional code.

   Example:
       >>> mes = np.array([1, 0, 1, 1])
       >>> impulse_responses = [np.array([1, 1, 1]), np.array([1, 0, 1])]
       >>> code = create_conv_code(mes, impulse_responses)
       >>> print(code)
       [1 1 0 1 1 0 1 0]
   """
  k = mes.shape[0]
  conv = []
  for imp in impulse_responses:
    conv1 = np.convolve(mes, imp) % 2
    conv.append(conv1[0:k])
  
  code = np.ravel(np.column_stack([conv1 for conv1 in conv]))

  return code

def save_model_weights(model, filepath):
  """
    Save the weights of the given model to the specified filepath.

    Parameters:
    - model (tf.keras.Model): The model whose weights are to be saved.
    - filepath (str): The filepath where the model weights will be saved.

    Returns:
    None

    Example:
    save_model_weights(my_model, "model_weights.h5")
    """
  
  model.save_weights(filepath)
  print('Model is succesfully saved to ', filepath)
  return

def load_model_weights(model, filepath):

  model = model.load_weights(filepath)
  print('Model is succesfully loaded from ', filepath)
  return model