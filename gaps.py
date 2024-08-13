import numpy as np


def binary_num_list(x):
    """
      Creates a list of all possible binary patterns of a size x
      Input: 
        x: gap length (int)
      Output: 
        list[list[int]]
    """
    if x == 0:
        return [[]]
    else:
        smaller_patterns = binary_num_list(x - 1)
        return [pattern + [0] for pattern in smaller_patterns] + \
               [pattern + [1] for pattern in smaller_patterns]
    
def split_data_set(data_set, leaf_train_size, tree_train_size, val_size, test_size):
   """
      Splits a full set up gaps into 4 different data sets: leaf train, tree train, val, test
      Input: 
        data_set: list[gaps]
        leaf_train_size: int
        tree_train_size: int
        val_size: int
        test_size: int
      Output: 
        leaf_train_data: list[gaps]
        tree_train_data: list[gaps]
        val_data: list[gaps]
        test_data: list[gaps]
   """
   leaf_train_data = data_set[:leaf_train_size]
   tree_train_data = data_set[leaf_train_size:leaf_train_size+tree_train_size]
   val_data = data_set[leaf_train_size+tree_train_size:leaf_train_size+tree_train_size+val_size]
   test_data = data_set[leaf_train_size+tree_train_size+val_size:leaf_train_size+tree_train_size+val_size+test_size]
   return leaf_train_data, tree_train_data, val_data, test_data


def assess_error(ground_truth_probs, test_tree_probs):
   """
    Assesses mean squared error for a list of predictions  
    Input:
      ground_truth_probs: list[float]
      test_tree_probs: list[floats]
    Output:
      mse: float
      std: float
   """
   total_se = 0
   list_trial_error = []
   for x in range(len(ground_truth_probs)):
      trial_se = (ground_truth_probs[x] - test_tree_probs[x])**2
      total_se += trial_se
      list_trial_error.append(trial_se)
   sample_size = len(ground_truth_probs)
   mse = total_se/sample_size
   std_dev = np.std(list_trial_error)
   return mse, std_dev

def assess_single_gap_error(ground_truth_prob, test_tree_prob):
  """
    Assesses squared error for one gap  
    Input:
      ground_truth_prob: float
      test_tree_prob: float
    Output:
      error: float
  """
  return (ground_truth_prob - test_tree_prob)**2

def running_average(data, window_size):
  """
    calculates running error of a data set  
    Input:
      data: list[int]
      window_size: list[int]
    Output:
      running_avg: float
  """
  ra = np.empty(len(data))
  for i in range(len(data)):
    if i < window_size:
      ra[i] = np.mean(data[:i+1])
    else:
      ra[i] = np.mean(data[i-window_size+1:i+1])
  return ra


def average(data):
  """
    calculates average of a data set  
    Input:
      data: list[int]
    Output:
      avg: float
  """
  avg = np.empty(len(data))
  for i in range(len(data)):
    avg[i] = np.mean(data[:i])
  return avg

def has_duplicates(lst):
    """
      Determines whether or not there are duplicates within a list
      Input:
        lst: list
      Output:
        duplicates: bool
    """
    return len(lst) != len(set(tuple(x) if isinstance(x, list) else x for x in lst))