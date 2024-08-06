import tree as t
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import os
import pickle

GAP_LENGTH = 15

np.random.seed(1)
random.seed(1)
now = datetime.now().strftime("%Y%m%d_%H%M%S")
# results_dir_path = os.path.join("./results", now)
# os.mkdir(results_dir_path)


def binary_num_list(x):
    """
      Creates a list of all possible binary patterns of a size x
      Input: x: gap length (int)
      Output: list[list[int]]
    """
    if x == 0:
        return [[]]
    else:
        smaller_patterns = binary_num_list(x - 1)
        return [pattern + [0] for pattern in smaller_patterns] + \
               [pattern + [1] for pattern in smaller_patterns]
    
def split_data_set(data_set, leaf_train_size, tree_train_size, val_size, test_size):
   leaf_train_data = data_set[:leaf_train_size]
   tree_train_data = data_set[leaf_train_size:leaf_train_size+tree_train_size]
   val_data = data_set[leaf_train_size+tree_train_size:leaf_train_size+tree_train_size+val_size]
   test_data = data_set[leaf_train_size+tree_train_size+val_size:leaf_train_size+tree_train_size+val_size+test_size]
   return leaf_train_data, tree_train_data, val_data, test_data


# def assess_error(trained_tree: t.Tree, ground_truth_tree: t.Tree, gaps_list):
#   """
#     Given a ground truth model, trained tree, and a list of gaps, assesses the average training error  
#     Input:
#       trained_tree: Tree
#       ground_truth_tree: Tree
#       train_gaps_list: list[list[int]]
#     Output:
#       returns avg. training error
#   """
#   ground_truth_probs = ground_truth_tree.run_tree_gaps(gaps_list, leaf_prob=True)
#   trained_tree_probs = trained_tree.run_tree_gaps(gaps_list, leaf_prob=True)

#   error = 0
#   for x in range(len(trained_tree_probs)):
#     error += np.abs(ground_truth_probs[x] - trained_tree_probs[x])
#   sample_size = len(gaps_list)
#   mean_error = error/sample_size
#   return mean_error

def assess_error(ground_truth_probs, test_tree_probs):
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

# if __name__ == "__main__":
#   full_data_set = binary_num_list(GAP_LENGTH)
#   split_data_sets = split_data_set(full_data_set, 500, 100, 100, 100)
#   save_lists_to_file(split_data_sets, "synthetic_gap_data.pkl")
