import gaps
import pickle
import random
import numpy as np

GAP_LENGTH=15

def save_lists_to_file(lists, file_name):
   """
    Inputs:
        lists = []
        file_name = string - name of file
    Outputs:
        File with given name including lists
   """
   with open(file_name, 'wb') as file:
    return pickle.dump(lists, file)

def load_gaps_from_file(file_name):
    """
    Inputs:
        file_name: string
    Output: 
        lists from given file
    """
    with open(file_name, 'rb') as file:
      return pickle.load(file)

if __name__ == "__main__":
  full_data_set = gaps.binary_num_list(GAP_LENGTH)
  np.random.shuffle(full_data_set)
  split_data_sets = gaps.split_data_set(full_data_set, 500, 100, 100, 100)
  save_lists_to_file(split_data_sets, "synthetic_gap_data.pkl")