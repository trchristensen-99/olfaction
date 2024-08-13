import tree as t
import gaps
import pickle

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
    #Load GAPs
    gap_data = load_gaps_from_file("synthetic_gap_data.pkl")
    gaps_leaf_trn = gap_data[0]

    #Build ground truth tree and generate data
    ground_truth = t.build_bald_tree(10, shuffle_nums=False)
    ground_truth.fill_tree_with_leaves(evenly_spaced=True)
    responses_leaf_trn = ground_truth.run_tree_gaps(gaps_leaf_trn, leaf_prob=False)
    data_leaf_train = (gaps_leaf_trn, responses_leaf_trn)

    #Store data for leaf trianing
    save_lists_to_file(data_leaf_train, "leaf_training_data.pkl")
  

