import tree as t
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import os


MUTATION_SIZE = 1000

TRAINING_SAMPLE_SIZE = 500
VALIDATION_SAMPLE_SIZE = 100
TESTING_SAMPLE_SIZE = 100

TREE_MUTATION_TRAINING_ERROR = []
TREE_MUTATION_VALIDATION_ERROR = []
TREE_MUTATION_TESTING_ERROR = []
UNSUCCESSFUL_MUTATION_COUNT = []

np.random.seed(1)
random.seed(1)
now = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir_path = os.path.join("./results", now)
os.mkdir(results_dir_path)


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


def run_GAPS(tree: t.Tree, gap_list):
  """
    Traverses a list of gaps through a tree and returns associated responses  
    Input: tree (Tree), gap_list(list[list[int]])
    Output: response_list (list[int])
  """
  response_list = []
  for gap in gap_list:
    prob = tree.root.child.traverse_GAP(gap)
    response = random.choices([0, 1], weights = (1-prob, prob))[0]
    response_list.append(response)
  return response_list


def run_single_GAP(tree, gap):
  """
    Traverses a single gap through the tree and returns the associated leaf probability  
    Input: tree (Tree), gap (list[int])
    Output: prob (float)
  """
  prob = tree.root.child.traverse_GAP(gap)
  return prob


def train_GAPS(tree, train_gap_list, train_response_list):
  """
  Trains a tree with a list of GAPS and a list of associated responses
  Input:
    tree: Tree object
    train_gap_list: list[list[int]]
    train_response_list: list[int] 
  Returns:
    No output, updates leaf probabilities in place
  """
  #given a tree and a dictionary with gaps as keys and successes as values, trains the tree
  for x in range(len(train_gap_list)):
    tree.root.child.traverse_and_update(train_gap_list[x], train_response_list[x])


def assess_training_error(trained_tree, ground_truth_tree, train_gaps_list):
  """
    Given a ground truth model, trained tree, and a list of gaps, assesses the average training error  
    Input:
      trained_tree: Tree
      ground_truth_tree: Tree
      train_gaps_list: list[list[int]]
    Output:
      returns avg. training error
  """
  ground_truth_probs = []
  trained_tree_probs = []
  for x in range(len(train_gaps_list)):
    ground_truth_pred = run_single_GAP(ground_truth_tree, train_gaps_list[x])
    ground_truth_probs.append(ground_truth_pred)
    trained_tree_pred = run_single_GAP(trained_tree, train_gaps_list[x])
    trained_tree_probs.append(trained_tree_pred)
  error = 0
  for x in range(len(trained_tree_probs)):
    error += np.abs(ground_truth_probs[x] - trained_tree_probs[x])
  sample_size = len(train_gaps_list)
  mean_error = error/sample_size
  return mean_error


def run_gaps_and_return_error(trained_tree, ground_truth_tree, gap_list):
  """
    Used for assessing validation and testing error. Runs gaps through trained tree and ground truth tree to 
    assess avg error.   
    Input:
      trained_tree: Tree
      ground_truth_tree: Tree
      gap_list: list[list[int]]
    Output:
      avg. error: float
  """
  error = 0
  gap_index = 0
  for gap in gap_list:
    ground_truth_pred = run_single_GAP(ground_truth_tree, gap_list[gap_index])
    trained_tree_pred = run_single_GAP(trained_tree, gap_list[gap_index])
    error += np.abs(ground_truth_pred - trained_tree_pred)
    gap_index += 1
  sample_size = len(gap_list)
  mean_error = error/sample_size
  return mean_error


def assess_error(test_tree_responses, ground_truth_responses):
  """
    Given two response lists, calculates the error between them  
    Input: test_tree_responses (list[int]), ground_truth_responses(list[int])
    Output: error (int)
  """
  error = np.abs(sum(ground_truth_responses) - sum(test_tree_responses))
  return error


def train_tree_with_mutations(start_tree:t.Tree, ground_truth_tree:t.Tree, train_list, ground_truth_response_list, validate_list, test_list):
  """
      
    Input:
    Output:
  """
  successful_tree_trace = []
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)

  train_GAPS(prev_tree, train_list, ground_truth_response_list[:TRAINING_SAMPLE_SIZE])
  prev_train_error = assess_training_error(prev_tree, ground_truth_tree, train_list)
  TREE_MUTATION_TRAINING_ERROR.append(prev_train_error)

  prev_validation_error = run_gaps_and_return_error(start_tree, ground_truth_tree, validate_list)
  TREE_MUTATION_VALIDATION_ERROR.append(prev_validation_error)

  prev_test_error = run_gaps_and_return_error(start_tree, ground_truth_tree, test_list)
  TREE_MUTATION_TESTING_ERROR.append(prev_test_error)

  num_successful_mutations = 0
  num_unsuccessful_mutations = 0
  while num_successful_mutations < MUTATION_SIZE and num_unsuccessful_mutations <5000:
    potential_trees = []
    potential_trees_validation_error = []
    portenital_trees_training_error = []
    potential_trees_test_error = []
  # while prev_train_error > 0.019:
    while len(potential_trees) < 11:
      curr_tree = prev_tree.mutate()
      train_GAPS(curr_tree, train_list, ground_truth_response_list[:TRAINING_SAMPLE_SIZE])
      curr_train_error = assess_training_error(curr_tree, ground_truth_tree, train_list)
      curr_validation_error = run_gaps_and_return_error(curr_tree, ground_truth_tree, validate_list)
      curr_test_error = run_gaps_and_return_error(curr_tree, ground_truth_tree, test_list)
      print(f"current tree validate error = {curr_validation_error}")
      print(f"prev tree validate error = {prev_validation_error}")
      print(f"successful tree mutations = {num_successful_mutations}")
      print(f"unsuccessful tree mutations = {num_unsuccessful_mutations}")
      potential_trees.append(curr_tree)

      potential_trees_validation_error.append(curr_validation_error)
      portenital_trees_training_error.append(curr_train_error)
      potential_trees_test_error.append(curr_test_error)

    best_tree_idx = 0
    best_tree_validation_error = np.inf
    for x in range(len(potential_trees_validation_error) - 1):
      if potential_trees_validation_error[x] < best_tree_validation_error:
        best_tree_validation_error = potential_trees_validation_error[x]
        best_tree_idx = x

    if best_tree_validation_error < prev_validation_error:
      TREE_MUTATION_TRAINING_ERROR.append(portenital_trees_training_error[best_tree_idx])
      TREE_MUTATION_VALIDATION_ERROR.append(best_tree_validation_error)
      TREE_MUTATION_TESTING_ERROR.append(potential_trees_test_error[best_tree_idx])

      prev_tree = copy.deepcopy(potential_trees[best_tree_idx])
      prev_validation_error = best_tree_validation_error
      prev_train_error = portenital_trees_training_error[best_tree_idx]
      prev_test_error = potential_trees_test_error[best_tree_idx]
      successful_tree_trace.append(prev_tree)
      num_successful_mutations+=1
      UNSUCCESSFUL_MUTATION_COUNT.append(num_unsuccessful_mutations)
      num_unsuccessful_mutations = 0
    else: 
      num_unsuccessful_mutations +=1
    

  return prev_tree


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


#after making a list of all possible gap patterns, randomly selects 500 test
#gaps and prints them out
full_gap_list = binary_num_list(10)
np.random.shuffle(full_gap_list)

training_list = full_gap_list[:TRAINING_SAMPLE_SIZE]
validate_list = full_gap_list[TRAINING_SAMPLE_SIZE:TRAINING_SAMPLE_SIZE + VALIDATION_SAMPLE_SIZE]
test_list = full_gap_list[TRAINING_SAMPLE_SIZE + VALIDATION_SAMPLE_SIZE:TRAINING_SAMPLE_SIZE + VALIDATION_SAMPLE_SIZE + TESTING_SAMPLE_SIZE]

#Create ground truth responses and run gaps
ground_truth = t.build_bald_tree(10, shuffle_nums=False)
ground_truth = t.fill_tree_with_leaves(ground_truth, evenly_spaced=True)
ground_truth_responses = run_GAPS(ground_truth, full_gap_list)

#Create random test tree
test_tree = t.build_bald_tree(10, shuffle_nums=True)
test_tree = t.fill_tree_with_leaves(test_tree, evenly_spaced=False)


# final_tree = train_GAPS(test_tree, train_gap_dict)
final_tree = train_tree_with_mutations(test_tree, ground_truth,training_list, ground_truth_responses, validate_list, test_list)
final_tree.print2D(final_tree.root.child)

#plot training, validation, and test error
x_axis = np.arange(len(TREE_MUTATION_TRAINING_ERROR))
plt.plot(x_axis, TREE_MUTATION_TRAINING_ERROR, c="blue", label="avg. training error")
plt.plot(x_axis, TREE_MUTATION_VALIDATION_ERROR, ls="--", c="red", label="avg. validation error")
plt.plot(x_axis, TREE_MUTATION_TESTING_ERROR, ls="--", c="green", label="avg. testing error")
plt.xlabel("tree mutations")
plt.ylabel("average error")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir_path, "tree_mutation_error"))
plt.clf()

x_axis = np.arange(len(UNSUCCESSFUL_MUTATION_COUNT))
plt.plot(x_axis, UNSUCCESSFUL_MUTATION_COUNT, c="purple", ls='--')
plt.xlabel('Successful Mutations')
plt.yscale(value='log')
plt.ylabel('Number of Mutations Tried')
plt.tight_layout()
plt.savefig(os.path.join(results_dir_path, "mutation_count_plot"))
plt.clf()