import tree as t
import nodes as n
import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import copy
from datetime import datetime
import os
import ast
from tqdm import tqdm

MUTATION_SIZE = 50

TREE_MUTATION_TRAINING_ERROR = []
TREE_MUTATION_VALIDATION_ERROR = []

np.random.seed(1)
random.seed(1)
now = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir_path = os.path.join("./results", now)
os.mkdir(results_dir_path)


def read_gap(gap):
  #converts a string into a readable list to traverse a tree
  new_gap = []
  for x in range(len(gap)):
    new_gap.append(int(gap[x]))
  return new_gap


def binary_num_list(x, prefix = ''):
  #creates all possible gap activation patterns of a certain size
  if x == 0:
    return [prefix]
  else:
    return (binary_num_list(x - 1, prefix + '0') +
            binary_num_list(x - 1, prefix + '1'))

def run_GAPS(tree, test_list):
  #Given a list of gaps, traverses a given tree for all gaps in the list
  #Fill out empty OUTPUT_DICT, used to calculate error for each iteration
  #BUG: currently only going to farthest right node
  response_list = []
  for gap in test_list:
    prob = tree.root.child.traverse_GAP(gap)
    if prob not in tree.OUTPUT_DICT:
      tree.OUTPUT_DICT[prob] = [0, 0]
  x = 0
  for gap in test_list:
    test_prob = tree.root.child.traverse_GAP(gap)
    test_val = random.choices([0, 1], weights = (1-test_prob, test_prob))[0]
    response_list.append(test_val)
    tree.OUTPUT_DICT[test_prob][0] += test_val
    tree.OUTPUT_DICT[test_prob][1] += 1
    if str(gap) not in tree.GAP_OUTPUT:
      tree.GAP_OUTPUT[str(gap)] = test_val
    x += 1
  return response_list

def run_single_GAP(tree, gap):
  prob = tree.root.child.traverse_GAP(gap)
  # response = random.choices([0, 1], weights = (1 - prob, prob))
  return prob

def train_GAPS(tree, train_dict: dict):
  #given a tree and a dictionary with gaps as keys and successes as values, trains the tree
  curr_tree = tree
  for gap,output in train_dict.items():
    gap_list = ast.literal_eval(gap)
    curr_tree.root.child.traverse_and_update(gap_list, output)


def assess_training_error(trained_tree, ground_truth_tree, train_dict):
  ground_truth_probs = []
  trained_tree_probs = []
  for gap in train_dict.keys():
    gap_list = ast.literal_eval(gap)
    ground_truth_pred = run_single_GAP(ground_truth_tree, gap_list)
    ground_truth_probs.append(ground_truth_pred)
    trained_tree_pred = run_single_GAP(trained_tree, gap_list)
    trained_tree_probs.append(trained_tree_pred)
  error = 0
  for x in range(len(trained_tree_probs)):
    error += np.abs(ground_truth_probs[x] - trained_tree_probs[x])
  sample_size = len(train_dict.keys())
  mean_error = error/sample_size
  return mean_error


def validate_and_return_error(trained_tree, ground_truth_tree, validate_list):
  error = 0
  gap_index = 0
  for gap in validate_list:
    ground_truth_pred = run_single_GAP(ground_truth_tree, validate_list[gap_index])
    trained_tree_pred = run_single_GAP(trained_tree, validate_list[gap_index])
    error += np.abs(ground_truth_pred - trained_tree_pred)
    gap_index += 1
  sample_size = len(validate_list)
  mean_error = error/sample_size
  return mean_error


def assess_error(test_tree_responses, ground_truth_responses):
  #given a trained tree and ground truth model, error of the trained tree is assessed and returned
  error = np.abs(sum(ground_truth_responses) - sum(test_tree_responses))
  return error


def train_tree_with_mutations(start_tree, ground_truth_tree, train_dict, validate_list):
  successful_tree_trace = []
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)

  train_GAPS(prev_tree, train_dict)
  prev_train_error = assess_training_error(prev_tree, ground_truth_tree, train_dict)
  TREE_MUTATION_TRAINING_ERROR.append(prev_train_error)

  prev_validation_error = validate_and_return_error(start_tree, ground_truth_tree, validate_list)
  TREE_MUTATION_VALIDATION_ERROR.append(prev_validation_error)

  num_successful_mutations = 0
  while num_successful_mutations < MUTATION_SIZE:
    curr_tree = prev_tree.mutate()
    train_GAPS(curr_tree, train_dict)
    curr_train_error = assess_training_error(curr_tree, ground_truth_tree, train_dict)
    curr_validation_error = validate_and_return_error(curr_tree, ground_truth_tree, validate_list)
    print(f"current tree error = {curr_train_error}")
    print(f"prev tree error = {prev_train_error}")
    if curr_validation_error < prev_validation_error:
      TREE_MUTATION_TRAINING_ERROR.append(curr_train_error)
      TREE_MUTATION_VALIDATION_ERROR.append(curr_validation_error)

      prev_tree = copy.deepcopy(curr_tree)
      prev_validation_error = curr_validation_error
      successful_tree_trace.append(prev_tree)
      num_successful_mutations+=1

  return prev_tree


def running_average(data, window_size):
  ra = np.empty(len(data))
  for i in range(len(data)):
    if i < window_size:
      ra[i] = np.mean(data[:i+1])
    else:
      ra[i] = np.mean(data[i-window_size+1:i+1])
  return ra


def average(data):
  avg = np.empty(len(data))
  for i in range(len(data)):
    avg[i] = np.mean(data[:i])
  return avg


#after making a list of all possible gap patterns, randomly selects 500 test
#gaps and prints them out
binary_list = binary_num_list(10)
test_list = []
for x in range(0,1000):
  test_list.append(read_gap(random.choice(binary_list)))


#Create, fill, and print tree
print("ground truth: ")
ground_truth = t.build_bald_tree(10, shuffle_nums=False)
ground_truth = t.fill_tree_with_leaves(ground_truth, evenly_spaced=True)
# ground_truth.print2D(ground_truth.root.child)
# print("__________________________________________")

ground_truth_results = run_GAPS(ground_truth, test_list)

test_tree = t.build_bald_tree(10, shuffle_nums=True)
test_tree = t.fill_tree_with_leaves(test_tree, evenly_spaced=False)
# test_tree.print2D(test_tree.root.child)
# print("__________________________________________")

num_train_gaps = 500
train_gap_keys = random.sample(list(ground_truth.GAP_OUTPUT.keys()), num_train_gaps)
train_gap_dict = {key: ground_truth.GAP_OUTPUT[key] for key in train_gap_keys}


num_test_gaps = 100
validate_set = {}
while len(validate_set) <= num_test_gaps:
  new_gap = random.choice(list(ground_truth.GAP_OUTPUT.keys()))
  if new_gap not in train_gap_dict and new_gap not in validate_set:
    validate_set[new_gap] = ground_truth.GAP_OUTPUT[new_gap]

validate_list = []
ground_truth_validate_responses = []
for key,val in validate_set.items():
    gap = ast.literal_eval(key)
    validate_list.append(gap)
    ground_truth_validate_responses.append(val)


# final_tree = train_GAPS(test_tree, train_gap_dict)
final_tree = train_tree_with_mutations(test_tree, ground_truth, train_gap_dict, validate_list)
final_tree.print2D(final_tree.root.child)

x_axis = np.arange(
  start=num_train_gaps-1,
  stop=len(TREE_MUTATION_TRAINING_ERROR)*num_train_gaps,
  step=num_train_gaps
)
plt.plot(x_axis, TREE_MUTATION_TRAINING_ERROR, ls="--", c="blue")
plt.scatter(x_axis, TREE_MUTATION_TRAINING_ERROR, c="blue")
plt.xlabel("total number of gaps trained on")
plt.ylabel("final training loss")
plt.tight_layout()
plt.savefig(os.path.join(results_dir_path, "mutation_training_error"))
plt.clf()


x_axis = np.arange(
  start=num_test_gaps-1,
  stop=len(TREE_MUTATION_VALIDATION_ERROR)*num_test_gaps,
  step=num_test_gaps
)
plt.plot(x_axis, TREE_MUTATION_VALIDATION_ERROR, ls="--", c="red")
plt.scatter(x_axis, TREE_MUTATION_VALIDATION_ERROR, c="red")
plt.xlabel("total number of gaps trained on")
plt.ylabel("final validation loss")
plt.tight_layout()
plt.savefig(os.path.join(results_dir_path, "mutation_validation_error"))
plt.clf()
