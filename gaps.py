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

MUTATION_SIZE = 7
# TREE_MUTATION_ERROR = {}
TREE_MUTATION_ERROR = []
TREE_BUILDING_TRAINING_ERROR_TRACE = [[] for _ in range(MUTATION_SIZE)]
TREE_BUILDING_TESTING_ERROR_TRACE = [[] for _ in range(MUTATION_SIZE)]

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
  response = random.choices([0, 1], weights = (1 - prob, prob))
  return response[0]

def train_GAPS(tree, train_dict: dict):
  #given a tree and a dictionary with gaps as keys and successes as values, trains the tree
  curr_tree = tree
  error_trace = []
  for gap,output in train_dict.items():
    gap_list = ast.literal_eval(gap)
    response = curr_tree.root.child.traverse_and_update(gap_list, output)
    # response = run_single_GAP(tree, gap_list)
    error = np.abs(output - response)
    error_trace.append(error)
  return error_trace

# def assess_error(test_tree_responses, ground_truth_responses):
#   #given a trained tree and ground truth model, error of the trained tree is assessed and returned
#   error = np.abs(sum(ground_truth_responses) - sum(test_tree_responses))
#   return error

def assess_error(test_tree_responses, ground_truth_responses):
  #given a trained tree and ground truth model, error of the trained tree is assessed and returned
  error = 0
  for x in range(len(test_tree_responses)):
    error += np.abs(ground_truth_responses[x] - test_tree_responses[x])
  # ground_truth_responses = np.array(ground_truth_responses)
  # test_tree_responses = np.array(test_tree_responses)
  # error = sum(np.abs(ground_truth_responses - test_tree_responses))
  return error

def train_tree_with_mutations(start_tree, ground_truth_responses, train_set, validate_set):
  #Starting with a random tree and a ground truth model, this functions modifies trees until 10 trees are constructed, each of which
  #performed better than the previous tree
  # trees_tested = 0
  successful_tree_trace = []
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)  
  start_tree_error_trace = train_GAPS(prev_tree, train_set)
  TREE_BUILDING_TRAINING_ERROR_TRACE[0] = start_tree_error_trace
  prev_responses = run_GAPS(prev_tree, validate_set)
  # TREE_BUILDING_TESTING_ERROR_TRACE[0] = prev_responses
  prev_tree_error = assess_error(prev_responses, ground_truth_responses)
  TREE_MUTATION_ERROR.append(prev_tree_error)
  num_successful_mutations = 0
  while num_successful_mutations < MUTATION_SIZE:
    # print(f"successful mutations: {num_successful_mutations}")
    curr_tree = prev_tree.mutate()
    curr_tree_train_error_trace = train_GAPS(curr_tree, train_set)
    curr_responses = run_GAPS(curr_tree, validate_set)
    curr_tree_error = assess_error(curr_responses, ground_truth_responses)
    print(f"num successful mutations = {num_successful_mutations}")
    print(f"prev tree error = {prev_tree_error}")
    print(f"curr tree error = {curr_tree_error}")
    if curr_tree_error < prev_tree_error: 
      prev_tree = copy.deepcopy(curr_tree)
      prev_tree_error = curr_tree_error
      successful_tree_trace.append(prev_tree)
      TREE_BUILDING_TRAINING_ERROR_TRACE[num_successful_mutations] = curr_tree_train_error_trace
      num_successful_mutations += 1
      # TREE_MUTATION_ERROR[num_successful_mutations] = curr_tree_error
      TREE_MUTATION_ERROR.append(curr_tree_error)
  return curr_tree


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

  # cumulative_sum = np.cumsum(np.insert(data, 0, 0))
  # return (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / float(window_size)


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
final_tree = train_tree_with_mutations(test_tree, ground_truth_validate_responses, train_gap_dict, validate_list)
final_tree.print2D(final_tree.root.child)

# for key,val in TREE_MUTATION_ERROR.items():
#   print(f"Mutation: {key} | Error: {val}")
for x in range(len(TREE_MUTATION_ERROR)):
  if x == 0:
    print(f"start tree error: {TREE_MUTATION_ERROR[x]}")
  else:
    print(f"mutation {x} error: {TREE_MUTATION_ERROR[x]}")

train_run_avg = []
for traces in TREE_BUILDING_TRAINING_ERROR_TRACE:
  trace_running_averages = []
  # running_avg = running_average(traces, window_size=5)
  avg = average(traces)
  # trace_running_averages.append(running_avg)
  trace_running_averages.append(avg)
  train_run_avg.append(trace_running_averages)


x_axis = np.arange(len(train_run_avg)* num_train_gaps)

run_train_avg_flat = np.array([])
for run_avg_i in train_run_avg:
  run_avg_i_flat = np.concatenate(run_avg_i)
  run_train_avg_flat = np.concatenate((run_train_avg_flat, run_avg_i_flat))

plt.plot(x_axis, run_train_avg_flat)
plt.xlabel("total number of gaps trained on")
plt.ylabel("average training loss")
plt.tight_layout()
plt.savefig(os.path.join(results_dir_path, "mutation_training_error"))
plt.clf()


print(len(TREE_MUTATION_ERROR))
x_axis = np.arange(
  start=num_train_gaps-1,
  stop=len(train_run_avg)*num_train_gaps,
  step=num_train_gaps
)
x_axis = np.insert(x_axis, 0, 0)

plt.plot(x_axis, TREE_MUTATION_ERROR, ls="--", c="red")
plt.scatter(x_axis, TREE_MUTATION_ERROR, c="red")
plt.xlabel("total number training GAPs")
plt.ylabel("total validation loss")
plt.tight_layout()
plt.savefig(os.path.join(results_dir_path, "mutation_testing_error"))
plt.clf()
# x = np.array(list(results.keys()))
# y = np.array([v[0] / v[1] for v in results.values()])
# fig, ax = plt.subplots()
# ax.scatter(x, y)
# plt.xlabel('p(lick)')
# plt.ylabel('p̂(lick)')
# b, a = np.polyfit(x, y, deg=1)
# plt.plot(x, a + b * x)
# plt.title('Simulated Prob vs. True Prob')
# # plt.show()


# #Print and graph model error
# x = np.array(list(tree1.ERROR_DICT.keys())) 
# y = np.array(list(tree1.ERROR_DICT.values()))
# fig, ax = plt.subplots()
# ax.scatter(x, y)
# plt.xlabel('iteration')
# plt.ylabel('error')
# b, a = np.polyfit(x, y, deg=1)
# plt.title('Error vs. Iteration')
# # plt.savefig(os.path.join(results_dir_path, "error_vs_itr.png"))
# plt.show()