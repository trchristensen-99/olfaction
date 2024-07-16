import Tree as t
import Nodes as n
import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import copy
from datetime import datetime
import os
import ast

TREE_MUTATION_ERROR = {}
# ERROR_DICT = {}
# OUTPUT_DICT = {}
# GAP_OUTPUT = {}
# TRAIN_ERROR = {}

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

# example

def train_GAPS(tree, train_dict: dict):
  #given a tree and a dictionary with gaps as keys and successes as values, trains the tree
  curr_tree = tree
  for key,val in train_dict.items():
    gap = ast.literal_eval(key)
    output = val
    curr_tree.root.child.traverse_and_update(gap, output)

def assess_error(test_tree_responses, ground_truth_responses):
  #given a trained tree and ground truth model, error of the trained tree is assessed and returned
  error = np.abs(sum(ground_truth_responses)- sum(test_tree_responses))
  # error = 0
  # for gaps in test_tree.GAP_OUTPUT.keys():
  #   error += np.abs(ground_truth.GAP_OUTPUT[gaps] - test_tree.GAP_OUTPUT[gaps])
  return error

def train_tree_with_mutations(start_tree, ground_truth_responses, train_set, validate_set):
  #Starting with a random tree and a ground truth model, this functions modifies trees until 10 trees are constructed, each of which
  #performed better than the previous tree
  # trees_tested = 0
  successful_tree_trace = []
  
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)  
  train_GAPS(prev_tree, train_set)
  prev_responses = run_GAPS(prev_tree, validate_set)
  prev_tree_error = assess_error(prev_responses, ground_truth_responses)
  num_successful_mutations = 0
  while num_successful_mutations < 5:
    curr_tree = prev_tree.mutate()
    train_GAPS(curr_tree, train_set)
    curr_responses = run_GAPS(curr_tree, validate_set)
    curr_tree_error = assess_error(curr_responses, ground_truth_responses)
    # print(f"prev tree error = {prev_tree_error}")
    # print(f"curr tree error = {curr_tree_error}")
    if curr_tree_error < prev_tree_error: 
      prev_tree = copy.deepcopy(curr_tree)
      prev_tree_error = curr_tree_error
      successful_tree_trace.append(prev_tree)
      num_successful_mutations += 1
      TREE_MUTATION_ERROR[num_successful_mutations] = curr_tree_error
  return prev_tree
   
    

#after making a list of all possible gap patterns, randomly selects 500 test
#gaps and prints them out
binary_list = binary_num_list(10)
test_list = []
for x in range(0,1000):
  test_list.append(read_gap(random.choice(binary_list)))
# print(test_list)


#adds leafs to the decision tree
leaf_list = []
leaf6 = n.LeafNode(0.55)
leaf_list.append(leaf6)
leaf7 = n.LeafNode(0.65)
leaf_list.append(leaf7)
leaf8 = n.LeafNode(0.75)
leaf_list.append(leaf8)
leaf9 = n.LeafNode(0.85)
leaf_list.append(leaf9)
leaf10 = n.LeafNode(0.95)
leaf_list.append(leaf10)
leaf1 = n.LeafNode(0.05)
leaf_list.append(leaf1)
leaf = n.LeafNode(0.10)
leaf_list.append(leaf)
leaf2 = n.LeafNode(0.15)
leaf_list.append(leaf2)
leaf3 = n.LeafNode(0.25)
leaf_list.append(leaf3)
leaf4 = n.LeafNode(0.35)
leaf_list.append(leaf4)
leaf5 = n.LeafNode(0.45)
leaf_list.append(leaf5)


#Create, fill, and print tree
print("ground truth: ")
ground_truth = t.build_bald_tree(10, shuffle_nums=False)
ground_truth = t.fill_tree_with_leaves(ground_truth, evenly_spaced=True)
ground_truth.print2D(ground_truth.root.child)
print("__________________________________________")

ground_truth_results = run_GAPS(ground_truth, test_list)

test_tree = t.build_bald_tree(10, shuffle_nums=True)
test_tree = t.fill_tree_with_leaves(test_tree, evenly_spaced=False)
test_tree.print2D(test_tree.root.child)
print("__________________________________________")


train_gap_keys = random.sample(list(ground_truth.GAP_OUTPUT.keys()), 500)
train_gap_dict = {key: ground_truth.GAP_OUTPUT[key] for key in train_gap_keys}
# for key,val in train_gap_dict.items():
#   print(f"key: {type(key)}, val: {val}")

# train_GAPS(test_tree, train_gaps)
# test_tree.print2D(test_tree.root.child)

validate_set = {}
while len(validate_set) <= 100:
  new_gap = random.choice(list(ground_truth.GAP_OUTPUT.keys()))
  if new_gap not in train_gap_dict and new_gap not in validate_set:
    validate_set[new_gap] = ground_truth.GAP_OUTPUT[new_gap]

validate_list = []
for key,val in validate_set.items():
    gap = ast.literal_eval(key)
    validate_list.append(gap)


final_tree = train_tree_with_mutations(test_tree, ground_truth_results, train_gap_dict, validate_list)
final_tree.print2D(final_tree.root.child)

for key,val in TREE_MUTATION_ERROR.items():
  print(f"Mutation: {key} | Error: {val}")

p = 0
# train_GAPS(test_tree, train_gaps)
# test_tree.print2D(test_tree.root.child)

# run_GAPS(test_tree, validate_set)
# print(test_tree.OUTPUT_DICT)


# test_tree = train_tree_with_mutations(test_tree, ground_truth, train_gaps, validate_list)

# run_GAPS(test_tree,validate_list)
# print(assess_error(test_tree, tree1))

# test_tree = test_tree.mutate()
# test_tree.print2D(test_tree.root.child)


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


# ##create tree to train: 
# train_tree = t.Tree(0, 10)
# train_tree.create_tree_random(8)
# for x in range(9):
#   new_leaf = n.LeafNode()
#   train_tree.root.child.add_node_in_order(new_leaf)
# t.print2D(train_tree.root.child)