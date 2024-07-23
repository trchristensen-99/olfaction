import tree as t
import numpy as np
import gaps
import copy

MUTATION_SIZE = 1000

TRAINING_SAMPLE_SIZE = 500
VALIDATION_SAMPLE_SIZE = 100
TESTING_SAMPLE_SIZE = 100

TREE_MUTATION_TRAINING_ERROR = []
TREE_MUTATION_VALIDATION_ERROR = []
TREE_MUTATION_TESTING_ERROR = []
UNSUCCESSFUL_MUTATION_COUNT = []

def greedy_mutations(start_tree:t.Tree, ground_truth_tree:t.Tree, train_list, ground_truth_response_list, validate_list, test_list):
  successful_tree_trace = []
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)

  gaps.train_GAPS(prev_tree, train_list, ground_truth_response_list[:TRAINING_SAMPLE_SIZE])
  prev_train_error = gaps.assess_training_error(prev_tree, ground_truth_tree, train_list)
  TREE_MUTATION_TRAINING_ERROR.append(prev_train_error)

  prev_validation_error = gaps.run_gaps_and_return_error(start_tree, ground_truth_tree, validate_list)
  TREE_MUTATION_VALIDATION_ERROR.append(prev_validation_error)

  prev_test_error = gaps.run_gaps_and_return_error(start_tree, ground_truth_tree, test_list)
  TREE_MUTATION_TESTING_ERROR.append(prev_test_error)

  num_successful_mutations = 0
  num_unsuccessful_mutations = 0
  while num_successful_mutations < MUTATION_SIZE and num_unsuccessful_mutations <10000:
    curr_tree = prev_tree.mutate()
    gaps.train_GAPS(curr_tree, train_list, ground_truth_response_list[:TRAINING_SAMPLE_SIZE])
    curr_train_error = gaps.assess_training_error(curr_tree, ground_truth_tree, train_list)
    curr_validation_error = gaps.run_gaps_and_return_error(curr_tree, ground_truth_tree, validate_list)
    curr_test_error = gaps.run_gaps_and_return_error(curr_tree, ground_truth_tree, test_list)
    print(f"current tree validate error = {curr_validation_error}")
    print(f"prev tree validate error = {prev_validation_error}")
    print(f"successful tree mutations = {num_successful_mutations}")
    print(f"unsuccessful tree mutations = {num_unsuccessful_mutations}")

    if curr_validation_error < prev_validation_error:
      TREE_MUTATION_TRAINING_ERROR.append(curr_train_error)
      TREE_MUTATION_VALIDATION_ERROR.append(curr_validation_error)
      TREE_MUTATION_TESTING_ERROR.append(curr_test_error)

      prev_tree = copy.deepcopy(curr_tree)
      prev_validation_error = curr_validation_error
      prev_train_error = curr_train_error
      prev_test_error = curr_test_error
      successful_tree_trace.append(prev_tree)
      num_successful_mutations+=1
      num_unsuccessful_mutations = 0
    else: 
      num_unsuccessful_mutations +=1

  return prev_tree


def pool_mutations(start_tree:t.Tree, ground_truth_tree:t.Tree, train_list, ground_truth_response_list, validate_list, test_list):
  """
      
    Input:
    Output:
  """
  successful_tree_trace = []
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)

  gaps.train_GAPS(prev_tree, train_list, ground_truth_response_list[:TRAINING_SAMPLE_SIZE])
  prev_train_error = gaps.assess_training_error(prev_tree, ground_truth_tree, train_list)
  TREE_MUTATION_TRAINING_ERROR.append(prev_train_error)

  prev_validation_error = gaps.run_gaps_and_return_error(start_tree, ground_truth_tree, validate_list)
  TREE_MUTATION_VALIDATION_ERROR.append(prev_validation_error)

  prev_test_error = gaps.run_gaps_and_return_error(start_tree, ground_truth_tree, test_list)
  TREE_MUTATION_TESTING_ERROR.append(prev_test_error)

  num_successful_mutations = 0
  num_unsuccessful_mutations = 0
  while num_successful_mutations < MUTATION_SIZE and num_unsuccessful_mutations <5000:
    potential_trees = []
    potential_trees_validation_error = []
    portenital_trees_training_error = []
    potential_trees_test_error = []
    while len(potential_trees) < 11:
      curr_tree = prev_tree.mutate()
      gaps.train_GAPS(curr_tree, train_list, ground_truth_response_list[:TRAINING_SAMPLE_SIZE])
      curr_train_error = gaps.assess_training_error(curr_tree, ground_truth_tree, train_list)
      curr_validation_error = gaps.run_gaps_and_return_error(curr_tree, ground_truth_tree, validate_list)
      curr_test_error = gaps.run_gaps_and_return_error(curr_tree, ground_truth_tree, test_list)
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

def chain_mutations(start_tree:t.Tree, ground_truth_tree:t.Tree, train_list, ground_truth_response_list, validate_list, test_list):
  """
      
    Input:
    Output:
  """
  successful_tree_trace = []
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)

  gaps.train_GAPS(prev_tree, train_list, ground_truth_response_list[:TRAINING_SAMPLE_SIZE])
  prev_train_error = gaps.assess_training_error(prev_tree, ground_truth_tree, train_list)
  TREE_MUTATION_TRAINING_ERROR.append(prev_train_error)

  prev_validation_error = gaps.run_gaps_and_return_error(start_tree, ground_truth_tree, validate_list)
  TREE_MUTATION_VALIDATION_ERROR.append(prev_validation_error)

  prev_test_error = gaps.run_gaps_and_return_error(start_tree, ground_truth_tree, test_list)
  TREE_MUTATION_TESTING_ERROR.append(prev_test_error)

  num_successful_mutations = 0
  num_unsuccessful_mutations = 0
  while num_successful_mutations < MUTATION_SIZE and num_unsuccessful_mutations <5000:
    potential_trees = []
    potential_trees_validation_error = []
    portenital_trees_training_error = []
    potential_trees_test_error = []

    curr_tree = prev_tree.mutate()
    curr_train_error = gaps.assess_training_error(curr_tree, ground_truth_tree, train_list)
    curr_validation_error = gaps.run_gaps_and_return_error(curr_tree, ground_truth_tree, validate_list)
    curr_test_error = gaps.run_gaps_and_return_error(curr_tree, ground_truth_tree, test_list)

    potential_trees.append(curr_tree)
    potential_trees_validation_error.append(curr_validation_error)
    portenital_trees_training_error.append(curr_train_error)
    potential_trees_test_error.append(curr_test_error)
    while len(potential_trees) < 11:
      curr_tree = curr_tree.mutate()
      gaps.train_GAPS(curr_tree, train_list, ground_truth_response_list[:TRAINING_SAMPLE_SIZE])
      curr_train_error = gaps.assess_training_error(curr_tree, ground_truth_tree, train_list)
      curr_validation_error = gaps.run_gaps_and_return_error(curr_tree, ground_truth_tree, validate_list)
      curr_test_error = gaps.run_gaps_and_return_error(curr_tree, ground_truth_tree, test_list)
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
