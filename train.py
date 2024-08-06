import tree as t
import numpy as np
import gaps
import copy

MAX_SUCCESSFUL_MUTATIONS = 1000
MAX_UNSUCCESSFUL_MUTATIONS = 1000

TRAINING_SAMPLE_SIZE = 500
VALIDATION_SAMPLE_SIZE = 100
TESTING_SAMPLE_SIZE = 100


def greedy_mutations(start_tree:t.Tree, ground_truth_tree:t.Tree, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list):
  successful_tree_trace = []
  tree_mutation_training_error = []
  tree_mutation_validation_error = []
  tree_mutation_testing_error = []
  unsuccessful_mutation_count = []
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)
  
  #collect probs to assess error
  ground_truth_probs_trn, ground_truth_probs_val, ground_truth_probs_tst = ground_truth_tree.get_probs_from_data(tree_train_list, validate_list, test_list)
  
  #calculate initial errors
  prev_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)
  prev_prob_trn, prev_prob_val, prev_prob_tst = prev_tree.get_probs_from_data(tree_train_list, validate_list, test_list)

  prev_train_error = gaps.assess_error(ground_truth_probs_trn, prev_prob_trn)
  tree_mutation_training_error.append(prev_train_error)

  prev_validation_error = gaps.assess_error(ground_truth_probs_val, prev_prob_val)
  tree_mutation_validation_error.append(prev_validation_error)

  prev_test_error = gaps.assess_error(ground_truth_probs_tst, prev_prob_tst)
  tree_mutation_testing_error.append(prev_test_error)

  num_successful_mutations = 0
  num_unsuccessful_mutations = 0

  local_min_val_error = [1,1]
  consecutive_validation_error_increases = 0
  while num_successful_mutations<MAX_SUCCESSFUL_MUTATIONS and consecutive_validation_error_increases<=100 and num_unsuccessful_mutations < MAX_UNSUCCESSFUL_MUTATIONS:
    curr_tree = prev_tree.mutate()
    curr_tree.reset_leaf_probs()
    curr_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)

    curr_prob_trn, curr_prob_val, curr_prob_tst = curr_tree.get_probs_from_data(tree_train_list, validate_list, test_list)

    curr_train_error = gaps.assess_error(ground_truth_probs_trn, curr_prob_trn)
    curr_validation_error = gaps.assess_error(ground_truth_probs_val, curr_prob_val)
    curr_test_error = gaps.assess_error(ground_truth_probs_tst, curr_prob_tst)
    print(f"current tree train error = {curr_train_error[0]}")
    print(f"prev tree train error = {prev_train_error[0]}")
    print(f"successful tree mutations = {num_successful_mutations}")
    print(f"unsuccessful tree mutations = {num_unsuccessful_mutations}")

    if curr_train_error[0] < prev_train_error[0]:
      tree_mutation_training_error.append(curr_train_error)
      tree_mutation_validation_error.append(curr_validation_error)
      tree_mutation_testing_error.append(curr_test_error)

      prev_tree = copy.deepcopy(curr_tree)
      
      prev_train_error = curr_train_error
      prev_validation_error = curr_validation_error
      prev_test_error = curr_test_error
      successful_tree_trace.append(prev_tree)
      num_successful_mutations+=1
      unsuccessful_mutation_count.append(num_unsuccessful_mutations)
      num_unsuccessful_mutations = 0
    else: 
      num_unsuccessful_mutations +=1

  unsuccessful_mutation_count.append(num_unsuccessful_mutations)
  min_val = min(tree_mutation_validation_error)
  final_tree_idx = tree_mutation_validation_error.index(min_val)
  final_tree = successful_tree_trace[final_tree_idx]  
  return final_tree, tree_mutation_training_error[:final_tree_idx+1], tree_mutation_validation_error[:final_tree_idx+1], tree_mutation_testing_error[:final_tree_idx+1], unsuccessful_mutation_count[:final_tree_idx+1]


def pool_mutations(start_tree:t.Tree, ground_truth_tree:t.Tree, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list):
  """
      
    Input:
    Output:
  """
  successful_tree_trace = []
  tree_mutation_training_error = []
  tree_mutation_validation_error = []
  tree_mutation_testing_error = []
  unsuccessful_mutation_count = []
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)

  ground_truth_probs_trn, ground_truth_probs_val, ground_truth_probs_tst = ground_truth_tree.get_probs_from_data(tree_train_list, validate_list, test_list)
  prev_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)
  prev_prob_trn, prev_prob_val, prev_prob_tst = prev_tree.get_probs_from_data(tree_train_list, validate_list, test_list)

  prev_train_error = gaps.assess_error(ground_truth_probs_trn, prev_prob_trn)
  tree_mutation_training_error.append(prev_train_error)

  prev_validation_error = gaps.assess_error(ground_truth_probs_val, prev_prob_val)
  tree_mutation_validation_error.append(prev_validation_error)

  prev_test_error = gaps.assess_error(ground_truth_probs_tst, prev_prob_tst)
  tree_mutation_testing_error.append(prev_test_error)

  num_successful_mutations = 0
  num_unsuccessful_mutations = 0
  
  local_min_val_error = [1,1]
  consecutive_validation_error_increases = 0
  while num_successful_mutations<MAX_SUCCESSFUL_MUTATIONS and consecutive_validation_error_increases<=100 and num_unsuccessful_mutations < MAX_UNSUCCESSFUL_MUTATIONS:
    potential_trees = []
    potential_trees_validation_error = []
    potential_trees_training_error = []
    potential_trees_test_error = []
    while len(potential_trees) < 11:
      curr_tree = prev_tree.mutate()
      curr_tree.reset_leaf_probs()
      curr_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)

      curr_prob_trn, curr_prob_val, curr_prob_tst = curr_tree.get_probs_from_data(tree_train_list, validate_list, test_list)

      curr_train_error = gaps.assess_error(ground_truth_probs_trn, curr_prob_trn)
      curr_validation_error = gaps.assess_error(ground_truth_probs_val, curr_prob_val)
      curr_test_error = gaps.assess_error(ground_truth_probs_tst, curr_prob_tst)
      print(f"current tree train error = {curr_train_error[0]}")
      print(f"prev tree train error = {prev_train_error[0]}")
      print(f"successful tree mutations = {num_successful_mutations}")
      print(f"unsuccessful tree mutations = {num_unsuccessful_mutations}")
      potential_trees.append(curr_tree)

      potential_trees_validation_error.append(curr_validation_error)
      potential_trees_training_error.append(curr_train_error)
      potential_trees_test_error.append(curr_test_error)

    best_tree_training_error = min(potential_trees_training_error)
    best_tree_idx = potential_trees_training_error.index(best_tree_training_error)

    if best_tree_training_error[0] < prev_train_error[0]:
      if potential_trees_validation_error[best_tree_idx][0] < local_min_val_error[0]:
        local_min_val_error = potential_trees_validation_error[best_tree_idx]
        consecutive_validation_error_increases = 0      
      else:
        consecutive_validation_error_increases += 1  
      tree_mutation_training_error.append(best_tree_training_error)
      tree_mutation_validation_error.append(potential_trees_validation_error[best_tree_idx])
      tree_mutation_testing_error.append(potential_trees_test_error[best_tree_idx])

      prev_tree = copy.deepcopy(potential_trees[best_tree_idx])
      prev_validation_error = potential_trees_validation_error[best_tree_idx]
      prev_train_error = best_tree_training_error
      prev_test_error = potential_trees_test_error[best_tree_idx]
      successful_tree_trace.append(prev_tree)
      num_successful_mutations+=1
      unsuccessful_mutation_count.append(num_unsuccessful_mutations)
      num_unsuccessful_mutations = 0
    else: 
      num_unsuccessful_mutations +=1
  unsuccessful_mutation_count.append(num_unsuccessful_mutations)
  min_val = min(tree_mutation_validation_error)
  final_tree_idx = tree_mutation_validation_error.index(min_val)
  final_tree = successful_tree_trace[final_tree_idx]  
  return final_tree, tree_mutation_training_error[:final_tree_idx+1], tree_mutation_validation_error[:final_tree_idx+1], tree_mutation_testing_error[:final_tree_idx+1], unsuccessful_mutation_count[:final_tree_idx+1]

def chain_mutations(start_tree:t.Tree, ground_truth_tree:t.Tree, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list):
  """
    FIX THIS  
    Input:
    Output:
  """

  # accepted_error_trn = []
  # accepted_error_tst = []
  # accepted_error_val = []

  successful_tree_trace = []
  tree_mutation_training_error = []
  tree_mutation_validation_error = []
  tree_mutation_testing_error = []
  unsuccessful_mutation_count = []
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)
  
  #collect probs to assess error
  ground_truth_probs_trn, ground_truth_probs_val, ground_truth_probs_tst = ground_truth_tree.get_probs_from_data(tree_train_list, validate_list, test_list)
  
  #calculate initial errors
  prev_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)
  prev_prob_trn, prev_prob_val, prev_prob_tst = prev_tree.get_probs_from_data(tree_train_list, validate_list, test_list)

  prev_train_error = gaps.assess_error(ground_truth_probs_trn, prev_prob_trn)
  tree_mutation_training_error.append(prev_train_error)

  prev_validation_error = gaps.assess_error(ground_truth_probs_val, prev_prob_val)
  tree_mutation_validation_error.append(prev_validation_error)

  prev_test_error = gaps.assess_error(ground_truth_probs_tst, prev_prob_tst)
  tree_mutation_testing_error.append(prev_test_error)

  num_successful_mutations = 0
  num_unsuccessful_mutations = 0

  local_min_val_error = [1,1]
  consecutive_validation_error_increases = 0
  while num_successful_mutations<MAX_SUCCESSFUL_MUTATIONS and consecutive_validation_error_increases<=100 and num_unsuccessful_mutations < MAX_UNSUCCESSFUL_MUTATIONS:
    potential_trees = []
    potential_trees_validation_error = []
    potential_trees_training_error = []
    potential_trees_test_error = []

    curr_tree = prev_tree.mutate()
    
    # curr_prob_trn, curr_prob_val, curr_prob_tst = curr_tree.get_probs_from_data(tree_train_list, validate_list, test_list)

    # curr_train_error = gaps.assess_error(ground_truth_probs_trn, curr_prob_trn)
    # curr_validation_error = gaps.assess_error(ground_truth_probs_val, curr_prob_val)
    # curr_test_error = gaps.assess_error(ground_truth_probs_tst, curr_prob_tst)

    # potential_trees.append(curr_tree)
    # potential_trees_validation_error.append(curr_validation_error)
    # potential_trees_training_error.append(curr_train_error)
    # potential_trees_test_error.append(curr_test_error)
    while len(potential_trees) < 11:
      if len(potential_trees) > 0:
        curr_tree = curr_tree.mutate()
      # curr_tree = curr_tree.mutate()
      curr_tree.reset_leaf_probs()
      curr_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)
      
      curr_prob_trn, curr_prob_val, curr_prob_tst = curr_tree.get_probs_from_data(tree_train_list, validate_list, test_list)

      curr_train_error = gaps.assess_error(ground_truth_probs_trn, curr_prob_trn)
      curr_validation_error = gaps.assess_error(ground_truth_probs_val, curr_prob_val)
      curr_test_error = gaps.assess_error(ground_truth_probs_tst, curr_prob_tst)
      print(f"current tree train error = {curr_train_error[0]}")
      print(f"prev tree train error = {prev_train_error[0]}")
      print(f"successful tree mutations = {num_successful_mutations}")
      print(f"unsuccessful tree mutations = {num_unsuccessful_mutations}")
      potential_trees.append(curr_tree)
      potential_trees_validation_error.append(curr_validation_error)
      potential_trees_training_error.append(curr_train_error)
      potential_trees_test_error.append(curr_test_error)

    best_tree_idx = 0
    best_tree_training_error = [np.inf, np.inf]
    for x in range(len(potential_trees_training_error)):
      if potential_trees_training_error[x][0] < best_tree_training_error[0]:
        best_tree_training_error = potential_trees_training_error[x]
        best_tree_idx = x

    if best_tree_training_error[0] < prev_train_error[0]:
      tree_mutation_training_error.append(best_tree_training_error)
      tree_mutation_validation_error.append(potential_trees_validation_error[best_tree_idx])
      tree_mutation_testing_error.append(potential_trees_test_error[best_tree_idx])

      prev_tree = copy.deepcopy(potential_trees[best_tree_idx])
      prev_validation_error = potential_trees_validation_error[best_tree_idx]
      prev_train_error = best_tree_training_error
      prev_test_error = potential_trees_test_error[best_tree_idx]
      successful_tree_trace.append(prev_tree)
      unsuccessful_mutation_count.append(num_unsuccessful_mutations)
      num_successful_mutations += 1
      num_unsuccessful_mutations = 0
    else:
      num_unsuccessful_mutations += 1
  unsuccessful_mutation_count.append(num_unsuccessful_mutations)
  min_val = min(tree_mutation_validation_error)
  final_tree_idx = tree_mutation_validation_error.index(min_val)
  final_tree = successful_tree_trace[final_tree_idx]  
  return final_tree, tree_mutation_training_error[:final_tree_idx+1], tree_mutation_validation_error[:final_tree_idx+1], tree_mutation_testing_error[:final_tree_idx+1], unsuccessful_mutation_count[:final_tree_idx+1]


def ensemble(num_trees, ground_truth, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list, tree_list = None, return_trees = False):
  #create a list to store all trees
  tree_outputs = []  
  final_error_tst = []
  if tree_list:
    for tree in tree_list:
      tree.reset_leaf_probs()
      final_tree, tree_mutation_training_error, tree_mutation_validation_error, tree_mutation_testing_error, unsuccessful_mutation_count = pool_mutations(tree, ground_truth, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list)
      tree_outputs.append(final_tree)
      final_error_tst.append(tree_mutation_testing_error[-1])
  else:
    forest = []
    for x in range(num_trees):
      new_tree = t.build_bald_tree(10, shuffle_nums=True)
      new_tree.fill_tree_with_leaves(evenly_spaced=False)
      forest.append(new_tree)
    for tree in forest:
      final_tree, tree_mutation_training_error, tree_mutation_validation_error, tree_mutation_testing_error, unsuccessful_mutation_count = pool_mutations(tree, ground_truth, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list)
      tree_outputs.append(final_tree)
      final_error_tst.append(tree_mutation_testing_error[-1])
  test_probs_ensemble = []
  test_probs_each_gap = []
  for gap in test_list:
    single_gap_probs = []
    for tree in tree_outputs:
      single_gap_probs.append(tree.run_tree_single_gap(gap, leaf_prob=True))
    test_probs_ensemble.append(np.mean(single_gap_probs))
    test_probs_each_gap.append(single_gap_probs)
  test_probs_ground_truth = ground_truth.run_tree_gaps(test_list, leaf_prob=True)
  single_gap_variance = []
  for prediction in test_probs_each_gap:
    single_gap_variance.append(np.var(prediction))
  single_gap_se = []
  for x in range(len(test_probs_ensemble)):
    single_gap_se.append(gaps.assess_single_gap_error(test_probs_ground_truth[x], test_probs_ensemble[x]))
  error = gaps.assess_error(test_probs_ground_truth, test_probs_ensemble)
  final_error_tst.append(error)
  if return_trees:
    return final_error_tst, tree_outputs, single_gap_se, single_gap_variance
  return final_error_tst, tree_outputs, single_gap_se, single_gap_variance
