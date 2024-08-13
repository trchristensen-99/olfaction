import tree as t
import numpy as np
import gaps
import copy

#Set hyperparameters to limit training runtime
MAX_SUCCESSFUL_MUTATIONS = 100
MAX_UNSUCCESSFUL_MUTATIONS = 1000


def greedy_mutations(start_tree:t.Tree, ground_truth_tree:t.Tree, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list):
  """
  Trains a given tree by performing one mutation per epoch. If the mutated tree is better, that will be the next tree to be mutated. Otherwise
  the original tree is mutated again. 
  Inputs:
    start_tree: tree to be trained, Tree
    ground_truth_tree: ground truth model, Tree
    leaf_train_list: GAPs to train leaves, [GAPs]
    leaf_train_responses: Ground truth responses to leaf train gaps, [int in (0,1)]
    tree_train_list: training GAPs, [GAPs]
    validate_list: validation GAPs, [GAPs]
    test_list: testing GAPs, [GAPs]
  Outputs:
    final_tree: final tree structure, Tree
    tree_mutation_training_error: trainig error trace, [[mean squared error (float), standard deviation (float)]]
    tree_mutation_validation_error: validation error trace [[mean squared error (float), standard deviation (float)]]
    tree_mutation_testing_error: testing error trace [[mean squared error (float), standard deviation (float)]]
    unsuccessful_mutation_count: trace of unsuccessful mutations per epoch [int]
  """
  #Intitialize trace lists
  successful_tree_trace = []
  tree_mutation_training_error = []
  tree_mutation_validation_error = []
  tree_mutation_testing_error = []
  unsuccessful_mutation_count = []

  #Specify starting tree
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)
  
  #Collect probs to assess error
  ground_truth_probs_trn, ground_truth_probs_val, ground_truth_probs_tst = ground_truth_tree.get_probs_from_data(tree_train_list, validate_list, test_list)
  
  #Calculate initial errors
  prev_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)
  prev_prob_trn, prev_prob_val, prev_prob_tst = prev_tree.get_probs_from_data(tree_train_list, validate_list, test_list)

  prev_train_error = gaps.assess_error(ground_truth_probs_trn, prev_prob_trn)
  tree_mutation_training_error.append(prev_train_error)

  prev_validation_error = gaps.assess_error(ground_truth_probs_val, prev_prob_val)
  tree_mutation_validation_error.append(prev_validation_error)

  prev_test_error = gaps.assess_error(ground_truth_probs_tst, prev_prob_tst)
  tree_mutation_testing_error.append(prev_test_error)

  #Initialize mutation counters
  num_successful_mutations = 0
  num_unsuccessful_mutations = 0

  #Mutation training loop
  while num_successful_mutations<MAX_SUCCESSFUL_MUTATIONS and num_unsuccessful_mutations < MAX_UNSUCCESSFUL_MUTATIONS:
    #mutate tree and retrain leaf probs
    curr_tree = prev_tree.mutate()
    curr_tree.reset_leaf_probs()
    curr_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)
    
    #calculate probabilities and assess error for data sets
    curr_prob_trn, curr_prob_val, curr_prob_tst = curr_tree.get_probs_from_data(tree_train_list, validate_list, test_list)
    curr_train_error = gaps.assess_error(ground_truth_probs_trn, curr_prob_trn)
    curr_validation_error = gaps.assess_error(ground_truth_probs_val, curr_prob_val)
    curr_test_error = gaps.assess_error(ground_truth_probs_tst, curr_prob_tst)

    if curr_train_error[0] < prev_train_error[0]: #if new tree is better, set the new tree as the previous tree and save errors
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

  #Find tree with lowest validation error and return that tree
  min_val = min(tree_mutation_validation_error)
  final_tree_idx = tree_mutation_validation_error.index(min_val)
  final_tree = successful_tree_trace[final_tree_idx]  
  return final_tree, tree_mutation_training_error[:final_tree_idx+1], tree_mutation_validation_error[:final_tree_idx+1], tree_mutation_testing_error[:final_tree_idx+1], unsuccessful_mutation_count[:final_tree_idx+1]


def pool_mutations(start_tree:t.Tree, ground_truth_tree:t.Tree, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list):
  """
  Trains a given tree by performing 10 different mutations, each only one mutation away from the original tree, per epoch. If the best mutated tree is better, that will be the next tree to be mutated.
  Otherwise the original tree is mutated again. 
  Inputs:
    start_tree: tree to be trained, Tree
    ground_truth_tree: ground truth model, Tree
    leaf_train_list: GAPs to train leaves, [GAPs]
    leaf_train_responses: Ground truth responses to leaf train gaps, [int in (0,1)]
    tree_train_list: training GAPs, [GAPs]
    validate_list: validation GAPs, [GAPs]
    test_list: testing GAPs, [GAPs]
  Outputs:
    final_tree: final tree structure, Tree
    tree_mutation_training_error: trainig error trace, [[mean squared error (float), standard deviation (float)]]
    tree_mutation_validation_error: validation error trace [[mean squared error (float), standard deviation (float)]]
    tree_mutation_testing_error: testing error trace [[mean squared error (float), standard deviation (float)]]
    unsuccessful_mutation_count: trace of unsuccessful mutations per epoch [int]
    tree_probs_trn: list of trained tree probabilties for all training GAPs per epoch, [[float]]
    tree_probs_val: list of trained tree probabilties for all validation GAPs per epoch, [[float]]
    tree_probs_tst: list of trained tree probabilties for all testing GAPs per epoch, [[float]]
  """
  #initialize trace lists
  successful_tree_trace = []
  tree_probs_trn = []
  tree_probs_val = []
  tree_probs_tst = []
  tree_mutation_training_error = []
  tree_mutation_validation_error = []
  tree_mutation_testing_error = []
  unsuccessful_mutation_count = []
  
  #Specify starting tree
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)

  #Collect probs and calculate initial error
  ground_truth_probs_trn, ground_truth_probs_val, ground_truth_probs_tst = ground_truth_tree.get_probs_from_data(tree_train_list, validate_list, test_list)
  prev_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)
  prev_prob_trn, prev_prob_val, prev_prob_tst = prev_tree.get_probs_from_data(tree_train_list, validate_list, test_list)
  tree_probs_trn.append(prev_prob_trn)
  tree_probs_val.append(prev_prob_val)
  tree_probs_tst.append(prev_prob_tst)
  
  prev_train_error = gaps.assess_error(ground_truth_probs_trn, prev_prob_trn)
  tree_mutation_training_error.append(prev_train_error)

  prev_validation_error = gaps.assess_error(ground_truth_probs_val, prev_prob_val)
  tree_mutation_validation_error.append(prev_validation_error)

  prev_test_error = gaps.assess_error(ground_truth_probs_tst, prev_prob_tst)
  tree_mutation_testing_error.append(prev_test_error)

  #Intialize mutation counters
  num_successful_mutations = 0
  num_unsuccessful_mutations = 0
  
  while num_successful_mutations<MAX_SUCCESSFUL_MUTATIONS and num_unsuccessful_mutations < MAX_UNSUCCESSFUL_MUTATIONS:
    #Initialize lists to store pool of mutations
    potential_trees = []
    potential_trees_validation_error = []
    potential_trees_training_error = []
    potential_trees_test_error = []
    while len(potential_trees) < 11: #perform 10 different mutations
      #Mutate and retrain leaf probs
      curr_tree = prev_tree.mutate()
      curr_tree.reset_leaf_probs()
      curr_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)

      #Calculate probabilities and assess error for datasets
      curr_prob_trn, curr_prob_val, curr_prob_tst = curr_tree.get_probs_from_data(tree_train_list, validate_list, test_list)
      curr_train_error = gaps.assess_error(ground_truth_probs_trn, curr_prob_trn)
      curr_validation_error = gaps.assess_error(ground_truth_probs_val, curr_prob_val)
      curr_test_error = gaps.assess_error(ground_truth_probs_tst, curr_prob_tst)

      #Add mutation to the pool
      potential_trees.append(curr_tree)
      potential_trees_validation_error.append(curr_validation_error)
      potential_trees_training_error.append(curr_train_error)
      potential_trees_test_error.append(curr_test_error)

    #Find best tree
    best_tree_training_error = min(potential_trees_training_error)
    best_tree_idx = potential_trees_training_error.index(best_tree_training_error)
    if best_tree_training_error[0] < prev_train_error[0]: #if best tree in the pool is better than previous tree, set that as new tree
      tree_probs_trn.append(curr_prob_trn)
      tree_probs_val.append(curr_prob_val)
      tree_probs_tst.append(curr_prob_tst) 
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
  #Find tree with lowest validation error and return that tree
  min_val = min(tree_mutation_validation_error)
  final_tree_idx = tree_mutation_validation_error.index(min_val)
  final_tree = successful_tree_trace[final_tree_idx]  
  return final_tree, tree_mutation_training_error[:final_tree_idx+1], tree_mutation_validation_error[:final_tree_idx+1], tree_mutation_testing_error[:final_tree_idx+1], unsuccessful_mutation_count[:final_tree_idx+1], tree_probs_trn[:final_tree_idx+1], tree_probs_val[:final_tree_idx+1], tree_probs_tst[:best_tree_idx+1]

def chain_mutations(start_tree:t.Tree, ground_truth_tree:t.Tree, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list):
  """
  Trains a given tree by performing a series of 10 mutations per epoch. If the best mutated chain is better, that will be the next tree to be mutated. Otherwise
  the original tree is mutated again. 
  Inputs:
    start_tree: tree to be trained, Tree
    ground_truth_tree: ground truth model, Tree
    leaf_train_list: GAPs to train leaves, [GAPs]
    leaf_train_responses: Ground truth responses to leaf train gaps, [int in (0,1)]
    tree_train_list: training GAPs, [GAPs]
    validate_list: validation GAPs, [GAPs]
    test_list: testing GAPs, [GAPs]
  Outputs:
    final_tree: final tree structure, Tree
    tree_mutation_training_error: trainig error trace, [[mean squared error (float), standard deviation (float)]]
    tree_mutation_validation_error: validation error trace [[mean squared error (float), standard deviation (float)]]
    tree_mutation_testing_error: testing error trace [[mean squared error (float), standard deviation (float)]]
    unsuccessful_mutation_count: trace of unsuccessful mutations per epoch [int]
  """
  #Intitialize trace lists
  successful_tree_trace = []
  tree_mutation_training_error = []
  tree_mutation_validation_error = []
  tree_mutation_testing_error = []
  unsuccessful_mutation_count = []
  prev_tree = start_tree
  successful_tree_trace.append(prev_tree)
  
  #Collect probs to assess error
  ground_truth_probs_trn, ground_truth_probs_val, ground_truth_probs_tst = ground_truth_tree.get_probs_from_data(tree_train_list, validate_list, test_list)
  
  #Calculate initial errors
  prev_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)
  prev_prob_trn, prev_prob_val, prev_prob_tst = prev_tree.get_probs_from_data(tree_train_list, validate_list, test_list)

  prev_train_error = gaps.assess_error(ground_truth_probs_trn, prev_prob_trn)
  tree_mutation_training_error.append(prev_train_error)

  prev_validation_error = gaps.assess_error(ground_truth_probs_val, prev_prob_val)
  tree_mutation_validation_error.append(prev_validation_error)

  prev_test_error = gaps.assess_error(ground_truth_probs_tst, prev_prob_tst)
  tree_mutation_testing_error.append(prev_test_error)

  #Initialize mutation counters
  num_successful_mutations = 0
  num_unsuccessful_mutations = 0

  #Mutation training loop
  while num_successful_mutations<MAX_SUCCESSFUL_MUTATIONS and num_unsuccessful_mutations < MAX_UNSUCCESSFUL_MUTATIONS:
    #Initialize lsits to store chains of mutations
    potential_trees = []
    potential_trees_validation_error = []
    potential_trees_training_error = []
    potential_trees_test_error = []

    #perform first mutation
    curr_tree = prev_tree.mutate()

    #Create chain of mutations
    while len(potential_trees) < 11:
      if len(potential_trees) > 0:
        curr_tree = curr_tree.mutate()
      
      #Retrain leaf probabilties
      curr_tree.reset_leaf_probs()
      curr_tree.train_leaf_prob(leaf_train_list, leaf_train_responses)
      
      #Run GAPs and assess error
      curr_prob_trn, curr_prob_val, curr_prob_tst = curr_tree.get_probs_from_data(tree_train_list, validate_list, test_list)
      curr_train_error = gaps.assess_error(ground_truth_probs_trn, curr_prob_trn)
      curr_validation_error = gaps.assess_error(ground_truth_probs_val, curr_prob_val)
      curr_test_error = gaps.assess_error(ground_truth_probs_tst, curr_prob_tst)
      
      #Add to chain of mutations
      potential_trees.append(curr_tree)
      potential_trees_validation_error.append(curr_validation_error)
      potential_trees_training_error.append(curr_train_error)
      potential_trees_test_error.append(curr_test_error)

    #Find best tree in chain
    best_tree_idx = 0
    best_tree_training_error = [np.inf, np.inf]
    for x in range(len(potential_trees_training_error)):
      if potential_trees_training_error[x][0] < best_tree_training_error[0]:
        best_tree_training_error = potential_trees_training_error[x]
        best_tree_idx = x

    #If best tree in chain is better than previous tree, set it as the new tree
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
  #Find tree with lowest validation error and return it
  min_val = min(tree_mutation_validation_error)
  final_tree_idx = tree_mutation_validation_error.index(min_val)
  final_tree = successful_tree_trace[final_tree_idx]  
  return final_tree, tree_mutation_training_error[:final_tree_idx+1], tree_mutation_validation_error[:final_tree_idx+1], tree_mutation_testing_error[:final_tree_idx+1], unsuccessful_mutation_count[:final_tree_idx+1]

def calculate_mean_predictions(all_trees_predictions):
    # Find the maximum number of epochs across all trees
    max_epochs = max(len(tree) for tree in all_trees_predictions)
    num_data_points = len(all_trees_predictions[0][0])
    num_trees = len(all_trees_predictions)
    
    mean_predictions = []
    
    for epoch in range(max_epochs):
        epoch_predictions = []
        for data_point in range(num_data_points):
            sum_predictions = 0
            for tree in all_trees_predictions:
                # If the tree has run out of epochs, use its last epoch's prediction
                tree_epoch = min(epoch, len(tree) - 1)
                sum_predictions += tree[tree_epoch][data_point]
            mean_prediction = sum_predictions / num_trees
            epoch_predictions.append(mean_prediction)
        mean_predictions.append(epoch_predictions)
    
    return mean_predictions

def ensemble(num_trees, ground_truth, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list, tree_list = None, return_trees = False):
  """
  Using one of the training methods, trains a number of different trees on the same datasets. Then takes the mean of predictions from each tree to create an ensemble prediction. 
  Helps determine disagreement between individual trees for certain datapoints.
  Inputs:
    num_trees: determines how many trees will be in the ensemble, int
    ground_truth: ground truth model, Tree
    leaf_train_list: leaf training GAPs, [GAPs]
    leaf_train_responses: leaf trainng responses [int in (0,1)]
    tree_train_list: training GAPs, [GAPs]
    validate_list: validation GAPs, [GAPs]
    test_list: testing GAPs, [GAPs]
    tree_list: pretrained trees to continue training, [Trees] - default is None
    return_trees: boolean, default is False
  Outputs:  
    ensemble_mse_trn: Final training MSE for each gap, [[mse (float), sd (float)]]
    ensemble_mse_val: Final validation MSE for each gap, [[mse (float), sd (float)]]
    ensemble_mse_val: Final testing MSE for each gap, [[mse (float), sd (float)]]
    all_trees_mse_trace_trn: List of mse traces for each epoch for each tree, [[[mse (float), sd (float)]]]
    all_trees_mse_trace_val: List of mse traces for each epoch for each tree, [[[mse (float), sd (float)]]]
    all_trees_mse_trace_tst: List of mse traces for each epoch for each tree, [[[mse (float), sd (float)]]]
    individual_tree_probs_tst: List of probability traces for each tree for testing set
  """
  #Initialize lists to store all individual tree information
  tree_outputs = []  
  individual_tree_probs_trn = []
  individual_tree_probs_val = []
  individual_tree_probs_tst = []
  all_trees_mse_trace_trn = []
  all_trees_mse_trace_val = []
  all_trees_mse_trace_tst = []
  final_error_tst = []

  #If given trees to train, train them and add them to lists
  if tree_list:
    for tree in tree_list:
      tree.reset_leaf_probs()
      final_tree, tree_mutation_training_error, tree_mutation_validation_error, tree_mutation_testing_error, unsuccessful_mutation_count, tree_probs_trn, tree_probs_val, tree_probs_tst = pool_mutations(tree, ground_truth, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list)
      tree_outputs.append(final_tree)
      final_error_tst.append(tree_mutation_testing_error[-1])
      individual_tree_probs_trn.append(tree_probs_trn)
      individual_tree_probs_val.append(tree_probs_val)
      individual_tree_probs_tst.append(tree_probs_tst)
      all_trees_mse_trace_trn.append(tree_mutation_training_error)
      all_trees_mse_trace_val.append(tree_mutation_validation_error)
      all_trees_mse_trace_tst.append(tree_mutation_testing_error)
  else:#if not given trees, create random trees and train them
    forest = []
    for x in range(num_trees):
      new_tree = t.build_bald_tree(10, shuffle_nums=True)
      new_tree.fill_tree_with_leaves(evenly_spaced=False)
      forest.append(new_tree)
    for tree in forest:
      final_tree, tree_mutation_training_error, tree_mutation_validation_error, tree_mutation_testing_error, unsuccessful_mutation_count, tree_probs_trn, tree_probs_val, tree_probs_tst = pool_mutations(tree, ground_truth, leaf_train_list, leaf_train_responses, tree_train_list, validate_list, test_list)
      tree_outputs.append(final_tree)
      final_error_tst.append(tree_mutation_testing_error[-1])
      individual_tree_probs_trn.append(tree_probs_trn)
      individual_tree_probs_val.append(tree_probs_val)
      individual_tree_probs_tst.append(tree_probs_tst)
      all_trees_mse_trace_trn.append(tree_mutation_training_error)
      all_trees_mse_trace_val.append(tree_mutation_validation_error)
      all_trees_mse_trace_tst.append(tree_mutation_testing_error)
  
  #Calculate ensemble probabilties for training, validation, and testing datasets
  ensemble_probs_trn = calculate_mean_predictions(individual_tree_probs_trn)
  ensemble_probs_val = calculate_mean_predictions(individual_tree_probs_val)
  ensemble_probs_tst = calculate_mean_predictions(individual_tree_probs_tst)

  #Find ground truth probabilties for training, validation, and testing datesets
  ground_truth_probs_trn = ground_truth.run_tree_gaps(tree_train_list, leaf_prob=True)
  ground_truth_probs_val = ground_truth.run_tree_gaps(validate_list, leaf_prob=True)
  ground_truth_probs_tst = ground_truth.run_tree_gaps(test_list, leaf_prob=True)

  #Calculate Mean Squared Error for all datasets
  ensemble_mse_trn = []
  for x in range(len(ensemble_probs_trn)):
    epoch_mse = gaps.assess_error(ground_truth_probs_trn, ensemble_probs_trn[x])
    ensemble_mse_trn.append(epoch_mse)
  ensemble_mse_val = []
  for x in range(len(ensemble_probs_val)):
    epoch_mse = gaps.assess_error(ground_truth_probs_val, ensemble_probs_val[x])
    ensemble_mse_val.append(epoch_mse)
  ensemble_mse_tst = []
  for x in range(len(ensemble_probs_tst)):
    epoch_mse = gaps.assess_error(ground_truth_probs_tst, ensemble_probs_tst[x])
    ensemble_mse_tst.append(epoch_mse)
  return ensemble_mse_trn, ensemble_mse_val, ensemble_mse_val, all_trees_mse_trace_trn, all_trees_mse_trace_val, all_trees_mse_trace_tst, individual_tree_probs_tst