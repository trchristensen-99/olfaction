import numpy as np
import pickle
import random
import os

# Save function from your code
def save_lists_to_file(lists, file_name):
    with open(file_name, 'wb') as file:
        return pickle.dump(lists, file)

# Create mock ensemble data with the expected structure
def generate_mock_ensemble_data():
    # Number of trees and mutation steps
    num_trees = 10
    mutation_steps = 100
    
    # Create ensemble MSE data for training, validation and testing
    ensemble_mse_trn = np.random.uniform(0.1, 0.5, mutation_steps) * np.exp(-np.arange(mutation_steps)/50)
    ensemble_mse_val = np.random.uniform(0.2, 0.6, mutation_steps) * np.exp(-np.arange(mutation_steps)/50)
    ensemble_mse_tst = np.random.uniform(0.3, 0.7, mutation_steps) * np.exp(-np.arange(mutation_steps)/50)
    
    # Reshape to match expected format
    ensemble_mse_trn = [[x] for x in ensemble_mse_trn]
    ensemble_mse_val = [[x] for x in ensemble_mse_val]
    ensemble_mse_tst = [[x] for x in ensemble_mse_tst]
    
    # Create individual tree MSE data
    all_trees_mse_trn = []
    all_trees_mse_val = []
    all_trees_mse_tst = []
    
    for i in range(num_trees):
        # Individual trees have slightly different performance
        tree_trn = np.random.uniform(0.2, 0.7, mutation_steps) * np.exp(-np.arange(mutation_steps)/40) 
        tree_val = np.random.uniform(0.3, 0.8, mutation_steps) * np.exp(-np.arange(mutation_steps)/40)
        tree_tst = np.random.uniform(0.4, 0.9, mutation_steps) * np.exp(-np.arange(mutation_steps)/40)
        
        all_trees_mse_trn.append([[x] for x in tree_trn])
        all_trees_mse_val.append([[x] for x in tree_val])
        all_trees_mse_tst.append([[x] for x in tree_tst])
    
    # Calculate variance and standard error across trees
    single_gap_variance = []
    single_gap_se = []
    
    for i in range(mutation_steps):
        # Get MSE values from all trees at this timestep
        mse_values = [tree[i][0] for tree in all_trees_mse_tst]
        
        variance = np.var(mse_values)
        se = np.std(mse_values) / np.sqrt(len(mse_values))
        
        single_gap_variance.append(variance)
        single_gap_se.append(se)
    
    # Create the final data structure
    ensemble_data = [
        ensemble_mse_trn,        # [0] ensemble training MSE
        ensemble_mse_val,        # [1] ensemble validation MSE
        ensemble_mse_tst,        # [2] ensemble testing MSE
        all_trees_mse_trn,       # [3] individual trees training MSE
        all_trees_mse_val,       # [4] individual trees validation MSE
        all_trees_mse_tst,       # [5] individual trees testing MSE
    ]
    
    return ensemble_data, single_gap_variance, single_gap_se

# Generate and save the data
ensemble_data, single_gap_variance, single_gap_se = generate_mock_ensemble_data()
save_lists_to_file(ensemble_data, "filename.pkl")

print("Generated filename.pkl with mock ensemble data")