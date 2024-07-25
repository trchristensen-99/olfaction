import tree as t
import gaps
import train
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

random_seed = 2
np.random.seed(random_seed)
random.seed(random_seed)
now = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir_path = os.path.join("./results", now)
os.mkdir(results_dir_path)


#after making a list of all possible gap patterns, randomly selects 500 test
#gaps and prints them out
full_gap_list = gaps.binary_num_list(10)
np.random.shuffle(full_gap_list)

#split_gaps[0]=training data set, split_gaps[1] = validate data set, split_gaps[2] = test data set
train_data, val_data, test_data = gaps.split_data_set(full_gap_list, train_size=500, val_size=100, test_size=100) 

#Create ground truth responses and run gaps
ground_truth = t.build_bald_tree(10, shuffle_nums=False)
ground_truth.fill_tree_with_leaves(evenly_spaced=True)
training_responses = ground_truth.run_tree_gaps(train_data, leaf_prob=False)

#Create random test tree
greedy_test_tree = t.build_bald_tree(10, shuffle_nums=True)
greedy_test_tree.fill_tree_with_leaves(evenly_spaced=False)

pool_test_tree = t.build_bald_tree(10, shuffle_nums=True)
pool_test_tree.fill_tree_with_leaves(evenly_spaced=False)

chain_test_tree = t.build_bald_tree(10, shuffle_nums=True)
chain_test_tree.fill_tree_with_leaves(evenly_spaced=False)


# greedy_final_tree, greedy_tree_mutation_training_error, greedy_tree_mutation_validation_error, greedy_tree_mutation_testing_error, greedy_unsuccessful_mutation_count = train.greedy_mutations(greedy_test_tree, ground_truth, training_responses, train_data, val_data, test_data)
# pool_final_tree, pool_tree_mutation_training_error, pool_tree_mutation_validation_error, pool_tree_mutation_testing_error, pool_unsuccessful_mutation_count = train.pool_mutations(pool_test_tree, ground_truth, training_responses, train_data, val_data, test_data)
chain_final_tree, chain_tree_mutation_training_error, chain_tree_mutation_validation_error, chain_tree_mutation_testing_error, chain_unsuccessful_mutation_count = train.chain_mutations(chain_test_tree, ground_truth, training_responses, train_data, val_data, test_data)
#rain.ensemble(5, ground_truth, training_responses, train_data, val_data, test_data)




#plot training, validation, and test error
fig_big, axs_big = plt.subplots(3, 1, figsize=(10,15))

fig_small, axs_small = plt.subplots(1, 1, figsize=(10,15))
# x_axis = np.arange(len(greedy_tree_mutation_training_error))
# ax1.plot(x_axis, greedy_tree_mutation_training_error, c="blue", label="avg. training error")
# ax1.plot(x_axis, greedy_tree_mutation_validation_error, ls="--", c="red", label="avg. validation error")
# ax1.plot(x_axis, greedy_tree_mutation_testing_error, ls="--", c="green", label="avg. testing error")
# ax1.set_xlabel("tree mutations")
# ax1.set_ylabel("average error")
# ax1.set_title("Greedy Mutation Performance")
# ax1.legend()

# x_axis = np.arange(len(pool_tree_mutation_training_error))
# ax2.plot(x_axis, pool_tree_mutation_training_error, c="blue", label="avg. training error")
# ax2.plot(x_axis, pool_tree_mutation_validation_error, ls="--", c="red", label="avg. validation error")
# ax2.plot(x_axis, pool_tree_mutation_testing_error, ls="--", c="green", label="avg. testing error")
# ax2.set_xlabel("tree mutations")
# ax2.set_ylabel("average error")
# ax2.set_title("Pool Mutation Performance")
# ax2.legend()

x_axis = np.arange(len(chain_tree_mutation_training_error))
ax3.plot(x_axis, chain_tree_mutation_training_error, c="blue", label="avg. training error")
ax3.plot(x_axis, chain_tree_mutation_validation_error, ls="--", c="red", label="avg. validation error")
ax3.plot(x_axis, chain_tree_mutation_testing_error, ls="--", c="green", label="avg. testing error")
ax3.set_xlabel("tree mutations")
ax3.set_ylabel("average error")
ax3.set_title("Chain Mutation Performance")
ax3.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir_path, "tree_mutation_error"))
plt.clf()

# x_axis = np.arange(len(UNSUCCESSFUL_MUTATION_COUNT))
# plt.plot(x_axis, UNSUCCESSFUL_MUTATION_COUNT, c="purple", ls='--')
# plt.xlabel('Successful Mutations')
# plt.yscale(value='log')
# plt.ylabel('Number of Mutations Tried')
# plt.tight_layout()
# plt.savefig(os.path.join(results_dir_path, "mutation_count_plot"))
# plt.clf()