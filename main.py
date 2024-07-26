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


greedy_final_tree, greedy_tree_mutation_training_error, greedy_tree_mutation_validation_error, greedy_tree_mutation_testing_error, greedy_unsuccessful_mutation_count = train.greedy_mutations(greedy_test_tree, ground_truth, training_responses, train_data, val_data, test_data)
pool_final_tree, pool_tree_mutation_training_error, pool_tree_mutation_validation_error, pool_tree_mutation_testing_error, pool_unsuccessful_mutation_count = train.pool_mutations(pool_test_tree, ground_truth, training_responses, train_data, val_data, test_data)
chain_final_tree, chain_tree_mutation_training_error, chain_tree_mutation_validation_error, chain_tree_mutation_testing_error, chain_unsuccessful_mutation_count = train.chain_mutations(chain_test_tree, ground_truth, training_responses, train_data, val_data, test_data)
ensemble_error = train.ensemble(5, ground_truth, training_responses, train_data, val_data, test_data)

final_errors = [greedy_tree_mutation_testing_error[-1], pool_tree_mutation_testing_error[-1], chain_tree_mutation_testing_error[-1], ensemble_error]

#plot training, validation, and test error
fig_big, axs_big = plt.subplots(3, 2, figsize=(30,20))

x_axis = np.arange(len(greedy_tree_mutation_training_error))
axs_big[0,0].plot(x_axis, greedy_tree_mutation_training_error, c="blue", label="avg. training error")
axs_big[0,0].plot(x_axis, greedy_tree_mutation_validation_error, ls="--", c="red", label="avg. validation error")
axs_big[0,0].plot(x_axis, greedy_tree_mutation_testing_error, ls="--", c="green", label="avg. testing error")
axs_big[0,0].set_xlabel("successful tree mutations")
axs_big[0,0].set_ylabel("average error")
axs_big[0,0].set_title("Greedy Mutation Performance")
axs_big[0,0].legend()

x_axis= np.arange(len(greedy_unsuccessful_mutation_count))
axs_big[0,1].plot(greedy_unsuccessful_mutation_count)
axs_big[0,1].set_xlabel('successful tree mutations')
axs_big[0,1].set_yscale(value='log')
axs_big[0,1].set_ylabel('unsuccessful tree mutations')
axs_big[0,1].set_title('Greedy Mutations Search Efficiency')


x_axis = np.arange(len(pool_tree_mutation_training_error))
axs_big[1,0].plot(x_axis, pool_tree_mutation_training_error, c="blue", label="avg. training error")
axs_big[1,0].plot(x_axis, pool_tree_mutation_validation_error, ls="--", c="red", label="avg. validation error")
axs_big[1,0].plot(x_axis, pool_tree_mutation_testing_error, ls="--", c="green", label="avg. testing error")
axs_big[1,0].set_xlabel("successful tree mutations")
axs_big[1,0].set_ylabel("average error")
axs_big[1,0].set_title("Pool Mutation Performance")
axs_big[1,0].legend()

x_axis= np.arange(len(pool_unsuccessful_mutation_count))
axs_big[1,1].plot(pool_unsuccessful_mutation_count)
axs_big[1,1].set_xlabel('successful tree mutations')
axs_big[1,1].set_yscale(value='log')
axs_big[1,1].set_ylabel('unsuccessful tree mutations')
axs_big[1,1].set_title('Pool Mutations Search Efficiency')


x_axis = np.arange(len(chain_tree_mutation_training_error))
axs_big[2,0].plot(x_axis, chain_tree_mutation_training_error, c="blue", label="avg. training error")
axs_big[2,0].plot(x_axis, chain_tree_mutation_validation_error, ls="--", c="red", label="avg. validation error")
axs_big[2,0].plot(x_axis, chain_tree_mutation_testing_error, ls="--", c="green", label="avg. testing error")
axs_big[2,0].set_xlabel("successful tree mutations")
axs_big[2,0].set_ylabel("average error")
axs_big[2,0].set_title("Chain Mutation Performance")
axs_big[2,0].legend()

x_axis= np.arange(len(chain_unsuccessful_mutation_count))
axs_big[2,1].plot(chain_unsuccessful_mutation_count)
axs_big[2,1].set_xlabel('successful tree mutations')
axs_big[2,1].set_yscale(value='log')
axs_big[2,1].set_ylabel('unsuccessful tree mutations')
axs_big[2,1].set_title('Chain Mutations Search Efficiency')


plt.tight_layout()
plt.savefig(os.path.join(results_dir_path, "tree_mutation_error"))
plt.clf()

model_names = ["Greedy Mutations", "Pool Mutations", "Chain Mutations", "Ensemble"]

plt.figure(figsize=(10, 6))
plt.scatter(model_names, final_errors, s=100)  # s=100 sets the size of the points
plt.ylabel('Final Error')
plt.title('Final Errors of Different Models')
plt.ylim(0, max(final_errors) * 1.1)  # Set y-axis limit to slightly above the maximum error

# Add value labels above each point
for i, v in enumerate(final_errors):
    plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(results_dir_path, "models_final_testing_error"))
plt.clf()


# x_axis = np.arange(len(UNSUCCESSFUL_MUTATION_COUNT))
# plt.plot(x_axis, UNSUCCESSFUL_MUTATION_COUNT, c="purple", ls='--')
# plt.xlabel('Successful Mutations')
# plt.yscale(value='log')
# plt.ylabel('Number of Mutations Tried')
# plt.tight_layout()
# plt.savefig(os.path.join(results_dir_path, "mutation_count_plot"))
# plt.clf()