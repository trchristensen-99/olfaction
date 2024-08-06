import tree as t
import gaps
import train
import random
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy import stats
import pickle

random_seed = 2
np.random.seed(random_seed)
random.seed(random_seed)
now = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir_path = os.path.join("./results", now)
os.mkdir(results_dir_path)

def save_lists_to_file(lists, file_name):
    with open(file_name, 'wb') as file:
      return pickle.dump(lists, file)

def load_gaps_from_file(file_name):
    with open(file_name, 'rb') as file:
      return pickle.load(file)

 
if __name__ == "__main__":

    data_gaps = load_gaps_from_file("synthetic_gap_data.pkl")
    gaps_tree_trn, gaps_val, gaps_tst = data_gaps[1], data_gaps[2], data_gaps[3]
    leaf_train_data = load_gaps_from_file("leaf_training_data.pkl")
    gaps_leaf_trn, responses_leaf_trn = leaf_train_data[0], leaf_train_data[1]
    #after making a list of all possible gap patterns, randomly selects 500 test
    #gaps and prints them out
    # full_gap_list = gaps.binary_num_list(10)
    # np.random.shuffle(full_gap_list)

    #split_gaps[0]=training data set, split_gaps[1] = validate data set, split_gaps[2] = test data set
    # leaf_train_data, tree_train_data, val_data, test_data = gaps.split_data_set(full_gap_list, leaf_train_size=500, tree_train_size=100, val_size=100, test_size=100) 

    #Create ground truth responses and run gaps
    ground_truth = t.build_bald_tree(10, shuffle_nums=False)
    ground_truth.fill_tree_with_leaves(evenly_spaced=True)
    # leaf_training_responses = ground_truth.run_tree_gaps(leaf_train_data, leaf_prob=False)

    #Create random test trees
    # greedy_test_tree = t.construct_from_prior(10, 0.5, 0.5)
    greedy_test_tree = t.build_bald_tree(10, shuffle_nums=True)
    greedy_test_tree.fill_tree_with_leaves(evenly_spaced=False)
    # pool_test_tree = t.construct_from_prior(10, 0.5, 0.5)
    pool_test_tree = t.build_bald_tree(10, shuffle_nums=True)
    pool_test_tree.fill_tree_with_leaves(evenly_spaced=False)
    # chain_test_tree = t.construct_from_prior(10, 0.5, 0.5)
    chain_test_tree = t.build_bald_tree(10, shuffle_nums=True)
    chain_test_tree.fill_tree_with_leaves(evenly_spaced=False)

    # pool_train_size_test_errors = [[],[],[],[],[],[],[],[],[],[]]
    # for x in range(len(training_data_sizes)):
    #     for j in range(20):
    #         pool_test_tree = t.build_bald_tree(15, shuffle_nums=True)
    #         pool_test_tree.fill_tree_with_leaves(evenly_spaced=False)
    #         pool_final_tree, pool_tree_mutation_training_error, pool_tree_mutation_validation_error, pool_tree_mutation_testing_error, pool_unsuccessful_mutation_count = train.pool_mutations(pool_test_tree, ground_truth, training_responses, training_data_sizes[x], val_data, test_data)
    #         pool_train_size_test_errors[x].append(pool_tree_mutation_testing_error[-1][0])
    # avg_test_errors_vary_size = []
    # for x in range(len(pool_train_size_test_errors)):
    #     avg_test_errors_vary_size.append(np.mean(pool_train_size_test_errors[x]))

    # plt.figure(figsize=(10, 6))
    # train_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # plt.plot(train_sizes, avg_test_errors_vary_size, ls="--", c="blue")
    # plt.xlabel("Training Data Size")
    # plt.ylabel("Final Average Error")
    # plt.title("Pool Mutation Performance")
    # plt.legend()



    # greedy_final_tree, greedy_tree_mutation_training_error, greedy_tree_mutation_validation_error, greedy_tree_mutation_testing_error, greedy_unsuccessful_mutation_count = train.greedy_mutations(greedy_test_tree, ground_truth, gaps_leaf_trn,responses_leaf_trn, gaps_tree_trn, gaps_val, gaps_tst)

    # greedy_mutation_meanerr_trn = []
    # greedy_mutation_meanerr_val = []
    # greedy_mutation_meanerr_tst = []
    # for x in range(len(greedy_tree_mutation_testing_error)):
    #     greedy_mutation_meanerr_trn.append(greedy_tree_mutation_training_error[x][0])
    #     greedy_mutation_meanerr_val.append(greedy_tree_mutation_validation_error[x][0])
    #     greedy_mutation_meanerr_tst.append(greedy_tree_mutation_testing_error[x][0])

    pool_final_tree, pool_tree_mutation_training_error, pool_tree_mutation_validation_error, pool_tree_mutation_testing_error, pool_unsuccessful_mutation_count = train.pool_mutations(pool_test_tree, ground_truth, gaps_leaf_trn,responses_leaf_trn, gaps_tree_trn, gaps_val, gaps_tst)

    pool_mutation_meanerr_trn = []
    pool_mutation_meanerr_val = []
    pool_mutation_meanerr_tst = []
    for x in range(len(pool_tree_mutation_testing_error)):
        pool_mutation_meanerr_trn.append(pool_tree_mutation_training_error[x][0])
        pool_mutation_meanerr_val.append(pool_tree_mutation_validation_error[x][0])
        pool_mutation_meanerr_tst.append(pool_tree_mutation_testing_error[x][0])

    # chain_final_tree, chain_tree_mutation_training_error, chain_tree_mutation_validation_error, chain_tree_mutation_testing_error, chain_unsuccessful_mutation_count = train.chain_mutations(chain_test_tree, ground_truth, leaf_train_data,leaf_training_responses, tree_train_data, val_data, test_data)

    # chain_mutation_meanerr_trn = []
    # chain_mutation_meanerr_val = []
    # chain_mutation_meanerr_tst = []
    # for x in range(len(chain_tree_mutation_testing_error)):
    #     chain_mutation_meanerr_trn.append(chain_tree_mutation_training_error[x][0])
    #     chain_mutation_meanerr_val.append(chain_tree_mutation_validation_error[x][0])
    #     chain_mutation_meanerr_tst.append(chain_tree_mutation_testing_error[x][0])

    # print(f"Greedy Final MSE: Train {greedy_mutation_meanerr_trn[-1]}, Val {greedy_mutation_meanerr_val[-1]}, Test {greedy_mutation_meanerr_tst[-1]}")
    # print(f"Pool Final MSE: Train {pool_mutation_meanerr_trn[-1]}, Val {pool_mutation_meanerr_val[-1]}, Test {pool_mutation_meanerr_tst[-1]}")
    # print(f"Chain Final MSE: Train {chain_mutation_meanerr_trn[-1]}, Val {chain_mutation_meanerr_val[-1]}, Test {chain_mutation_meanerr_tst[-1]}")

    # plot training, validation, and test error
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # x_axis = np.arange(len(greedy_tree_mutation_training_error))
    # ax1.plot(x_axis, greedy_mutation_meanerr_trn, c="blue", label="Training")
    # ax1.plot(x_axis, greedy_mutation_meanerr_val, ls="--", c="red", label="Validation")
    # ax1.plot(x_axis, greedy_mutation_meanerr_tst, ls="--", c="green", label="Testing")
    # ax1.set_xlabel("Mutations", fontsize=14)
    # ax1.set_ylabel("MSE", fontsize=14)
    # # ax1.set_title("Greedy Mutation Performance")
    # ax1.legend()

    # x_axis= np.arange(len(greedy_unsuccessful_mutation_count))
    # axs_big[0,1].plot(greedy_unsuccessful_mutation_count)
    # axs_big[0,1].set_xlabel('successful tree mutations')
    # axs_big[0,1].set_yscale(value='log')
    # axs_big[0,1].set_ylabel('unsuccessful tree mutations')
    # axs_big[0,1].set_title('Greedy Mutations Search Efficiency')

    # x_axis = np.arange(len(pool_tree_mutation_training_error))
    # ax2.plot(x_axis, pool_mutation_meanerr_trn, c="blue", label="Training")
    # ax2.plot(x_axis, pool_mutation_meanerr_val, ls="--", c="red", label="Validation")
    # ax2.plot(x_axis, pool_mutation_meanerr_tst, ls="--", c="green", label="Testing")
    # ax2.set_xlabel("Mutations", fontsize=14)
    # ax2.set_ylabel("MSE", fontsize=14)
    # # ax2.set_title("Pool Mutation Performance")
    # ax2.legend()
    plt.figure(figsize=(10, 6))
    base_x_axis = np.arange(len(pool_tree_mutation_training_error))
    x_axis = base_x_axis * 500
    plt.plot(base_x_axis, pool_mutation_meanerr_trn, c="blue", label="Training")
    plt.plot(base_x_axis, pool_mutation_meanerr_val, ls="--", c="red", label="Validation")
    plt.plot(base_x_axis, pool_mutation_meanerr_tst, ls="--", c="green", label="Testing")
    plt.xlabel("Mutations", fontsize=14)
    plt.ylabel("Mean squared error (MSE)", fontsize=14)
    plt.yscale(value='log')
    # plt.title("Tree Training", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_path, "pool_performance.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    plt.clf()

    # x_axis= np.arange(len(pool_unsuccessful_mutation_count))
    # axs_big[1,1].plot(pool_unsuccessful_mutation_count)
    # axs_big[1,1].set_xlabel('successful tree mutations')
    # axs_big[1,1].set_yscale(value='log')
    # axs_big[1,1].set_ylabel('unsuccessful tree mutations')
    # axs_big[1,1].set_title('Pool Mutations Search Efficiency')


    # x_axis = np.arange(len(chain_tree_mutation_training_error))
    # ax3.plot(x_axis, chain_mutation_meanerr_trn, c="blue", label="Training")
    # ax3.plot(x_axis, chain_mutation_meanerr_val, ls="--", c="red", label="Validation")
    # ax3.plot(x_axis, chain_mutation_meanerr_tst, ls="--", c="green", label="Testing")
    # ax3.set_xlabel("Mutations", fontsize=14)
    # ax3.set_ylabel("MSE", fontsize=14)
    # # ax3.set_title("Chain Mutation Performance")
    # ax3.legend()

    # for ax in (ax1, ax2, ax3):
    #     ax.set_yscale('log')

    # Find the global min and max y values
    # y_min = min(np.min(greedy_mutation_meanerr_trn), np.min(greedy_mutation_meanerr_val), np.min(greedy_mutation_meanerr_tst), np.min(pool_mutation_meanerr_trn), np.min(pool_mutation_meanerr_val), np.min(pool_mutation_meanerr_tst), np.min(chain_mutation_meanerr_trn), np.min(chain_mutation_meanerr_val), np.min(chain_mutation_meanerr_tst))
    # y_max = max(np.max(greedy_mutation_meanerr_trn), np.max(greedy_mutation_meanerr_val), np.max(greedy_mutation_meanerr_tst), np.max(pool_mutation_meanerr_trn), np.max(pool_mutation_meanerr_val), np.max(pool_mutation_meanerr_tst), np.max(chain_mutation_meanerr_trn), np.max(chain_mutation_meanerr_val), np.max(chain_mutation_meanerr_tst))

    # y_min_extended = y_min * 0.8  # Adjust this factor to extend lower
    # y_max_extended = y_max * 1.1  # Optionally extend upper limit

    # # Set the same y-axis limits for all subplots
    # for ax in (ax1, ax2, ax3):
    #     ax.set_ylim(y_min_extended, y_max_extended)

    # # x_axis= np.arange(len(chain_unsuccessful_mutation_count))
    # # axs_big[2,1].plot(chain_unsuccessful_mutation_count)
    # # axs_big[2,1].set_xlabel('successful tree mutations')
    # # axs_big[2,1].set_yscale(value='log')
    # # axs_big[2,1].set_ylabel('unsuccessful tree mutations')
    # # axs_big[2,1].set_title('Chain Mutations Search Efficiency')

    # ax1.annotate('(a)', xy=(-0.10, 1.10), xycoords='axes fraction', fontsize=12, ha='center', va='center')
    # ax2.annotate('(b)', xy=(-0.10, 1.10), xycoords='axes fraction', fontsize=12, ha='center', va='center')
    # ax3.annotate('(c)', xy=(-0.10, 1.10), xycoords='axes fraction', fontsize=12, ha='center', va='center')

    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir_path, "approaches_performance.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    # plt.clf()

    # final_ensemble_error, tree_outputs, single_gap_se, single_gap_variance = train.ensemble(10, ground_truth, leaf_train_data, leaf_training_responses, tree_train_data, val_data, test_data)




    # error_mean = []
    # for x in range(len(final_ensemble_error)):
    #     error_mean.append(final_ensemble_error[x][0])

    # error_std = []
    # for x in range(len(final_ensemble_error)):
    #     error_std.append(final_ensemble_error[x][1])

    # x_labels = []
    # for x in range(10):
    #     x_labels.append(f"Tree {x + 1}")
    # x_labels.append("Ensemble")

    # plt.figure(figsize=(10,6))
    # x_axis = x_labels
    # plt.scatter(x_axis[:-1], error_mean[:-1], c='blue')
    # plt.scatter(x_axis[-1], error_mean[-1], c='red')
    # plt.errorbar(x_axis[:-1], error_mean[:-1], yerr=error_std[:-1], fmt='o', c='b', capsize=5)
    # plt.errorbar(x_axis[-1], error_mean[-1], yerr=error_std[-1],fmt='o', c='r', capsize=5)
    # plt.xlabel("Trees")
    # plt.ylabel("MSE")
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir_path, "ensemble_error.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    # plt.clf()

    # zipped_lists = zip(single_gap_variance, single_gap_se)

    # # Sort the zipped list by variance (the first element of each pair)
    # sorted_pairs = sorted(zipped_lists)

    # # Unzip the sorted pairs back into two lists
    # sorted_variance, sorted_se = zip(*sorted_pairs)

    # # Convert them back to lists
    # sorted_variance = list(sorted_variance)
    # sorted_se = list(sorted_se)

    # # variance = []
    # # for x in range(len(error_std)):
    # #     variance.append(np.square(error_std[x]))

    # # model_names = []
    # # for x in range(len(final_ensemble_error) - 1):
    # #     model_names.append(f"Tree {x}")
    # # model_names.append("Ensemble")



    # plt.figure(figsize=(10, 6))
    # plt.scatter(sorted_variance, sorted_se, s=100, color='blue')

    # # # Scatter plot for the last point (Ensemble) with a different color
    # # plt.scatter(variance[-1], error_mean[-1], s=100, color='red', label='Ensemble')
    # # plt.plot(variance, p(variance), "r--")
    # # plt.errorbar(x=model_names, y=error_mean, yerr=error_std, fmt='none', ecolor='gray', elinewidth=2, capsize=5, capthick=2)
    # plt.ylabel('SE')
    # plt.xlabel('Variance')
    # slope, intercept, r_value, p_value, std_err = stats.linregress(sorted_variance, sorted_se)
    # line = slope * np.array(sorted_variance) + intercept
    # plt.plot(sorted_variance, line, color='r', linestyle='--', alpha=0.8)

    # # Improve readability
    # # plt.grid(True, linestyle='--', alpha=0.7)

    # # Calculate and add R-squared
    # r_squared = r_value**2

    # # Get the current axes
    # ax = plt.gca()

    # # Add text for R-squared
    # ax.text(0.05, 0.95, f'r = {r_value:.4f}', 
    #         transform=ax.transAxes, 
    #         fontsize=10, 
    #         verticalalignment='top')
    # # plt.title('MSE of Ensemble Trees')
    # # plt.ylim(0, max(error_mean) * 1.1)  # Set y-axis limit to slightly above the maximum error

    # # # Add value labels above each point
    # # for i, v in enumerate(error_mean):
    # #     plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')

    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir_path, "variance.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    # plt.clf()



