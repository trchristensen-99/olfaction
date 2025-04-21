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
results_dir = "./results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    results_dir = os.path.join(results_dir, now)
    os.mkdir(results_dir)

def save_lists_to_file(lists, file_name):
    with open(file_name, 'wb') as file:
      return pickle.dump(lists, file)

def load_gaps_from_file(file_name):
    with open(file_name, 'rb') as file:
      return pickle.load(file)

if __name__ == "__main__":

    #load and create testing data
    data_gaps = load_gaps_from_file("./synthetic_gap_data.pkl")
    gaps_tree_trn, gaps_val, gaps_tst = data_gaps[1], data_gaps[2], data_gaps[3]
    leaf_train_data = load_gaps_from_file("./leaf_training_data.pkl")
    gaps_leaf_trn, responses_leaf_trn = leaf_train_data[0], leaf_train_data[1]

    #Create ground truth responses and run gaps
    ground_truth = t.build_bald_tree(10, shuffle_nums=False)
    ground_truth.fill_tree_with_leaves(evenly_spaced=True)

    #Create random test trees
    greedy_test_tree = t.construct_from_prior(10, 0.5, 0.5)
    pool_test_tree = t.construct_from_prior(10, 0.5, 0.5)
    chain_test_tree = t.construct_from_prior(10, 0.5, 0.5)

    #Train greedy model
    greedy_final_tree, greedy_tree_mutation_training_error, greedy_tree_mutation_validation_error, greedy_tree_mutation_testing_error, greedy_unsuccessful_mutation_count = train.greedy_mutations(greedy_test_tree, ground_truth, gaps_leaf_trn,responses_leaf_trn, gaps_tree_trn, gaps_val, gaps_tst)

    #Store greedy errors
    greedy_mutation_meanerr_trn = []
    greedy_mutation_meanerr_val = []
    greedy_mutation_meanerr_tst = []
    for x in range(len(greedy_tree_mutation_testing_error)):
        greedy_mutation_meanerr_trn.append(greedy_tree_mutation_training_error[x][0])
        greedy_mutation_meanerr_val.append(greedy_tree_mutation_validation_error[x][0])
        greedy_mutation_meanerr_tst.append(greedy_tree_mutation_testing_error[x][0])

    #Train pool model
    pool_final_tree, pool_tree_mutation_training_error, pool_tree_mutation_validation_error, pool_tree_mutation_testing_error, pool_unsuccessful_mutation_count, pool_tree_probs_trn, pool_tree_probs_val, pool_tree_probs_tst = train.pool_mutations(pool_test_tree, ground_truth, gaps_leaf_trn,responses_leaf_trn, gaps_tree_trn, gaps_val, gaps_tst)

    #Store pool errors
    pool_mutation_meanerr_trn = []
    pool_mutation_meanerr_val = []
    pool_mutation_meanerr_tst = []
    for x in range(len(pool_tree_mutation_testing_error)):
        pool_mutation_meanerr_trn.append(pool_tree_mutation_training_error[x][0])
        pool_mutation_meanerr_val.append(pool_tree_mutation_validation_error[x][0])
        pool_mutation_meanerr_tst.append(pool_tree_mutation_testing_error[x][0])

    #Train chain model
    chain_final_tree, chain_tree_mutation_training_error, chain_tree_mutation_validation_error, chain_tree_mutation_testing_error, chain_unsuccessful_mutation_count = train.chain_mutations(chain_test_tree, ground_truth, leaf_train_data, responses_leaf_trn, gaps_tree_trn, gaps_val, gaps_tst)

    #Store chain errors
    chain_mutation_meanerr_trn = []
    chain_mutation_meanerr_val = []
    chain_mutation_meanerr_tst = []
    for x in range(len(chain_tree_mutation_testing_error)):
        chain_mutation_meanerr_trn.append(chain_tree_mutation_training_error[x][0])
        chain_mutation_meanerr_val.append(chain_tree_mutation_validation_error[x][0])
        chain_mutation_meanerr_tst.append(chain_tree_mutation_testing_error[x][0])

    #plot greedy loss plot along with unsuccessful mutation count
    fig, axs_big = plt.subplots(3, 2, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    x_axis = np.arange(len(greedy_mutation_meanerr_trn))
    axs_big[0,0].plot(x_axis, greedy_mutation_meanerr_trn, c="blue", label="Training")
    axs_big[0,0].plot(x_axis, greedy_mutation_meanerr_val, ls="--", c="red", label="Validation")
    axs_big[0,0].plot(x_axis, greedy_mutation_meanerr_tst, ls="--", c="green", label="Testing")
    axs_big[0,0].set_xlabel("Mutations", fontsize=14)
    axs_big[0,0].set_ylabel("MSE", fontsize=14)
    axs_big[0,0].set_title("Greedy Mutation Performance")
    axs_big[0,0].legend()

    x_axis= np.arange(len(greedy_unsuccessful_mutation_count))
    axs_big[0,1].plot(greedy_unsuccessful_mutation_count)
    axs_big[0,1].set_xlabel('successful tree mutations')
    axs_big[0,1].set_yscale(value='log')
    axs_big[0,1].set_ylabel('unsuccessful tree mutations')
    axs_big[0,1].set_title('Greedy Mutations Search Efficiency')

    #plot pool loss plot along with unsuccessful mutation count
    x_axis = np.arange(len(pool_tree_mutation_training_error))
    axs_big[1,0].plot(x_axis, pool_mutation_meanerr_trn, c="blue", label="Training")
    axs_big[1,0].plot(x_axis, pool_mutation_meanerr_val, ls="--", c="red", label="Validation")
    axs_big[1,0].plot(x_axis, pool_mutation_meanerr_tst, ls="--", c="green", label="Testing")
    axs_big[1,0].set_xlabel("Mutations", fontsize=14)
    axs_big[1,0].set_ylabel("MSE", fontsize=14)
    axs_big[1,0].set_title("Pool Mutation Performance")
    axs_big[1,0].legend()

    x_axis= np.arange(len(pool_unsuccessful_mutation_count))
    axs_big[1,1].plot(pool_unsuccessful_mutation_count)
    axs_big[1,1].set_xlabel('successful tree mutations')
    axs_big[1,1].set_yscale(value='log')
    axs_big[1,1].set_ylabel('unsuccessful tree mutations')
    axs_big[1,1].set_title('Pool Mutations Search Efficiency')

    #plot greedy loss plot along with unsuccessful mutation count
    x_axis = np.arange(len(chain_tree_mutation_training_error))
    axs_big[2,0].plot(x_axis, chain_mutation_meanerr_trn, c="blue", label="Training")
    axs_big[2,0].plot(x_axis, chain_mutation_meanerr_val, ls="--", c="red", label="Validation")
    axs_big[2,0].plot(x_axis, chain_mutation_meanerr_tst, ls="--", c="green", label="Testing")
    axs_big[2,0].set_xlabel("Mutations", fontsize=14)
    axs_big[2,0].set_ylabel("MSE", fontsize=14)
    axs_big[2,0].set_title("Chain Mutation Performance")
    axs_big[2,0].legend()


    x_axis= np.arange(len(chain_unsuccessful_mutation_count))
    axs_big[2,1].plot(chain_unsuccessful_mutation_count)
    axs_big[2,1].set_xlabel('successful tree mutations')
    axs_big[2,1].set_yscale(value='log')
    axs_big[2,1].set_ylabel('unsuccessful tree mutations')
    axs_big[2,1].set_title('Chain Mutations Search Efficiency')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "approaches_performance.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    plt.clf()