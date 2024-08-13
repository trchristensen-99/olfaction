import numpy as np
import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
import pickle

def save_lists_to_file(lists, file_name):
    """
    Inputs:
        lists = []
        file_name = string - name of file
    Outputs:
        File with given name including lists
    """
    with open(file_name, 'wb') as file:
      return pickle.dump(lists, file)

def load_lists_from_file(file_name):
    """
    Inputs:
        file_name: string
    Output: 
        lists from given file
    """
    with open(file_name, 'rb') as file:
      return pickle.load(file)

if __name__ == '__main__':

    #load ensemble data
    ensemble_data = load_lists_from_file("filename".pkl")
    
    #save mse from ensemble data for training, validation, and testing data sets
    ensemble_mse_trn = []
    for x in range(len(ensemble_data[0])):
        ensemble_mse_trn.append(ensemble_data[0][x][0])
    
    ensemble_mse_val = []
    for x in range(len(ensemble_data[0])):
        ensemble_mse_val.append(ensemble_data[1][x][0])
    
    ensemble_mse_tst = []
    for x in range(len(ensemble_data[2])):
        ensemble_mse_tst.append(ensemble_data[2][x][0])
   
   #Download mse traces for all individual tree for training, validation, and testing sets
    all_trees_mse_trn = []
    for x in range(len(ensemble_data[3])):
        tree_mse_trace = []
        for mse in ensemble_data[3][x]:
            tree_mse_trace.append(mse[0])
        all_trees_mse_trn.append(tree_mse_trace)
    
    all_trees_mse_val = []
    for x in range(len(ensemble_data[4])):
        tree_mse_trace = []
        for mse in ensemble_data[4][x]:
            tree_mse_trace.append(mse[0])
        all_trees_mse_val.append(tree_mse_trace)

    all_trees_mse_tst = []
    for x in range(len(ensemble_data[5])):
        tree_mse_trace = []
        for mse in ensemble_data[5][x]:
            tree_mse_trace.append(mse[0])
        all_trees_mse_tst.append(tree_mse_trace)

    #Create loss plot for training 
    plt.figure(figsize=(10,6))

    x_axis = np.arange(len(ensemble_mse_trn))
    plt.plot(x_axis, ensemble_mse_trn, c='red', label="Ensemble")

    for i, tree_mse in enumerate(all_trees_mse_trn):
        tree_epochs = len(tree_mse)
        plt.plot(np.arange(tree_epochs), tree_mse, c='blue', alpha=0.5, label=f'Individual Trees' if i == 0 else "")

    plt.xlabel('Mutations')
    plt.xscale(value='log')
    plt.ylabel('MSE')
    plt.yscale(value='log')
    plt.title('Ensemble Training Performance')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Create loss plot for validation 
    plt.figure(figsize=(10,6))

    x_axis = np.arange(len(ensemble_mse_val))
    plt.plot(x_axis, ensemble_mse_val, c='red', label="Ensemble")

    for i, tree_mse in enumerate(all_trees_mse_val):
        tree_epochs = len(tree_mse)
        plt.plot(np.arange(tree_epochs), tree_mse, c='blue', alpha=0.5, label=f'Individual Trees' if i == 0 else "")

    plt.xlabel('Mutations')
    plt.xscale(value='log')
    plt.ylabel('MSE')
    plt.yscale(value='log')
    plt.title('Ensemble Validation Performance')
    plt.legend()
    plt.tight_layout()
    plt.show()


    #Create loss plot for testing
    plt.figure(figsize=(10,6))

    x_axis = np.arange(len(ensemble_mse_tst))
    plt.plot(x_axis, ensemble_mse_tst, c='red', label="Ensemble")

    for i, tree_mse in enumerate(all_trees_mse_tst):
        tree_epochs = len(tree_mse)
        plt.plot(np.arange(tree_epochs), tree_mse, c='blue', alpha=0.5, label=f'Individual Trees' if i == 0 else "")

    plt.xlabel('Mutations')
    plt.xscale(value='log')
    plt.ylabel('MSE')
    plt.yscale(value='log')
    plt.title('Ensemble Testing Performance')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Sort variance and standard error to plot
    zipped_lists = zip(single_gap_variance, single_gap_se)
    sorted_pairs = sorted(zipped_lists)
    sorted_variance, sorted_se = zip(*sorted_pairs)
    sorted_variance = list(sorted_variance)
    sorted_se = list(sorted_se)

    #Scatter plot for Squared Error vs Variance
    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_variance, sorted_se, s=100, color='blue')
    plt.ylabel('SE')
    plt.xlabel('Variance')
    slope, intercept, r_value, p_value, std_err = stats.linregress(sorted_variance, sorted_se)
    line = slope * np.array(sorted_variance) + intercept
    plt.plot(sorted_variance, line, color='r', linestyle='--', alpha=0.8)

    # Calculate and add R-squared
    r_squared = r_value**2
    ax = plt.gca()
    ax.text(0.05, 0.95, f'r = {r_value:.4f}', 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_path, "variance.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    plt.clf()