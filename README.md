# Bayesian CART for Mouse Olfaction

Implementation of the Bayesian CART model described by Chipman et al (1998) for predicting behavioral responses in a odor discrimination task. 
Given glomerulus activation patterns, this decision tree model outputs the predicted probability for the mouse's response.

## Overview:

1. Generate synthetic data:
   - generate_synthetic_gaps.py
   - generate_leaf_training_data.py
2. Train and plot single trees:
   - single_tree_loss_plots.py
3. Train and plot tree ensemble
   - ensemble_loss_plot.py
  
   
## Classes

nodes.py:
  - Defines node classes to be used in constructing a tree (RootNode, Node, LeafNode)
  - Includes functions to traverse gaps, add nodes, update node values, etc.

tree.py:
  - Uses nodes.py to construct a binary decision tree.
  - Includes functions to construct trees, mutate tree structure, run data, etc.

gaps.py:
  - Used to create synthetic GAPs
  - Includes funtions to generate GAPs, assess GAP errors, split GAP dataset, etc.

train.py:
  - Model class, holds functins to mutate trees in 3 different approaches as well as ensemble function.
  - Includes single greeedy mutation approach, greedy pool mutation approach, chain mutation approach, and ensemble. 
