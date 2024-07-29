import gaps
import random
import nodes as n
import copy
import numpy as np
import matplotlib.pyplot as plt

class Tree:
 
  COUNT = [10]

  def __init__(self, glom_size, tree_id=0):
    #Initializes a tree with a root node
    self.GLOM_LIST = []
    self.tree_id = tree_id
    self.root = n.RootNode()
    self.GLOM_LIST.append(self.root)
    self.glom_size = glom_size
    self.num_terminal_nodes = 0

  def add_to_tree(self, item):
    """
    Adds given node to a given tree in order
    Inputs: tree (Tree), item (Node)
    Returns: true if tree is full, false otherwise
    """
    self.GLOM_LIST.append(item)
    if self.root.child is None:
      self.root.child = item
      item.parent = self.root
      return isinstance(item, n.LeafNode)
    queue = [self.root.child]
    while queue:
      current = queue.pop(0)
      if isinstance(current, n.LeafNode):
          continue
      if current.left is None:
          current.left = item
          item.parent = current
          return False
      elif current.right is None:
          current.right = item
          item.parent = current
          return False
      else: 
          queue.append(current.left)
          queue.append(current.right)
    

  def fill_tree_with_leaves(self, evenly_spaced=False):
    """
      Given a bald tree, adds LeafNode objects to it until it is full  
      Input: tree (Tree), evenly_spaced(bool): if True, leaves will be assinged probs evenly spaced between each other, 
        otherwise, leaves will be assigned 0 as probs
      Output:
    """
    if self.root.child is None:
      new_leaf = n.LeafNode(0)
      self.root.child = new_leaf
      new_leaf.parent = self.root.child
      self.GLOM_LIST.append(new_leaf)
      return self
    queue = [self.root.child]
    leaf_count = 0
    nodes_to_fill = []
    # First pass: count existing leaves and find nodes to fill
    while queue:
      current = queue.pop(0)
      if isinstance(current, n.LeafNode):
          leaf_count += 1
      elif isinstance(current, n.Node):
          if current.left is None:
              nodes_to_fill.append((current, 'left'))
          else:
              queue.append(current.left)
          if current.right is None:
              nodes_to_fill.append((current, 'right'))
          else:
              queue.append(current.right)
    # Calculate total leaves after filling
    total_leaves = leaf_count + len(nodes_to_fill)
    # Second pass: fill empty slots with leaves
    for i, (node, side) in enumerate(nodes_to_fill):
      if evenly_spaced:
          prob = i / (total_leaves - 1) if total_leaves > 1 else 0
      else:
          prob = 0
      new_leaf = n.LeafNode(prob)
      new_leaf.parent = node
      setattr(node, side, new_leaf)
      self.GLOM_LIST.append(new_leaf)

    return self

  def print2DUtil(self, root, space):
    """
      Recursively prints out a tree given a defined spacing
      Input: self (Tree), root (Node, LeafNode), space (int)
      Output: None when no more further nodes exist
    """
    if (root == None):
        return
    space += self.COUNT[0]
    if not isinstance(root, n.LeafNode):
      self.print2DUtil(root.right, space)
    print()
    for i in range(self.COUNT[0], space):
        print(end=" ")
    if isinstance(root, n.LeafNode):
       print(root.prob)
    else:
       print(root.number)
    if not isinstance(root, n.LeafNode):
      self.print2DUtil(root.left, space)

  def print2D(self, root):
    """
      Calls the print2DUtil function on a root node
      Input: Self (Tree), root (Node)
      Output: No output, prints node values
    """
    self.print2DUtil(root, 0)


  # def pSPLIT(self, node):
  #   """
  #     Calculates the probability of a node splitting, considering defined alpha and beta values and node depth
  #     Used to construct a tree by drawing from the prior      
  #     Input: Self (Tree), node (Node)
  #     Output: Split: 0=no split and 1=split (int)
  #   """
  #   alpha = 0.5
  #   beta = 0.5
  #   p_split = alpha * (1 + node.depth()) ** -beta
  #   return random.choices([0, 1], weights = (1-p_split, p_split))


  # def pRULE(self, node, used_param):
  #   """
  #     Determines the probability of assigning a given value to a node once it has split
  #     Input: self (Tree), node (Node), used_param ([int])
  #     Output: Assigned node value (int)
  #   """
  #   if not used_param:
  #     return
  #   next_rule = random.choice(used_param)
  #   used_param.remove(next_rule)
  #   node.number = next_rule
  #   return used_param

  # def construct_node_from prior(self, node, used_param):
  #   if self.pSPLIT(node) == 1:
  #     self.pRULE(node, used_param)

  

  # def draw_from_prior(self):
  #    first_node = n.Node()
  #    used_num = []
  #    if len(self.GLOM_LIST) == 1:
  #     self.pRULE(first_node)
  #     self.root.child = first_node
  #     used_num.append(first_node.number)
  #    else:
  #       while first_node.depth() < 7:
  #          new_node = n.Node()
  #    return first_node



  def grow(self, node_id = None):
    """
      From possible nodes (leaf nodes) selects a new node to grow. Then creates a new node replaces it with the
      selected grow node. New node is given 2 empty leaves
      Input: self (Tree), node_id (int)
      Output: No output, grows tree in place
    """
    #select grow node
    available_nodes = []
    for node in self.GLOM_LIST:
        if isinstance(node, n.LeafNode) and node is not None:
            available_nodes.append(node)
    grow_node = random.choice(available_nodes)

    #select value for grow node
    available_values = []
    for x in range(0, self.glom_size):
        for glom in self.GLOM_LIST:
            in_list = False
            if not isinstance(glom, n.LeafNode) and glom.number == x:
                in_list = True
        if in_list is False:
            available_values.append(x) 

    #create grow node and assign parents and children           
    new_node = n.Node(number = random.choice(available_values))
    self.GLOM_LIST.append(new_node)
    new_node.parent = grow_node.parent
    if new_node.parent.left == grow_node:
       new_node.parent.left = new_node
    elif new_node.parent.right == grow_node:
       new_node.parent.right = new_node
    left_new_leaf = n.LeafNode()
    new_node.add_node(0, left_new_leaf)
    self.GLOM_LIST.append(left_new_leaf)
    right_new_leaf = n.LeafNode()
    new_node.add_node(1, right_new_leaf)
    self.GLOM_LIST.append(right_new_leaf)
    self.GLOM_LIST.remove(grow_node)
    

  def prune(self):
    """
      Randomly selects a node within the tree to prune. Possible prune nodes are limited to nodes with 2 leaves and cannot be the root node
      Input: Self (Tree)
      Output: No output, prunes tree in place
    """
    #select prune node
    possible_prune_nodes = []
    for node in self.GLOM_LIST:
        if not isinstance(node, n.RootNode) and not isinstance(node, n.LeafNode):
            if (isinstance(node.left, n.LeafNode) and isinstance(node.right, n.LeafNode)) and not isinstance(node.parent, n.RootNode):
                possible_prune_nodes.append(node)
    prune_node = random.choice(possible_prune_nodes)

    #prune node and replace with a leaf
    if prune_node.parent.left == prune_node:
        new_leaf = n.LeafNode()
        prune_node.parent.left = new_leaf
        new_leaf = prune_node.parent
        self.GLOM_LIST.append(new_leaf)
    else:
        new_leaf = n.LeafNode()
        prune_node.parent.right = new_leaf
        new_leaf = prune_node.parent
        self.GLOM_LIST.append(new_leaf)
    self.GLOM_LIST.remove(prune_node)
    if not isinstance(prune_node, n.LeafNode):
        if prune_node.left in self.GLOM_LIST:
            self.GLOM_LIST.remove(prune_node.left)
        if prune_node.right in self.GLOM_LIST:
            self.GLOM_LIST.remove(prune_node.right)

  def change(self):
    """
      Assigns a random internal node a new value
      Input: self (Tree)
      Output: No output, changes tree in place
    """
    #Randomly select change node
    available_change_nodes = []
    for node in self.GLOM_LIST:
       if not isinstance(node, n.LeafNode) and not isinstance(node, n.RootNode):
          available_change_nodes.append(node)
    change_node = random.choice(available_change_nodes)

    #assign new value
    change_node.number = random.randint(0, self.glom_size-1)


  def swap(self):
    """
      Randomly swaps a parent/child pair (both internal ndoes) within the tree
      Input: Self (Tree)
      Output: No output, swaps nodes in place
    """
    #determine location of child to swap
    child_location = random.choice([0, 1]) #0 represents left, 1 represents right
    
    #looks for swappable nodes. Needs to be an internal node with another internal
    #node as its child. If no swappable nodes, function stops. 
    swappable_nodes = []
    for glom in self.GLOM_LIST[1:]:
       if not isinstance(glom, n.LeafNode): 
          if child_location == 0:
             if not isinstance(glom.left, n.LeafNode):
                swappable_nodes.append(glom)
          else:
             if not isinstance(glom.right, n.LeafNode):
                swappable_nodes.append(glom)
    if not swappable_nodes:
        return
    
    #stores 2 nodes to swap and creates a temporary node of the parent getting swapped
    swap_node = random.choice(swappable_nodes)
    swap_child = swap_node.left if child_location == 0 else swap_node.right
    # print(f"swapping {swap_node} with {swap_child}")
    temp_node = swap_node.create_temp_node()

    #updates pointers from parents to children
    if isinstance(temp_node.parent, n.RootNode):
        temp_node.parent.child = swap_child
    else: 
        if temp_node.parent.left == swap_node:
            temp_node.parent.left = swap_child
        elif temp_node.parent.right == swap_node:
            temp_node.parent.right = swap_child
    
    #swaps children
    swap_node.left = swap_child.left
    swap_node.left.parent = swap_node
    swap_node.right = swap_child.right
    swap_node.right.parent = swap_node
    if temp_node.left == swap_child:
        swap_child.left = swap_node
        swap_child.right = temp_node.right
    else:
        swap_child.right = swap_node
        swap_child.left = temp_node.left
    
    #update parents
    swap_child.parent = temp_node.parent
    swap_node.parent = swap_child


  def mutate(self):
    """
      Creates a copy of given tree, and randomly selects one of four mutations to perform. Each mutation
      has an equal probability of being picked. Mutation is performed on copy of the given tree. 
      Input: self (Tree)
      Output: mutated_tree (Tree)
    """
    mutated_tree = copy.deepcopy(self)
    random_mutations = [mutated_tree.grow, mutated_tree.prune, mutated_tree.change, mutated_tree.swap]
    chosen_mutation = random.choice(random_mutations)
    chosen_mutation() 
    return mutated_tree
  

  def get_leaf_probs(self):
     """
      Creates a list of probabilities of all leaves in a tree
      Input: self (Tree)
      Output: leaf_prob (list)
    """
     leaf_probs = []
     for node in self.GLOM_LIST:
        if isinstance(node, n.LeafNode):
           leaf_probs.append(node.number)
     return leaf_probs
  
  def run_tree_gaps(self, gap_list, leaf_prob = False):
    """
      Traverses a list of gaps through a tree and returns associated responses  
      Input: tree (Tree), gap_list(list[list[int]])
      Output: response_list (list[int])
    """
    response_list = []
    for gap in gap_list:
      if leaf_prob:
        response_list.append(self.run_tree_single_gap(gap, leaf_prob=True))
      else:
        response_list.append(self.run_tree_single_gap(gap, leaf_prob=False))
    return response_list


  def run_tree_single_gap(self, gap, leaf_prob = True):
    """
      Traverses a single gap through the tree and returns the associated leaf probability  
      Input: tree (Tree), gap (list[int])
      Output: prob (float)
    """
    prob = self.root.child.traverse_GAP(gap)
    if leaf_prob:
      return prob
    else:
      response = random.choices([0, 1], weights = (1-prob, prob))[0]
      return response


  def train_leaf_prob(self, train_gap_list, train_response_list):
    """
    Trains a tree with a list of GAPS and a list of associated responses
    Input:
      self: Tree object
      train_gap_list: list[list[int]]
      train_response_list: list[int] 
    Returns:
      No output, updates leaf probabilities in place
    """
    for x in range(len(train_gap_list)):
      self.root.child.traverse_and_update(train_gap_list[x], train_response_list[x])

  def get_probs_from_data(self, list_trn, list_val, list_tst):
     probs_trn = self.run_tree_gaps(list_trn, leaf_prob=True)
     probs_val = self.run_tree_gaps(list_val, leaf_prob=True)
     probs_tst = self.run_tree_gaps(list_tst, leaf_prob=True)
     return probs_trn, probs_val, probs_tst

def build_bald_tree(num_nodes, shuffle_nums=False):
    """
    Creates a tree with a given number of nodes. No LeafNode objects are added to the tree
      Input: num_nodes (int), shuffle nums: False if node are in order, True otherwise (bool)
      Output: created tree (Tree)
    """
    tree = Tree(glom_size=10)
    num_order = [i for i in range(num_nodes)]
    if shuffle_nums:
      random.shuffle(num_order)
    for num in num_order:
      node = n.Node(num)
      tree.add_to_tree(node)
    return tree

def construct_from_prior(glom_size, alpha, beta):
  possible_params = [i for i in range(glom_size)]
  tree = Tree(10)
  start_node = n.Node()
  tree.root.child = start_node
  start_node.parent = tree.root
  start_node.construct_nodes_from_prior(possible_params, tree, alpha, beta)
  return tree

  

if __name__ == "__main__":
  # test_tree_ordered_even = build_bald_tree(10, shuffle_nums=False)
  # test_tree_ordered_even = test_tree_ordered_even.fill_tree_with_leaves(evenly_spaced=True)
  # test_tree_ordered_even.print2D(test_tree_ordered_even.root.child)
  # mutated_test_tree_ordered_even = test_tree_ordered_even.mutate()
  # mutated_test_tree_ordered_even.print2D(mutated_test_tree_ordered_even.root.child)



  # test_tree_shuffled_even = build_bald_tree(10, shuffle_nums=True)
  # test_tree_shuffled_even = test_tree_shuffled_even.fill_tree_with_leaves(evenly_spaced=True)
  # test_tree_ordered_zero = build_bald_tree(10, shuffle_nums=False)
  # test_tree_ordered_zero = test_tree_ordered_zero.fill_tree_with_leaves(evenly_spaced=False)
  # test_tree_shuffled_zero = build_bald_tree(10, shuffle_nums=True)
  # test_tree_shuffled_zero = test_tree_shuffled_zero.fill_tree_with_leaves(evenly_spaced=False)
  alpha_vals = [0.5, 0.95, 0.95, 0.95]
  beta_vals = [0.5, 0.5, 1.0, 1.5]

  bin_width = 1
  fig, axs = plt.subplots(2, 2, figsize=(10,10))

  plot_number = 0


  for (alpha, beta) in zip(alpha_vals, beta_vals):
    print(alpha)
    print(beta)
    total_num_terminal_nodes = []
    for x in range (200000):
      test_tree = construct_from_prior(30, alpha, beta)
      total_num_terminal_nodes.append(test_tree.num_terminal_nodes)

    mean = np.mean(total_num_terminal_nodes)
    row_index = plot_number // 2
    col_index = plot_number % 2  
    
    bins = np.arange(min(total_num_terminal_nodes), max(total_num_terminal_nodes) + bin_width, bin_width)
    counts, bins, patches = plt.hist(total_num_terminal_nodes, bins=bins, density=False, alpha=0.7, color='blue', edgecolor='black')
    probabilities = counts / counts.sum()
    
    axs[row_index, col_index].cla()  # Clear the current axes
    axs[row_index, col_index].bar(bins[:-1], probabilities, width=bin_width, alpha=0.7, color='blue', edgecolor='black', align='edge')
    axs[row_index, col_index].set_title(f'Alpha: {alpha}, Beta: {beta}, mean = {mean}', fontsize=10)
    axs[row_index, col_index].set_xlabel('Number of Terminal Nodes', fontsize=10)
    axs[row_index, col_index].set_ylabel('Probability', fontsize=10)
    axs[row_index, col_index].set_xlim(0, 30)

    plot_number += 1
  # Add titles and labels
  # plt.title('Histogram of Terminal Nodes')
  # plt.xlabel('Number of Terminal Nodes')
  # plt.ylabel('Probability')
  


  # Show plot
  plt.tight_layout()
  plt.savefig("test")
  plt.clf()
  

  p=0
  # test_tree.print2D(test_tree.root.child)   
     

    
   

