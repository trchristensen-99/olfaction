import random
import nodes as n
import copy
import numpy as np

class Tree:
 
  COUNT = [10]

  def __init__(self, glom_size, tree_id=0):
    #Initializes a tree with a root node
    self.GLOM_LIST = []
    self.tree_id = tree_id
    self.root = n.RootNode()
    self.GLOM_LIST.append(self.root)
    self.glom_size = glom_size


  def create_tree_in_order(self, num_nodes):
    """
      Creates an tree of a desired size in which internal nodes are given ordered values
      Input: Self (Tree), num_nodes (int)
      Output: No output, works in place on given tree
    """
    for x in range(0,num_nodes):
      new_node = n.Node(x)
      if x == 0:
       self.root.child = new_node
       new_node.parent = self.root
      else:
        self.root.child.add_node_in_order(new_node)
      self.GLOM_LIST.append(new_node)

  def create_tree_random(self, num_nodes):
    """
      Creates a tree of a desired size in which internal nodes are given random values within the range of total nodes
      Input: self (Tree), num_nodes(int)
      Output: No output, works in place on given tree
    """
    available_nodes = []
    i = 0
    for x in range(0,num_nodes):
      available_nodes.append(x)
    while len(available_nodes) != 0:
      new_node_val = random.choice(available_nodes)
      available_nodes.remove(new_node_val)
      new_node = n.Node(new_node_val)
      if i == 0:
        self.root.child = new_node
        new_node.parent =self.root
      else:
        self.root.child.add_node_in_order(new_node)
      self.GLOM_LIST.append(new_node)
      i+=1
  
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


  def pSPLIT(self, node):
    """
      Calculates the probability of a node splitting, considering defined alpha and beta values and node depth
      Used to construct a tree by drawing from the prior      
      Input: Self (Tree), node (Node)
      Output: Split: 0=no split and 1=split (int)
    """
    alpha = 0.5
    beta = 0.5
    p_split = alpha * (1 + node.depth()) ** -beta
    return random.choices([0, 1], weights = (1-p_split, p_split))


  def pRULE(self, node, used_param = []):
    """
      Determines the probability of assigning a given value to a node once it has split
      Input: self (Tree), node (Node), used_param ([int])
      Output: Assigned node value (int)
    """
    next_rule = random.choice([0, self.glom_size])
    if next_rule not in used_param:
        used_param.append(next_rule)
        node.number = next_rule
        return next_rule
    return self.pRULE(self, node, used_param)
  

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


def add_to_tree(tree, item):
  """
  Adds given node to a given tree in order
  Inputs: tree (Tree), item (Node)
  Returns: true if tree is full, false otherwise
  """
  tree.GLOM_LIST.append(item)
  if tree.root.child is None:
    tree.root.child = item
    item.parent = tree.root
    return isinstance(item, n.LeafNode)
  queue = [tree.root.child]
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
     add_to_tree(tree, node)
  return tree

def fill_tree_with_leaves(tree, evenly_spaced=False):
  """
    Given a bald tree, adds LeafNode objects to it until it is full  
    Input: tree (Tree), evenly_spaced(bool): if True, leaves will be assinged probs evenly spaced between each other, 
      otherwise, leaves will be assigned 0 as probs
    Output:
  """
  if tree.root.child is None:
    new_leaf = n.LeafNode(0)
    tree.root.child = new_leaf
    new_leaf.parent = tree.root.child
    tree.GLOM_LIST.append(new_leaf)
    return tree
  queue = [tree.root.child]
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
    tree.GLOM_LIST.append(new_leaf)

  return tree
  

if __name__ == "__main__":
  test_tree_ordered_even = build_bald_tree(10, shuffle_nums=False)
  test_tree_ordered_even = fill_tree_with_leaves(test_tree_ordered_even, evenly_spaced=True)
  test_tree_ordered_even.print2D(test_tree_ordered_even.root.child)
  mutated_test_tree_ordered_even = test_tree_ordered_even.mutate()
  mutated_test_tree_ordered_even.print2D(mutated_test_tree_ordered_even.root.child)



  test_tree_shuffled_even = build_bald_tree(10, shuffle_nums=True)
  test_tree_shuffled_even = fill_tree_with_leaves(test_tree_shuffled_even, evenly_spaced=True)
  test_tree_ordered_zero = build_bald_tree(10, shuffle_nums=False)
  test_tree_ordered_zero = fill_tree_with_leaves(test_tree_ordered_zero, evenly_spaced=False)
  test_tree_shuffled_zero = build_bald_tree(10, shuffle_nums=True)
  test_tree_shuffled_zero = fill_tree_with_leaves(test_tree_shuffled_zero, evenly_spaced=False)
  p=0
  # test_tree.print2D(test_tree.root.child)   
     

    
   

