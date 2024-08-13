import random

"""
Contains RootNode, Node, and LeafNode classes. To be used to construct trees. 
"""


class RootNode:
  """
    Represents the root node of a tree structure.

    Attributes:
        number (int): A numerical value associated with the root node, default is 1000.
        child: Reference to the child node, initially None.
        parent: Reference to the parent node, always None for a root node.
  """
  def __init__(self, number = 1000):
    self.number = number
    self.child = None
    self.parent = None


  def __str__(self):
    return "root"


class Node:
  """
    Represents an internal node in a tree structure.

    Node class holds methods for tree manipulation, traversal, and probability calculations.

    Attributes:
        left: Reference to the left child node, initially None.
        right: Reference to the right child node, initially None.
        parent: Reference to the parent node, initially None.
        number (int): Integer representation of Node value.

    Methods:
        height(): Calculates the height of node.
        full(): Determines if tree below node is full.
        traverse_GAP(gap): Traverses the tree based on a given GAP.
        traverse_and_update(gap, response): Traverses the tree and updates leaf probabilities.
        add_node(location, new_node): Adds a new child node at a given location.
        create_temp_node(): Creates a temporary copy of this node.
        depth(): Calculates the depth of node.
        replace_with_leaf(): Replaces Node object with a LeafNode object.
        pSPLIT(tree, alpha, beta): Calculates the probability of this node splitting.
        pRULE(possible_params, tree): Assigns a value to this node from possible parameters.
        construct_nodes_from_prior(possible_params, tree, alpha, beta): Constructs child nodes based on pSPLIT and pRULE.
  """

  def __init__(self, number = 0):
    self.left = None
    self.right = None
    self.parent = None
    self.number = number


  def height(self):
    """
    Recursively tracks the maximum height of a given node by traversing through its child nodes
      Input: Self (Node)
      Output: Maximum node height (int)
    """
    if isinstance(self, LeafNode):
      return 0
    elif isinstance(self, Node):
      if self.left is not None and self.right is not None:
        return max(1 + self.left.height(), 1 + self.right.height())
      elif self.left is not None:
        return 1 + self.left.height()
      elif self.right is not None:
        return 1 + self.right.height()
      else:
        return 0
    

  def full(self):
    """
      Recursively determines if everything below the given node is full. Full is defined as every node having 0 or 2 children. 
      Input: Self (Node)
      Output: Fullness (bool)
    """
    if isinstance(self, LeafNode):
      return True
    elif self.left is not None and self.right is not None:
      return True
    elif self.left is not None and self.right is not None:
      if self.left.height() != self.right.height():
        return False
      else:
        return self.left.full() and self.right.full()
    else:
      return False #node with one child is not full


  def traverse_GAP(self, gap):
    """
      Traverses through the tree for a given GAP - if the gap idx of the node is 0, it traverses left, if 1, traverses right
      Input: Self (Node), GAP binary list (list)
      Output: Final leaf probability (float)
    """
    if isinstance(self, LeafNode):
      return self.prob
    if gap[self.number] == 0:
      return self.left.traverse_GAP(gap)
    elif gap[self.number] == 1: 
      return self.right.traverse_GAP(gap)
    else:
      raise ValueError("Invalid glomerulus state")


  def traverse_and_update(self, gap, response):
    """
      Traverses through the tree for a given GAP - if the gap idx of the node is 0, it traverses left, if 1, traverses right. 
      Once the final leaf is reached, the leaf probability is updated
      Input: Self (Node), GAP binary list (list), Response: 0=no lick and 1=lick (int)
      Output:
    """
    if isinstance(self, LeafNode):
      self.update_probability(response)
      return self.prob
    if gap[self.number] == 0:
      return self.left.traverse_and_update(gap, response)
    elif gap[self.number] == 1:
      return self.right.traverse_and_update(gap, response)


  def add_node(self, location, new_node):
    """
      Manually adds a new node as a child of the given node
      Input: Self (Node), location: 0=left and 1=right (int), new_node (Node, LeafNode) 
      Output:
    """
    if location == 0:
      self.left = new_node
      new_node.parent = self
    else:
      self.right = new_node
      new_node.parent = self
  

  def create_temp_node(self):
    """
      Creates a temporary node, identical to the node called on, to be used for tree mutations
      Input: Self (Node)
      Output: temp_node (Node)
    """
    #creates a temporary node with the same pointers as the given node
    temp_node = Node(number = self.number)
    temp_node.parent = self.parent
    temp_node.left = self.left
    temp_node.right = self.right
    return temp_node


  def depth(self):
    """
      Recursively finds the depth of a node by traversing its parents
      Input: Self(Node, LeafNode)
      Output: Depth (int)
    """
    if self.parent is None or isinstance(self.parent, RootNode):
      return 0
    else:
      return 1 + self.parent.depth()

  def replace_with_leaf(self):
    """
    Converts node object into a leaf. To be used for building tree, if a node doesnt split
    it is turned into a leaf. 
    Input:
      self: Node
    Output: 
      No ouput, modifies in place
    """
    new_leaf = LeafNode()
    if not isinstance(self.parent, RootNode):
      if self.parent.left == self:
        self.parent.left = new_leaf
        new_leaf.parent = self.parent
      elif self.parent.right == self:
        self.parent.right = new_leaf
        new_leaf.parent = self.parent
    else: 
      self.parent.child = new_leaf
      new_leaf.parent = self.parent

    
  def __str__(self):
    return f"Node {self.number}"

  def pSPLIT(self, tree, alpha=0.5, beta=0.5):
    """
      Calculates the probability of a node splitting, considering defined alpha and beta values and node depth
      Used to construct a tree by drawing from the prior      
      Input: Self (Tree), node (Node)
      Output: Split: 0=no split and 1=split (int)
    """
    p_split = alpha * (1 + self.depth()) ** -beta
    split = random.choices([0, 1], weights = (1-p_split, p_split))[0]
    return split


  def pRULE(self, possible_params):
    """
      Determines the probability of assigning a given value to a node once it has split
      Input: 
        Self: Node, 
        used_param [int]
      Output: 
        Assigned node value (int)
    """
    if not possible_params:
      return
    next_rule = random.choice(possible_params)
    possible_params.remove(next_rule)
    self.number = next_rule
    return possible_params

  def construct_nodes_from_prior(self, possible_params, tree, alpha, beta):
    """
      Constructs a tree via probability of splitting and rule assignment probabilites
      Inputs:
        Self: Node
        possible_params: list[int]
        tree: Tree
        alpha: float < 1
        beta: float >= 0
    """
    if not possible_params:
      return
    if self.pSPLIT(tree, alpha, beta) == 1:
      updated_params = self.pRULE(possible_params, tree)
      new_node_left = Node()
      self.add_node(0, new_node_left)
      new_node_right = Node()
      self.add_node(1, new_node_right)
      return new_node_left.construct_nodes_from_prior(updated_params, tree, alpha, beta), new_node_right.construct_nodes_from_prior(updated_params, tree, alpha, beta)
    self.replace_with_leaf()
    tree.num_terminal_nodes += 1
    

class LeafNode(Node):
  """
    Represents a leaf node in a tree structure.
    This class inherits from Node, but has no children and value is a probability rather than glom representation.

    Attributes:
        parent (Node, RootNode): Parent reference, initially None.
        licks (int): Lick count, initially 0.
        total_trials (int): Total number of trials, initially 0.
        prob (float): Probability associated with this leaf node, initially set to init_prob.

    Methods:
        update_probability(response): Updates the probability based on a new behavioral response.
    """
  def __init__(self, init_prob = 0):
    self.parent = None
    self.licks = 0
    self.total_trials = 0
    self.prob = init_prob


  def update_probability(self, response):
    """
      Updates the current leaf's probability given a new behavioral response
      Input: Self (LeafNode), response: 0=behavior did not occur and 1=behavior did occur (int)
      Output: No output
    """
    if response == 1:
      self.licks += 1
    self.total_trials += 1
    self.prob = self.licks/self.total_trials
 

  def __str__(self):
    return "leaf"