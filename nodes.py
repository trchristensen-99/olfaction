"""
"""

class RootNode:
  def __init__(self, number = 1000):
    self.number = number
    self.child = None
    self.parent = None


  def __str__(self):
    return "root"


#Node Class
#Functions include __init__() - intializes object, height() - determines height
#of node, full()- determines if a specific node is full, add_node(new_node) -
#adds a new node to the node it is called on, traverse_GAP(GAP) - given a GAP,
#traverses from that node,
class Node:
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
    if self.parent is None:
      return 0
    else:
      return 1 + self.parent.depth()
    
  def __str__(self):
    return f"Node {self.number}"
  

class LeafNode(Node):
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