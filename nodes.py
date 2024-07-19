import numpy as np
import random

#RootNode Class
#Initializes a node with a pointer to 1 child
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
    #calculates the height of a given node
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
    else:
      p=0


  def full(self):
    #checks to see if the tree is full
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


  # def add_node_in_order(self, new_node):
  #   #given a current node and a new node will add the new node in the first
  #   #available position to maintain the fullness of the tree
  #   if isinstance(self.left, LeafNode, self.right, LeafNode):
  #     return
  #   if self.left is None:
  #     self.left = new_node
  #     new_node.parent = self
  #     self.leaf = False
  #   elif self.right is None:
  #     self.right = new_node
  #     new_node.parent = self
  #     self.leaf = False
  #   else:
  #     if self.left.full() is True and self.right.full() is True:
  #       if self.left.height() > self.right.height():
  #         self.right.add_node_in_order(new_node)
  #       else:
  #         self.left.add_node_in_order(new_node)
  #     elif self.left.full() is False:
  #       self.left.add_node_in_order(new_node)
  #     else:
  #       self.right.add_node_in_order(new_node)


  def traverse_GAP(self, gap):
    #Recursive function to traverse tree until a leaf is reached. When a leaf
    #is reached the probability associated with that node is returned
    # if self.leaf is True:
    #   # print(f'''The predicted probability of the mouse
    #   # licking based on {gap} is: {self.probability}''')
    #   return self.probability
    if isinstance(self, LeafNode):
      return self.prob
    if gap[self.number] == 0: # or gap[self.number] == "0":
      return self.left.traverse_GAP(gap)
    elif gap[self.number] == 1: # or gap[self.number] =="1":
      return self.right.traverse_GAP(gap)
    else:
      raise ValueError("Invalid glomerulus state")

  def traverse_and_update(self, gap, response):
    # print("traversing")
    #traverses tree with a gap/sucess combo and updates leaf probabilities
    if isinstance(self, LeafNode):
    #   test_response = random.choices([0,1], weights = (1-self.prob, self.prob))[0]
      self.update_probability(response)
      return self.prob
    # for some reason, this is reading the gap as a string
    if gap[self.number] == 0:
      return self.left.traverse_and_update(gap, response)
    elif gap[self.number] == 1:
      return self.right.traverse_and_update(gap, response)


  def add_node(self, location, new_node):
    #Manually adds a new node at a given location to the current node 
    #Location = 0: left, Location = 1: right.
    if location == 0:
      self.left = new_node
      new_node.parent = self
    else:
      self.right = new_node
      new_node.parent = self
  
  def create_temp_node(self):
    #creates a temporary node with the same pointers as the given node
    temp_node = Node(number = self.number)
    temp_node.parent = self.parent
    temp_node.left = self.left
    temp_node.right = self.right
    return temp_node

  def depth(self):
    #returns the depth of a node within a tree
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
    # if self.total_trials:
    #     self.number = self.licks/self.total_trials

  def update_probability(self, response):
    #Based on the success or failure (1 or 0) of a trail, a leafs probability will be updated
    if response == 1:
      self.licks += 1
    self.total_trials += 1
    self.prob = self.licks/self.total_trials
 
  def __str__(self):
    return "leaf"
  
# start_node = Node(0)
# start_left = Node(1)
# start_node.left = start_left
# start_right = Node(2)
# start_node.right = start_right
# leaf1 = LeafNode(0.1)
# start_left.left = leaf1
# leaf2 = LeafNode(0.2)
# start_left.right = leaf2
# leaf3 = LeafNode(0.3)
# start_right.left = leaf3
# leaf4 = LeafNode(0.4)
# start_right.right = leaf4
# print(start_node.traverse_GAP("000"))
# print("_____________________")
# print(start_node.traverse_GAP([0, 0, 0]))
# print("_____________________")
# print(start_node.traverse_GAP("110"))
# print("_____________________")
# print(start_node.traverse_GAP([1, 1, 0]))

  
