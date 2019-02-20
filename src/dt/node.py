# Node of a Decision Tree

class Node:
  totalcount = 0

  def __init__(self, dataset, parent):
    Node.totalcount += 1
    #
    self.id = Node.totalcount
    self.level = 1 if (parent is None) else parent.level + 1
    #
    self.data = dataset
    #
    self.predicate = None
    self.line = None
    self.answer = None
    #
    self.childSAT = None
    self.childUNSAT = None
    self.parent = parent


  def is_root(self):
    return (self.level == 1)


  def is_leaf(self):
    assert((self.childSAT is None) == (self.childUNSAT is None))
    return (self.childSAT is None)


  def is_predicate(self):
    if self.predicate is not None:
      assert(self.line is None and self.answer is None)
      return True
    return False


  def is_line(self):
    if self.line is not None:
      assert(self.predicate is None and self.answer is None)
      return True
    return False


  def is_answer(self):
    if self.answer is not None:
      assert(self.predicate is None and self.line is None)
      return True
    return False


  def name(self):
    if self.is_predicate():
      return self.predicate.name
    if self.is_line():
      return self.line.name
    if self.is_answer():
      return self.answer.name
    return 'NOT SET'
