class Node:
    #each node has s,g,p,? value and two children
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    #calculates the data of the node from the children input
    def calc(self):
        if self.left.data == 's' or self.left.data == 'g':
            self.data = self.left.data
        elif self.left.data == "p":
            self.data = self.right.data

    #second phase of the algorithm
    def propagateDown(self, data):
        # if self.data is p then it will propagate the data from its father
        if self.data != 's' and self.data != 'g':
            self.data = data

        # if self is a node with a left child
        if self.left is not None:
            # counting only left childs as they would descend simultaneously
            # in a parallel enviroment
            self.left.propagateDown(self.data)

        # if self is a node with a right child
        if self.right is not None:
            self.right.propagateDown(self.data)

    #return the leaves of the subtree under self
    def leaves(self):
        ret = ''
        if self.left is not None:
            ret += self.left.leaves()

        if self.right is not None:
            ret += self.right.leaves()

        # only the leaf nodes
        if self.right is None and self.left is None:
            ret += self.data

        return ret

    #helper function used for printing
    def printNode(self, depth):
        if self.right is not None:
            self.right.printNode(depth + 1)
        print '\t' * depth + self.data
        if self.left is not None:
            self.left.printNode(depth + 1)
