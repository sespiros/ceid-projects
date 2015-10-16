#!/usr/bin/python
"""
.. moduleauthor:: Spyros Seimenis <seimenis@ceid.upatras.gr>
"""
import db

# defines tree type ex. B = 2*TREE_TYPE-1
TREE_TYPE = 4

# counts the number of nodes in the btree currently working on
nodeCounter = 0

# defines the Node of a BTree
class BTreeNode:
    """This is the class that implements the btree nodes

        a BTreeNode consists of:

        * a nodeCounter that is given to him and it is unique that represents its relative position
          with the others and the line of the node in the
          index file
        * a boolean leaf that indicates if the current node is a leaf or not
        * a variable t that holds the type of the b-tree this node belongs to
        * a list with nodekeys. A nodekey is a list with its first element being the actual key and
          the rest elements are the positions in the record file in which this key exists
        * a list with children. For each of the children of the node, its
          num(nodeCounter) is stored in that list
    """
    def __init__(self, t = TREE_TYPE, leaf = True, jsontemp = False):
        """Constructor of a BTreeNode

        :param t: (integer) the tree type
        :param leaf: (boolean) becomes true if the node is a leaf
        :param jsontemp: (boolean) becomes True if we want to create a temporary BTreeNode so that the nodeCounter wont increment
        """
        global nodeCounter
        self.num = nodeCounter
        if not jsontemp: nodeCounter += 1
        self.leaf = leaf
        self.t = t
        self.keys = []
        self.children = []
        if db.DEBUG and not jsontemp: print ('\n\tNew Node! ' + str(self.num), end='')

    def print(self):
        """Prints info about the BTreeNode"""
        print ('Node: '+ str(self.num) + ' Number of keys: ' + str(len(self.keys)) + ' Number of children: ' + str(len(self.children)) +
            '( ',end='')
        for i in self.keys:
            print(i[0], end='')
            print ('('+ str(len(i)-1)+')',end='')
            print (', ',end='')
        print(')')

    def split_child(self, i, file):
        """This function splits a node in the following way:
            self: (...x, key[i]=k, ...) ==> self: (...x, key'[i], y, key'[i+1]=k...)
        """
        t = self.t
        x = db.disk_read(self.children[i], file)

        y = BTreeNode(t, x.leaf)
        if db.DEBUG: print ('\t\t\t\t Current right child num = ', str(y.num))
        if db.DEBUG: print (' from split_child')
        self.keys.insert(i, x.keys[t-1])
        self.children.insert(i+1, y.num)
        y.keys = x.keys[t:]
        x.keys = x.keys[:t-1]
        if db.DEBUG:
            print ("\t\t\t parent node " + str(self.num)+" from split = ",end='')
            self.print()
            print ("\t\t\t existing node " + str(x.num)+" from split = ",end='')
            x.print()
            print ("\t\t\t new node " + str(y.num)+" from split = ",end='')
            y.print()
        if not y.leaf:
            y.children = x.children[t:]
            x.children = x.children[:t]
        db.disk_write(x, file)
        db.disk_write(self, file)
        db.disk_write(y, file)

    def merge_children(self, i, file):
        """ Merges children[i] and children[i+1] of the given node by pushing keys[i] down
        """
        x = db.disk_read(self.children[i], file)
        y = db.disk_read(self.children[i+1], file)
        x.keys += [self.keys[i]]+y.keys
        x.children += y.children
        self.keys.pop(i)
        self.children.pop(i+1)
        db.disk_write(self, file)
        db.disk_write(x, file)
        db.disk_write(y, file)

    def optimize(self, i, file):
        """Optimizes and rebalances the current node

        .. warning::
            Its use is experimental and it is not yet fully working
        """
        x = db.disk_read(self.children[i], file)
        y = db.disk_read(self.children[i+1], file)
        mod = len(y.keys+x.keys)+1
        if mod<=2*TREE_TYPE-1:
            x.keys += [self.keys[i]]+y.keys
            x.children += y.children
            self.keys.pop(i)
            self.children.pop(i+1)
        else:
            tmpK = x.keys + [self.keys[i]]+y.keys
            tmpC = x.children + y.children
            x.keys = tmpK[:2*TREE_TYPE-1]
            x.children = tmpC[:2*TREE_TYPE-1]
            self.replace_key(i,tmpK[2*TREE_TYPE-1], file)
            y.keys = tmpK[2*TREE_TYPE:]
            y.children = tmpC[2*TREE_TYPE:]
        db.disk_write(self, file)
        db.disk_write(x, file)
        db.disk_write(y, file)

    def keysremove(self, key):
        """Removes the given nodekey from the node"""
        for i in self.keys:
            if i[0] == key:
                self.keys.remove(i)

    def replace_key(self, i, key, file):
        """Replaces keys[i] with the given key and write changes"""
        self.keys[i] = key
        db.disk_write(self, file)
        return key

    def is_full(self):
        """Returns true if node is full"""
        return len(self.keys) >= 2*self.t-1

    def can_remove(self):
        """Returns true if the current node can remove keys"""
        return len(self.keys) >= self.t

# insertion
def B_tree_insert(tr, key, file):
    """It operates as a wrapper for :func:`B_tree_insert_nonfull`, that checks if parent node is full and do the necessary operations for insertion

    :param tr: the root of the subtree
    :type tr: :class:`BTreeNode`
    :param key: the key to delete
    :type key: integer / string
    :param file: the index file
    :type file: file descriptor
    """
    root = tr
    if root.is_full():
        s = BTreeNode(root.t, False)
        if db.DEBUG: print (' because ' + str(root.num) + ' is full', end='')
        s.children.insert(0, root.num)
        db.disk_write(s, file)
        s.split_child(0, file)
        root = s
    B_tree_insert_nonfull(root, key, file)
    return root

def ordered_insert(lst, x):
    """Helper function that inserts element x in lst in ascending order

    :param lst: list to insert to
    :type lst: list
    :param x: object to insert
    :type x: object
    """
    i = len(lst)
    lst.append(x)
    while i>0 and lst[i]<lst[i-1]:
        (lst[i-1], lst[i]) = (lst[i], lst[i-1])
        i=i-1

def B_tree_insert_nonfull(tr, key, file):
    """ Inserts a key in a given subtree

    :param tr: the root of the subtree
    :type tr: :class:`BTreeNode`
    :param key: the key to delete
    :type key: integer / string
    :param file: the index file
    :type file: file descriptor

    if the given node is a leaf inserts the key into the node calling :func:`ordered_insert`
    otherwise tries to insert the key into the child nodes of the given node
    """
    if tr.leaf:
        ordered_insert(tr.keys, key)
        if db.DEBUG: print (' in node ' + str(tr.num))
        db.disk_write(tr, file)
    else:
        i = len(tr.keys)

        while i>0 and key < tr.keys[i-1]:
            i = i-1
        x = db.disk_read(tr.children[i], file)
        if x.is_full():
            tr.split_child(i, file)
            if key>tr.keys[i]:
                i = i+1
        x = db.disk_read(tr.children[i], file)
        B_tree_insert_nonfull(x, key, file)

# deletion
def B_tree_delete(tr, key, file):
    """ Deletes a key in a given btree

    :param tr: the root of the btree
    :type tr: :class:`BTreeNode`
    :param key: the key to delete
    :type key: integer / string
    :param file: the index file
    :type file: file descriptor

    """
    i = len(tr.keys)
    while i>0:
        if key == tr.keys[i-1][0]:
            if tr.leaf:  # case 1 in CLRS
                tr.keysremove(key)
                db.disk_write(tr, file)
            else: # case 2 in CLRS
                left = db.disk_read(tr.children[i-1], file)
                right = db.disk_read(tr.children[i], file)
                if left.can_remove(): # case 2a
                    nodekey = tr.replace_key(i-1, left.keys[-1], file)
                    B_tree_delete(left, nodekey[0], file)
                elif right.can_remove(): # case 2b
                    nodekey = tr.replace_key(i-1, right.keys[0], file)
                    B_tree_delete(right, nodekey[0], file)
                else: # case 2c
                    tr.merge_children(i-1, file)
                    left = db.disk_read(i-1, file)
                    B_tree_delete(left, nodekey[0], file)
                    if tr.keys==[]: # tree shrinks in height
                        tr = left
            return tr
        elif key > tr.keys[i-1][0]:
            break
        else:
            i = i-1
    # case 3
    if tr.leaf:
        print ('Error: The key ' + str(key) + ' does not match any record in index ' + file.name)
        return tr #key doesn't exist at all
    x = db.disk_read(tr.children[i], file)
    if not x.can_remove():
        #print ('cannot remove ' + str(key.data[0]))
        left = db.disk_read(tr.children[i-1], file)
        try:
            right = db.disk_read(tr.children[i+1], file)
            rightCanRemove = right.can_remove()
        except IndexError:
            rightCanRemove = False
        if i>0 and left.can_remove(): #left sibling
            x.keys.insert(0, tr.keys[i-1])
            tr.keys[i-1] = left.keys.pop()
            if not x.leaf:
                x.children.insert(0, left.children.pop())
            db.disk_write(x, file)
            db.disk_write(left, file)
        elif i<len(tr.children) and rightCanRemove: #right sibling
            x.keys.append(tr.keys[i])
            tr.keys[i]=right.keys.pop(0)
            if not x.leaf:
                x.children.append(right.children.pop(0))
            db.disk_write(x, file)
            db.disk_write(right, file)
        else: # case 3b
            if i>0:
                tr.merge_children(i-1, file)
                x = db.disk_read(x.num-1, file)
            else:
                tr.merge_children(i, file)
                x = db.disk_read(x.num, file)

    B_tree_delete(x, key, file)
    if tr.keys==[]: # tree shrinks in height
        tr = tr.children[0]

    db.disk_write(tr, file)

    return tr

# search
def B_tree_search(tr, key, file):
    """ Searches for a key in a given btree

    :param tr: the root of the btree
    :type tr: :class:`BTreeNode`
    :param key: the key to search for
    :type key: integer / string
    :param file: the index file
    :type file: file descriptor

    """
    if len(tr.keys) != 0:
        for i in range(len(tr.keys)):
            if key <= tr.keys[i][0]:
                break
        if key == tr.keys[i][0]:
            return (tr, i)
        if tr.leaf:
            return None
        else:
            if key > tr.keys[-1][0]:
                i=i+1
            x = db.disk_read(tr.children[i], file)
        return B_tree_search(x, key, file)
    return None
