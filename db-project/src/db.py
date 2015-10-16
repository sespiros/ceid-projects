#!/usr/bin/python
"""
.. moduleauthor:: Spyros Seimenis <seimenis@ceid.upatras.gr>
"""
import btree
import string
import random
import json

# if this set to True adds verbosity in the output of the program !!very usefull!!
DEBUG = False

# default json line size
SIZE_SMALL = 300
SIZE_BIG = 1000

# It is used as an enum for fast processing
_attribute = ['ID','surname','name','year','age']

# creation
def createDatabase(name, size):
    """Function for generating an example record file used for testing

    :param name: the name of the database
    :type name: str.
    :param size: the number of records in the record file
    :type size: integer

    It is used to generate random IDs, surnames, names, year and age in order to create a test database,
    every record is a dictionary that first is jsoned and then dumped into the file

    .. note::
        The databases files have .db extension

    """

    years = [x for x in range(2000,2050)]
    age = [x for x in range(18,60)]
    names = ['forrest', 'kostas', 'spiros', 'dimitris',
             'agathwn', 'marcello', 'markos', 'sotiris',
             'petros', 'nikos', 'pelopidas', 'loukas', 'john',
              'ntinos', 'makis', 'marinos', 'root', 'mitsos',
              'giannis', 'vagelis', 'pietros','spyro','flkj', 'richard','linus']

    surnames = ['souloumelitis', 'tsakalos', 'lamarinas', 'zathuras',
                'pelopiou', 'maderas', 'gump', 'robocopios',
                'danezos', 'aggelou', 'papakonstantinou', 'venizelos',
                'snow','ROOT','lanister','asdf','surnamer','maragos',
                'kalyvas','derek','dales','lemos','otinanai','stallman','torvalds']

    def id_generator(size=6, chars=string.ascii_lowercase):
        return ''.join(random.choice(chars) for x in range(size))

    file = open (name, 'w')
    for i in range(size):
        record = {}
        record['ID'] = i+1
        record['surname'] = random.choice(surnames)
        record['name'] = random.choice(names)
        record['year'] = random.choice(years)
        record['age'] = random.choice(age)
        json.dump(record, file)
        file.write('\n')

    file.close()


def createIndex(name,filename,attr):
    """Middle function that creates index based on the attribute given

    :param name: name of the database
    :type name: string
    :param filename: name of the index
    :type filename: string

    .. note::
        The index files have .json extension and their names follows the following rule {attribute}_{dbname}.json
        ,also an index's first line contains the number of the node that is root, namely the line number of this node

    """
    if attr=='ID':
        createPrimaryIndex(name,filename)
    else:
        createSecondaryIndex(name,filename,attr)
    # resets the nodeCounter in btree.py because the creation of the index has
    # finished
    btree.nodeCounter = 0

def createPrimaryIndex(name,filename):
    """Function that creates primary key indexes

    :param name: name of the database
    :type name: string
    :param filename: name of the index
    :type filename: string

    """
    f = open(name, 'r')
    out = open(filename, 'w+')

    counter = 0
    list = []
    tr = btree.BTreeNode(btree.TREE_TYPE)   # B-tree initialization of root
    out.write('1   \n')
    for line in f:
        # checks for deleted records
        if line[0]=='-':
            counter += 1
            continue
        # loads the record
        t = json.loads(line)
        # constructs the key
        key = [t['ID'], counter]
        if DEBUG: print (' Pushing key ' + str(key[0]), end  = '')
        tr = btree.B_tree_insert(tr, key, out)
        counter+=1
    out.seek(0)
    out.write(str(tr.num)+(4-len(str(tr.num)))*' ' + '\n')
    out.close()
    f.close()

def createSecondaryIndex(name,filename,attr):
    """Function that creates secondary key indexes based on the given attribute(not primary key)

    :param name: name of the database
    :type name: string
    :param filename: name of the index
    :type filename: string
    :param attr: the attribute (surname, name, year, age)
    :param type: string

    """
    f = open(name, 'r')
    out = open(filename, 'w+')

    counter = 0
    list = []
    tr = btree.BTreeNode(btree.TREE_TYPE)   # B-tree initialization of root
    out.write('1   \n')
    for line in f:
        if line[0]=='-':
            counter += 1
            continue
        t = json.loads(line)
        nodekey = [t[attr], counter]
        if DEBUG:  print ('Pushing key ' + str(nodekey[0]), end='')
        tr = checkInsert(tr, nodekey, out)
        counter+=1
    out.seek(0)
    out.write(str(tr.num)+(4-len(str(tr.num)))*' ' + '\n')
    out.close()
    f.close()

# insertion
def insertInDatabase(record, filename):
    """ Inserts a given record in the given database

    :param record: the record
    :type record: dictionary
    :param filename: the database's filename
    :type filename: string
    :raises: ValueError
    """
    file = open(filename, 'r+')
    numlines = 0
    dmap = [record['ID']]
    for i in file:
        numlines += 1
        try:
            l = json.loads(i)['ID']
        except ValueError:
            pass
        dmap.append(l)

    # check if record's ID matches any existing ID in record file(duplicate
    # primary key)
    dmap = set(dmap)
    if numlines + 1 == len(dmap):
        found = False
    else:
        found = True

    if not found:
        print('Inserting record with ID:' + str(record['ID']) + ' in database \'' + filename + '\'')
        json.dump(record, file)
        file.write('\n')
        file.close()
        return numlines
    else:
        print ('Error: trying to insert duplicate primary key in database ' + filename)
        return -1

def insertInIndex(nodekey, indexname):
    """Inserts a key into an index

    :param nodekey: the nodekey to insert in the database
    :type nodekey: list
    :param indexname: filename of the index
    :type indexname: string
    """
    file = open(indexname,'r+')

    # Initializes the nodeCounter for use with the specified index
    numlines = 0
    for i in file:
        numlines += 1
    btree.nodeCounter = numlines
    file.seek(0)

    root = int(file.readline())
    tr = disk_read(root, file)
    if DEBUG:print ('Pushing key ' + str(nodekey[0]))
    tr = checkInsert(tr, nodekey, file)
    btree.nodeCounter = 0
    file.seek(0)
    file.write(str(tr.num)+(4-len(str(tr.num)))*' ' + '\n')
    file.close()

def checkInsert( tr, nodekey, file ):
    """ Wrapper for :func:`btree.B_tree_insert`

    :param tr: The root of the btree we want to insert to
    :type tr: :class:`btree.BTreeNode`
    :param nodekey: the nodekey we want to insert
    :type nodekey: list
    :param file: The file descriptor of the index
    :type file: file

    If the nodekey found in the index:

    * If index is for primary key checks for duplicate and throws an error
    * If index is for secondary key instead of inserting a new nodekey appends the
      record in the existing key
    """
    res = btree.B_tree_search(tr, nodekey[0], file)
    if res == None:
        tr = btree.B_tree_insert(tr, nodekey, file)
    else:
        if DEBUG: print ('\nFound collision for ', nodekey[0])
        if file.name.split('_')[0] == 'ID':
            print ('Error: trying to insert duplicate primary key ' + str(nodekey[0]) + ' in index ' + file.name)
        else:
            new_offset = nodekey[1]
            res[0].keys[res[1]].append(new_offset)
            disk_write(res[0], file)
    return tr

# deletion
def deleteFromIndex(indexname, keyP, keyS=None):
    """ Deletes a record from an index

    :param indexname: the index's filename
    :type indexname: string
    :param keyP: A primary key
    :type keyP: integer
    :param keyS: A secondary key
    :type keyS: string/integer

    If the index is a primary key index it just deletes the record with the primary keyP
    If the index is a secondary key index:

    * searches in the btree the keyS using :func:`db.find`
    * if found uses :func:`db.retrieve` to retrieve the records that the nodekey contains
      else prints a message that the given keyS didn't found in the index
    * if found checks the records to find matching records with the given keyP
      then if the nodekey has only one record deletes the nodekey completely,
      if the nodekey has multiple records with the keyS deletes only the one matching the keyP
      if none of the records matches keyP prints error message

    """
    file = open(indexname,'r+')

    file.seek(0)
    root = int(file.readline())
    tr = disk_read(root, file)

    name = indexname.split('_')[0]
    if name == 'ID':
        tr = btree.B_tree_delete(tr, keyP, file)
        file.seek(0)
        file.write(str(tr.num)+(4-len(str(tr.num)))*' ' + '\n')
        file.close()
    else:
        found = False
        res = find(keyS, file.name)
        if not res == None:
            lines = retrieve(res[0].keys[res[1]], indexname.split('_')[1].split('.')[0] + '.db')
            for i in range(len(lines)):
                if keyP == json.loads(lines[i])['ID']:
                    found = True
                    if len(res[0].keys[res[1]]) == 2:
                        tr = btree.B_tree_delete(tr, keyS, file)
                        file.seek(0)
                        file.write(str(tr.num)+(4-len(str(tr.num)))*' ' + '\n')
                    else:
                        fileD = open(indexname.split('_')[1].split('.')[0] + '.db', 'r')
                        data = fileD.readlines()
                        k = 1
                        while k < len(data):
                            jsoned = json.loads(data[k])
                            if jsoned['ID'] == keyP:
                                break
                            k+=1
                        res[0].keys[res[1]].remove(k)
                        disk_write(res[0], file)
            if found == False:
                print ('The key \'' + str(keyS) + '\' exists but does not match any record in index ' + file.name)

def delete(key, filename):
    """ Given a primary key and an database file deletes the key from the database and all the indexes

    :param key: the ID of the user to be deleted
    :type key: integer
    :param filename: the name of the database file
    :type filename: string

    First searches the record file(database) for the given primary key and deletes it from the database
    Then it uses :func:`db.deleteFromIndex` to delete the key from all the indexes that this database has
    """
    file = open(filename,'r+')
    file.seek(0)
    found = False
    count = 0
    while not found:
        a = file.tell()
        line = file.readline()
        count += 1
        if line == '':break
        if line[0] == '-':continue
        record = json.loads(line)

        if record['ID']==key:
            found = True

    if found:
        for i in range(5):
            indexname = _attribute[i] + '_' + filename.split('.')[0] + '.json'
            try:
                with open(indexname) as f:
                    deleteFromIndex(indexname, key, record[_attribute[i]])
            except IOError as e:
                pass
        # the new implementation of deleteFromIndex uses the record file for
        # the process. Because of that the record is deleted from the record
        # file after the deleteFromIndex has finished
        file.seek(a)
        pad = '-'*30
        file.write(pad)
        file.close()
    else:
        print('Primary key ' + str(key) + ' not found or has already deleted')

# disk access
def disk_write(object, file):
    """Function for writing a b-tree node into a file

    :param object: a btree node
    :type object: :class:`btree.BTreeNode`
    :param file: the file to write to
    :type file: :class: file

    It is used between the processing of an index to gain
    write access of the given index. It uses the json module
    to write the given object into the file.

    .. note::
        It uses constant json size of 200 for ID primary key index
        and 500 for secondary key index. It may fail if node is too big

    """
    if file.name.split('_')[0]=='ID':
        size = SIZE_SMALL 
    else:
        size = SIZE_BIG
    file.seek(0)
    file.readline()
    file.seek( file.tell() + object.num*size )
    dumped = json.dumps(object.__dict__)
    file.write(dumped)
    file.write(' '*(size-2-len(dumped))+'\n')

def disk_read(block, file):
    """Function for reading a b-tree node into a file

    :param block: a btree node number of line in the index
    :type block: integer
    :param file: the file to write to
    :type file: class _io.TextIOWrapper
    :rtype: :class:`btree.BTreeNode`

    It is used between the processing of an index to gain
    read access of the given index. It fins the line in the index file
    then uses json to recreate a BTreeNode and return it

    .. note::
        It uses constant json size of 200 for ID primary key index
        and 500 for secondary key index. It may fail if node is too big

    """
    if file.name.split('_')[0]=='ID':
        size = SIZE_SMALL 
    else:
        size = SIZE_BIG
    file.seek(0)
    file.readline()
    file.seek( file.tell() + block*size )
    l = file.readline()
    try:
        d = json.loads(l)
    except ValueError:
       print (l)
    object = btree.BTreeNode(jsontemp=True)
    try:
        object.__dict__ = d
    except UnboundLocalError:
        print ("Error: Node size exceeds max limit of json node in the file")
        print(file.name)
        exit()
    return object

# searching
def find(key, indexname):
    """Finds a record in an index given a key

    :param key: The key to look for
    :type key: integer/string
    :param indexname: filename of the index
    :type indexname: string
    :rtype: tuple that contains a node and an index that points to a key in this node

    .. note::
        the difference between a key and a nodekey is that the key is a single value (integer or string) that is used to traverse the tree.
        the nodekey is a struct-like object (list) that contains the key and the offset of the certain key in the record file. Each node has nodekeys
        but it is traversed using the keys in the nodekeys
    """
    file = open(indexname,'r')
    file.seek(0)
    root = int(file.readline())
    tr = disk_read(root, file)
    res = btree.B_tree_search(tr, key, file)

    if res == None:
        print('Error: Key \'' + str(key) + '\' not found in index ' + indexname)

    file.close()
    return res

def findRange(key1, key2,  indexname):
    """Finds records in an index, in the range of two keys

    :param key1: The key down border
    :type key1: integer/string
    :param key2: The key up border
    :type key2: integer/string
    :param indexname: filename of the index
    :type indexname: string

    """
    # recursive function. Generates a list with the nodekeys that are between
    # the two given keys
    def _traverse(tr, list, key1, key2, file):
        tr = disk_read(tr, file)
        for i in range(len(tr.keys)):
            if tr.keys[i][0] >= key1 and tr.keys[i][0] <= key2:
                list.append((tr, i))
        for i in tr.children:
            _traverse(i, list, key1, key2, file)

    file = open(indexname,'r+')
    file.seek(0)
    root = int(file.readline())

    # Swaps the keys if the key2 is smaller than key1
    (MIN, MAX) = (key1, key2)
    if (key1>key2): (MIN, MAX) = (key2, key1)

    list = []
    _traverse(root, list, MIN, MAX, file)

    if list == None:
        print('Keys in range \'' + str(MIN) + '\', \'' + str(MAX) + '\' not found.')

    file.close()
    return list

# misc
def makeRecord(line):
    """Function for creating record dictionary given a properly formatted string

    :param line: properly formatted string
    :type line: str.
    :return record: the generated record
    :type record: dictionary

    Example of its function

    >>> makeRecord('5070 spyros seimenis 2 20')
    {'age': 20, 'year': 2, 'surname': 'spyros', 'ID': 5070, 'name': 'seimenis'}
    """
    _size = { 0:5, 1:20, 2:10, 3:4, 4:2 }
    line = line.split()
    record = {}
    record['ID'] = int(line[0][:_size[0]])
    record['surname'] = line[1][:_size[1]]
    record['name'] = line[2][:_size[2]]
    record['year'] = int(line[3][:_size[3]])
    record['age'] = int(line[4][:_size[4]])

    return record

def retrieve(nodekey, filename):
    """Function for retrieving the lines from the database that a key of Btreenode is pointing

    :param nodekey: a key of the class.BtreeNode
    :type nodekey: list
    :param filename: the file to write to
    :type filename: str.
    :return lines: The lines from the database that the given key is pointing
    :type lines: list

    """
    file = open(filename,'r')
    lines = nodekey[1:]
    lines.sort()
    lines[1:] = [ lines[i+1]-lines[i]-1 for i in range(len(lines)-1)]
    li = []
    for i in lines:
        for k in range(i):file.readline()
        li.append(file.readline().strip())
    return li
    file.close()

def tree_print(filename):
    """ Helper function that prints the b-tree that an index contains

    :param filename: the index's name
    :type filename: string
    """
    # recursive function
    def _recPrint(block, file, deep=0):
        tr = disk_read(block, file)
        for i in range(deep):print('\t|',end='')
        print('-->',end='')
        tr.print()
        for i in tr.children:
            _recPrint(i, file, deep + 1)
    try:
        file = open(filename,'r+')
    except IOError:
        return
    block = int(file.readline())
    _recPrint(block, file)
    file.close()

def printRecords(list):
    """ Helper function that formats a list of jsoned records into records(dictionaries) and prints them

    :param list: list with jsoned records (from the record file)
    :type list: list
    """
    for i in list:
        jsoned = json.loads(i)
        for k in _attribute:
           print (k + ': ' + str(jsoned[k]) + ' ', end='')
        print()
    print()

# experimental
def optimize(filename):
    """ This function is experimentally used to optimize/rebalance an index after creation

    :param filename: index's filename
    :type filename: string

    .. warning::
        Experimental function, currently not working properly

    """
    # recursice function
    def _optimize(tr, file):
        i = 0
        while i < len(tr.keys)-1:
            leng = len(tr.keys)
            tr.optimize(i,file)
            tr = disk_read(tr.num, file)
            if len(tr.keys)==leng:i+=1
            if len(tr.keys)==0:
                file.seek(0)
                file.write(str(tr.children[0])+(4-len(str(tr.children[0])))*' ' + '\n')
                tr = disk_read(tr.children[0], file)
        i = 0
        for i in tr.children:
            left = disk_read(i, file)
            if not left.leaf:
                _optimize(left, file)

    file = open(filename, 'r+')

    numlines = 0
    for i in file:
        numlines += 1
    btree.nodeCounter = numlines
    file.seek(0)

    block = int(file.readline())
    tr = disk_read(block, file)
    _optimize(tr, file)
    file.close()
