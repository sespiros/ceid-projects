#!/usr/bin/python
"""
.. moduleauthor:: Spyros Seimenis <seimenis@ceid.upatras.gr>
"""
import db
import btree
import json
import time

def full_testbench(name, amount):
    """ This testbench tests the full functionality of the b-tree database and its relying functions

    :param name: Name of the database
    :type name: string
    :param amount: Amount of records that will be created in database
    :type amount: integer

    .. note::
        This function is fully commented in the source. click [source] on the right
    """
    database = name + '.db'

    # CreateDatabase
    db.createDatabase(database,amount)

    # For the time i use padding to store the nodes
    # in the file although its a bad technique
    # here i set the db.SIZE_SMALL and db.SIZE_BIG according to
    # the amount of records in the database
    # db.SIZE_SMALL = 200 + amount//1000
    # db.SIZE_BIG = 1000 + amount

    # CreatePrimaryIndex and CreateSecondary Index
    for i in db._attribute:
        indexname = i + '_' + name + '.json'
        x = db.createIndex(database,indexname, i)

    # Insert record in database
    record = db.makeRecord("10070 seimenis souloumenis 50 80")
    num = db.insertInDatabase(record, database)

    # Insert record in database with duplicate primary key
    record_error = db.makeRecord("20 error error 2 20")
    num_error = db.insertInDatabase(record_error, database)                   # ERROR 1

    # Insert
    if num != -1:
        for i in db._attribute:
            indexname = i + '_' + name + '.json'
            db.insertInIndex([record[i], num], indexname)
            db.insertInIndex([record_error[i], num_error], indexname)

    # All of the following function can be called in order to test their
    # functionality

    # Test function for finding in Primary and Secondary key indexes
    def test_find():
        query = db.makeRecord("20 ROOT ROOT 2000 24")
        for i in db._attribute:
            indexname = i + '_' + name + '.json'
            res = db.find(query[i], indexname)
            if not res == None:
                lines = db.retrieve(res[0].keys[res[1]], database)
                print ('Records found in ' + indexname)
                db.printRecords(lines)

    # Test function for finding in Primary and Secondary key indexes with range find
    def test_rangefind():
        range_query = {'ID':(0,10),
                       'surname':('pelopiou', 'ROOT'),
                       'name':('john', 'sotiris'),
                       'year':(2000, 2005),
                       'age':(20, 21)}

        for i in db._attribute:
            indexname = i + '_' + name + '.json'
            key1 = range_query[i][0]
            key2 = range_query[i][1]
            list = db.findRange(key1, key2, indexname)

            (MIN, MAX) = (key1, key2)
            if (key1>key2): (MIN, MAX) = (key2, key1)

            if not list == None:
                print ('--------------RECORDS FOUND IN ' + indexname + ' IN RANGE \'' + str(MIN) + '\', \'' + str(MAX) + '\'--------------')
                for i in list:
                    lines = db.retrieve(i[0].keys[i[1]], database)
                    db.printRecords(lines)

    # Test function for deleting in Primary and Secondary key indexes
    def test_delete():
        # Deletes a record from the primary index
        indexname = 'ID' + '_' + name + '.json'
        db.deleteFromIndex(indexname, 5)
        db.deleteFromIndex(indexname, 1000)          # this does not exist   # ERROR 2

        # Deletes a record from the secondary index
        indexname = 'name' + '_' + name + '.json'
        db.deleteFromIndex(indexname, 1, 'seimenis') # this does not exist   # ERROR 3
        db.deleteFromIndex(indexname, 5070, 'souloumeni')
        db.deleteFromIndex(indexname, 5, 'hadfadd')

        # deletes completely the primary key from all the indexes of the given database
        db.delete(5070, database)

    #test_find()
    #test_rangefind()
    #test_delete()

    print_all(name)

def print_all(name):
    """Given an database prints all the indexes created on this database"""
    #Printing
    print ('-----------------------------------------------------------------------------------')
    for i in db._attribute:
        indexname = i + '_' + name + '.json'
        print ('INDEX ' + indexname)
        db.tree_print(indexname)
    print()

def timingProfile(name):
    """ It is a test function that shows the difference between the linear search in a db file
        and btree search.

        .. note::
            For the test i use a search on a secondary key index because it is the most general case
            the function can test the times between all the indexes too..

        .. warning::
            To have a more accurate timing it is preferred to use externally the cProfile python module for profiling
            usage: python -m cProfile testbench.py

    """

    def treefind(value):
        res = db.find(value, indexname)
        if not res == None:
            lines = db.retrieve(res[0].keys[res[1]], database)
            print ('Records found in ' + indexname)
            db.printRecords(lines)

    def normalfind(value, attribute):
        file = open(database,'r')
        found = False
        end = False
        lines = []
        for i in file:
            obj = json.loads(i.strip())
            if obj[attribute]==value:
                lines.append(i)
        db.printRecords(lines)

    # Initialization
    #db.createDatabase(database,9000)
    #db.createIndex(database, indexname, attribute)

    database = name + '.db'
    attribute = 'surname'   # set it for the preferred attribute
    indexname = attribute + '_' + name + '.json'
    searchval = 'stallman' # set it for the preferred key to search

    start = time.time()
    normalfind(searchval, attribute)
    normalTime = time.time()-start

    start = time.time()
    treefind(searchval)
    btreeTime = time.time()-start

    if (btreeTime < normalTime):
        print ('Btree is ' + str(normalTime/btreeTime) + ' times faster!!')
    else:
        print ('Normal search is faster.......secret: never happens :P')

def main():
    """ The main function that is called for testbench and profiling

    .. warning::
        There is a malfunction because i use :func:`db.createDatabase` to create my test database files.
        When i produce a very large file(ex 10000), i have only 30 names so it is normal for the creation
        of the secondary key indexes to have VERY large nodes that exceeds tha db.SIZE_BIG.
        For the time i set it manually

    """
    db.SIZE_SMALL = 300
    db.SIZE_BIG = 20000
    #full_testbench('test', 10000)
    timingProfile('test')

if __name__ == '__main__':
    main()
