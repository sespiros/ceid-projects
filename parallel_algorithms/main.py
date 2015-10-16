# -*- coding: utf-8 -*-
from clbtree import Node
from math import floor
import sys
import matplotlib.pyplot as p


def CLadd(a, b):
    totalSteps = 0
    listA = [int(x) for x in list(a)]
    listB = [int(x) for x in list(b)]
    result = ''

    # Phase 1 -
    N = len(listA)
    leaves = []
    # initialize leaves
    for x in xrange(0, N):
        if listA[x] == 0 and listB[x] == 0:
            # stop carry
            node = Node('s')
        elif listA[x] == 1 and listB[x] == 1:
            # generate carry
            node = Node('g')
        else:
            node = Node('p')

        leaves.append(node)

    # propagate up
    newlevel = leaves
    while len(newlevel) > 1:
        level = newlevel
        newlevel = []

        if (len(level) % 2 != 0):
            pnode = Node(level[0])
            newlevel.insert(0, pnode.data)

        mid = int(floor(len(level) / 2.0))
        for i in xrange(mid - 1, -1, -1):
            pnode = Node('?')
            pnode.right = level[- (2 * i + 1)]
            pnode.left = level[-(2 * i + 2)]
            pnode.calc()

            # in the same time, update left children with current value
            # (happens on next step)
            pnode.left.data = pnode.right.data
            pnode.right.data = '?'
            newlevel.append(pnode)
        #counting each level as 1 step because they would happen
        #simultaneousy in parallel environment
        totalSteps = totalSteps + 2

    root = newlevel[-1]
    if root.data == 'g':
        result += '1'
    else:
        result += '0'
    root.data = 's'

    # Phase 2 -
    #root.printNode(0)
    root.propagateDown('s')
    #root.printNode(0)

    newlevel = root.leaves()

    for i in range(0, N):
        if newlevel[i] == 'g':
            carry = 1
        else:
            carry = 0
        result += str((int(listA[i]) + int(listB[i]) + carry) % 2)

    totalSteps += 1
    return (result, totalSteps)


def compare(N, c):
    result = ''
    a = ''
    b = ''
    if c == 0:
        a = raw_input('a: ')
        b = raw_input('b: ')

        if len(a) != N or len(b) != N:
            print "Can't work like this!"
            sys.exit(0)
    else:
        # it generates output for a 1111111....111
        for i in range(0, N):
            a += '1'
        # it generates output for b 0000000....001
        for i in range(0, N - 1):
            b += '0'
        b += '1'

    # performing sequential addition
    sequentialSteps = 0
    carry = 0
    for i in xrange(1, N + 1):
        res = int(a[N - i]) + int(b[N - i]) + carry
        sumRes = res % 2
        carry = res / 2
        result = str(sumRes) + result
        sequentialSteps = sequentialSteps + 1
    # last bit
    result = str(carry) + result
    sequentialSteps = sequentialSteps + 1

    #performing carry lookahead addition
    (clresult, clSteps) = CLadd(a, b)

    if c == 0:
        print
        print "Sequential result      = ", result
        print "Carry-lookahead result = ", clresult
        print

    return (sequentialSteps, clSteps)


def main():
    choice = 0
    while choice < 2:
        if choice == 0:
            N = []
            N.append(int(raw_input('Enter number of bits: ')))
        else:
            print
            print "Test results for suggested number of bits:"
            print "=============================================="
            print
            N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        seqresults = []
        clresults = []
        for i in N:
            res = compare(i, choice)
            seqresults.append(res[0])
            clresults.append(res[1])

        print N, "bits"
        print seqresults, "sequential steps"
        print clresults, "Carry-Lookahead steps"

        if choice == 1:
            # plotting the steps of the two algorithms using matplotlib
            # uncomment to plot
            p.plot(N, seqresults, 'r^:', N, clresults, 'b*:')
            p.ylabel("Steps")
            p.xlabel("Bits")
            p.legend(('Sequential', 'Carry-Lookahead'))
            p.show()
            pass

        choice += 1

    print
    raw_input('Press Enter to end...')

if __name__ == '__main__':
    main()
