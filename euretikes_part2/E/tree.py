#/usr/bin/python
import itertools
import random

iteration = [(10,0,0)]
capacity =[10,7,3]
operators = [i for i in list(itertools.product([0,1,2],[0,1,2])) if i[0]!=i[1]]

def drop_water(state_tp, src, dest):

    if state_tp[src]>0 and state_tp[dest]<capacity[dest]:

        state = list(state_tp) # make it list

        if state[src] > capacity[dest]-state[dest]:     # with remain
            state[src] -= capacity[dest]-state[dest]
            state[dest] = capacity[dest]
        else:                                           # no remain
            state[dest] += state[src]
            state[src] = 0
        new_state_tp = tuple(state)   # make it tuple again

        return new_state_tp
    return None


def add_to_graph(t1, t2, s):
    colorMap = {(0,1):'black',
                (1,2):'red',
                (2,1):'green',
                (2,0):'orange',
                (1,0):'pink',
                (0,2):'blue'}

    str1 = '"('+str(t1[0])+','+str(t1[1])+','+str(t1[2])+')"'
    if t2 is not None:
        str2 = '"('+str(t2[0])+','+str(t2[1])+','+str(t2[2])+')"'
        f.write(str1 + '->' + str2 + ' [color = ' + colorMap[s] + ']'+'[label="'+str(s) +'"]\n')
    else:
        f.write(str1 + '\n')

f = open('tree.dot','w')
f.write('digraph {\n')
f.write('labelloc="t";\nlabel="E";\n')
f.write('rankdir=LR\n')

largest = [0,1,2]
fullest = []
possiblel = [[(0,2),(0,1),(2,1),(2,0),(1,2),(1,0)],
            [(1,2),(1,0),(2,1),(2,0),(0,2),(0,1)],
            [(2,1),(2,0),(1,2),(1,0),(0,2),(0,1)]]
possible = []
fitness = []

for i in iteration:
    fullest = [x for (y,x) in sorted(zip(list(i),[0,1,2]))]
    possible = []
    fitness = []
    best_ops = possiblel[fullest[0]]

    for op in best_ops:
        new_state_tp = drop_water(i,op[0],op[1])

        possible.append(new_state_tp)

    #evaluate fitness function
    for node in possible:
        if node is None:
            fitness.append(1000)
        else:
            fitness.append(abs(5-node[0]) + abs(5-node[1]) + abs(0-node[2]))

    print("### Node  " + str(i) + "#########################")
    for j in range(len(fitness)):
        if possible[j] is not None and possible[j] not in iteration:
            print("Node " + str(possible[j]) + " with fitness " + str(fitness[j]))

    for factor in fitness:
        index = fitness.index(min(fitness))
        node = possible[index]

        if node is None: break

        if node not in iteration:
            iteration.append(node)
            add_to_graph(i, node, (best_ops[index][0], best_ops[index][1]))
            break
        else:
            fitness[index]=1000

    if iteration[-1]==(5,5,0):break

f.write('}\n')
f.close()
