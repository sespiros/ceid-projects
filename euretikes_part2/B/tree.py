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
        if new_state_tp not in iteration:
            iteration.append(new_state_tp)
            add_to_graph(state_tp, new_state_tp, (src, dest))
            print ("from" + str(state_tp)+"found new " + str(new_state_tp))
            return 1
        return 0

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
f.write('labelloc="t";\nlabel="B";\n')

largest = [0,1,2]
fullest = []

for i in iteration:
    fullest = [x for (y,x) in sorted(zip(list(i),[0,1,2]))][::-1]
    fullest.remove(0)
    best_ops = [(0,1),(0,2),(fullest[0],0),(fullest[0],fullest[1]),(fullest[1],0),(fullest[1],fullest[0])]

    for op in best_ops:
        if drop_water(i,op[0],op[1])==1:break

    if iteration[-1]==(5,5,0):break

f.write('}\n')
f.close()
