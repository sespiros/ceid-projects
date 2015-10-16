#/usr/bin/python
import itertools
import random
import string

explored = [(10,0,0)]
exp_ids = [((10,0,0),'start')]
levels = [[((10,0,0),'start')]]
capacity =[10,7,3]
operators = {i for i in list(itertools.product([0,1,2],[0,1,2])) if i[0]!=i[1]}

def drop_water(state_tp, src, dest, level):

    if state_tp[0][src]>0 and state_tp[0][dest]<capacity[dest]:

        state = list(state_tp[0]) # make it list

        if state[src] > capacity[dest]-state[dest]:     # with remain
            state[src] -= capacity[dest]-state[dest]
            state[dest] = capacity[dest]
        else:                                           # no remain
            state[dest] += state[src]
            state[src] = 0

        new_state_tp = tuple(state)   # make it tuple again

        if new_state_tp in explored:
            id = ''.join(random.choice(string.ascii_lowercase) for x in range(10))
            node = (new_state_tp, id)
        #    add_to_graph(state_tp, node, (src, dest),1)

        if new_state_tp is not None and new_state_tp not in explored:
            explored.append(new_state_tp)
            id = ''.join(random.choice(string.ascii_lowercase) for x in range(10))
            node = (new_state_tp, id)
            exp_ids.append(node)
            levels[level].append(node)
            add_to_graph(state_tp, node, (src, dest),0)

        if (new_state_tp[0]==5 and new_state_tp[1]==5):return 1
        return 0

def add_to_graph(t1, t2, s, c):
    colorMap = {(0,1):'black',
                (1,2):'red',
                (2,1):'green',
                (2,0):'orange',
                (1,0):'pink',
                (0,2):'blue'}

    label1 = '"('+str(t1[0][0])+','+str(t1[0][1])+','+str(t1[0][2])+')"'
    node1 = t1[1]
    f.write( node1 + ' [label='+label1+']\n')
    label2 = '"('+str(t2[0][0])+','+str(t2[0][1])+','+str(t2[0][2])+')"'
    node2 = t2[1]
    f.write( node2 + ' [label='+label2+']')
    if c==1: f.write('[style=filled]')
    f.write( '\n')
    f.write(node1 + '->' + node2 + ' [color = ' + colorMap[s] + ']'+'[label="'+str(s) +'"]\n')

def gentree():
    index = 0
    for level in levels:
        levels.append([])
        index += 1
        for node in level:
            for op in operators:
                if drop_water(node, op[0], op[1], index)==1:return

        f.write("{rank=same; ")
        for i in level:
            f.write(i[1]+' ')
        f.write("}\n")



f = open('tree.dot','w')
f.write('digraph unix {\n')
f.write('rankdir = LR \n')
f.write('labelloc="t";\nlabel="C";\n')
gentree()
# monopati skepsis
for i in range(len(exp_ids)-1):
    f.write(exp_ids[i][1] + '->' + exp_ids[i+1][1] + ' [label = ' + str(i+1) +'][penwidth=2.0][color = purple][fontcolor = purple]\n')
f.write('}\n')
f.close()
