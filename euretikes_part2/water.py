#/usr/bin/python
import itertools

#p1a
space = [i for i in list(itertools.product(range(0,11), range(0,8), range(0,4))) if i[0]+i[1]+i[2]==10]

#p2a
capacity =[10,7,3]

def drop_water(state_tp, src, dest):

    if state_tp[src]>0 and state_tp[dest]<capacity[dest]:

        state = list(state_tp)

        if state[src] > capacity[dest]-state[dest]:     # with remain
            state[src] -= capacity[dest]-state[dest]
            state[dest] = capacity[dest]
        else:                                           # no remain
            state[dest] += state[src]
            state[src] = 0

        new_state_tp = tuple(state)
        add_to_graph(state_tp, new_state_tp)

def add_to_graph(t1, t2):
    str1 = '"('+str(t1[0])+','+str(t1[1])+','+str(t1[2])+')"'
    if t2 is not None:
        str2 = '"('+str(t2[0])+','+str(t2[1])+','+str(t2[2])+')"'
        f.write(str1 + '->' + str2 + '\n')
    else:
        f.write(str1 + '\n')

f = open('graph.dot','w')
f.write('digraph {\n')

operators = [i for i in list(itertools.product([0,1,2],[0,1,2])) if i[0]!=i[1]]
for state in space:
    for op in operators:
        drop_water(state, op[0], op[1])

f.write('}\n')
f.close()
