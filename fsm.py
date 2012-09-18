#!/usr/bin/python

def main():
	nodes = [(0,1),(2,3),(3,0),(4,2),(1,4)]

	for inp in range(1000):
		st = bin(inp)[2:]
		st = st[::-1]
		k = int(st[0])
		for i in st[1:]:
			k = nodes[k][int(i)]
		if inp%5==0 and k!=0:
			print ('Error number mod 5 == 0 not recognized')
		if k==0 and inp%5!=0:
			print ('Error number mod 5 != 0 recognized')
	
if __name__=='__main__':
	main()