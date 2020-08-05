import numpy as np
from collections import Counter

def Adj2Pairs(adj,filename):
    f = open(filename, 'w')
    for i in range(len(adj)):
        for j in range(len(adj)):
            f.write(str(i)+'\t'+str(j)+'\n')
    f.close()

def Pairs2Adj(Pairspath):
    f = open(Pairspath,'r')
    line = f.readline()
    count  = 0
    array = []
    while line:
        wordlist = line.split()
        array.append(int(wordlist[0]))
        array.append(int(wordlist[1]))
        count = count+1
        line = f.readline()
    f.close()
    count = np.max(np.asarray(array))

    adj = np.zeros((count,count))
    f = open(Pairspath, 'r')
    line = f.readline()
    while line:
        wordlist = line.split()
        a = wordlist[0]
        b = wordlist[1]
        adj[int(a)-1,int(b)-1] = 1
        line = f.readline()
    f.close()
    #print(adj)

    return adj

