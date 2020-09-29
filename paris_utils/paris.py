# -*- coding: utf-8 -*-
#
#    Copyright (C) 2018 by
#    Thomas Bonald <thomas.bonald@telecom-paristech.fr>
#    Bertrand Charpentier <bertrand.charpentier@live.fr>
#    All rights reserved.
#    BSD license.

import numpy as np
import networkx as nx

def paris(G, copy_graph = True):
    """
    :param G: graph in nx.graph format
    :param copy_graph:
    :return: dendrogram after cluster
    """
    n = G.number_of_nodes()
    if copy_graph:
        F = G.copy()
    else:
        F = G
        
    # index nodes from 0 to n - 1
    if set(F.nodes()) != set(range(n)):
        F = nx.convert_node_labels_to_integers(F)
        
    # node weights
    w = {u: 0 for u in range(n)}
    wtot = 0
    for (u,v) in F.edges():
        if 'weight' not in F[u][v]:
            F[u][v]['weight'] = 1
        weight = F[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += weight
        if u != v:
            wtot += weight

    # cluster sizes
    s = {u: 1 for u in range(n)}
    
    # connected components
    cc = []
        
    # dendrogram as list of merges
    D = []
    
    # cluster index
    u = n 
    while n > 0:
        # nearest-neighbor chain
        chain = [list(F.nodes())[0]]
        while chain != []:
            a = chain.pop()
            # nearest neighbor 
            dmin = float("inf")
            b = -1
            for v in F.neighbors(a):
                if v != a:
                    d = w[v] * w[a] / float(F[a][v]['weight']) / float(wtot)
                    if d < dmin:
                        b = v
                        dmin = d
                    elif d == dmin:
                        b = min(b,v)
            d = dmin
            if chain != []:
                c = chain.pop()
                if b == c:
                    # calculate the probability
                    p = F[a][b]['weight'] / s[a] * s[b]
                    # merge a,b
                    D.append([a, b, d, s[a] + s[b], p])
                    # update graph
                    F.add_node(u)
                    neighbors_a = list(F.neighbors(a))
                    neighbors_b = list(F.neighbors(b))
                    for v in neighbors_a:
                        F.add_edge(u,v,weight = F[a][v]['weight'])
                    for v in neighbors_b:
                        if F.has_edge(u,v):
                            F[u][v]['weight'] += F[b][v]['weight']
                        else:
                            F.add_edge(u,v,weight = F[b][v]['weight'])
                    F.remove_node(a)
                    F.remove_node(b)
                    n -= 1
                    # update weight and size
                    w[u] = w.pop(a) + w.pop(b)
                    s[u] = s.pop(a) + s.pop(b)
                    # change cluster index
                    u += 1
                else:
                    chain.append(c)
                    chain.append(a)
                    chain.append(b)
            elif b >= 0:
                chain.append(a)
                chain.append(b)   
            else:
                # remove the connected component
                cc.append((a,s[a]))
                F.remove_node(a)
                w.pop(a)
                s.pop(a)
                n -= 1
    
    # add connected components to the dendrogram
    a,s = cc.pop()
    for b,t in cc:
        s += t
        D.append([a,b,float("inf"),s])
        a = u
        u += 1
        
    return reorder_dendrogram(np.array(D))

def private_paris(G, copy_graph = True, numofcluster = 1):
    """
        :param G: graph in nx.graph format
        :param copy_graph:
        :return: the cluster graph, the size of each node in dendrogram, the top nodes, the dendrogram
         after cluster, with differential privacy
        """
    n = G.number_of_nodes()
    N = n
    if copy_graph:
        F = G.copy()
    else:
        F = G

    # index nodes from 0 to n - 1
    if set(F.nodes()) != set(range(n)):
        F = nx.convert_node_labels_to_integers(F)

    # node weights
    w = {u: 0 for u in range(n)}
    wtot = 0
    for (u, v) in F.edges():
        if 'weight' not in F[u][v]:
            F[u][v]['weight'] = 1
        weight = F[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += weight
        if u != v:
            wtot += weight

    # cluster sizes
    s = {u: 1 for u in range(n)}

    # node sizes
    ss = {u: 1 for u in range(n)}

    #top nodes
    top = [i for i in range(n)]

    # connected components
    cc = []

    # dendrogram as list of merges, initialize with leaf nodes of the dendrogram
    # left , right, probability, left, right internals children, node number, parent, distance, size, left and right leaf children
    D = {u: [None, None, None, [], [], u, None, None, None, [], []] for u in range(n)}

    # scores of each cluster level
    if numofcluster == 1:
        scores = {i+1: 0 for i in range(n)}

    # cluster index
    u = n
    while (n > numofcluster and len(cc) == 0) or n > numofcluster - 1:  # when there is at least one connected
        # component, the times of merge should add one
        # nearest-neighbor chain, agglomerate two nodes if they are mutual nearest
        chain = [list(F.nodes())[0]]  # the first node
        while chain != []:
            a = chain.pop()
            # nearest neighbor
            dmin = float("inf")
            b = -1
            for v in F.neighbors(a):
                if v != a:
                    d = w[v] * w[a] / float(F[a][v]['weight']) / float(wtot)
                    if d < dmin:
                        b = v
                        dmin = d
                    elif d == dmin:
                        b = min(b, v)
            d = dmin
            if chain != []:
                c = chain.pop()
                if b == c:  # merge a,b
                    # calculate the probability
                    p = F[a][b]['weight'] / (s[a]*s[b])
                    # record the nodes under this merge
                    if a < N:
                        left_leaf = [a]
                        left = []
                    else:
                        left = D[a][3] + D[a][4] + [a]
                        left_leaf = D[a][9] + D[a][10]
                    if b < N:
                        right_leaf = [b]
                        right = []
                    else:
                        right = D[b][3] + D[b][4] + [b]
                        right_leaf = D[b][9] + D[b][10]
                    # merge a,b
                    D[u] = [a, b, p, left, right, u, None, d, s[a]+s[b], left_leaf, right_leaf]
                    D[a][6] = u
                    D[b][6] = u
                    # renew top nodes
                    top.remove(a)
                    top.remove(b)
                    top.append(u)

                    # update graph
                    F.add_node(u)
                    neighbors_a = list(F.neighbors(a))
                    neighbors_b = list(F.neighbors(b))
                    for v in neighbors_a:
                        F.add_edge(u, v, weight=F[a][v]['weight'])
                    for v in neighbors_b:
                        if F.has_edge(u, v):
                            F[u][v]['weight'] += F[b][v]['weight']
                        else:
                            F.add_edge(u, v, weight=F[b][v]['weight'])
                    F.remove_node(a)
                    F.remove_node(b)
                    n -= 1
                    # update weight, size and node size
                    w[u] = w.pop(a) + w.pop(b)
                    s[u] = s.pop(a) + s.pop(b)
                    ss[u] = s[u]
                    # change cluster index
                    u += 1
                    if not (n > numofcluster and len(cc) == 0) or n > numofcluster - 1:
                        break
                else:
                    chain.append(c)
                    chain.append(a)
                    chain.append(b)
            elif b >= 0:  # the chain is empty, and a has neighbour
                chain.append(a)
                chain.append(b)
            else: # a is a connected component
                # remove the connected component
                cc.append((a, s[a]))
                F.remove_node(a)
                top.remove(a)
                w.pop(a)
                s.pop(a)
                n -= 1
                if not (n > numofcluster and len(cc) == 0) or n > numofcluster - 1:
                    break
    if not len(cc) == 0:
        # add connected components to the dendrogram, all the connected components are combined to one cluster
        a, s = cc.pop()
        if len(cc) == 0:
            top.append(a)
        else:
            for b, t in cc:
                s += t
                D[u] = [a, b, 0, D[a][3]+D[a][4], D[b][3]+D[b][4], u, None, float("inf"), s, D[a][9]+D[a][10], D[b][9]+D[b][10]]
                ss[u] = s
                D[a][6] = u
                D[b][6] = u
                a = u
                u += 1
            top.append(a)
        F.add_node(a)
    if numofcluster == 1:
        return scores
    else:
        return [F, ss, top, D]

def reorder_dendrogram(D):
    n = np.shape(D)[0] + 1
    order = np.zeros((2,n - 1),float)
    order[0] = range(n - 1)
    order[1] = np.array(D)[:, 2] # sort by distance
    index = np.lexsort(order)
    nindex = {i:i for i in range(n)}
    nindex.update({n + index[t]:n + t for t in range(n - 1)})
    return np.array([[nindex[int(D[t][0])],nindex[int(D[t][1])],D[t][2],D[t][3]] for t in range(n - 1)])[index,:]
