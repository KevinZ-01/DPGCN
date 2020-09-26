import os
import networkx as nx
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
a = {1:1}
print(1 in a)
g = nx.Graph()
g.add_edge(0,1)
g[0][1]['pro'] = 1
print(g[1][0]['pro'])