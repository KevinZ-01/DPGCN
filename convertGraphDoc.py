import os
import networkx as nx
import pandas as pd

def graph_convert_write(data_dir, save_path):
    edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"], engine='python')
    edgelist["label"] = "cites"

    edgelist.sample(frac=1).head(5)

    Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
    nx.set_node_attributes(Gnx, "paper", "label")

    nx.write_graphml(Gnx, save_path)

graph_convert_write('D:/junior/科研/UIUC暑研/proj2/DP-GCN-master/data/cora', 'D:/junior/科研/UIUC暑研/proj2/DP-GCN-master/data/cora/cora.graphml')
