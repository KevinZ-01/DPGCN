import os
import logging
import numpy as np
import scipy.sparse as sp
import time
import math
import torch
import random
import networkx as nx
import timeit
from utils import feature_reader, graph_reader, \
                normalize, sparse_mx_to_torch_sparse_tensor, AdjPairs
from paris_utils import paris, plot_best_clusterings, private_paris, plot_dendrogram, plot_private_dendrogram,\
                plot_k_clusterings

class Worker():
    def __init__(self, args, dataset=''):
        self.args = args
        self.dataset = dataset
        self.load_data()

    def load_data(self):
        self.features, self.labels, self.idx_train, self.idx_val, self.idx_test \
            = feature_reader(dataset=self.dataset, scale=self.args.scale,
                            train_ratio=self.args.train_ratio, feature_size=self.args.feature_size)

        # print('feature_size', self.features.shape)
        self.n_nodes = len(self.labels)
        self.n_features = self.features.shape[1]
        self.n_classes = self.labels.max().item() + 1

        self.edges = graph_reader(dataset=self.dataset)

        self.adj = self.build_adj_mat()

        # self.calculate_connectivity()

        if torch.cuda.is_available():
            self.features = self.features.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            if hasattr(self, 'prj'):
                self.prj = self.prj.cuda()

    def build_adj_mat(self):
        adj = self.build_adj_original()

        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def build_adj_original(self):
        adj = sp.coo_matrix((np.ones(self.edges.shape[0]), (self.edges[:, 0], self.edges[:, 1])),
                            shape=(self.n_nodes, self.n_nodes),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        return adj


class HRGWorker(Worker):
    def __init__(self, args, dataset=''):
        super(HRGWorker, self).__init__(args, dataset)

    def Pairs2Graph(self):
        pairsname = "output/" + self.dataset + "_random_graph_1.pairs"
        self.adj = AdjPairs.Pairs2Adj(pairsname)

    def Graph2Pairs(self):
        self.filename = 'data/' + self.dataset + '.pairs'
        AdjPairs.Adj2Pairs(self.adj, self.filename)

    def runSample(self):
        self.Graph2Pairs()
        command = "./privHRG -f "+ self.filename +" -epsilonHRG 0.5 -epsilonE 0.5 -eq " + str(self.args.eq) + \
                  " -stop " + str(self.args.stop)
        os.system(command)
        self.Pairs2Graph()


class GraphClusterWorker():
    def __init__(self, args, dataset=''):
        self.args = args
        self.dataset = dataset
        self.load_data()
        self.runCluster()
        if self.args.non_private:
            self.dentoadj()
        else:
           self.runSample()
           self.addLaplacian()
           self.dentoadj()

    def load_data(self):
        print('loading data...')
        self.features, self.labels, self.idx_train, self.idx_val, self.idx_test \
            = feature_reader(dataset=self.dataset, scale=self.args.scale,
                             train_ratio=self.args.train_ratio, feature_size=self.args.feature_size)

        # print('feature_size', self.features.shape)
        self.n_features = self.features.shape[1]
        self.n_classes = self.labels.max().item() + 1

        self.edges = graph_reader(dataset=self.dataset)
        # transform graph to nxnetwork
        self.G = nx.Graph()
        self.G.add_edges_from(self.edges)
        self.n = self.G.number_of_nodes()
        self.E = self.G.number_of_edges()  # total edges
        print('dataset load finish')
        print('number of nodes:', self.n)
        print('number of edges:', self.E)

    def runCluster(self):
        print('clustering...')
        time1 = timeit.timeit()
        if self.args.non_private == True:
            self.D = paris(self.G)
        else:
            if not self.args.num_cluster == 1:
                # the graph of top nodes, the size of each node in dendrogram, the top nodes, the dendrogram
                self.F, self.size, self.top, self.D = private_paris(self.G, True, self.args.num_cluster)
            else:
                scores = private_paris(self.G)
                num_cluster = np.argmax(np.asarray(scores)) + 1
                self.F, self.size, self.top, self.D = private_paris(self.G, True, num_cluster)
        time2 = timeit.timeit()
        print('clustering finished, time cost:', time2 - time1)
        self.numofcluster = 2*self.n - len(self.D)
        # plot and save the dendrogram
        #print('plotting clusterings and dendrogram...')
#        pos = nx.spring_layout(self.G)
#        plot_k_clusterings(self.G, self.D, self.top, pos, 'private_cluster.jpg')
        time3 = timeit.timeit()
#        plot_private_dendrogram(self.D, 'private_dendrogram')
        print('setting F and node_edge pairs...')
        # set up F: calculate probability
        for node1 in self.F.nodes():
            for node2 in self.F.nodes():
                if not self.F.has_edge(node1, node2):
                    self.F.add_edge(node1, node2)
                    self.F[node1][node2]['pro'] = 0
                    self.F[node1][node2]['weight'] = 0
                elif 'pro' not in self.F[node1][node2]:
                    self.F[node1][node2]['pro'] = self.F[node1][node2]['weight'] / (self.size[node1]*self.size[node2])
        # build node_edge pairs, that is related every edge of the graph to one node in the dendrogram,
        # or related to a pair of two top nodes. For each node in the dendrogram, see if the neighbour of each left children
        # is a node in the right children O(N*N)
        self.edge_node = {self.D[i][5]: [] for i in range(len(self.D))}
        for i in range(len(self.D)):
            node = self.D[i]
            for node1 in node[9]:
                for neighbour in self.G.neighbors(node1):
                    if neighbour in node[10]:
                        self.edge_node[node[5]].append((node1, neighbour))
        # add edges between clusters
        for i in range(self.numofcluster):
            for j in range(i+1, self.numofcluster):
                self.edge_node[(self.top[i], self.top[j])] = []
                for node in self.D[self.top[i]][9]+self.D[self.top[i]][10]:
                    for neighbour in self.G.neighbors(node):
                        if neighbour in self.D[self.top[j]][9]+self.D[self.top[j]][10]:
                            self.edge_node[(self.top[i], self.top[j])].append((node, neighbour))
        print('F and node_edge pairs set up finish, time cost:', timeit.timeit() - time3)

    def runSample(self):
        # calculate sensitivity
        self.sensitivity_cluster = 2 / self.args.min_cluster_size
        if self.args.utility_sample == 'ave_pro':
            self.sensitivity_sample = 2/self.E
        elif self.args.utility_sample == 'log likelihood':
            self.sensitivity_sample = 2*math.log(self.n) - math.log(4) + 1
        # set two threshold parameters to manually control convergence
        thresh_eq = np.max((self.args.eq, 100)) if self.args.eq else 100
        thresh_stop = np.max((self.args.stop, 3000)) if self.args.stop else 3000
        # parameters
        newMeanL = -1e49
        interval = 65536
        max_len = np.max((1e8, thresh_eq * self.n))
        trace = np.zeros(int(max_len))
        self.len = 0
        self.check = np.zeros(10000)
        check_num = 0
        self.score = 0

        convergence = 0
        t = 0
        print('Getting initial score...')
        t0 = time.time()
        ## get initial score
        if self.args.utility_sample == 'ave_pro':
            numerator = 0
            for i in range(len(self.D)):
                r = self.D[i]
                if r[5] >= self.n:
                    numerator += r[2] ** 2 * len(r[9]) * len(r[10])
            for (u,v) in self.F.edges():
                numerator += self.F[u][v]['pro']* 2 *self.size[u]*self.size[v]
            self.score = numerator / self.E
        elif self.args.utility_sample == 'log likelihood':
            for i in range(len(self.D)):
                r = self.D[i]
                if r[5] >= self.n and not r[2] == 0:
                    self.score += -len(r[3]) * len(r[4]) * (r[2] * math.log(r[2]) + (1 - r[2]) * math.log(1 - r[2]))
                # add top nodes log likelihoods
            for (u, v) in self.F.edges():
                if not self.F[u][v]['pro'] == 0:
                    self.score += -(len(self.D[u][3])+len(self.D[u][4]) )* (len(self.D[v][3]+len(self.D[v][4]))) * (
                                self.F[u][v]['pro'] * math.log(self.F[u][v]['pro'])
                                + (1 - self.F[u][v]['pro']) * math.log(1 - self.F[u][v]['pro']))
        print('Start samping...')
        while convergence == 0:
            oldMeanL = newMeanL
            newMeanL = 0
            for i in range(interval):
                # choose 2 nodes in the dendrogram to change position
                nodes = random.sample(range(0, 2 * self.n - self.numofcluster), 2)
                p = self.D[nodes[0]][6]
                top1 = nodes[0]
                while p is not None:
                    top1 = p
                    p = self.D[top1][6]
                p = self.D[nodes[1]][6]
                top2 = nodes[1]
                while p is not None:
                    top2 = p
                    p = self.D[top2][6]
                # to make sure two nodes are from different clusters
                while top1 == top2:
                    nodes = random.sample(range(0, 2 * self.n - self.numofcluster), 2)
                    p = self.D[nodes[0]][6]
                    top1 = nodes[0]
                    while p is not None:
                        top1 = p
                        p = self.D[top1][6]
                    p = self.D[nodes[1]][6]
                    top2 = nodes[1]
                    while p is not None:
                        top2 = p
                        p = self.D[top2][6]
#                print(self.score)
                self.exchange(nodes)
                newMeanL += self.score

                if (self.len < max_len):
                    trace[self.len] = self.score
                    self.len = self.len + 1
                t = t + 1

                if t % 1000 == 0:
                    print('1e3 MCMC moves take {}s'.format(time.time() - t0), 'score: ', self.score)
                    t0 = time.time()

            self.check[check_num] = np.abs(newMeanL - oldMeanL) / interval
            print('newMean-oldMean:{}'.format(self.check[check_num]))
            check_num = check_num + 1
            judge_convergence = self.judge_converge()

            if (judge_convergence and (t > thresh_eq * self.n)):
                convergence = 1
            elif t > thresh_stop * self.n or t >= max_len:
                convergence = 1

    def exchange(self, nodes):
        """
        :param nodes: nodes to be changed
        :return:
        """
        len_edge_node = 0
        for key in self.edge_node:
            len_edge_node += len(self.edge_node[key])
        print(len(self.edge_node), len_edge_node)
        if nodes[0] in self.top and nodes[1] in self.top:
            # top nodes change will not affect the utility function
            return
        elif nodes[0] in self.top or nodes[1] in self.top:  # one node is top node
            new_record = {}  # a dict that record the renewed probability and the renewed left and right length
            # of relative parent nodes
            local_edge_node = {}  # a dict that record the renewed edge_node pairs
            if nodes[0] in self.top:
                a = 1
                b = 0
            else:
                a = 0
                b = 1
            len_a = np.max((len(self.D[nodes[a]][9]) + len(self.D[nodes[a]][10]), 1))
            len_b = np.max((len(self.D[nodes[b]][9]) + len(self.D[nodes[b]][10]), 1))
            ## renew the dendrogram
            #find all the parents of nodes[a], O(logN)
            parents = []
            this_top = 0
            p = self.D[nodes[a]][6]
            while not p==None:
                parents.append(p)
                this_top = p
                p = self.D[p][6]
            ############################################################
            # renew left and right leaf children, O(logN*logN)
            remove = self.D[nodes[a]][9] + self.D[nodes[a]][10]
            if not remove:
                remove.append(nodes[a])
            add = self.D[nodes[b]][9] + self.D[nodes[b]][10]
            if not add:
                add.append(nodes[b])
            child = nodes[a]
            for p in parents:
                if child == self.D[p][0]:
                    new_record[p] = [0, len(self.D[p][9])-len(remove)+len(add), len(self.D[p][10])]
                else:
                    new_record[p] = [0, len(self.D[p][9]), len(self.D[p][10]) - len(remove)+len(add)]
                child = p
            ##############################################################
            # renew the probabilities of nodes O(logN*logN)
            new_key = (np.min((this_top, nodes[b])), np.max((this_top, nodes[b])))
            for p in parents:
                pro, new_edge_node = self.calc_prob(new_record[p][1:3], p, nodes[b], nodes[a], new_key)
                new_record[p][0] = pro
                local_edge_node[p] = new_edge_node

            #renew the probabilities between clusters on new_record O(N)
            for node in self.F.nodes():
                if not (node==nodes[a] or node == this_top or node == nodes[b]):
                    key1 = (np.min((node, nodes[a])), np.max((node, nodes[a])))
                    pro, new_edge_node = self.calc_cluster_prob([len(self.D[nodes[a]][9]), len(self.D[nodes[a]][10])],
                                                                [node, nodes[a]], None, nodes[a], this_top)
                    local_edge_node[key1] = new_edge_node
                    new_record[(node, nodes[a])] = pro

                    key1 = (np.min((node, this_top)), np.max((node, this_top)))
                    pro, new_edge_node = self.calc_cluster_prob(new_record[this_top][1:3],[node, this_top],
                                                                nodes[a], nodes[b], nodes[b])
                    local_edge_node[key1] = new_edge_node
                    new_record[(node, this_top)] = pro
            # calculate the probabilities of (this_top, nodes[a])
            total = len_a * (new_record[this_top][1]+new_record[this_top][2])
            new_edge_node = []
            key = (np.min((this_top, nodes[b])), np.max((nodes[b], this_top)))
            for (u, v) in self.edge_node[key]:
                if (u in self.D[nodes[a]][9] + self.D[nodes[a]][10] and v in self.D[nodes[b]][9] + self.D[nodes[b]][10]) or \
                        (v in self.D[nodes[a]][9] + self.D[nodes[a]][10] and u in self.D[nodes[b]][9] + self.D[nodes[b]][10]):
                    new_edge_node.append((u, v))
            for node in remove:
                for neighbour in self.G.neighbors(node):
                    if neighbour in self.D[nodes[a]][9]+self.D[nodes[a]][10]:
                        new_edge_node.append((node, neighbour))
            key = (np.min((this_top, nodes[a])), np.max((nodes[a], this_top)))
            local_edge_node[key] = new_edge_node
            num_edge = len(new_edge_node)
            new_record[key] = num_edge/total
            #################################################################
            #calculate the utility function O(logN)
            if self.args.utility_sample == 'ave_pro':
                for p in parents:
                    after_score = self.score - \
                                  (self.D[p][2] ** 2 * len(self.D[p][9])*len(self.D[p][10]) -
                                   new_record[p][0]** 2 * new_record[p][1]*new_record[p][2])/self.E

                for node in self.F.nodes():
                    if not (node == nodes[b] or node == this_top):
                        len_node = np.max((len(self.D[node][9]) + len(self.D[node][10]), 1))
                        after_score += (new_record[(node, nodes[a])] ** 2 * len_node * len_a - \
                                       self.F[node][nodes[b]]['pro'] ** 2 * len_node * len_b)/self.E

                        after_score += (new_record[(node, this_top)] ** 2 * len_node * (new_record[this_top][1] + new_record[this_top][2]) - \
                                       self.F[node][this_top]['pro'] ** 2 * len_node * (len(self.D[this_top][9]) + len(self.D[this_top][10]))) / self.E
                # calculate the edges between this top and nodes[a], at the same time minus this top and nodes[b]
                after_score += (new_record[key] ** 2 * (new_record[this_top][1] + new_record[this_top][2]) * len_a - \
                               self.F[this_top][nodes[b]]['pro'] ** 2 * (len(self.D[this_top][9])+len(self.D[this_top][10])) * len_b) / self.E

            elif self.args.utility_sample == 'log likelihood':
                for p in parents:
                    after_score = self.score - \
                                  len(self.D[p][9])*len(self.D[p][10])* (self.D[p][2] * math.log(self.D[p][2]) + (1 - self.D[p][2]) * math.log(1 - self.D[p][2])) + \
                                  len(new_record[p][2])*len(new_record[p][1])* (new_record[p][0] * math.log(new_record[p][0]) + (1 - new_record[p][0]) * math.log(1 - new_record[p][0]))
                # calculate the between-cluster information entropy that has been changed
                for node in self.F.nodes():
                    if not (node == nodes[a] or node == this_top):
                        len_node = np.max((len(self.D[node][9])+len(self.D[node][10]), 1))
                        pro_node_a = new_record[(node, nodes[a])]
                        pro_node_b = self.F[node][nodes[b]]['pro']
                        after_score += \
                            len_node*len_a*(pro_node_a * math.log(pro_node_a) + (1 - pro_node_a) * math.log(1 - pro_node_a)) - \
                            len_node*len_a*(pro_node_b * math.log(pro_node_b) + (1 - pro_node_b) * math.log(1 - pro_node_b))
                        pro_node_thistop = new_record[(node, this_top)]
                        pro_node_this_before = self.F[node][this_top]['pro']
                        after_score += \
                            len_node * (new_record[this_top][1]+new_record[this_top][2]) * \
                            (pro_node_thistop * math.log(pro_node_thistop) + (1 - pro_node_thistop) * math.log( 1 - pro_node_thistop)) - \
                            len_node * (len(self.D[this_top][9]) + len(self.D[this_top][10])) * \
                            (pro_node_this_before * math.log(pro_node_this_before) + (1 - pro_node_this_before) * math.log(1 - pro_node_this_before))
                # calculate the score between (thistop, nodes[a])
                pro_this_b = self.F[this_top][nodes[b]]['pro']
                after_score += \
                            (new_record[this_top][1]+new_record[this_top][2])*len_a*\
                            (new_record[key] * math.log(new_record[key]) + (1 - new_record[key]) * math.log(1 - new_record[key])) - \
                            (len(self.D[this_top][9])+len(self.D[this_top][10]))*len_b*\
                            (pro_this_b * math.log(pro_this_b) + (1 - pro_this_b) * math.log(1 - pro_this_b))
            # probability to exchange these two nodes O(1)
            prob = math.exp(self.args.epsilon2*(after_score-self.score)/self.sensitivity_sample)
            if random.uniform(0, 1) < prob:
                self.score = after_score
                # renew the children of all the relative nodes
                remove = self.D[nodes[a]][3] + self.D[nodes[a]][4]
                if nodes[a] >= self.n:
                    remove.append(nodes[a])
                add = self.D[nodes[b]][3] + self.D[nodes[b]][4]
                if nodes[b] >= self.n:
                    add.append(nodes[b])
                child = nodes[a]
                for p in parents:
                    if child == self.D[p][0]:
                        left_right = 3
                    else:
                        left_right = 4

                    self.D[p][left_right] = [e for e in self.D[p][left_right] if e not in remove]
                    self.D[p][left_right] += add
                    child = p
                # renew the leaf children of all the relative nodes
                remove = self.D[nodes[a]][9] + self.D[nodes[a]][10]
                if not remove:
                    remove.append(nodes[a])
                add = self.D[nodes[b]][9] + self.D[nodes[b]][10]
                if not add:
                    add.append(nodes[b])
                child = nodes[a]
                for p in parents:
                    if child == self.D[p][0]:
                        left_right = 9
                    else:
                        left_right = 10

                    self.D[p][left_right] = [e for e in self.D[p][left_right] if e not in remove]
                    self.D[p][left_right] += add
                    child = p
                #################################################################################
                # renew the probability of nodes in dendrogram, and the edge_node pairs
                for p in parents:
                    self.D[p][2] = new_record[p][0]
                    self.edge_node[p] = local_edge_node[p]
                # renew the probability between clusters
                self.F.remove_node(nodes[b])
                self.F.add_node(nodes[a])
                for node in self.F.nodes():
                    if not (node == this_top or node == nodes[a]):
                        key1 = (np.min((node, this_top)), np.max((node, this_top)))
                        self.F[node][this_top]['pro'] = new_record[(node, this_top)]
                        self.edge_node[key1] = local_edge_node[key1]

                        key1 = (np.min((node, nodes[a])), np.max((node, nodes[a])))
                        self.F[node][nodes[a]]['pro'] = new_record[(node, nodes[a])]
                        self.edge_node[key1] = local_edge_node[key1]

                self.F[this_top][nodes[a]]['pro'] = new_record[key]
                self.edge_node[key] = local_edge_node[key]
                # delete the edge_node pairs that relative to nodes[b]
                for key in self.edge_node:
                    if key is not nodes[b] and nodes[b] in key:
                        self.edge_node.pop(key)
                # renew the left/right direct child of the parent of the non-top node
                p = self.D[nodes[a]][6]
                if not p == None:
                    if self.D[p][0] == nodes[a]:
                        self.D[p][0] = nodes[b]
                    elif self.D[p][1] == nodes[a]:
                        self.D[p][1] = nodes[b]
                self.D[nodes[a]][6] = None
                self.D[nodes[b]][6] = p
                    ####################################################################
        else:  # both the nodes are not top nodes
            new_record = {}  # a dict that record the renewed probability and the renewd left and right length
            # of relative parent nodes
            local_edge_node = {} # a dict record the renewed edge_node pairs
            ## renew the dendrogram
            # find all the parents of node[1] and node[2]
            parents1 = []
            parents2 = []
            p = self.D[nodes[0]][6]
            while not p == None:
                parents1.append(p)
                top1 = p
                p = self.D[p][6]
            p = self.D[nodes[1]][6]
            while not p is None:
                parents2.append(p)
                top2 = p
                p = self.D[p][6]
            #############################################################################
            # renew left and right leaf child nodes
            remove = self.D[nodes[0]][9] + self.D[nodes[0]][10]
            if nodes[0] < self.n:
                remove.append(nodes[0])
            add = self.D[nodes[1]][9] + self.D[nodes[1]][10]
            if nodes[1] < self.n:
                add.append(nodes[1])
            child = nodes[0]
            for p in parents1:
                if child == self.D[p][0]:
                    new_record[p] = [0, len(self.D[p][9]) - len(remove) + len(add), len(self.D[p][10])]
                else:
                    new_record[p] = [0, len(self.D[p][9]), len(self.D[p][10]) - len(remove) + len(add)]
                child = p

            remove = self.D[nodes[1]][9] + self.D[nodes[1]][10]
            if nodes[1] < self.n:
                remove.append(nodes[1])
            add = self.D[nodes[0]][9] + self.D[nodes[0]][10]
            if nodes[0] < self.n:
                add.append(nodes[0])
            child = nodes[1]
#            print(len(self.D[5293][10]))
            for p in parents2:
                if child == self.D[p][0]:
                    new_record[p] = [0, len(self.D[p][9]) - len(remove) + len(add), len(self.D[p][10])]
                else:
                    new_record[p] = [0, len(self.D[p][9]), len(self.D[p][10]) - len(remove) + len(add)]
                child = p
            ############################################################################
            # renew the probabilities of nodes
            new_key = (np.min((top1, top2)), np.max((top1, top2)))
            for p in parents1:
                # find the key of edge_node from which the new edges come from
                pro, new_edge_node = self.calc_prob(new_record[p][1:3], p, nodes[0], nodes[1], new_key)
                local_edge_node[p] = new_edge_node
                new_record[p][0] = pro
            for p in parents2:
                # find the key of edge_node from which the new edges come from
                pro, new_edge_node = self.calc_prob(new_record[p][1:3], p, nodes[1], nodes[0], new_key)
                local_edge_node[p] = new_edge_node
                new_record[p][0] = pro
            # renew the probabilities between clusters
            for node in self.F.nodes():
                if not (node == top1 or node == top2):
                    key1 = (np.min((node, top1)), np.max((node, top1)))
                    pro, new_edge_node = self.calc_cluster_prob(new_record[top1][1:3], [node, top1], nodes[0], nodes[1], top2)
                    local_edge_node[key1] = new_edge_node
                    new_record[(node, top1)] = pro

                    key1 = (np.min((node, top2)), np.max((node, top2)))
                    pro, new_edge_node = self.calc_cluster_prob(new_record[top2][1:3], [node, top2], nodes[1], nodes[0], top1)
                    local_edge_node[key1] = new_edge_node
                    new_record[(node, top2)] = pro
            # renew the probability between top1 and top2
            pro, new_edge_node = self.calc_tops_pro(top1, top2, nodes[0], nodes[1], new_record[top1][1:3], new_record[top2][1:3])
            local_edge_node[new_key] = new_edge_node
            new_record[(top1, top2)] = pro
            #####################################################################################
            # calculate the utility function
            if self.args.utility_sample == 'ave_pro':
                for p in parents1 + parents2:
#                    print('before: ', self.D[p][2], len(self.D[p][9]) * len(self.D[p][10]), 'after: ', new_record[p][0], new_record[p][1] * new_record[p][2])
                    after_score = self.score - \
                                  (self.D[p][2] ** 2 * len(self.D[p][9]) * len(self.D[p][10]) -
                                   new_record[p][0] ** 2 * new_record[p][1] * new_record[p][2]) / self.E
#                print('score diff1: ', after_score - self.score)
                for node in self.F.nodes():
                    if not (node == top1 or node == top2):
                        len_node = np.max((len(self.D[node][9]) + len(self.D[node][10]), 1))
                        after_score += (new_record[(node, top1)] ** 2 * len_node * (
                                               new_record[top1][1] + new_record[top1][2]) - \
                                       self.F[node][top1]['pro'] ** 2 * len_node * (
                                               len(self.D[top1][9]) + len(self.D[top1][10]))) / self.E

                        after_score += (new_record[(node, top2)] ** 2 * len_node * (
                                               new_record[top2][1] + new_record[top2][2]) - \
                                       self.F[node][top2]['pro'] ** 2 * len_node * (
                                               len(self.D[top2][9]) + len(self.D[top2][10]))) / self.E
                # calculate probability of (top1, top2)
                after_score += (new_record[(top1, top2)] ** 2 *
                               (new_record[top1][1] + new_record[top1][2]) * (
                                       new_record[top2][1] + new_record[top2][2]) -
                               self.F[top1][top2]['pro'] ** 2 * (
                                       len(self.D[top2][9]) + len(self.D[top2][10])) * (
                                       len(self.D[top1][9]) + len(self.D[top1][10]))) / self.E

            elif self.args.utility_sample == 'log likelihood':
                for p in parents1 + parents2:
                    after_score = self.score - \
                                  len(self.D[p][9]) * len(self.D[p][10]) * (
                                          self.D[p][2] * math.log(self.D[p][2]) + (1 - self.D[p][2]) * math.log(
                                      1 - self.D[p][2])) + \
                                  new_record[p][1] * new_record[p][2] * (
                                          new_record[p][0] * math.log(new_record[p][0]) + (
                                              1 - new_record[p][0]) * math.log(
                                      1 - new_record[p][0]))
                # calculate the between-cluster information entropy that has been changed
                for node in self.F.nodes():
                    if not (node == top1 or node == top2):
                        len_node = np.max((len(self.D[node][9]) + len(self.D[node][10]), 1))
                        after_score += \
                            len_node * (new_record[top1][1] + new_record[top1][2]) * \
                            (new_record[(node, top1)] * math.log(new_record[(node, top1)]) + (
                                    1 - new_record[(node, top1)]) * math.log(
                                1 - new_record[(node, top1)])) - \
                            len_node * (len(self.D[top1][9]) + len(self.D[top1][10])) * \
                            (self.F[node][top1]['pro'] * math.log(self.F[node][top1]['pro']) + (
                                    1 - self.F[node][top1]['pro']) * math.log(
                                1 - self.F[node][top1]['pro']))

                        after_score += \
                            len_node * (new_record[top2][1] + new_record[top2][2]) * \
                            (new_record[(node, top2)] * math.log(new_record[(node, top2)]) + (
                                    1 - new_record[(node, top2)]) * math.log(
                                1 - new_record[(node, top2)])) - \
                            len_node * (len(self.D[top2][9]) + len(self.D[top2][10])) * \
                            (self.F[node][top2]['pro'] * math.log(self.F[node][top2]['pro']) + (
                                    1 - self.F[node][top2]['pro']) *
                             math.log(1 - self.F[node][top2]['pro']))
                # calculate probability of (top1, top2)
                after_score += \
                    (new_record[top1][1] + new_record[top1][2]) * (
                            new_record[top2][1] + new_record[top2][2]) * \
                    (new_record[(top1, top2)] * math.log(new_record[(top1, top2)]) + (
                            1 - new_record[(top1, top2)]) * math.log(
                        1 - new_record[(top1, top2)])) - \
                    (len(self.D[top2][9]) + len(self.D[top2][10])) * (
                            len(self.D[top1][9]) + len(self.D[top1][10])) * \
                    (self.F[top1][top2]['pro'] * math.log(self.F[top1][top2]['pro']) + (
                            1 - self.F[top1][top2]['pro']) * math.log(
                        1 - self.F[top1][top2]['pro']))
            # probability to exchange these two nodes
            prob = math.exp(self.args.epsilon2 * (after_score - self.score) / self.sensitivity_sample)
            #print(prob)
            if random.uniform(0, 1) < prob:
                ## renew the dendrogram and F, and edge_node pairs
                self.score = after_score
                ##################################################################
                # renew the non-leaf children of all the parents of two nodes
                remove = self.D[nodes[0]][3] + self.D[nodes[0]][4]
                if nodes[0] >= self.n:
                    remove.append(nodes[0])
                add = self.D[nodes[1]][3] + self.D[nodes[1]][4]
                if nodes[1] >= self.n:
                    add.append(nodes[1])
                child = nodes[0]
                for p in parents1:
                    if child == self.D[p][0]:
                        left_right = 3
                    else:
                        left_right = 4

                    self.D[p][left_right] = [e for e in self.D[p][left_right] if e not in remove]
                    self.D[p][left_right] += add
                    child = p

                remove = self.D[nodes[1]][3] + self.D[nodes[1]][4]
                if nodes[1] >= self.n:
                    remove.append(nodes[1])
                add = self.D[nodes[0]][3] + self.D[nodes[0]][4]
                if nodes[0] >= self.n:
                    add.append(nodes[0])
                child = nodes[1]
                for p in parents2:
                    if child == self.D[p][0]:
                        left_right = 3
                    else:
                        left_right = 4

                    self.D[p][left_right] = [e for e in self.D[p][left_right] if e not in remove]
                    self.D[p][left_right] += add
                    child = p

                # renew the leaf children of all the relative nodes
                remove = self.D[nodes[0]][9] + self.D[nodes[0]][10]
                if nodes[0] < self.n:
                    remove.append(nodes[0])
                add = self.D[nodes[1]][9] + self.D[nodes[1]][10]
                if nodes[1] < self.n:
                    add.append(nodes[1])
                child = nodes[0]
                for p in parents1:
                    if child == self.D[p][0]:
                        left_right = 9
                    else:
                        left_right = 10

                    self.D[p][left_right] = [e for e in self.D[p][left_right] if e not in remove]
                    self.D[p][left_right] += add
                    child = p

                remove = self.D[nodes[1]][9] + self.D[nodes[1]][10]
                if nodes[0] < self.n:
                    remove.append(nodes[1])
                add = self.D[nodes[0]][9] + self.D[nodes[0]][10]
                if nodes[0] < self.n:
                    add.append(nodes[0])
                child = nodes[1]
                for p in parents2:
                    if child == self.D[p][0]:
                        left_right = 9
                    else:
                        left_right = 10

                    self.D[p][left_right] = [e for e in self.D[p][left_right] if e not in remove]
                    self.D[p][left_right] += add
                    child = p
                ###################################################################
                # renew the left/right direct child
                p1 = self.D[nodes[0]][6]
                if not p1 is None:
                    if self.D[p1][0] == nodes[0]:
                        self.D[p1][0] = nodes[1]
                    elif self.D[p1][1] == nodes[0]:
                        self.D[p1][1] = nodes[1]
                p2 = self.D[nodes[1]][6]
                if not p2 is None:
                    if self.D[p2][0] == nodes[1]:
                        self.D[p2][0] = nodes[0]
                    elif self.D[p2][1] == nodes[1]:
                        self.D[p2][1] = nodes[0]
                self.D[nodes[0]][6] = p2
                self.D[nodes[1]][6] = p1
                ###################################################################
                # renew the probability of nodes in dendrogram, and the edge_node pairs
                for p in parents1 + parents2:
                    self.D[p][2] = new_record[p][0]
                    self.edge_node[p] = local_edge_node[p]
                # renew the probability between clusters
                for node in self.F.nodes():
                    if not (node == top1 or node == top2):
                        key1 = (np.min((node, top1)), np.max((node, top1)))
                        self.F[node][top1]['pro'] = new_record[(node, top1)]
                        self.edge_node[key1] = local_edge_node[key1]

                        key1 = (np.min((node, top2)), np.max((node, top2)))
                        self.F[node][top2]['pro'] = new_record[(node, top2)]
                        self.edge_node[key1] = local_edge_node[key1]
                self.F[top1][top2]['pro'] = new_record[(top1, top2)]
                self.edge_node[new_key] = local_edge_node[new_key]
                ####################################################################

    def calc_prob(self, new_len, node, child_remove, child_add, new_key):
        """
        :param D: Dendrogram
        :param node: node to be calculated
        :param child_remove: child code removed from the node
        :param child_add: child code added to the code
        :param new_key: the index of where the new edge come from, in (node1, node2) form
        :return: probability
        """
        total = new_len[0] * new_len[1]
        ## calculate edges
        if child_remove in self.D[node][9] or child_remove in self.D[node][3]:
            left_right = 10
        else:
            left_right = 9
        # delete edges from edge_node
        to_remove = []
#        print(len(self.edge_node[node]))
        for (u, v) in self.edge_node[node]:  # O(logN)
            if (u in self.D[child_remove][9] + self.D[child_remove][10] + [child_remove] and v in self.D[node][left_right]) or \
                    (v in self.D[child_remove][9] + self.D[child_remove][10] + [child_remove] and u in self.D[node][left_right]):
                to_remove.append((u, v))
        edge_node = [edge for edge in self.edge_node[node] if edge not in to_remove]
        # add edges brought from new node
#        print(len(self.edge_node[new_key]))
        for (u, v) in self.edge_node[new_key]:
            if (u in self.D[child_add][9] + self.D[child_add][10] + [child_add] and v in self.D[node][left_right]) or \
                    (v in self.D[child_add][9] + self.D[child_add][10] + [child_add] and u in self.D[node][left_right]):
                edge_node.append((u, v))
        num_edge = len(edge_node)
        return num_edge / total, edge_node

    def calc_cluster_prob(self, new_len, nodes, child_remove, child_add, origin_top):
        """
        :param nodes: two top nodes, the last one is the one who changed
        :param child_remove: child that removed from the first cluster
        :param child_add: child that added to the first cluster
        :return: probability
        """
        children1 = self.D[nodes[0]][9] + self.D[nodes[0]][10]
        if not children1:
            children1.append(nodes[0])
        if new_len[0] + new_len[1] == 0:
            total = len(children1)
        else:
            total = len(children1) * (new_len[0] + new_len[1])
        # delete edges
        to_remove = []
        if child_remove is not None:
            key1 = (np.min((nodes[0], nodes[1])), np.max((nodes[0], nodes[1])))
            for (u, v) in self.edge_node[key1]:
                if (u in self.D[child_remove][9] + self.D[child_remove][10] + [child_remove] and v in children1) or \
                        (v in self.D[child_remove][9] + self.D[child_remove][10] + [child_remove] and u in children1):
                    to_remove.append((u, v))
            edge_node = [edge for edge in self.edge_node[key1] if edge not in to_remove]
        else:
            edge_node = []
        # add edges
        key = (np.min((nodes[0], origin_top)), np.max((nodes[0], origin_top)))
#        print(len(self.edge_node[key]))
        for (u, v) in self.edge_node[key]:
            if (u in self.D[child_add][9] + self.D[child_add][10] + [child_add] and v in children1) or \
                    (v in self.D[child_add][9] + self.D[child_add][10] + [child_add] and u in children1):
                edge_node.append((u, v))
        num_edge = len(edge_node)
        return num_edge / total, edge_node

    def calc_tops_pro(self, top1, top2, node1, node2, new_rec_top1, new_rec_top2):
        total = (new_rec_top1[0]+new_rec_top1[1])*(new_rec_top2[0]+new_rec_top2[1])
        key = (np.min((top1, top2)), np.max((top1, top2)))
        remove = []
        for (u,v) in self.edge_node[key]:
            if (u in self.D[node1][9]+self.D[node1][10] and v in self.D[top2][9]+self.D[top2][10] and v not in self.D[node2][9]+self.D[node2][10]) \
                or (v in self.D[node1][9]+self.D[node1][10] and u in self.D[top2][9]+self.D[top2][10] and u not in self.D[node2][9]+self.D[node2][10]) \
                or (u in self.D[node2][9]+self.D[node2][10] and v in self.D[top1][9]+self.D[top1][10] and v not in self.D[node1][9]+self.D[node1][10]) \
                or(v in self.D[node2][9]+self.D[node2][10] and u in self.D[top1][9]+self.D[top1][10] and u not in self.D[node1][9]+self.D[node1][10]):
                remove.append((u,v))
        edge_node = [e for e in self.edge_node[key] if e not in remove]

        for node in self.D[node1][9]+self.D[node1][10]:
            for neighbour in self.G.neighbors(node):
                if neighbour in self.D[top1][9]+self.D[top2][10]:
                    edge_node.append((node,neighbour))
        for node in self.D[node2][9]+self.D[node2][10]:
            for neighbour in self.G.neighbors(node):
                if neighbour in self.D[top2][9]+self.D[top2][10]:
                    edge_node.append((node,neighbour))
        num_edge = len(edge_node)
        return num_edge / total, edge_node

    def judge_converge(self):
        stationary = 0
        thresh = 10
        non_stationary = 0
        if self.len >= thresh:
            for i in range(thresh):
                if self.check[self.len - i - 1] < 0.02 * self.n:
                    stationary = stationary + 1
                if self.check[self.len - i - 1] > 0.05 * self.n:
                    non_stationary = non_stationary + 1

            if non_stationary <= 0 and stationary >= thresh * 0.8:
                return 1
            else:
                return 0
        else:
            return 0

    def addLaplacian(self):
        # python implementation of Erd˝os-R´enyi model
        epsilon = self.args.epsilon3
        for top_node in self.top:
            self.calculateNoisyProb(top_node, epsilon)

    def calculateNoisyProb(self, node, epsilon):
        lambda_b = 1 / (epsilon * len(self.D[node][3]) * len(self.D[node][4]))
        lambda_c = 2 / (epsilon * len(self.D[node][3] + self.D[node][4]) * len(self.D[node][3] + self.D[node][4] - 1))
        if lambda_b > self.args.pi1 and lambda_c > self.args.pi2:
            edges_below = self.calculate_edge_below(node)
            noisy_prob = edges_below + np.random.laplace(scale=epsilon)
            noisy_prob = np.clip(noisy_prob, 0, 1)
            for child in self.D[node][3] + self.D[node][4]:
                if child >= self.n:
                    self.D[child][2] = noisy_prob
        else:
            noisy_prob = self.D[node][2] + np.random.laplace(scale=epsilon) / (
                        len(self.D[node][3]) * len(self.D[node][4]))
            self.D[node][2] = np.clip(noisy_prob, 0, 1)
            self.calculateNoisyProb(self.D[node][0], epsilon)
            self.calculateNoisyProb(self.D[node][1], epsilon)

    def calculate_edge_below(self, node):
        # calculate how many edges are under the node
        if node < self.n:
            return 0
        else:
            return self.D[node][2]*len(self.D[node][3])*len(self.D[node][4]) +\
                    self.calculate_edge_below( self.D[node][0]) + self.calculate_edge_below(self.D[node][1])

    def dentoadj(self):
        self.adj = sp.coo_matrix(0, shape=(self.n, self.n), dtype=np.float32)
        for i in range(np.asarray(self.D).shape[0]):
            p = self.D[i][2]
            for j in self.D[i][3]:
                for k in self.D[i][4]:
                    self.adj[j][k] = 1 if np.random.random() < p else 0

        self.adj = self.adj + self.adj.T


class graph_sampleWorker(Worker):
    def __init__(self, args, dataset=''):
        super(graph_sampleWorker, self).__init__(args, dataset)

    def sample_graph(self):
        """
        use M-H sampling, minimum neighborhood method to sample a graph
        """
        x = random.randint(0, self.N-1)
        y = random.randint(0, self.N-1)
        adj_before = self.adj

        previous_score = self.GetScore()
        self.adj[x,y] = 1 if self.adj[x,y] == 0 else 0
        after_score = self.GetScore()

        if np.random.uniform(0,1) > min(1, math.exp((after_score-previous_score))/2):
            self.adj = adj_before

    def judge_converge(self):
        stationary = 0
        thresh = 10
        non_stationary = 0
        if self.len >= thresh:
            for i in range(thresh):
                if self.check[self.len-i-1]< 0.02*self.N:
                    stationary = stationary+1
                if self.check[self.len-i-1]> 0.05*self.N:
                    non_stationary = non_stationary+1

            if non_stationary<=0 and stationary>= thresh*0.8:
                return 1
            else:
                return 0
        else:
            return 0

    def GetScore(self):
        if self.args.loss_sample == 'A':
            return -np.sum(np.abs(self.adj-self.realadj))
        elif self.args.loss_sample == 'logA':
            loss_A = np.sum(np.abs(self.adj - self.realadj))
            if loss_A==0:
                return 1
            else:
                return -math.log(loss_A)
        elif self.args.loss_sample == 'AX':
            self.AHat = self.normalization(self.adj)
            return -np.sum(np.abs(np.matmul(self.AHat, self.features)-self.AHatX))
        elif self.args.loss_sample == 'logAX':
            loss_AX = np.sum(np.abs(np.matmul(self.AHat, self.features)-self.AHatX))
            if loss_AX == 0:
                return 1
            else:
                return -math.log(loss_AX)

    def normalization(self, matrix):
        ret = matrix + np.diag(self.N)
        rowsum = np.array(ret.sum(1)) * 1.0
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(ret)

        return mx

    def run(self):
        #set two threshold parameters to manually control convergence
        thresh_eq = np.max(self.args.eq, 1000) if self.args.eq else 1000
        thresh_stop = np.max(self.args.stop, 3000) if self.args.stop else 3000
        #parameters
        self.realadj = self.adj
        newMeanL = -1e49
        interval = 65536
        self.N = self.adj.shape[0]
        self.realAHat = self.normalization(self.adj)
        self.AHatX = np.matmul(self.realAHat, self.features)
        max_len = np.max(1e8, thresh_eq*self.N)
        trace = np.zeros(max_len)
        self.len = 0
        self.check = np.zeros((10000))
        check_num = 0

        convergence = 0
        t=0
        t0 = time.time()
        while(convergence == 0):
            oldMeanL = newMeanL
            newMeanL = 0
            for i in range(interval):
                self.sample_graph() # make a MCMC move
                score = self.GetScore()
                newMeanL += score

                if(self.len<max_len):
                    trace[self.len] = score
                    self.len = self.len+1
                t = t+1

                if t%1000000 == 0:
                    print('1e6 MCMC moves take {}s'.format(time.time()-t0))
                    t0 = time.time()

            self.check[check_num] = np.abs(newMeanL-oldMeanL)/interval
            print('newMean-oldMean:{}'.format(self.check[check_num]))
            check_num = check_num+1
            judge_convergence = self.judge_converge()

            if(judge_convergence and (t>thresh_eq*self.N)):
                convergence = 1
            elif t>thresh_stop*self.N or t>=max_len:
                convergence = 1
