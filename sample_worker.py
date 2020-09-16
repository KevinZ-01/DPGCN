import os
import logging
import numpy as np
import scipy.sparse as sp
import time
import math
import torch
import random
import networkx as nx
from utils import feature_reader, graph_reader, \
                normalize, sparse_mx_to_torch_sparse_tensor, AdjPairs
from paris_utils import paris, plot_best_clusterings, private_paris

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
        self.sensitivity_cluster = 
        self.sensitivity_sample =
        self.runCluster()
        self.runSample()
        self.addLaplacian()
        self.dentoadj()

    def runCluster(self):
        file = self.dataset + '.graphml'
        self.G = nx.read_graphml(file, node_type=int)
        self.n = self.G.number_of_nodes()
        if self.args.non_private == True:
            self.D = paris(self.G)
        else:
            # the cluster graph, the size of each node in dendrogram, the top nodes, the dendrogram
            self.F, self.size, self.top, self.D = private_paris(self.G)

        self.numofcluster = self.n - np.asarray(self.D).shape[0]

        pos_x = nx.get_node_attributes(self.G, 'pos_x')
        pos_y = nx.get_node_attributes(self.G, 'pos_y')
        pos = {u: (pos_x[u], pos_y[u]) for u in self.G.nodes()}
        plot_best_clusterings(self.G, self.D, 4, pos)

    def runSample(self):
        # set two threshold parameters to manually control convergence
        thresh_eq = np.max(self.args.eq, 1000) if self.args.eq else 1000
        thresh_stop = np.max(self.args.stop, 3000) if self.args.stop else 3000
        # parameters
        newMeanL = -1e49
        interval = 65536
        max_len = np.max(1e8, thresh_eq * self.n)
        trace = np.zeros(max_len)
        self.len = 0
        self.check = np.zeros(10000)
        check_num = 0
        self.score = 0

        convergence = 0
        t = 0
        t0 = time.time()
        ## get initial score
        if self.args.utility_sample == 'ave_pro':
            self.E = 0  # total edges
            numerator = 0
            for r in self.D:
                self.E += r[2] * self.size[r[5]]
                numerator += (r[2] * self.D[r][3] * self.D[r][4]) ** 2 / (self.size[r[0]] * self.size[r[1]])
            for (u,v) in self.F.edges():
                numerator += (self.F[u][v]['pro']*self.size[u]*self.size[v]) ** 2 / (self.size[u]*self.size[v])
            self.score = numerator / self.E
        elif self.args.utility_sample == 'log likelihood':
            for r in self.D:
                self.score += -len(r[3]) * len(r[4]) * (r[2] * math.log(r[2]) + (1 - r[2]) * math.log(1 - r[2]))
                # add top nodes log likelihoods
            for (u, v) in self.F.edges():
                if not self.F[u][v]['pro'] == 0:
                    self.score += -(len(self.D[u][3])+len(self.D[u][4]) )* (len(self.D[v][3]+len(self.D[v][4]))) * (
                                self.F[u][v]['pro'] * math.log(self.F[u][v]['pro'])
                                + (1 - self.F[u][v]['pro']) * math.log(1 - self.F[u][v]['pro']))

        while convergence == 0:
            oldMeanL = newMeanL
            newMeanL = 0
            for i in range(interval):
                # choose 2 nodes in the dendrogram to change position
                nodes = random.sample(range(0, 2 * self.n - self.numofcluster), 2)
                # in case there is parent and child relation between the nodes to be changed
                while nodes[0] in self.D[nodes[1]][3] or nodes[0] in self.D[nodes[1]][4] or \
                    nodes[1] in self.D[nodes[0]][3] or nodes[1] in self.D[nodes[0]][3]:
                    nodes = random.sample(range(0, 2 * self.n - self.numofcluster), 2)

                self.exchange(nodes)
                newMeanL += self.score

                if (self.len < max_len):
                    trace[self.len] = self.score
                    self.len = self.len + 1
                t = t + 1

                if t % 1000000 == 0:
                    print('1e6 MCMC moves take {}s'.format(time.time() - t0))
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
        if nodes[0] in self.top and nodes[1] in self.top:
            # top nodes change will not affect the utility function
            return
        elif nodes[0] in self.top:  # one node is top node
            D_copy = self.D
            top_copy = self.top
            F_copy = self.F
            # renew the top list
            top_copy.remove(nodes[0])
            top_copy.append(nodes[1])
            ## renew the dendrogram
            #find all the parents of node[1]
            parents = []
            this_top = 0
            for r in self.D:
                if nodes[1] in r[3] or nodes[1] in r[4]:
                    parents.append(r[5])
                    if r[5] in self.top:
                        this_top = r[5] # record the relative top nodes
                    # update child if it's a direct parent
                    if r[0] == nodes[1]:
                        D_copy[r[5]][0] = nodes[0]
                    elif r[1] == nodes[1]:
                        D_copy[r[5]][1] = nodes[0]
            # renew left and right child nodes
            for p in parents:
                if nodes[1] in self.D[p][3]:
                    left_right = 3
                else:
                    left_right = 4

                D_copy[p][left_right].remove(nodes[1])
                D_copy[p][left_right].remove(D_copy[nodes[1]][3])
                D_copy[p][left_right].remove(D_copy[nodes[1]][4])
                D_copy[p][left_right].append(nodes[0])
                D_copy[p][left_right].append(D_copy[nodes[0]][3])
                D_copy[p][left_right].append(D_copy[nodes[0]][4])
             # renew the probabilities of nodes
            for p in parents:
                pro = self.calc_prob(D_copy, p)
                D_copy[p][2] = pro
            #renew the probabilities between clusters
            F_copy.remove_node(nodes[0])
            F_copy.add_node(nodes[1])
            for node in F_copy.nodes:
                if not node==nodes[1]:
                    pro = self.calc_cluster_prob(D_copy,[node, nodes[1]])
                    F_copy.add_edge(node, nodes[1])
                    F_copy[node][nodes[1]]['pro'] = pro
                if not node==this_top:
                    pro = self.calc_cluster_prob(D_copy,[node, this_top])
                    if not F_copy.has_edge(node, this_top):
                        F_copy.add_edge(node, this_top)
                    F_copy[node][this_top]['pro'] = pro

            #calculate the utility function
            if self.args.utility_sample == 'ave_pro':
                for p in parents:
                    after_score = self.score - \
                                  (self.D[p][2] ** 2 * len(self.D[p][3])*len(self.D[p][4]) +
                                   D_copy[p][2]** 2 * len(D_copy[p][3])*len(D_copy[p][4]))/self.E

                for node in F_copy.nodes():
                    if not node == nodes[1]:
                        after_score += F_copy[node][nodes[1]]['pro'] ** 2 * (len(D_copy[node][3]) + len(D_copy[node][4])) * (len(D_copy[nodes[1]][3]) + len(D_copy[nodes[1]][4])) - \
                                       self.F[node][nodes[0]]['pro'] ** 2 * (len(self.D[node][3])+len(self.D[node][4])) * (len(self.D[nodes[0]][3])+len(self.D[nodes[0]][4]))
                    if not node == this_top:
                        after_score += F_copy[node][this_top]['pro'] ** 2 * (len(D_copy[node][3]) + len(D_copy[node][4])) * (
                                                   len(D_copy[this_top][3]) + len(D_copy[this_top][4])) - \
                                       self.F[node][this_top]['pro'] ** 2 * (len(self.D[node][3]) + len(self.D[node][4])) * (
                                                   len(self.D[this_top][3]) + len(self.D[this_top][4]))

            elif self.args.utility_sample == 'log likelihood':
                after_score = 0
                for p in parents:
                    after_score = self.score + \
                                  len(self.D[p][3])*len(self.D[p][4])* (self.D[p][2] * math.log(self.D[p][2]) + (1 - self.D[p][2]) * math.log(1 - self.D[p][2])) - \
                                  len(D_copy[p][3])*len(D_copy[p][4])* (D_copy[p][2] * math.log(D_copy[p][2]) + (1 - D_copy[p][2]) * math.log(1 - D_copy[p][2]))
                # calculate the between-cluster information entropy that has been changed
                for node in F_copy.nodes():
                    if not node == nodes[1]:
                        after_score += \
                            (len(D_copy[node][3])+len(D_copy[node][4]))*(len(D_copy[nodes[1]][3])+len(D_copy[nodes[1]][4]))*\
                            (F_copy[node][nodes[1]]['pro'] * math.log(F_copy[node][nodes[1]]['pro']) + (1 - F_copy[node][nodes[1]]['pro']) * math.log(1 - F_copy[node][nodes[1]]['pro'])) - \
                            (len(self.D[node][3])+len(self.D[node][4]))*(len(self.D[nodes[1]][3])+len(self.D[nodes[1]][4]))*\
                            (self.F[node][nodes[0]]['pro'] * math.log(self.F[node][nodes[0]]['pro']) + (1 - self.F[node][nodes[0]]['pro']) * math.log(1 - self.F[node][nodes[0]]['pro']))
                    if not node == this_top:
                        after_score += \
                            (len(D_copy[node][3]) + len(D_copy[node][4])) * (len(D_copy[this_top][3]) + len(D_copy[this_top][4])) * \
                            (F_copy[node][this_top]['pro'] * math.log(F_copy[node][this_top]['pro']) + (1 - F_copy[node][this_top]['pro']) *
                             math.log( 1 - F_copy[node][this_top]['pro'])) - \
                            (len(self.D[node][3]) + len(self.D[node][4])) * (len(self.D[this_top][3]) + len(self.D[this_top][4])) * \
                            (self.F[node][this_top]['pro'] * math.log(self.F[node][this_top]['pro']) + (1 - self.F[node][this_top]['pro']) *
                             math.log(1 - self.F[node][this_top]['pro']))
            # probability to exchange these two nodes
            prob = math.exp()
            if random.uniform(0, 1) < prob:
                self.D = D_copy
                self.F = F_copy
                self.top = top_copy
        elif nodes[1] in self.top:
        else: # both the nodes are not top nodes
            a = 2

    def calc_prob(self, D, node):
        """
        :param D: dendrogram
        :param node: nodes to be calculated
        :return: probability
        """
        total = len(D[node][3])*len(D[node][4])
        # calculate edges
        num_edge = 0
        for edge in self.G.edges:
            if (edge[0] in D[node][3] and edge[1] in D[node][4]) or (edge[1] in D[node][3] and edge[0] in D[node][4]):
                num_edge += 1
        return num_edge/total

    def calc_cluster_prob(self, D, nodes):

    def calc_log_likelihood(self, D):
        return np.min(1, )

    def calc_ave_prob(self, D):
        return np.min(1, )

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

    def GetScore(self, Dendrogram):
        if self.args.loss_sample == 'A':
            return -np.sum(np.abs(self.adj - self.realadj))
        elif self.args.loss_sample == 'logA':
            loss_A = np.sum(np.abs(self.adj - self.realadj))
            if loss_A == 0:
                return 1
            else:
                return -math.log(loss_A)
        elif self.args.loss_sample == 'AX':
            self.AHat = self.normalization(self.adj)
            return -np.sum(np.abs(np.matmul(self.AHat, self.features) - self.AHatX))
        elif self.args.loss_sample == 'logAX':
            loss_AX = np.sum(np.abs(np.matmul(self.AHat, self.features) - self.AHatX))
            if loss_AX == 0:
                return 1
            else:
                return -math.log(loss_AX)

    def addLaplacian(self):
        self.D[:][2] = self.D[:][2] + np.random.laplace(scale=self.args.epsilon3, size=np.asarray(self.D).shape[0])

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
