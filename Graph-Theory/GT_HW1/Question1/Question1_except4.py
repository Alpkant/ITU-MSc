############################
# Author: Alperen Kantarcı #
# Last Edit: 28/10/2019    #
############################
import matplotlib.pyplot as plt
import snap as s # Graph library
import scipy.io # To read matlab file
import numpy as np
import argparse


def read_adj_mat(path=""):
    mat = scipy.io.loadmat(path)
    adj_mat = mat['A']
    binarized_mat = np.where(adj_mat>0, 1, 0)
    return  adj_mat , binarized_mat

# Create random binarized matrix with given arguments
def create_random_mat():
    nodes = 100
    edges = 2555
    bin_graph = s.GenRndGnm(s.PUNGraph, nodes, edges)
    adj_mat = np.zeros((nodes,nodes))
    np.fill_diagonal(adj_mat, 1)
    
    for edge in bin_graph.Edges():
        random_weight = np.random.uniform(-0.2,1)
        adj_mat[edge.GetSrcNId(),edge.GetDstNId()] = random_weight 
        adj_mat[edge.GetDstNId(),edge.GetSrcNId()] = random_weight
    return bin_graph, adj_mat

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="random graph else relative path of .mat file of the adjacency matrix")
parser.add_argument("--random", help="True if you want to test on random graph")
args = parser.parse_args()
if not any(vars(args).values()):
    raise Exception("Please give an argument. Give -h to see all the arguments.")
if(args.random):
    bin_graph, adj_mat = create_random_mat() 
    binarized_mat = np.where(adj_mat>0, 1, 0)
else:
    # Step 1
    if args.path == "":
        raise Exception("Path is not given. For random graph please give \"--random True\" argument")
    adj_mat, binarized_mat  = read_adj_mat(args.path)
    num_nodes = binarized_mat.shape[0]
    bin_graph = s.TUNGraph.New()
    [bin_graph.AddNode(i) for i in range(binarized_mat.shape[0])]
    for i in range(binarized_mat.shape[0]):
        for j in range(i+1,binarized_mat.shape[1]):
            if binarized_mat[i,j]:
                bin_graph.AddEdge(i,j)
            
# Double check node and edge numbers
print("Graph: Nodes %d, Edges %d" % (bin_graph.GetNodes(), bin_graph.GetEdges()))
# Visualize the thresholded matrix
c = 'viridis'
fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot(111)
# Plot binarized adcaceny matrix
ax.matshow(binarized_mat,cmap=c)
ax.set_title("Binarized thresholded adjacency matrix")
plt.savefig("Binarized_thresholded_adj.png")
plt.show()

fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot(111)
# Since it doesn't look informative
# I plotted the binarized graph with weights
# So thresholding made on the graph 
cm = ax.matshow(adj_mat*binarized_mat,cmap=c)
fig.colorbar(cm)
ax.set_title("Weights of thresholded adjacency matrix")
plt.savefig("Weighted_thresholded_adj.png")
plt.show()

# Step 2
# Store centralities for each node 
degree_cent = []
closeness_cent = []
eigen_cent = []
# Calculate eigen centralities
NIdEigenH = s.TIntFltH()
s.GetEigenVectorCentr(bin_graph, NIdEigenH)
# Calculate centralities for each nodes
for id,node in enumerate(bin_graph.Nodes()):
    degree_cent.append(s.GetDegreeCentr(bin_graph, node.GetId()))
    closeness_cent.append(s.GetClosenessCentr(bin_graph, node.GetId()))
    eigen_cent.append(NIdEigenH[id])

fig = plt.figure(figsize=(64,32))
ax = fig.add_subplot(111)
indices = np.arange(len(degree_cent))
width = 0.9
ax.bar(indices, closeness_cent,width=width, color='r', alpha=1, label='Closeness Centrality')
ax.bar(indices, degree_cent,width=width, color='g', alpha=1, label='Degree Centrality')
ax.bar(indices, eigen_cent, width=width, color='b', alpha=1, label='Eigen Centrality')
plt.xticks(indices, 
        ['{}'.format(i) for i in range(len(degree_cent))] )
ax.tick_params(axis='x', which='major', labelsize=6)
ax.tick_params(axis='y', which='major', labelsize=12)
ax.set_xlabel("Node")
ax.set_ylabel("Centrality value")
ax.legend(fontsize='large',loc=2)
plt.show()


fig = plt.figure(figsize=(64,32))
ax = fig.add_subplot(111)
ax.hist(closeness_cent,bins="rice",range=(0,1.),alpha=0.7,rwidth=1,label = "Closeness distribution")
ax.hist(degree_cent,bins='rice',range=(0,1.),alpha=0.7,rwidth=0.8,label = "Degree distribution")
ax.hist(eigen_cent,bins='rice',range=(0,1.),alpha=0.7,rwidth=0.6,label = "Eigen distribution")
ax.tick_params(axis='x', which='major',labelrotation=-90,labelsize=12)
ax.tick_params(axis='y', which='major',labelsize=14)
ax.set_xticks(np.arange(0, 1, 1e-2))
ax.set_xlabel("Metric value")
ax.set_ylabel("Frequency")
ax.legend(fontsize='xx-large')
plt.show()
