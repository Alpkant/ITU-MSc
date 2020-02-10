############################
# Author: Alperen Kantarcı #
# Last Edit: 31/10/2019    #
############################
import matplotlib.pyplot as plt
import snap as s # Graph library
import scipy.io # To read matlab file
import numpy as np
import argparse

def delete_20_percent(graph):
    for node in ep_graph.Nodes():
        # For every node 20% possiblity to delete it
        # Since my dataset is not really big only 20%  is enough for thresholding
        # After the for loop nearly 20% of the nodes would be deleted
        should_delete = np.random.choice(2, 1, p=[0.8,0.2])[0]
        if should_delete:
            graph.DelNode(node.GetId())
    return graph



parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Path for the Epinion dataset file")
args = parser.parse_args()
if not any(vars(args).values()):
    raise Exception("Please give an argument. Give -h to see all the arguments.")

if args.path == "":
    raise Exception("Path is not given. Please specify the relative path of Epinion dataset txt file.")


# Read the graph
ep_graph = s.LoadEdgeList(s.PUNGraph, args.path, 0, 1)
print("Number of nodes: {}, Number of edges {}".format(ep_graph.GetNodes(),ep_graph.GetEdges()))

# Step 1.2
# Remove 20% of the nodes to use thresholded version
ep_graph = delete_20_percent(ep_graph)
num_node = ep_graph.GetNodes()
num_edge = ep_graph.GetEdges()
print("After removing 20\% of the nodes, Number of nodes: {}, Number of edges {}".format(num_node,num_edge))

# Step 2
# Calculate centralities 
node_ids = []
degree_cent = []
closeness_cent = []
eigen_cent = []

NIdEigenH = s.TIntFltH()
s.GetEigenVectorCentr(ep_graph, NIdEigenH)
print("Please wait for calculation, it takes time.")
# Calculate centralities for each nodes
for id,node in enumerate(ep_graph.Nodes()):
    if id % 1000 == 0:
        print("{} / {}".format(id,num_node))
    node_ids.append(node.GetId())
    degree_cent.append(s.GetDegreeCentr(ep_graph, node.GetId()))
    closeness_cent.append(s.GetClosenessCentr(ep_graph, node.GetId()))
    eigen_cent.append(NIdEigenH[node.GetId()])

# Step 2.1
# Plot distribution of the centralities for each centrality metric
fig = plt.figure(figsize=(32,16))
ax = fig.add_subplot(111)
ax.hist(closeness_cent,bins=100,range=(0,0.25),alpha=1,rwidth=1,label = "Closeness distribution")
ax.hist(degree_cent,bins=100,range=(0,.25),alpha=0.7,rwidth=0.8,label = "Degree distribution")
ax.hist(eigen_cent,bins=100,range=(0,0.25),alpha=0.5,rwidth=0.6,label = "Eigen distribution")
ax.tick_params(axis='x', which='major',labelrotation=-90,labelsize=12)
ax.tick_params(axis='y', which='major',labelsize=14)
#ax.set_xticks(np.arange(0, 0.5, 1e-3))
ax.set_xlabel("Metric value")
ax.set_ylabel("Frequency")
ax.legend(fontsize='xx-large')
plt.show()

# Prepare three different graph for three different resilience attack
# Each attack will attack to most central nodes according to the
# three different centrality measure.
n_degree_cent = np.array(degree_cent)
node_ids_degree = np.array(node_ids.copy())

n_closeness_cent = np.array(closeness_cent)
node_ids_closeness = np.array(node_ids.copy())

n_eigen_cent = np.array(eigen_cent)
node_ids_eigen = np.array(node_ids.copy())

idx_degree = np.argsort(n_degree_cent)
idx_closeness = np.argsort(n_closeness_cent)
idx_eigen = np.argsort(n_eigen_cent)

n_degree_cent = np.array(n_degree_cent)[idx_degree]
node_ids_degree = np.array(node_ids_degree)[idx_degree]
n_closeness_cent = np.array(n_closeness_cent)[idx_closeness]
node_ids_closeness = np.array(node_ids_closeness)[idx_closeness]
n_eigen_cent = np.array(n_eigen_cent)[idx_eigen]
node_ids_eigen = np.array(node_ids_eigen)[idx_eigen]

ep_graph_1 = s.ConvertGraph(type(ep_graph), ep_graph)
ep_graph_2 = s.ConvertGraph(type(ep_graph), ep_graph)
ep_graph_3 = s.ConvertGraph(type(ep_graph), ep_graph)
# Create a list of strongly connected component sizes

degree_components = []
closeness_components = []
eigen_components = []

# At each attack, remove top  5% most central nodes until 60%
num_delete = int(node_ids_degree.shape[0] * 0.05)
# First loop would hold the connected compenent size of original graph
# Therefore loop stop when 60% of the most central nodes removed
for topk in range(0,65,5):
    ComponentDist = s.TIntPrV()
    s.GetSccSzCnt(ep_graph_1, ComponentDist)
    degree_components.append(ComponentDist.Last().GetVal1())

    ComponentDist = s.TIntPrV()
    s.GetSccSzCnt(ep_graph_2, ComponentDist)
    closeness_components.append(ComponentDist.Last().GetVal1())

    ComponentDist = s.TIntPrV()
    s.GetSccSzCnt(ep_graph_3, ComponentDist)
    eigen_components.append(ComponentDist.Last().GetVal1())
    
    
    # Delete top 5 percent for each centralities
    for i in range(num_delete): 
        ep_graph_1.DelNode(int(node_ids_degree[-1-i]))
        ep_graph_2.DelNode(int(node_ids_closeness[-1-i]))
        ep_graph_3.DelNode(int(node_ids_eigen[-1-i]))
    n_degree_cent = n_degree_cent[:-num_delete]
    n_closeness_cent = n_closeness_cent[:-num_delete]
    n_eigen_cent = n_eigen_cent[:-num_delete]
    
    node_ids_degree = node_ids_degree[:-num_delete]
    node_ids_closeness = node_ids_closeness[:-num_delete]
    node_ids_eigen = node_ids_eigen[:-num_delete]


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.tick_params(axis='x', which='major',labelrotation=-90,labelsize=12)
ax.tick_params(axis='y', which='major',labelsize=14)
ax.set_xticks(np.arange(0, 65, 5))
ax.set_yticks(np.arange(0, 60000, 3000))
ax.set_ylim((0,60000))
ax.scatter(range(0,65,5),degree_components,color="r",marker="v",alpha=1,s=48,label= "Degree centrality connected components")
ax.scatter(range(0,65,5),eigen_components,color="c",marker="o",alpha=1,s=56,label= "Eigen centrality connected components")
ax.scatter(range(0,65,5),closeness_components,color="m",marker="*",alpha=1,s=48,label= "Closeness centrality connected components")
    
ax.set_xlabel("Removed Percentage")
ax.set_ylabel("Size of largest strongly connected components")
ax.legend(fontsize='xx-large')
plt.show()