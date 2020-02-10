############################
# Author: Alperen Kantarcı #
# Last Edit: 28/10/2019    #
############################
import matplotlib.pyplot as plt
import snap as s # Graph library
import scipy.io # To read matlab file
import numpy as np
import argparse


def get_density(edges,nodes):
    return edges / (nodes * (nodes-1))

def findPercolationThreshold(adj_matrix):
    adj_matrix = np.array(adj_matrix)
    # Remove self connections from adj matrix
    np.fill_diagonal(adj_matrix, 0)
    
    # Create an empty graph
    graph = s.TUNGraph.New()
    # Create all nodes but no connections between them
    [graph.AddNode() for i in range(adj_matrix.shape[0])]

    percolation_threshold = 0
    # Add every edge in order of descending weights
    while np.amax(adj_matrix) > 0 :
        result = np.where(adj_matrix == np.amax(adj_matrix))[0]
        # Add the edge with biggest weight
        graph.AddEdge(int(result[0]),int(result[1]))
        # Mark (make weight 0) the biggest weighted edge in adj_matrix
        adj_matrix[result[0]][result[1]] = 0.
        adj_matrix[result[1]][result[0]] = 0.
        # Calculate biggest connected compenent
        fraction_val = s.GetMxWccSz(graph)
        
        if fraction_val > 0.95 and percolation_threshold == 0:
            percolation_threshold = get_density(graph.GetEdges(),graph.GetNodes())

    return percolation_threshold

def generate_connected_component_plot(adj_matrix):
    adj_matrix = np.array(adj_matrix)
    # Remove self connections from adj matrix
    np.fill_diagonal(adj_matrix, 0)
    
    # Create an empty graph
    graph = s.TUNGraph.New()
    # Create all nodes but no connections between them
    [graph.AddNode() for i in range(adj_matrix.shape[0])]
    percolation_percents = []
    connectivity_percents = []
    
    # Add every edge in order of descending weights
    while np.amax(adj_matrix) > 0 :
        result = np.where(adj_matrix == np.amax(adj_matrix))[0]
        # Add the edge with biggest weight
        graph.AddEdge(int(result[0]),int(result[1]))
        # Mark (make weight 0) the biggest weighted edge in adj_matrix
        adj_matrix[result[0]][result[1]] = 0.
        adj_matrix[result[1]][result[0]] = 0.
        # Calculate biggest connected compenent
        fraction_val = s.GetMxWccSz(graph)
        percolation_percents.append(fraction_val)
        connectivity_percents.append(get_density(graph.GetEdges(),graph.GetNodes()))

    plt.plot(connectivity_percents,percolation_percents)
    plt.xlabel("Connection Density")
    plt.ylabel("Proportion of nodes in largest component")
    plt.savefig("connected_component_plot.png")
    plt.show()


def read_adj_mat(path=""):
    mat = scipy.io.loadmat(path)
    adj_mat = mat['A']
    binarized_mat = np.where(adj_mat>0, 1, 0)
    return  adj_mat , binarized_mat

# Create random binarized matrix with given arguments
def create_random_mat():
    nodes = 120
    edges = 2560
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

print("Percolation Threshold = {}".format(findPercolationThreshold(adj_mat)))
generate_connected_component_plot(adj_mat)

