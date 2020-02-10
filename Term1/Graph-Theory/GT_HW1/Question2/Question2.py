############################
# Author: Alperen Kantarcı #
# Last Edit: 30/10/2019    #
############################
import matplotlib.pyplot as plt
import scipy.io # To read matlab file
import numpy as np
import argparse


def read_inc_mat(path=""):
    mat = scipy.io.loadmat("H.mat")
    inc_mat = mat['A']
    mat = scipy.io.loadmat(path)
    return  inc_mat

# Create random binarized matrix with given arguments
def create_random_mat():
    nodes = 50
    edges = 120
    inc_mat = np.zeros((nodes,edges))
    
    for i in range(edges):
        for j in range(nodes):
            inc_mat[j][i] = np.random.choice(2, 1, p=[0.35,0.65]) 
    return inc_mat

def calculate_hyperedge_weights(inc_mat,weight_type=1):
    weights = []
    # Two types of weight calculation is considered 
    # normalization is different for each weights
    # because max. sum of centralities differ with different weights
    # Weight 1 = constant weight for each edge
    # Weight 2 = multiplicity of a edge is the weight of the edge
    if weight_type == 1: #Constant weights
        weights = [1 for i in range(inc_mat.shape[1])]
        normalizer = inc_mat.shape[0] * (inc_mat.shape[0]-1)

    elif weight_type == 2:  #Frequency based (multiplicity) weights
        for i in range(inc_mat.shape[1]): # For each edge
            weight = 1
             #Calculate multiplicity by comparing different edges with same vertex
            for j in range(inc_mat.shape[1]):
                if i == j: continue
                flag = 0
                for k in range(inc_mat.shape[0]):
                    if inc_mat[k][i] == inc_mat[k][j]:
                        flag+=1
                weight += flag
            weights.append(weight)
    else:
        raise Exception("Wrong type of weight. Please select weight as 1 or 2!")
    normalizer = inc_mat.shape[0]*(inc_mat.shape[0]-1)*inc_mat.shape[1]*(inc_mat.shape[1]-1)
    return weights, normalizer

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="random graph else relative path of .mat file of the adjacency matrix")
parser.add_argument("--random", help="True if you want to test on random graph")
parser.add_argument("--weight", default=2, help="1 for constant weight, 2 for multiplicity weights")

args = parser.parse_args()
if not any(vars(args).values()):
    raise Exception("Please give an argument. Give -h to see all the arguments.")

if(args.random):
    inc_mat = create_random_mat()
else:
    if args.path == "":
        raise Exception("Path is not given. For random graph please give \"--random True\" argument")
    inc_mat  = read_inc_mat(args.path)

num_node = inc_mat.shape[0]
num_hyperedge = inc_mat.shape[1]        
# Double check node and edge numbers
print("Graph: Nodes %d, Edges %d" % (num_node, num_hyperedge))
fig = plt.figure(figsize=(16,32))
ax = fig.add_subplot(111)
# Plot binarized adcaceny matrix
ax.matshow(inc_mat)
ax.set_title("Incidence matrix")
plt.savefig("incidence_matrix.png")
plt.show()

# Calculate edge weights for the chosen weight type
centralities = []
weights, normalizer = calculate_hyperedge_weights(inc_mat,int(args.weight))
for i in range(num_node):
    weight_sum = 0
    for j in range(num_node):
        if i == j: continue
        for k in range(num_hyperedge):
            if(inc_mat[i][k] == 1 and inc_mat[j][k] == 1):
                weight_sum += weights[k] 
    centralities.append(weight_sum/normalizer)

fig = plt.figure(figsize=(64,32))
ax = fig.add_subplot(111)
indices = np.arange(len(centralities))
width = 0.9
ax.bar(indices, centralities,width=width, color='r', alpha=1, label='Weighted Node Degree Centrality')
plt.xticks(indices, 
        ['{}'.format(i) for i in range(len(centralities))] )
ax.tick_params(axis='x', which='major', labelsize=6)
ax.tick_params(axis='y', which='major', labelsize=12)
ax.set_xlabel("Node")
ax.set_ylabel("Centrality value")
ax.legend(fontsize='large',loc=2)
plt.savefig("node_centralities.png")
plt.show()


fig = plt.figure(figsize=(64,32))
ax = fig.add_subplot(111)
ax.hist(centralities,range=(min(centralities),max(centralities)+1e-5),alpha=1,label='Weighted Node Degree Centrality')
ax.tick_params(axis='x', which='major',labelrotation=-90,labelsize=12)
ax.tick_params(axis='y', which='major',labelsize=14)

ax.set_xlabel("Metric value")
ax.set_ylabel("Frequency")
ax.legend(fontsize='xx-large')
plt.savefig("node_centrality_dist.png")
plt.show()
